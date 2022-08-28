# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from ast import arg
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.pcbdataset import PCBDataset
from pysot.core.config import cfg

import ipdb
import wandb

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml', help='configuration of tracking')
parser.add_argument('--epoch', type=int, help='epoch')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--template_bg', type=str, help='whether crop template with bg')
parser.add_argument('--template_context_amount', type=int, help='how much bg for template')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='compulsory for pytorch launcer')
args = parser.parse_args()

DEBUG = cfg.DEBUG


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader(validation_split):
    logger.info("build train dataset")
    dataset = PCBDataset(args)
    logger.info("build dataset done")

    # split train & val dataset
    dataset_size = len(dataset)
    split = dataset_size - int(np.floor(validation_split * dataset_size))
    indices = list(range(dataset_size))
    train_indices, val_indices = indices[:split], indices[split:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # train_sampler = None
    # if get_world_size() > 1:
    #     train_sampler = DistributedSampler(dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # collate_fn=train_dataset.collate_fn,      # 參考: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/train.py#L72
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        # sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_opt_lr(model, current_epoch=0):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    print(f"current_epoch: {current_epoch}")

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.rpn_head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    if cfg.MASK.MASK:
        trainable_params += [{'params': model.mask_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    if cfg.REFINE.REFINE:
        trainable_params += [{'params': model.refine_head.parameters(),
                              'lr': cfg.TRAIN.LR.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=args.epoch)
    # lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    # lr_scheduler.step()         # https://github.com/allenai/allennlp/issues/3922
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)


def train(train_loader, val_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // args.epoch // (args.batch_size * world_size)
    print(f"train_loader.dataset number: {len(train_loader.dataset)}")

    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.MODEL_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.MODEL_DIR)

    # logger.info("model\n{}".format(describe(model.module)))
    end = time.time()

    # 改成訓練多個 epoch (原版只訓練一個 epoch)
    for epoch in range(args.epoch):
        epoch = epoch + 1
        print(f"epoch: {epoch}")
        logger.info('epoch: {}'.format(epoch))
        # start training backbone at epoch=10, 他是把 backbone model 的 params 變成 requires_grad=True
        if cfg.BACKBONE.TRAIN_EPOCH == epoch:
            logger.info('start training backbone.')
            optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
        cur_lr = lr_scheduler.get_cur_lr()

        train_loss = dict(cls=0, loc=0, total=0)
        val_loss = dict(cls=0, loc=0, total=0)
        epoch_start = time.time()

        ####################################################################
        # Training
        ####################################################################
        model.train()
        for idx, data in enumerate(train_loader):
            tb_idx = idx
            # if idx % num_per_epoch == 0 and idx != 0:
            #     for idx, pg in enumerate(optimizer.param_groups):
            #         logger.info('epoch {} lr {}'.format(epoch, pg['lr']))
            #         if rank == 0:
            #             tb_writer.add_scalar('lr/group{}'.format(idx+1),
            #                                 pg['lr'], tb_idx)

            data_time = average_reduce(time.time() - end)
            # if rank == 0:
            #     tb_writer.add_scalar('time/data', data_time, tb_idx)

            # Forwarding
            outputs = model(data)

            batch_cls_loss = outputs['cls_loss']
            batch_loc_loss = outputs['loc_loss']
            batch_total_loss = outputs['total_loss']
            print(f"cls_loss: {batch_cls_loss:<6.3f} | loc_loss: {batch_loc_loss:<6.3f} | total_loss: {batch_total_loss:<6.3f}")

            train_loss['cls'] += batch_cls_loss
            train_loss['loc'] += batch_loc_loss
            train_loss['total'] += batch_total_loss

            if is_valid_number(batch_total_loss.data.item()):
                optimizer.zero_grad()
                batch_total_loss.backward()
                reduce_gradients(model)

                if rank == 0 and cfg.TRAIN.LOG_GRADS:
                    log_grads(model.module, tb_writer, tb_idx)

                # clip gradient
                clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                optimizer.step()

        epoch_end = time.time()
        print(f"=== epoch duration: {epoch_end - epoch_start} s ===")

        lr_scheduler.step()

        train_loss['cls'] = train_loss['cls'] / len(train_loader)
        train_loss['loc'] = train_loss['loc'] / len(train_loader)
        train_loss['total'] = train_loss['total'] / len(train_loader)
        print("Train")
        print(f"cls_loss: {train_loss['cls']:<6.3f} | loc_loss: {train_loss['loc']:<6.3f} | total_loss: {train_loss['total']:<6.3f}")

        # wandb.log({
        #     "train_cls_loss": train_loss['cls'],
        #     "train_loc_loss": train_loss['loc'],
        #     "train_total_loss": train_loss['total']
        # })

        ####################################################################
        # Validating
        ####################################################################
        model.eval()
        for idx, data in enumerate(val_loader):
            outputs = model(data)

            batch_cls_loss = outputs['cls_loss']
            batch_loc_loss = outputs['loc_loss']
            batch_total_loss = outputs['total_loss']
            val_loss['cls'] += batch_cls_loss
            val_loss['loc'] += batch_loc_loss
            val_loss['total'] += batch_total_loss

        val_loss['cls'] = val_loss['cls'] / len(val_loader)
        val_loss['loc'] = val_loss['loc'] / len(val_loader)
        val_loss['total'] = val_loss['total'] / len(val_loader)
        print("Validation")
        print(f"cls_loss: {val_loss['cls']:<6.3f} | loc_loss: {val_loss['loc']:<6.3f} | total_loss: {val_loss['total']:<6.3f}")

        # wandb.log({
        #     "val_cls_loss": val_loss['cls'],
        #     "val_loc_loss": val_loss['loc'],
        #     "val_total_loss": val_loss['total']
        # })

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.data.item())

        average_meter.update(**batch_info)

        if rank == 0:
            # for k, v in batch_info.items():
            #     tb_writer.add_scalar(k, v, tb_idx)
            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                            epoch, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch,
                            average_meter.batch_time.avg,
                            args.epoch * num_per_epoch)

        # save model
        if get_rank() == 0 and epoch % cfg.TRAIN.SAVE_MODEL_FREQ == 0:
            # save model directory
            save_model_dir = os.path.join(
                cfg.TRAIN.MODEL_DIR,
                # k{}_r{}_e{}_b{}_{bg0.5}
                f"k{cfg.ANCHOR.ANCHOR_NUM}_r{cfg.TRAIN.SEARCH_SIZE}_e{args.epoch}_b{args.batch_size}_{args.template_bg}{args.template_context_amount}"
            )
            if not os.path.exists(save_model_dir):
                os.makedirs(save_model_dir)
                print(f"create new model dir: {save_model_dir}")
            # save model path
            save_model_path = os.path.join(
                save_model_dir,
                f"model_e{epoch}.pth"
            )
            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                save_model_path
            )
            print(f"save model to: {save_model_path}")
        end = time.time()


def main():
    rank, world_size = dist_init()
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        # logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # build dataset loader
    train_loader, val_loader = build_data_loader(cfg.DATASET.VALIDATION_SPLIT)

    # create model
    model = ModelBuilder().cuda()

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)

    # # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model,
                                           cfg.TRAIN.START_EPOCH)

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)
    dist_model = DistModule(model)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, val_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)

    # constants = {
    #     "anchor": cfg.ANCHOR.ANCHOR_NUM,
    #     "score_size": cfg.TRAIN.OUTPUT_SIZE,
    #     "epochs": args.epoch,
    #     "batch_size": args.batch_size,
    #     "lr": cfg.TRAIN.BASE_LR,
    #     "weight_decay": cfg.TRAIN.WEIGHT_DECAY
    # }
    # wandb.init(
    #     project="siamrpnpp",
    #     entity="adrien88",
    #     name=f"e{args.epoch}-b{args.batch_size}-{args.template_bg}{args.template_context_amount}",
    #     config=constants
    # )

    main()
