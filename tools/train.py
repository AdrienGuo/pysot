# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import json
import logging
import math
import os
import random
import time

import ipdb
import numpy as np
import torch
import torch.nn as nn
import wandb
from pysot.core.config import cfg
# === 這裡選擇要老師 or 亭儀的裁切出來的資料集 ===
from pysot.datasets.pcbdataset_new import PCBDataset
from pysot.models.model_builder import ModelBuilder
from pysot.utils.average_meter import AverageMeter
from pysot.utils.check_image import create_dir
from pysot.utils.distributed import (DistModule, average_reduce, dist_init,
                                     get_rank, get_world_size,
                                     reduce_gradients)
from pysot.utils.log_helper import add_file_handler, init_log, print_speed
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.misc import commit, describe
from pysot.utils.model_load import load_pretrain, restore_from
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--crop_method', type=str, help='teachr / amy')
parser.add_argument('--bg', type=str, nargs='?', const='', help='background')
parser.add_argument('--anchors', type=int, help='number of anchors')
parser.add_argument('--epoch', type=int, help='epoch')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--dataset', type=str, help='training dataset')
parser.add_argument('--cfg', type=str, default='config.yaml', help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='compulsory for pytorch launcer')
# parser.add_argument('--template_bg', type=str, default='', help='whether crop template with bg')
# parser.add_argument('--template_context_amount', type=int, default='', help='how much bg for template')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_loader(validation_split, random_seed):
    logger.info("build train dataset")
    dataset = PCBDataset(args)
    logger.info("build dataset done")

    # split train & val dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.seed(random_seed)
    random.shuffle(indices)
    split = dataset_size - int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[:split], indices[split:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # train_sampler = None
    # if get_world_size() > 1:
    #     train_sampler = DistributedSampler(dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
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

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    assert len(train_loader.dataset) != 0, "Error, there is no data for training."

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

    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.MODEL_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.MODEL_DIR)

    # logger.info("model\n{}".format(describe(model.module)))
    end = time.time()

    for epoch in range(args.epoch):
        # one epoch
        epoch = epoch + 1
        print(f"epoch: {epoch}")
        logger.info('epoch: {}'.format(epoch))
        # Start training backbone at epoch: n,
        # 他是把 backbone model 的 params 變成 requires_grad=True
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
            # one batch
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
            print(f"cls_lossb: {batch_cls_loss:<6.3f} | loc_loss: {batch_loc_loss:<6.3f} | total_loss: {batch_total_loss:<6.3f}")

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
        print(f"--- epoch duration: {epoch_end - epoch_start} s ---")

        lr_scheduler.step()

        train_loss['cls'] = train_loss['cls'] / len(train_loader)
        train_loss['loc'] = train_loss['loc'] / len(train_loader)
        train_loss['total'] = train_loss['total'] / len(train_loader)
        print("--- Train ---")
        print(f"cls_loss: {train_loss['cls']:<6.3f} | loc_loss: {train_loss['loc']:<6.3f} | total_loss: {train_loss['total']:<6.3f}")

        ##########################################
        # Validating
        ##########################################
        model.eval()
        for idx, data in enumerate(val_loader):
            with torch.no_grad():
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
        print("--- Validation ---")
        print(f"cls_loss: {val_loss['cls']:<6.3f} | loc_loss: {val_loss['loc']:<6.3f} | total_loss: {val_loss['total']:<6.3f}")

        wandb.log({
            "train_cls_loss": train_loss['cls'],
            "train_loc_loss": train_loss['loc'],
            "train_total_loss": train_loss['total'],
            "val_cls_loss": val_loss['cls'],
            "val_loc_loss": val_loss['loc'],
            "val_total_loss": val_loss['total']
        })

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

        ##########################################
        # Save model
        ##########################################
        if get_rank() == 0 and epoch % cfg.TRAIN.SAVE_MODEL_FREQ == 0:
            # save model directory
            save_model_dir = os.path.join(
                cfg.TRAIN.MODEL_DIR,
                # x{}_{}_k{}_e{}_b{}_relu
                f"x{cfg.TRAIN.SEARCH_SIZE}_{args.crop_method}_bg{args.bg}_k{args.anchors}_e{args.epoch}_b{args.batch_size}_norm"
            )
            if not os.path.exists(save_model_dir):
                os.makedirs(save_model_dir)
                print(f"Create new dir: {save_model_dir}")
            # save model path
            model_path = os.path.join(save_model_dir, f"model_e{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }, model_path)
            print(f"Save model to: {model_path}")

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
    train_loader, val_loader = build_loader(cfg.DATASET.VALIDATION_SPLIT, random_seed=42)

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

    # Create save model directory
    model_dir = os.path.join(
        cfg.TRAIN.MODEL_DIR,
        # x{}_{}_k{}_e{}_b{}_relu
        f"x{cfg.TRAIN.SEARCH_SIZE}_{args.crop_method}_bg{args.bg}_k{args.anchors}_e{args.epoch}_b{args.batch_size}_norm"
    )
    create_dir(model_dir)

    # Start training
    train(train_loader, val_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)

    # constants = {
    #     "anchor": cfg.ANCHOR.ANCHOR_NUM,
    #     "score_size": cfg.TRAIN.OUTPUT_SIZE,
    #     "epochs": args.epoch,
    #     "batch_size": args.batch_size,
    #     "lr": cfg.TRAIN.BASE_LR,
    #     "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
    #     "normalized": True
    # }
    # wandb.init(
    #     project="siamrpnpp",
    #     entity="adrien88",
    #     name=f"x{cfg.TRAIN.SEARCH_SIZE}_{args.crop_method}_bg{args.bg}_k{args.anchors}_e{args.epoch}_b{args.batch_size}_norm",
    #     config=constants
    # )

    main()
