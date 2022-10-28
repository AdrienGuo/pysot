# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, annotations, division, print_function,
                        unicode_literals)

import argparse
import os
import random
import time

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from pysot.core.config import cfg
from pysot.datasets.pcbdataset_new import PCBDataset
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.check_image import draw_preds, save_image
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.statistics import overlap_ratio_one
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--model', default='', type=str, help='model of models to eval')
parser.add_argument('--crop_method', type=str, help='teachr / amy')
parser.add_argument('--bg', type=str, nargs='?', const='', help='background')
parser.add_argument('--neg', type=float, help='negative sample ratio')
parser.add_argument('--anchors', type=int, help='number of anchors')
parser.add_argument('--epoch', type=int, help='epoch')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--dataset_path', type=str, help='training dataset path')
parser.add_argument('--dataset_name', type=str, help='training dataset name')
parser.add_argument('--criteria', type=str, help='sample criteria for dataset')
parser.add_argument('--bk', type=str, help='whether use pretrained backbone')
parser.add_argument('--cfg', type=str, default='config.yaml', help='configuration of tracking')
parser.add_argument('--test_dataset', type=str, help='testing dataset path')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='compulsory for pytorch launcer')
args = parser.parse_args()

torch.set_num_threads(1)


def calculate_metrics(pred_scores, pred_boxes, label_boxes):
    """
    Args:
        pred_boxes: (data_num, pred_num, 4) #list
        label_boxes: (data_num, label_num, 4) #list
    """
    assert len(pred_scores) == len(pred_boxes), "length of pred_scores and pred_boxes should be the same"
    assert len(pred_boxes) == len(label_boxes), "length of pred_boxes and label_boxes should be the same"

    tp = list()
    fp = list()
    boxes_num = list()

    lens = len(pred_scores)
    for idx in range(lens):
        # 一個 data
        tp_one = torch.zeros(len(pred_boxes[idx]))
        fp_one = torch.zeros(len(pred_boxes[idx]))
        boxes_one_num = len(label_boxes[idx])
        for pred_idx in range(len(pred_boxes[idx])):
            best_iou = 0
            for label_idx in range(len(label_boxes[idx])):
                iou = overlap_ratio_one(pred_boxes[idx][pred_idx], label_boxes[idx][label_idx])
                if iou > best_iou:
                    best_iou = iou
            # 所有預測出來的 pred_boxes，他們都已經是 positive 了；而且不是 true (tp) 就是 false (fp)
            if best_iou >= 0.5:
                tp_one[pred_idx] = 1
            else:
                fp_one[pred_idx] = 1

        tp_one_sum = sum(tp_one)
        fp_one_sum = sum(fp_one)

        tp.append(tp_one_sum)
        fp.append(fp_one_sum)
        boxes_num.append(boxes_one_num)

    if (sum(tp) + sum(fp) == 0):
        precision = 0
    else:
        precision = sum(tp) / (sum(tp) + sum(fp))
    recall = sum(tp) / sum(boxes_num)

    return precision, recall


def evaluate(test_loader, tracker):
    pred_scores = list()
    pred_boxes = list()
    pred_classes = list()
    label_boxes = list()
    label_classes = list()

    clocks = 0
    period = 0
    idx = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            img_path = data['img_path'][0]
            z_box = data['z_box'][0]    # 不要 batch
            z_img = data['z_img'].cuda()
            x_img = data['x_img'].cuda()
            gt_boxes = data['gt_boxes'][0]    # 用在算 precision, recall
            scale = data['scale'][0].cpu().numpy()
            spatium = [x.cpu().item() for x in data['spatium']]
            # cls = [torch.from_numpy(cls).cuda() for cls in data['cls']]

            # print(f"Load image from: {img_path}")
            image = cv2.imread(img_path)

            ######################################
            # 調整 z_box, gt_boxes
            # (x1, y1, x2, y2) -> (x1, y1, w, h)
            ######################################
            z_box = z_box.cpu().numpy().squeeze()
            z_box[2] = z_box[2] - z_box[0]
            z_box[3] = z_box[3] - z_box[1]
            # z_box = np.around(z_box, decimals=2)

            gt_boxes = gt_boxes.cpu().numpy()
            gt_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
            gt_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]

            ######################################
            # Init tracker
            ######################################
            # tic = cv2.getTickCount()
            start = time.time()

            # 用 template image 將 tracker 初始化
            # z_crop = tracker.init(image, gt_box)
            _ = tracker.init(z_img, z_box)

            ######################################
            # Do tracking
            ######################################
            # 用 search image 進行 "track" 的動作
            outputs = tracker.track(image, x_img, scale, spatium)

            # toc = cv2.getTickCount()
            # clocks += toc - tic    # 總共有多少個 clocks (clock cycles)

            end = time.time()
            period += end - start

            pred_scores.append(outputs['top_scores'])
            pred_boxes.append(outputs['pred_boxes'])
            label_boxes.append(gt_boxes.tolist())    # gt_boxes[0], 不要 batch

            # calculate_metrics([outputs['top_scores']], [outputs['pred_boxes']], [gt_boxes[0].cpu().tolist()])
        precision, recall = calculate_metrics(pred_scores, pred_boxes, label_boxes)
        print(f"precision: {precision * 100}")
        print(f"recall: {recall * 100}")

        # period = clocks / cv2.getTickFrequency()
        fps = (idx + 1) / period
        print(f"Speed: {fps} fps")

        return {
            "precision": precision,
            "recall": recall,
            "fps": fps
        }


if __name__ == "__main__":
    cfg.merge_from_file(args.cfg)        # 不加 ModelBuilder() 會出問題ㄟ??

    test_dataset = PCBDataset(args, "eval")
    print(f"Dataset number: {len(test_dataset)}")

    # train_dataset = PCBDataset(args, "train")
    # val_dataset = PCBDataset(args, "val")

    # split train & val dataset
    # dataset_size = len(train_dataset)
    # indices = list(range(dataset_size))
    # random.seed(42)
    # random.shuffle(indices)
    # split = dataset_size - int(np.floor(0.1 * dataset_size))
    # train_indices, val_indices = indices[:split], indices[split:]
    # train_dataset = Subset(train_dataset, train_indices)
    # val_dataset = Subset(val_dataset, val_indices)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     num_workers=0
    # )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,    # 只能設 1
        num_workers=0
    )

    # Create model
    model = ModelBuilder()
    # Load model
    model = load_pretrain(model, args.model).cuda().eval()

    # Build tracker
    tracker = build_tracker(model)

    # evaluate(test_loader, tracker)
    print("Start evaluating...")
    metrics = evaluate(test_loader, tracker)

    print("=" * 20, "Done!", "=" * 20, "\n")
