# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, annotations, division, print_function,
                        unicode_literals)

import argparse
import os

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from toolkit.utils.statistics import overlap_ratio_one
from pysot.utils.check_image import draw_preds, save_image
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from pysot.core.config import cfg
from pysot.datasets.pcbdataset_test import PCBDatasetTest
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_float2str, vot_overlap
from torch.utils.data import DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--model', default='', type=str, help='model of models to eval')
parser.add_argument('--template_bg', type=str, help='whether crop template with bg')
parser.add_argument('--template_context_amount', type=float, help='how much bg for template')
parser.add_argument('--config', default='', type=str, help='config file')
parser.add_argument('--dataset', type=str, help='datasets')
parser.add_argument('--annotation', type=str, help='annotation for testing')
args = parser.parse_args()

torch.set_num_threads(1)


def calculate_metrics(pred_scores, pred_boxes, label_boxes):
    """
    Args:
        pred_boxes: (data_num, pred_num, 4)
        label_boxes: (data_num, label_num, 4)
    """
    assert len(pred_scores) == len(pred_boxes), "length of pred_scores and pred_boxes should be the same"
    assert len(pred_boxes) == len(label_boxes), "length of pred_boxes and label_boxes should be the same"

    tp_total = list()
    fp_total = list()
    total_boxes_num = 0

    lens = len(pred_scores)
    for idx in range(lens):
        tp = torch.zeros(len(pred_boxes[idx]))
        fp = torch.zeros(len(pred_boxes[idx]))
        label_boxes_num = len(label_boxes[idx])
        for pred_idx in range(len(pred_boxes[idx])):
            best_iou = 0
            for label_idx in range(len(label_boxes[idx])):
                iou = overlap_ratio_one(pred_boxes[idx][pred_idx], label_boxes[idx][label_idx])
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= 0.5:
                tp[pred_idx] = 1
            else:
                fp[pred_idx] = 1

        tp_sum = sum(tp)
        fp_sum = sum(fp)

        tp_total.append(tp_sum)
        fp_total.append(fp_sum)
        total_boxes_num = total_boxes_num + label_boxes_num

    recall = sum(tp_total) / total_boxes_num
    precision = sum(tp_total) / (sum(tp_total) + sum(fp_total))
    print(f"recall: {recall}")
    print(f"precision: {precision}")


def evaluate(test_loader, tracker):
    pred_scores = list()
    pred_boxes = list()
    pred_classes = list()
    label_boxes = list()
    label_classes = list

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            image_path = data['image_path'][0]
            template = data['template'][0]
            template_image = data['template_image'].cuda()
            search_image = data['search_image'].cuda()
            gt_boxes = data['gt_boxes'].cuda()    # 用在算 precision, recall
            scale_ratio = data['r'][0].cpu().numpy()
            # cls = [torch.from_numpy(cls).cuda() for cls in data['cls']]
            # search = [torch.from_numpy(search).cuda() for search in data['search']]

            print(f"load image from: {image_path}")
            image = cv2.imread(image_path)

            ####################################################################
            # pred_boxes, scores
            ####################################################################
            template_box = template.cpu().numpy().squeeze()

            # template_box: (x1, y1, x2, y2) -> (x1, y1, w, h)
            template_box = [
                template_box[0],
                template_box[1],
                template_box[2] - template_box[0],
                template_box[3] - template_box[1]
            ]
            template_box = np.around(template_box, decimals=2)
            # pred_boxes.append(template_box)

            ####################################################################
            # init tracker
            # save template image to ./results/images/{image_name}/template/{idx}.jpg
            ####################################################################
            # 用 template image 將 tracker 初始化
            # z_crop = tracker.init(image, gt_box)
            _ = tracker.init(template_image, template_box)

            ####################################################################
            # tracking
            ####################################################################
            # 用 search image 進行 "track" 的動作
            outputs = tracker.track(image, search_image, scale_ratio)

            pred_scores.append(outputs['top_scores'])
            pred_boxes.append(outputs['pred_boxes'])
            label_boxes.append(gt_boxes[0].cpu().tolist())    # gt_boxes[0] 不要 batch

            # calculate_metrics([outputs['top_scores']], [outputs['pred_boxes']], [gt_boxes[0].cpu().tolist()])
        calculate_metrics(pred_scores, pred_boxes, label_boxes)


if __name__ == "__main__":
    test_dataset = PCBDatasetTest(args)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,    # 只能設 1
                             num_workers=0)

    cfg.merge_from_file(args.config)        # 不加 ModelBuilder() 會出問題ㄟ??

    # create model
    model = ModelBuilder()
    # load model
    model = load_pretrain(model, args.model).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    # dir = os.path.join(args.save_dir, model_name)
    # if not os.path.isdir(dir):
    #     os.makedirs(dir)

    evaluate(test_loader, tracker)

    print("=" * 20, "Done!", "=" * 20, "\n")
