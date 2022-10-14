# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, annotations, division, print_function,
                        unicode_literals)

import argparse
import os
import re
from unicodedata import decimal

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from pysot.core.config import cfg
# === 這裡選擇要老師 or 亭儀的裁切出來的資料集 ===
from pysot.datasets.pcbdataset_new import PCBDataset
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.check_image import (create_dir, draw_box, draw_preds,
                                     save_image)
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_float2str, vot_overlap
from torch.utils.data import DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--model', default='', type=str, help='model of models to eval')
parser.add_argument('--crop_method', default='', type=str, help='teacher / amy')
parser.add_argument('--bg', type=str, nargs='?', const='', help='background')
parser.add_argument('--neg', type=float, help='negative sample ratio')
parser.add_argument('--dataset_name', type=str, help='datasets name')
parser.add_argument('--dataset_path', type=str, help='datasets path')
parser.add_argument('--criteria', type=str, help='sample criteria for dataset')
parser.add_argument('--save_dir', type=str, help='save to which directory')
parser.add_argument('--config', default='', type=str, help='config file')
args = parser.parse_args()

torch.set_num_threads(1)


def test(test_loader, tracker, dir):
    clocks = 0
    for idx, data in enumerate(test_loader):
        # only one data in a batch (batch_size=1)
        img_path = data['img_path'][0]
        img_name = data['img_name'][0]
        z_box = data['z_box'][0]
        # origin_template_box = data['origin_template_box'][0]
        z_img = data['z_img'].cuda()
        x_img = data['x_img'].cuda()
        gt_boxes = data['gt_boxes'][0]
        scale = data['scale'][0].cpu().numpy()
        spatium = [x.cpu().item() for x in data['spatium']]

        ##########################################
        # Load image
        ##########################################
        print(f"Load image from: {img_path}")
        image = cv2.imread(img_path)

        ##########################################
        # Creat directories
        ##########################################
        # 用圖片檔名當作 sub_dir 的名稱
        # img_name = img_path.split('/')[-1]
        sub_dir = os.path.join(dir, img_name)
        create_dir(sub_dir)

        # --- 創 sub_dir/origin，裡面存 original image ---
        origin_dir = os.path.join(sub_dir, "origin")
        create_dir(origin_dir)
        # --- 創 sub_dir/search，裡面存 search image ---
        search_dir = os.path.join(sub_dir, "search")
        create_dir(search_dir)
        # 創 sub_dir/template，裡面存 template image
        z_dir = os.path.join(sub_dir, "template")
        create_dir(z_dir)
        # >>>>>>>>>>>>>>>>>>
        # 創 sub_dir/pred_annotation，裡面存 pred_annotation
        anno_dir = os.path.join(sub_dir, "pred_annotation")
        create_dir(anno_dir)
        # 創 sub_dir/origin_pred_annotation，裡面存 origin_pred_annotation
        # origin_anno_dir = os.path.join(sub_dir, "origin_pred_annotation")
        # create_dir(origin_anno_dir)
        # <<<<<<<<<<<<<<<<<<
        # >>>>>>>>>>>>>>>>>>
        # 創 sub_dir/pred，裡面存 pred image
        pred_dir = os.path.join(sub_dir, "pred")
        create_dir(pred_dir)
        # 創 sub_dir/origin_pred，裡面存 original pred image
        # origin_pred_dir = os.path.join(sub_dir, "origin_pred")
        # create_dir(origin_pred_dir)
        # <<<<<<<<<<<<<<<<<<

        ##########################################
        # Save original image
        ##########################################
        origin_path = os.path.join(origin_dir, f"{img_name}.jpg")
        save_image(image, origin_path)

        ##########################################
        # Save search image
        # 為什麼不要下面做完 track 在存就好了勒 ??
        ##########################################
        search_image_cpu = x_img[0].cpu().numpy().copy()
        search_image_cpu = search_image_cpu.transpose(1, 2, 0)      # (3, 255, 255) -> (255, 255, 3)
        search_path = os.path.join(search_dir, f"{idx}.jpg")
        save_image(search_image_cpu, search_path)

        ##########################################
        # pred_boxes, scores
        ##########################################
        pred_boxes = []
        origin_pred_boxes = []
        scores = None
        # z_box: (1, [x1, y1, x2, y2]) -> ([x1, y1, x2, y2])
        z_box = z_box.cpu().numpy().squeeze()
        # origin_template_box = origin_template_box.cpu().numpy().squeeze()
        gt_boxes = gt_boxes.cpu().numpy()    # gt_boxes: (n, 4) #corner

        # z_box: (x1, y1, x2, y2) -> (x1, y1, w, h)
        z_box[2] = z_box[2] - z_box[0]
        z_box[3] = z_box[3] - z_box[1]
        z_box = np.around(z_box, decimals=2)
        pred_boxes.append(z_box)

        gt_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]

        # origin_template_box: (x1, y1, x2, y2) -> (x1, y1, w, h)
        # origin_template_box = [
        #     origin_template_box[0],
        #     origin_template_box[1],
        #     origin_template_box[2] - origin_template_box[0],
        #     origin_template_box[3] - origin_template_box[1]
        # ]
        # origin_template_box = np.around(origin_template_box, decimals=2)
        # origin_pred_boxes.append(origin_template_box)

        ##########################################
        # Init tracker
        # Save template image to ./results/{dataset_name}/{img_name}/template/{idx}.jpg
        ##########################################
        tic = cv2.getTickCount()

        # 用 template image 將 tracker 初始化
        # 其實這裡可以不用回傳 z_img，因為 z_img 在裡面完全不會變
        with torch.no_grad():
            z_img = tracker.init(z_img, z_box)

        ##########################################
        # tracking
        ##########################################
        # 用 search image 進行 "track" 的動作
        with torch.no_grad():
            outputs = tracker.track(image, x_img, scale, spatium)

        toc = cv2.getTickCount()
        clocks += toc - tic    # 總共有多少個 clocks (clock cycles)

        z_img = np.transpose(z_img, (1, 2, 0))    # (3, 127, 127) -> (127, 127, 3)
        z_path = os.path.join(z_dir, f"{idx}.jpg")
        save_image(z_img, z_path)

        scores = np.around(outputs['top_scores'], decimals=2)
        # === pred_boxes on "search" image ===
        for box in outputs['pred_boxes']:
            box = np.around(box, decimals=2)
            pred_boxes.append(box)
        # === pred_boxes on "original" image ===
        for origin_box in outputs['origin_pred_boxes']:
            origin_box = np.around(origin_box, decimals=2)
            origin_pred_boxes.append(origin_box)

        # Save search image
        x_img = outputs['x_img']
        x_img = np.transpose(x_img, (1, 2, 0))
        # x_path = os.path.join(x_dir, f"{idx}.jpg")
        # save_image(x_img, x_path)
        # print(f"save x_img image to: {x_path}")

        ##########################################
        # Save annotation file
        ##########################################
        # === pred_boxes on "search" image ===
        anno_path = os.path.join(anno_dir, f"{idx}.txt")
        with open(anno_path, 'w') as f:
            f.write(', '.join(map(str, pred_boxes[0])) + '\n')    # template
            for i, x in enumerate(pred_boxes[1:]):
                # format: [x1, y1, w, h, score]
                f.write(', '.join(map(str, x)) + ', ' + str(scores[i]) + '\n')
        print(f"Save annotation result to: {anno_path}")
        # === pred_boxes on "original" image ===
        # origin_anno_path = os.path.join(origin_anno_dir, f"{idx}.txt")
        # with open(origin_anno_path, 'w') as f:
        #     f.write(', '.join(map(str, origin_pred_boxes[0])) + '\n')    # template
        #     for i, x in enumerate(origin_pred_boxes[1:]):
        #         # format: [x1, y1, w, h, score]
        #         f.write(', '.join(map(str, x)) + ', ' + str(scores[i]) + '\n')
        # print(f"save origin annotation result to: {origin_anno_path}")

        ##########################################
        # draw the gt boxes
        ##########################################
        # === gt_boxes on "search" image ===
        x_img = draw_box(x_img, gt_boxes, type="gt")

        ##########################################
        # draw the pred boxes
        ##########################################
        # === pred_boxes on "search" image ===
        pred_path = os.path.join(pred_dir, f"{idx}.jpg")
        pred_image = draw_preds(sub_dir, x_img, scores, anno_path, idx)
        if pred_image is None:    # 如果沒偵測到物件，存 search image
            save_image(x_img, pred_path)
        else:
            save_image(pred_image, pred_path)
        # === pred_boxes on "original" image ===
        # origin_pred_path = os.path.join(origin_pred_dir, f"{idx}.jpg")
        # origin_pred_image = draw_preds(sub_dir, image, scores, origin_anno_path, idx)
        # if origin_pred_image is None:      # 如果沒偵測到物件，存 original image
        #     save_image(image, origin_pred_path)
        # else:
        #     save_image(origin_pred_image, origin_pred_path)

        print("=" * 20)

        # ipdb.set_trace()

    period = clocks / cv2.getTickFrequency()
    fps = idx / period
    print(f"Speed: {fps} fps")


if __name__ == "__main__":
    test_dataset = PCBDataset(args, "eval")
    print(f"Test dataset number: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset,
                             batch_size=1,    # 只能設 1
                             num_workers=0)

    cfg.merge_from_file(args.config)    # 不加 ModelBuilder() 會出問題ㄟ??

    # Create model
    model = ModelBuilder()
    # Load model
    model = load_pretrain(model, args.model).cuda().eval()
    print(f"Load model from: {args.model}")

    # model_name = args.model.split("/")[2].split(".")[0]
    model_name = args.model.split('/')[-2]
    print(f"model_name: {model_name}")

    dir = os.path.join(args.save_dir, model_name)
    create_dir(dir)
    print(f"Test results saved in: {dir}")

    # Build tracker
    tracker = build_tracker(model)

    test(test_loader, tracker, dir)

    print("=" * 20, "Done!", "=" * 20, "\n")
