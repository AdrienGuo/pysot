# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import logging
import os
import sys
import time

import cv2
import numpy as np
from pysot.pysot.datasets.pcb_crop_amy import PCBCropAmy
import torch
from pysot.core.config import cfg
# from pysot.datasets.anchor_target import AnchorTarget
from pysot.pysot.datasets.pcb_crop import PCBCrop
from pysot.utils.bbox import Center, Corner, center2corner
from pysot.utils.check_image import draw_box, save_image
# from torch.utils.data import Dataset

logger = logging.getLogger("global")

import ipdb
# import pysot.utils.check_image as check_image
# from PIL import Image
# from pysot.datasets.crop_image import crop_like_SiamFC
# from torch.utils.data import DataLoader
# from torchvision import transforms

DEBUG = cfg.DEBUG

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class PCBDataset():
    def __init__(self, args) -> None:
        """ 這裡就是負責讀取資料
        """
        super(PCBDataset, self).__init__()

        data_cfg = getattr(cfg.DATASET, "CUSTOM")
        
        self.root = data_cfg.ROOT
        self.anno = data_cfg.ANNO
        images, template, search = self._make_dataset(self.root)
        self.images = images
        self.template = template
        self.search = search
        self.max_num_box = self._find_max_num_box(self.search)    # targets 最多的數量

        # data PCBCrop (preprocess)
        self.template_crop = PCBCropAmy(
            template_size=cfg.TRAIN.EXEMPLAR_SIZE,
            search_size=cfg.TRAIN.SEARCH_SIZE,
            type="template"
        )
        self.search_crop = PCBCropAmy(
            template_size=cfg.TRAIN.EXEMPLAR_SIZE,
            search_size=cfg.TRAIN.SEARCH_SIZE,
            type="search"
        )

        self.template_bg = args.template_bg
        self.template_context_amount = args.template_context_amount

        # self.search_aug = Augmentation(
        #         cfg.DATASET.SEARCH.SHIFT,
        #         cfg.DATASET.SEARCH.SCALE,
        #         cfg.DATASET.SEARCH.BLUR,
        #         cfg.DATASET.SEARCH.FLIP,
        #         cfg.DATASET.SEARCH.COLOR
        #     )

    def _make_dataset(self, dir_path):
        """ 回傳資料
        Return:
            images (list): [ [path, 類別], ...]
            template (list): 
            search (list): 
        """
        images = []
        search = []         # search image 上 bbox 的座標
        template = []       # template 的意思

        directory = os.path.expanduser(dir_path)
        for root, _, files in os.walk(directory, followlinks=True):
            files = sorted(files)               # 將 files 排序
            for file in files:
                box = []
                if file.endswith(('.jpg', '.png', 'bmp')):
                    path = os.path.join(root, file)
                    anno_path = os.path.join(self.anno, file[:-3] + "txt")        # 改成 .txt (annotation 檔案)
                    # 不是 text 的類型
                    if os.path.isfile(anno_path):
                        f = open(anno_path, 'r')
                        lines = f.readlines()
                        anno = []
                        for line in lines:
                            line = line.strip('\n')
                            line = line.split(' ')
                            line = list(map(lambda x: float(x), line))
                            anno.append(line)

                        for i in range(len(anno)):
                            if anno[i][0] != 26:                  # 如果是電子元件類型，第26類為文字要忽略
                                item = path, anno[i][0]
                                images.append(item)
                                template.append([anno[i][1], anno[i][2], anno[i][3], anno[i][4]])
                                box = []
                            
                            if anno[i][0] != 26:
                                for j in range(len(anno)):              # 這裡應該可以直接改成用 filter 來做
                                    if anno[j][0] == anno[i][0]:
                                        box.append([anno[j][1], anno[j][2], anno[j][3], anno[j][4]])
                                box = np.stack(box).astype(np.float32)
                                search.append(box)
                    # text 類型
                    else:
                        anno_path = os.path.join(self.anno, file[:-3] + "label")
                        assert os.path.isfile(anno_path), f"{anno_path} does not exist!!"
                        f = open(anno_path, 'r')
                        img = cv2.imread(path)
                        imh, imw = img.shape[:2]
                        lines = f.readlines()
                        anno = []
                        for line in lines:
                            line = line.strip('\n')
                            line = line.split(',')
                            line = list(line)
                            anno.append(line)
                        
                        for i in range(len(anno)):
                            if (float(anno[i][1]) > 0) and (float(anno[i][2]) > 0):
                                item = path, anno[i][0]
                                images.append(item)
                                cx = float(anno[i][1]) + (float(anno[i][3]) - float(anno[i][1]))/2
                                cy = float(anno[i][2]) + (float(anno[i][4]) - float(anno[i][2]))/2
                                w = float(anno[i][3]) - float(anno[i][1])
                                h = float(anno[i][4]) - float(anno[i][2])
                                template.append([cx/imw, cy/imh, w/imw, h/imh])
                                box = []
                                for j in range(len(anno)):
                                    if anno[j][0] == anno[i][0]:
                                        cx = float(anno[i][1]) + (float(anno[i][3]) - float(anno[i][1]))/2
                                        cy = float(anno[i][2]) + (float(anno[i][4]) - float(anno[i][2]))/2
                                        w = float(anno[i][3]) - float(anno[i][1])
                                        h = float(anno[i][4]) - float(anno[i][2])

                                        box.append([cx/imw, cy/imh, w/imw, h/imh])
                                box = np.stack(box).astype(np.float32)
                                search.append(box)
        return images, template, search

    def _find_max_num_box(self, boxes):
        num_boxes = list(map(lambda x: x.shape[0], boxes))
        max_num_boxes = max(num_boxes)
        return max_num_boxes
    
    def get_image_anno(self, idx, arg):
        """
        Return:
            imgage_path: 
            image_anno: 
        """
        image_path, template_cls = self.images[idx]
        image_anno = arg[idx]
        return image_path, image_anno

    def get_positive_pair(self, idx):
        return self.get_image_anno(idx, self.template), \
               self.get_image_anno(idx, self.search)
    
    def get_neg_pair(self, type, idx=None):
        if type == "template":
            return self.get_image_anno(idx, self.template)
        elif type == "search":
            idx = np.random.randint(low=0, high=len(self.images))
            return self.get_image_anno(idx, self.search)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        logger.debug("__getitem__")

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()

        # 加入 neg 的原因要去看 [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        # 這個 if 還需要修改成 template 和 search "一定" 不會互相對應到同一張圖片的
        neg = False
        if neg:
            template = self.get_neg_pair("template", idx)
            search = self.get_neg_pair("search")
        else:
            template, search = self.get_positive_pair(idx)

        ####################################################################
        # Step 1.
        # get template and search images (raw data)
        ####################################################################
        image_path = template[0]
        image_name = image_path.split('/')[-1].split('.')[0]
        # print(f"load image from: {image_path}")
        image = cv2.imread(image_path)
        template_image = cv2.imread(image_path)    # cv2 讀進來的檔案是 BGR (一般是 RGB)
        search_image = cv2.imread(image_path)

        if DEBUG:
            print(f"template image path: {template[0]}")
            print(f"search image path: {search[0]}")
            print(f"shape template, search: {template[1].shape}, {search[1].shape}")
        assert template_image is not None, f"error image: {template[0]}"

        ####################################################################
        # Step 2.
        # crop the template and search images
        # -------------------------------------------------------
        # === 定義代號 ===
        # z: template
        # x: search
        ####################################################################
        template_box = template[1]    # template_box: (cx, cy, w, h) #ratio
        search_boxes = search[1]

        template_image, template_ratio, _, _ = self.template_crop(
            template_image,
            template_box,
            bg=self.template_bg,
            context_amount=self.template_context_amount
        )

        # gt_boxes ((x1, y1, x2, y2), num)
        search_image, gt_boxes, _, _ = self.search_crop(
            search_image,
            search_boxes,
            ratio=template_ratio
        )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ########################
        # 創 directory
        ########################
        # image_name = image_path.split('/')[-1].split('.')[0]
        # sub_dir = os.path.join("./image_check/train/", image_name)
        # if not os.path.isdir(sub_dir):
        #     os.makedirs(sub_dir)
        #     print(f"create dir: {sub_dir}")

        # # 創 sub_dir/origin，裡面存 origin image
        # origin_dir = os.path.join(sub_dir, "origin")
        # if not os.path.isdir(origin_dir):
        #     os.makedirs(origin_dir)
        #     print(f"create dir: {origin_dir}")
        # # 創 sub_dir/search，裡面存 search image
        # search_dir = os.path.join(sub_dir, "search")
        # if not os.path.isdir(search_dir):
        #     os.makedirs(search_dir)
        #     print(f"create dir: {search_dir}")
        # # 創 sub_dir/template，裡面存 template image
        # template_dir = os.path.join(sub_dir, "template")
        # if not os.path.isdir(template_dir):
        #     os.makedirs(template_dir)
        #     print(f"Create dir: {template_dir}")

        # #########################
        # # 存圖片
        # #########################
        # origin_path = os.path.join(origin_dir, f"{idx}.jpg")
        # save_image(image, origin_path)
        # print(f"save original image to: {origin_path}")

        # template_path = os.path.join(template_dir, f"{idx}.jpg")
        # save_image(template_image, template_path)
        # print(f"save template image to: {template_path}")

        # # draw gt_boxes on search image
        # search_path = os.path.join(search_dir, f"{idx}.jpg")
        # tmp_gt_boxes = gt_boxes.copy()
        # tmp_gt_boxes[2] = tmp_gt_boxes[2] - tmp_gt_boxes[0]
        # tmp_gt_boxes[3] = tmp_gt_boxes[3] - tmp_gt_boxes[1]
        # gt_image = draw_box(search_image, np.transpose(tmp_gt_boxes, (1, 0)))
        # save_image(gt_image, search_path)
        # print(f"save search image to: {search_path}")
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        ####################################################################
        # Step 3.
        # get the gt_boxes for calculating labels in training
        ####################################################################
        gt_boxes = np.asarray(gt_boxes)
        # gt_boxes = np.transpose(gt_boxes, (1, 0))
        gt_boxes = Corner(gt_boxes[0], gt_boxes[1], gt_boxes[2], gt_boxes[3])
        gt_boxes = np.array(gt_boxes)
        gt_boxes = np.transpose(gt_boxes, (1, 0))   # (4, num) -> (num, 4)
        gt_boxes = torch.from_numpy(gt_boxes)

        # check the bounding box (這照理來說不應該發生吧...):
        not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        # 為了要解決 gt_boxes 數量不一致的問題
        # 將所有的 gt_boxes 都 padding 成數量最多的那個 gt_boxes 的數量
        # 確定這裡的 gt_boxes_padding 沒問題 (08/27/2022)
        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()

        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0
            assert False, "=== There are no targets!! ==="

        # cls, delta, delta_weight, overlap = self.anchor_target(
        #     bbox, cfg.TRAIN.OUTPUT_SIZE, neg, idx)

        # (127, 127, 3) -> (3, 127, 127) for CNN using
        template_image = template_image.transpose((2, 0, 1)).astype(np.float32)
        search_image = search_image.transpose((2, 0, 1)).astype(np.float32)

        return {
            'template_image': template_image,
            'search_image': search_image,
            'gt_boxes': gt_boxes_padding,
            'num_boxes': num_boxes,
            # 下面是檢查 anchor 的時候，要存檔用的
            'image_name': image_name,
            'idx': idx
        }


if __name__ == "__main__":
    print("Loading dataset...")
    train_dataset = PCBDataset()

    # train_loader = DataLoader(train_dataset,
    #                           batch_size=cfg.TRAIN.BATCH_SIZE,
    #                           num_workers=cfg.TRAIN.NUM_WORKERS,
    #                           collate_fn=train_dataset.collate_fn,      # 參考: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/train.py#L72
    #                           pin_memory=True,
    #                           sampler=None)

    print("="*20 + " Done!! " + "="*20 + "\n")
