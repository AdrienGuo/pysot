# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import logging
import os
import sys
import time

import cv2
import ipdb
import numpy as np
import torch
from pysot.core.config import cfg
from pysot.datasets.pcb_crop_old import PCBCrop
from pysot.utils.bbox import Center, Corner, center2corner
from pysot.utils.check_image import create_dir, draw_box, save_image

logger = logging.getLogger("global")


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

        self.root = args.dataset
        self.anno = args.dataset
        images, template, search = self._make_dataset(self.root)
        self.images = images
        self.template = template
        self.search = search
        self.max_num_box = self._find_max_num_box(self.search)    # targets 最多的數量

        # crop template & search (preprocess)
        self.pcb_crop = PCBCrop(
            template_size=cfg.TRAIN.EXEMPLAR_SIZE,
            search_size=cfg.TRAIN.SEARCH_SIZE,
        )

        self.crop_method = args.crop_method
        # self.template_bg = args.template_bg
        # self.template_context_amount = args.template_context_amount

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
        assert template_image is not None, f"error image: {image_path}"

        ####################################################################
        # Step 2.
        # Crop the template and search images
        # -------------------------------------------------------
        # === 定義代號 ===
        # z: template
        # x: search
        ####################################################################
        z_box = template[1]    # z_box: [cx, cy, w, h] #ratio
        z_box = np.asarray(z_box)
        z_box = z_box[np.newaxis, :]    # [cx, cy, w, h] -> (1, [cx, cy, w, h]) 轉成跟 gt_boxes 一樣是二維的
        gt_boxes = search[1]    # gt_boxes: (num, [cx, cy, w, y]) #ratio
        gt_boxes = np.asarray(gt_boxes)

        # 亭儀的舊方法是先做出 template，再去調整 search
        z_img = self.pcb_crop.get_template(template_image, z_box.copy())    # array 真的要小心處理，因為他們的 address 都是一樣的
        # gt_boxes: (num, [x1, y1, x2, y2])
        x_img, gt_boxes, z_box, r, spatium = self.pcb_crop.get_search(search_image, gt_boxes.copy(), z_box.copy())

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ########################
        # 創 directory
        ########################
        # dir = f"./image_check/train/x{cfg.TRAIN.SEARCH_SIZE}_{self.crop_method}"
        # image_name = image_path.split('/')[-1].split('.')[0]

        # sub_dir = os.path.join(dir, image_name)
        # create_dir(sub_dir)

        # # 創 sub_dir/origin，裡面存 origin image
        # origin_dir = os.path.join(sub_dir, "origin")
        # create_dir(origin_dir)
        # # 創 sub_dir/template，裡面存 template image
        # template_dir = os.path.join(sub_dir, "template")
        # create_dir(template_dir)
        # # 創 sub_dir/search，裡面存 search image
        # search_dir = os.path.join(sub_dir, "search")
        # create_dir(search_dir)

        # #########################
        # # 存圖片
        # #########################
        # origin_path = os.path.join(origin_dir, f"{idx}.jpg")
        # save_image(image, origin_path)

        # template_path = os.path.join(template_dir, f"{idx}.jpg")
        # save_image(z_img, template_path)

        # # draw gt_boxes on search image
        # search_path = os.path.join(search_dir, f"{idx}.jpg")
        # tmp_gt_boxes = gt_boxes.copy()
        # tmp_gt_boxes[:, 2] = tmp_gt_boxes[:, 2] - tmp_gt_boxes[:, 0]
        # tmp_gt_boxes[:, 3] = tmp_gt_boxes[:, 3] - tmp_gt_boxes[:, 1]
        # gt_image = draw_box(x_img, tmp_gt_boxes, type="gt")
        # tmp_z_box = z_box.copy()
        # tmp_z_box[:, 2] = tmp_z_box[:, 2] - tmp_z_box[:, 0]
        # tmp_z_box[:, 3] = tmp_z_box[:, 3] - tmp_z_box[:, 1]
        # z_gt_image = draw_box(gt_image, tmp_z_box, type="template")
        # save_image(z_gt_image, search_path)
        # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ipdb.set_trace()

        ####################################################################
        # Step 3.
        # Transform the gt_boxes format to fix the problem of
        # the inconsistency of number of gt_boxes
        ####################################################################
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
        # TODO: check whether the values are between 0~1 (toTensor)
        z_img = z_img.transpose((2, 0, 1)).astype(np.float32)
        x_img = x_img.transpose((2, 0, 1)).astype(np.float32)

        return {
            'image_path': image_path,
            'z_img': z_img,
            'x_img': x_img,
            'z_box': z_box,    # 算 kmeans 也會用到
            'gt_boxes': gt_boxes,    # test 上畫圖用，在 train 的時候要拿掉 (因為數量不一致的關係)
            'gt_boxes_padding': gt_boxes_padding,    # train 的 label
            'num_boxes': num_boxes,
            # === 下面是檢查 anchor 的時候，要存檔用的 ===
            'image_name': image_name,
            'idx': idx,
            # === 下面是 test 的時候，要將 box 轉成原圖上用的 ===
            'scale': r,
            'spatium': spatium
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
