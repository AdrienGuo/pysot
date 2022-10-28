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
from pysot.datasets.augmentation import Augmentation
import torch
from pysot.core.config import cfg
from pysot.datasets.pcb_crop_new import PCBCrop
from pysot.utils.bbox import Center, Corner, center2corner
from pysot.utils.check_image import create_dir, draw_box, save_image
from torchvision import transforms

logger = logging.getLogger("global")


# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class PCBDataset():
    def __init__(self, args, mode) -> None:
        """ 代號
        z: template
        x: search

        Args:
            args: User setting arguments.
            mode: (train / val / eval / test), Set get pair method.
        """

        super(PCBDataset, self).__init__()

        if mode == "test":
            self.root = args.test_dataset
            self.anno = args.test_dataset
        else:
            self.root = args.dataset_path
            self.anno = args.dataset_path

        images, template, search = self._make_dataset(self.root)
        images, template, search = self._filter_dataset(
            images, template, search, args.criteria)
        assert len(images) != 0, "Error, dataset is empty!"
        self.images = images
        self.template = template
        self.search = search
        self.max_num_box = self._find_max_num_box(self.search)    # targets 最多的數量

        self.crop_method = args.crop_method
        self.bg = args.bg
        self.neg = args.neg
        self.criteria = args.criteria
        self.mode = mode
        # self.template_context_amount = args.template_context_amount

        # transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),    # range [0, 255] -> [0.0, 1.0]
        ])

    def _make_dataset(self, dir_path):
        """ 回傳資料
        Return:
            images (list): [ [path, 類別], ...]
            template (list):
            search (list):
        """
        images = []
        search = []    # search image 上 box 的座標
        template = []

        directory = os.path.expanduser(dir_path)
        for root, _, files in os.walk(directory, followlinks=True):
            files = sorted(files)    # 將 files 排序
            for file in files:
                box = []
                if file.endswith(('.jpg', '.png', 'bmp')):
                    path = os.path.join(root, file)
                    anno_path = os.path.join(self.anno, file[:-3] + "txt")
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
                            # 如果是電子元件類型，第 26 類為文字要忽略
                            # 現在變很煩，pcb 和 text 的標籤重疊了...
                            if anno[i][0] != 26:
                                item = path, anno[i][0]
                                images.append(item)
                                template.append([anno[i][1], anno[i][2], anno[i][3], anno[i][4]])
                                box = []
                            if anno[i][0] != 26:
                                for j in range(len(anno)):    # 這裡應該可以直接改成用 filter 來做
                                    if anno[j][0] == anno[i][0]:
                                        box.append([anno[j][1], anno[j][2], anno[j][3], anno[j][4]])
                                box = np.stack(box).astype(np.float32)
                                search.append(box)
                    # text 類型
                    else:
                        anno_path = os.path.join(self.anno, file[:-3] + "label")
                        assert os.path.isfile(anno_path), f"Error, {anno_path} does not exist!!"
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
                                        cx = float(anno[j][1]) + (float(anno[j][3]) - float(anno[j][1]))/2
                                        cy = float(anno[j][2]) + (float(anno[j][4]) - float(anno[j][2]))/2
                                        w = float(anno[j][3]) - float(anno[j][1])
                                        h = float(anno[j][4]) - float(anno[j][2])
                                        box.append([cx/imw, cy/imh, w/imw, h/imh])

                                box = np.stack(box).astype(np.float32)
                                search.append(box)
        return images, template, search

    def _filter_dataset(self, images, template, search, criteria):
        # criteria == all
        if criteria == "all":
            return images, template, search
        # for
        inds_match = list()
        for idx, image in enumerate(images):
            # read image
            img = cv2.imread(image[0])
            # get w & h
            img_h, img_w = img.shape[:2]
            z_w = template[idx][2] * img_w
            z_h = template[idx][3] * img_h
            # calculate r by resize to 255
            long_side = max(img_w, img_h)
            r = cfg.TRAIN.SEARCH_SIZE / long_side
            # calculate template new w, h
            z_w = z_w * r
            z_h = z_h * r
            if criteria == "small":
                if max(z_w, z_h) <= 32:
                    inds_match.append(idx)
            elif criteria == "mid":
                if 32 < max(z_w, z_h) <= 64:
                    inds_match.append(idx)
            elif criteria == "big":
                if max(z_w, z_h) > 64:
                    inds_match.append(idx)
            else:
                assert False, "ERROR, chosen criteria is wrong!"
        images = [images[i] for i in inds_match]
        template = [template[i] for i in inds_match]
        search = [search[i] for i in inds_match]

        return images, template, search

    def _find_max_num_box(self, boxes):
        num_boxes = list(map(lambda x: x.shape[0], boxes))
        max_num_boxes = max(num_boxes)
        return max_num_boxes

    def _get_image_anno(self, idx, data):
        img_path, template_cls = self.images[idx]
        image_anno = data[idx]
        return img_path, image_anno

    def _get_positive_pair(self, idx):
        return self._get_image_anno(idx, self.template), \
               self._get_image_anno(idx, self.search)

    def _get_negative_pair(self, idx):
        while True:
            idx_neg = np.random.randint(0, len(self.images))
            if self.images[idx][0] != self.images[idx_neg][0]:
                # idx 和 idx_neg 不是對應到同一張圖
                break
        return self._get_image_anno(idx, self.template), \
               self._get_image_anno(idx_neg, self.search)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            gt_boxes_padding: (G=100, 4)
        """

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()

        # 加入 neg 的原因要去看 [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
        neg = self.neg > np.random.random()

        # Get one dataset
        if self.mode == "train" and neg:
            template, search = self._get_negative_pair(idx)
        else:
            template, search = self._get_positive_pair(idx)

        ##########################################
        # Step 1.
        # Get image
        ##########################################
        img_path = template[0]
        img_name = img_path.split('/')[-1].rsplit('.', 1)[0]
        img = cv2.imread(img_path)
        # print(f"Load image from: {img_path}")

        ##########################################
        # Data augmentation (preprocess?)
        ##########################################
        aug = Augmentation()
        # img = aug._gray(img)
        # img = aug._contrast(img)    # 效果不好

        z_img = img
        x_img = img
        assert z_img is not None, f"Error image: {template[0]}"

        ##########################################
        # Step 2.
        # Crop the template and search images
        # ----------------------------------------
        # === 定義代號 ===
        # z: template
        # x: search
        ##########################################
        # crop template & search (preprocess)
        pcb_crop = PCBCrop(
            template_size=cfg.TRAIN.EXEMPLAR_SIZE,
            search_size=cfg.TRAIN.SEARCH_SIZE,
        )

        z_box = template[1]    # z_box: [cx, cy, w, h] #ratio
        z_box = np.asarray(z_box)
        z_box = z_box[np.newaxis, :]    # [cx, cy, w, h] -> (1, [cx, cy, w, h]) 轉成跟 gt_boxes 一樣是二維的
        gt_boxes = search[1]    # gt_boxes: (num, [cx, cy, w, h]) #ratio
        gt_boxes = np.asarray(gt_boxes)

        # 先做好 search image (x_img)，
        # 再從 x_img 上面切出 template (z_img)
        # gt_boxes: (num, [x1, y1, x2, y2])
        x_img, gt_boxes, z_box, r, spatium = pcb_crop.get_search(x_img, gt_boxes.copy(), z_box.copy())
        z_img = pcb_crop.get_template(x_img, z_box.copy(), self.bg)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ########################
        # 創 directory
        ########################
        # dir = f"./image_check/train/x{cfg.TRAIN.SEARCH_SIZE}_{self.crop_method}_bg{self.bg}"
        # dir = "./image_check/train/"
        # img_name = img_path.split('/')[-1].split('.')[0]
        # # 以 “圖片名稱” 當作 sub_dir 的名稱
        # sub_dir = os.path.join(dir, img_name)
        # create_dir(sub_dir)
        # # 創 sub_dir/origin，裡面存 origin image
        # origin_dir = os.path.join(sub_dir, "origin")
        # create_dir(origin_dir)
        # # 創 sub_dir/search，裡面存 search image
        # search_dir = os.path.join(sub_dir, "search")
        # create_dir(search_dir)
        # # 創 sub_dir/template，裡面存 template image
        # template_dir = os.path.join(sub_dir, "template")
        # create_dir(template_dir)

        # #########################
        # # 存圖片
        # #########################
        # origin_path = os.path.join(origin_dir, "origin.jpg")
        # save_image(img, origin_path)
        # template_path = os.path.join(template_dir, f"{idx}.jpg")
        # save_image(z_img, template_path)

        # # draw gt_boxes on search image
        # dummy_boxes = gt_boxes.copy()
        # dummy_boxes[:, 2] = dummy_boxes[:, 2] - dummy_boxes[:, 0]
        # dummy_boxes[:, 3] = dummy_boxes[:, 3] - dummy_boxes[:, 1]
        # dummy_img = draw_box(x_img, dummy_boxes, type="gt")
        # dummy_box = z_box.copy()
        # dummy_box[:, 2] = dummy_box[:, 2] - dummy_box[:, 0]
        # dummy_box[:, 3] = dummy_box[:, 3] - dummy_box[:, 1]
        # dummy_img = draw_box(dummy_img, dummy_box, type="template")
        # search_path = os.path.join(search_dir, f"{idx}.jpg")
        # save_image(dummy_img, search_path)

        # ipdb.set_trace()
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        ##########################################
        # Step 3.
        # Transform the gt_boxes format to fix the problem of
        # the inconsistency of number of gt_boxes
        ##########################################
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
        z_img = z_img.transpose((2, 0, 1)).astype(np.float32)
        x_img = x_img.transpose((2, 0, 1)).astype(np.float32)

        # 有沒有做 normalize 的 loss 曲線都一樣
        # z_img = self.transform(z_img)
        # x_img = self.transform(x_img)

        # train 不需要 gt_boxes
        if self.mode == "train" or self.mode == "val":
            gt_boxes = np.array([[0, 0, 0, 0]])

        return {
            'z_img': z_img,
            'x_img': x_img,
            'neg': neg,
            'gt_boxes_padding': gt_boxes_padding,    # train 的 label

            'img_path': img_path,
            'z_box': z_box,    # 算 kmeans 也會用到
            'gt_boxes': gt_boxes,    # test 上畫圖用，在 train 的時候要拿掉 (因為數量不一致的關係)
            'num_boxes': num_boxes,
            # === 下面是檢查 anchor 的時候，要存檔用的 ===
            'img_name': img_name,
            'idx': idx,
            # === 下面是 test 的時候，要將 box 轉成原圖上用的 ===
            'scale': r,
            'spatium': spatium
        }
