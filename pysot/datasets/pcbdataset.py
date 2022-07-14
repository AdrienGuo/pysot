# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

class PCBDataset():
    def __init__(self) -> None:
        """ 這裡就是負責讀取資料
        """
        super(PCBDataset, self).__init__()
        
        # create anchor target
        self.anchor_target = AnchorTarget()

        # 可以不用迴圈
        for name in cfg.DATASET.NAMES:
            data_cfg = getattr(cfg.DATASET, name)
        
        self.root = data_cfg.ROOT
        self.anno = data_cfg.ANNO
        images, template, search = self._make_dataset(self.root)
        self.images = images
        self.template = template
        self.search = search

    def _make_dataset(self, dir_path):
        """ 回傳資料
        Return:
            images (list): [[path, 類別], ...]\n
            template (list): \n
            search (list): \n
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
                    anno_path = os.path.join(self.anno, file[:-3]+"txt")        # 改成 .txt (annotation 檔案)

                    # 非 text 類型
                    if os.path.isfile(anno_path):
                        f = open(anno_path, 'r')
                        lines = f.readlines()
                        anno = []
                        for line in lines:
                            line=line.strip('\n')
                            line=line.split(' ')
                            line = list(map(float, line))
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
                    elif os.path.isfile(os.path.join(self.anno, file[:-3]+"label")):
                        f = open(os.path.join(self.anno, file[:-3]+"label"), 'r')
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
    
    def get_image_anno(self, index, typeName):
        """ 
        Return:
            imgage_path: \n
            image_anno: \n
        """
        image_path, _ = self.images[index]
        if typeName=="template":
            image_anno = self.template[index]
        elif typeName=="search":
            image_anno = self.search[index]
        image_anno = np.stack(image_anno).astype(np.float32)        # 這要幹嘛? 回傳的image_anno不是只有一個物件嗎?
        return image_path, image_anno

    def get_positive_pair(self, index):
        return self.get_image_anno(index, "template"), \
               self.get_image_anno(index, "search")
    
    def get_neg_pair(self, typeName, index=None):
        if typeName == "template":
            return self.get_image_anno(index, "template")
        elif typeName == "search":
            index = np.random.randint(low=0, high=len(self.images))
            return self.get_image_anno(index, "search")
    
    def _get_bbox(self, image, shape):
        """
        Args:
            image: 實際影像\n
            shape: bbox 的位置 ([x1, y1, x2, y2])，是比例值\n
        """
        imh, imw = image.shape[:2]          # iamge 的 height, width
        cx, cy = imw//2, imh//2
        print(shape)
        w, h = imw * (shape[:, 2]-shape[:, 0]), imh * (shape[:, 3]-shape[:, 1])     # 將比例值乘以實際大小轉成實際位置的數值
        bbox = center2corner(Center(cx, cy, w, h))      # Center 有可能不能這樣讀資料...
        return bbox
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        # 加入 neg 的原因要去看 [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        # 這個 if 還需要修改成 template 和 search "一定" 不會互相對應到同一張圖片的
        if neg:
            template = self.get_neg_pair("template", index)
            search = self.get_neg_pair("search")
        else:
            template, search = self.get_positive_pair(index)

        
        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])
        assert template_image is not None, f"error image: {template[0]}"
        # if template_image is None:
        #     print('error image:',template[0])

        # get bounding box
        # 先用 255*255 就好 (跑起來比較快)
        # template_box = self._get_bbox(template_image, template[1])      # 暫時沒用
        search_box = self._get_bbox(search_image, search[1])

        # get labels
        cls, delta, delta_weight, overlap = self.anchor_target(
                search_box, cfg.TRAIN.OUTPUT_SIZE, neg)
        
        return None

if __name__ == "__main__":
    dataset = PCBDataset()
    dataset.__getitem__(2)
    print("="*20 + " Done!! " + "="*20)
