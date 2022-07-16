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

# debug mode
from torch.utils.data import DataLoader
DEBUG = cfg.DEBUG

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

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )

    def _make_dataset(self, dir_path):
        """ 回傳資料
        Return:
            images (list): [[path, 類別], ...]
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
                    anno_path = os.path.join(self.anno, file[:-3]+"txt")        # 改成 .txt (annotation 檔案)
                    assert os.path.isfile(anno_path), f"This annotation path doesn't exist: {anno_path}"
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
    
    def get_image_anno(self, index, type):
        """ 
        Return:
            imgage_path: 
            image_anno: 
        """
        image_path, _ = self.images[index]
        if type=="template":
            image_anno = self.template[index]
        elif type=="search":
            image_anno = self.search[index]
        image_anno = np.stack(image_anno).astype(np.float32)        # 這要幹嘛? 回傳的image_anno不是只有一個物件嗎?
        return image_path, image_anno

    def get_positive_pair(self, index):
        return self.get_image_anno(index, "template"), \
               self.get_image_anno(index, "search")
    
    def get_neg_pair(self, type, index=None):
        if type == "template":
            return self.get_image_anno(index, "template")
        elif type == "search":
            index = np.random.randint(low=0, high=len(self.images))
            return self.get_image_anno(index, "search")
    
    def _get_bbox(self, image, shape, type):
        """
        Args:
            image: 實際影像
            shape: bbox 的位置 ([cx, cy, w, h])，是比例值
        """
        # 是先高度再寬度 !!!
        imh, imw = image.shape[:2]          # image 的 height, width
        if DEBUG:
            print(f"imh, imw: {imh}, {imw}")
        if type == "template":
            cx, w = imw*shape[0], imw*shape[2]
            cy, h = imh*shape[1], imh*shape[3]
        elif type == "search":
            cx, w = imw*shape[:, 0], imw*shape[:, 2]
            cy, h = imh*shape[:, 1], imh*shape[:, 3]
        bbox = center2corner(Center(cx, cy, w, h))      # Center 有可能不能這樣讀資料...
        return bbox
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        logger.debug("__getitem__")
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        # 加入 neg 的原因要去看 [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()
        
        neg = False     # 先不要有 negative pair

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
        if DEBUG:
            print(f"template image path: {template[0]}")
            print(f"search image path: {search[0]}")
            print(f"shape template, search: {template[1].shape}, {search[1].shape}")
        assert template_image is not None, f"error image: {template[0]}"
        
        # get bounding box
        # 先用 255*255 就好 (跑起來比較快)
        template_box = self._get_bbox(template_image, template[1], "template")
        search_box = self._get_bbox(search_image, search[1], "search")
        assert template_box != [], f"template_box is empty"
        assert search_box != [], f"search_box is empty"

        if DEBUG:
            print(f"search_image shape: {search_image.shape}")
            print(f"search_box type: {type(search_box)}")
            print(f"search_box:\n {search_box}")
        
        # (image, bbox) is the return data type
        template_image, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search_image, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)
        if DEBUG:
            print(f"adjusted search_bbox: {bbox}")

        """ 
        savedimage_path = "./image_check/train/" + search[0].split("/")[-1]
        cv2.imwrite(savedimage_path, search_image)
        print(f"save image to: {savedimage_path}")
        """
        # TODO: 畫 bbox

        # get labels
        cls, delta, delta_weight, overlap = self.anchor_target(
                bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
        
        template_image = template_image.transpose((2, 0, 1)).astype(np.float32)
        search_image = search_image.transpose((2, 0, 1)).astype(np.float32)
        print(f"template_image shape: {template_image.shape}")
        
        return {
                'template_image': template_image,
                'search_image': search_image,
                'label_cls': cls,
                'label_loc': delta,
                'label_loc_weight': delta_weight,
                'bbox': np.array(bbox)
               }
    
    def collate_fn(self, batch):
        """ 因為每個 template 會有 "不同數量" 的 targets，we need a collate function (to be passed to the DataLoader).
            不然會跳出 RuntimeError: stack expects each tensor to be equal size, but got [4, 1] at entry 0 and [4, 2] at entry 2
            參考: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/datasets.py#L60
        args:
            batch: an iterable of N sets from __getitem__()
        return:
            a tensor of images, lists of varying-size tensors of bounding boxes, etc
        """
        template_image = list()
        search_image = list()
        cls = list()
        delta = list()
        delta_weight = list()
        bbox = list()

        for b in batch:
            template_image.append(b['template_image'])
            search_image.append(b['search_image'])
            cls.append(b['label_cls'])
            delta.append(b['label_loc'])
            delta_weight.append(b['label_loc_weight'])
            bbox.append(b['bbox'])
                
        return {
                'template_image': template_image,
                'search_image': search_image,
                'label_cls': cls,
                'label_loc': delta,
                'label_loc_weight': delta_weight,
                'bbox': bbox
               }
    

if __name__ == "__main__":
    dataset = PCBDataset()
    dataset.__getitem__(7)

    train_loader = DataLoader(dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              collate_fn=dataset.collate_fn,      # 參考: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/train.py#L72
                              pin_memory=True,
                              sampler=None)
    print(len(dataset))
    print(len(train_loader))

    for data in enumerate(train_loader):
        pass

    print("="*20 + " Done!! " + "="*20 + "\n")
