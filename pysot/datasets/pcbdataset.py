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

from pysot.utils.bbox import Corner, center2corner, Center
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.pcb_augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

import pysot.datasets.check_image as check_image
import ipdb
from pysot.datasets.crop_image import crop_like_SiamFC
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
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
        # for name in cfg.DATASET.NAMES:
        #     data_cfg = getattr(cfg.DATASET, name)
        data_cfg = getattr(cfg.DATASET, "CUSTOM")
        
        self.root = data_cfg.ROOT
        self.anno = data_cfg.ANNO
        images, template, search = self._make_dataset(self.root)
        self.images = images
        self.template = template
        self.search = search

        # data augmentation
        self.template_aug = Augmentation(
                                "template"
                            )
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
    
    def get_image_anno(self, index, arg):
        """ 
        Return:
            imgage_path: 
            image_anno: 
        """
        image_path, _ = self.images[index]
        # print(f"type: {type(arg)}, {type(arg)}")
        image_anno = arg[index]
        # if type=="template":
        #     image_anno = self.template[index]
        # elif type=="search":
        #     image_anno = self.search[index]
        # print(f"image_anno: {image_anno}")
        image_anno = np.stack(image_anno).astype(np.float32)        # 這要幹嘛? 回傳的image_anno不是只有一個物件嗎?
        return image_path, image_anno

    def get_positive_pair(self, index):
        return self.get_image_anno(index, self.template), \
               self.get_image_anno(index, self.search)
    
    def get_neg_pair(self, type, index=None):
        if type == "template":
            return self.get_image_anno(index, self.template)
        elif type == "search":
            index = np.random.randint(low=0, high=len(self.images))
            return self.get_image_anno(index, self.search)
    
    def _get_bbox(self, image, shape, type):
        """
        Args:
            image: 實際影像
            shape: bbox 的位置 ([cx, cy, w, h])，是比例值
        """
        # 是先高度再寬度 !!!
        imh, imw = image.shape[:2]          # image 的 height, width
        if type == "template":
            cx, w = imw*shape[0], imw*shape[2]
            cy, h = imh*shape[1], imh*shape[3]
        elif type == "search":
            cx, w = imw*shape[:, 0], imw*shape[:, 2]
            cy, h = imh*shape[:, 1], imh*shape[:, 3]
        bbox = center2corner(Center(cx, cy, w, h))      # Center 有可能不能這樣讀資料...
        return bbox
    
    def _get_bbox_amy(self, shape, typeName, direction, origin_size, temp):#原本cx,cy,w,h
        bbox=[]
        length = len(shape)
        if typeName=="template":                    # 但亭儀的 typeName 完全沒有等於 template，應該是因為 template 也不需要 bbox
            shape[0]=shape[0]*origin_size[0]        # shape 是比例，origin_size 是實際影像大小
            shape[1]=shape[1]*origin_size[1]
            shape[2]=shape[2]*origin_size[0]
            shape[3]=shape[3]*origin_size[1]
            # 這裡是要把物體移到中間
            if direction=='x':
                x1 = (shape[0]-shape[2]/2)+temp     # temp 這個取名我不懂...
                y1 = (shape[1]-shape[3]/2)
                x2 = (shape[0]+shape[2]/2)+temp
                y2 = (shape[1]+shape[3]/2)
            else:
                x1 = (shape[0]-shape[2]/2)
                y1 = (shape[1]-shape[3]/2)+temp
                x2 = (shape[0]+shape[2]/2)
                y2 = (shape[1]+shape[3]/2)+temp
            bbox.append(center2corner(Center((x1+(x2-x1)/2),(y1+(y2-y1)/2), (x2-x1), (y2-y1))))
        else:
            
            shape[:,0]=shape[:,0]*origin_size[0]
            shape[:,1]=shape[:,1]*origin_size[1]
            shape[:,2]=shape[:,2]*origin_size[0]
            shape[:,3]=shape[:,3]*origin_size[1]

            for i in range (len(shape)):
                if direction=='x':
                    x1 = (shape[i][0]-shape[i][2]/2)+temp
                    y1 = (shape[i][1]-shape[i][3]/2)
                    x2 = (shape[i][0]+shape[i][2]/2)+temp
                    y2 = (shape[i][1]+shape[i][3]/2)
                else:
                    x1 = (shape[i][0]-shape[i][2]/2)
                    y1 = (shape[i][1]-shape[i][3]/2)+temp
                    x2 = (shape[i][0]+shape[i][2]/2)
                    y2 = (shape[i][1]+shape[i][3]/2)+temp
                bbox.append(center2corner(Center((x1+(x2-x1)/2),(y1+(y2-y1)/2), (x2-x1), (y2-y1))))
            
        return bbox
    
    
    # 使用 image_crop 511
    def _search_gt_box(self, image, shape, scale, check):
        # print("box:",shape.shape)
        imh, imw = image.shape[:2]
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        bbox=[]
        cx=shape[:,0]*imw
        cy=shape[:,1]*imh
        w=shape[:,2]*imw
        h=shape[:,3]*imh
            
        for i in range (len(shape)):
            
            w1 ,h1 = w[i],h[i]
            if check==0:
                cx1 = cx[i]*scale[0]
                cy1 = cy[i]*scale[1]
            else:
                cx1 = cx[i]*scale[0]+scale[2]
                cy1 = cy[i]*scale[1]+scale[3]
   
            wc_z = w1 + context_amount * (w1+h1)
            hc_z = h1 + context_amount * (w1+h1)
            s_z = np.sqrt(wc_z * hc_z)
            scale_z = exemplar_size / s_z
            w1 = w1*scale_z
            h1 = h1*scale_z
            
            bbox.append(center2corner(Center(cx1, cy1, w1, h1)))
        return bbox
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        logger.debug("__getitem__")
        # 就只是隨機 gray 而已 (在 augmentation 才會用到，那為甚麼不要在 .xxx_aug() 在做就好啊...)
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        
        # 加入 neg 的原因要去看 [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        # 這個 if 還需要修改成 template 和 search "一定" 不會互相對應到同一張圖片的
        neg = False
        if neg:
            template = self.get_neg_pair("template", index)
            search = self.get_neg_pair("search")
        else:
            template, search = self.get_positive_pair(index)
        
        ####################################################################
        # Step 1.
        # get template and search images (raw data)
        ####################################################################
        template_image = cv2.imread(template[0])        # cv2 讀進來的檔案是 BGR (一般是 RGB)
        search_image = cv2.imread(search[0])
        
        # image_h, image_w = search_image.shape[:2]
        # template_box = center2corner(template[1])
        # tmplt_x1, tmplt_x2 = image_w * template_box[0], image_w * template_box[2]
        # tmplt_y1, tmplt_y2 = image_h * template_box[1], image_h * template_box[3]
        # print(f"image path: {template[0]}")
        # print(f"[{tmplt_x1, tmplt_y1, tmplt_x2, tmplt_y2}]")
        # print(f"search: \n{search[1]}")
        
        if DEBUG:
            print(f"template image path: {template[0]}")
            print(f"search image path: {search[0]}")
            print(f"shape template, search: {template[1].shape}, {search[1].shape}")
        assert template_image is not None, f"error image: {template[0]}"


        ####################################################################
        # Step 2.
        # crop template image, 
        # and get the scale which will be used in Step 3.
        # 
        # === 定義代號 ===
        # z: template
        # x: search
        ####################################################################
        z_h, z_w = template_image.shape[:2]         # need to save the original size of template image
        template_box = template[1]
        search_bbox = search[1]
        # template_bbox_corner = center2corner(template_box)
        # template_image, scale = crop_like_SiamFC(search_image, template_bbox_corner)
        
        template_image, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE)
        
        cv2.imwrite("image_check/trash/template.jpg", template_image)
        
        ipdb.set_trace()
        

        ####################################################################
        # Step 3.
        # crop search image according to scale
        # search crop (like SiamFC) 但影像都放在左上角 (不懂??)
        # 同樣要處理 search 的 bbox
        ####################################################################
        if (z_h * search_image.shape[0] > cfg.TRAIN.SEARCH_SIZE) or (z_w * search_image.shape[1] > cfg.TRAIN.SEARCH_SIZE):
            centercrop = transforms.CenterCrop(cfg.TRAIN.SEARCH_SIZE)
            resize = transforms.Resize([cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE])
            search_image = Image.fromarray((cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB)))     # 因為 transforms.centercrop 不能用 cv2
            s_resize = resize(search_image)
            origin_size = s_resize.size
            search_image = centercrop(s_resize)
        
            if origin_size[0] < cfg.TRAIN.SEARCH_SIZE: #x
                temp = (cfg.TRAIN.SEARCH_SIZE-origin_size[0]) / 2
                direction = 'x'
            elif origin_size[1] < cfg.TRAIN.SEARCH_SIZE: #y
                temp = (cfg.TRAIN.SEARCH_SIZE-origin_size[1]) / 2
                direction = 'y'
            else:
                temp=0
                direction = 'x'

            # get bounding box
            # 亭儀的 _get_bbox 要這樣寫應該是因為他有把 search image 移動過，所以 bbox 也要相對地移動
            search_box = self._get_bbox_amy(search[1],"search",direction,origin_size,temp)
            search_image = cv2.cvtColor(np.asarray(search_image), cv2.COLOR_RGB2BGR)
            bbox = search_box
            # bbox 和 search_box 竟然會是不一樣的 data type!!?
            # bbox: corner, search_box: array
        else:
            cx = (z_w/2)*scale[0]+scale[2]
            cy = (z_h/2)*scale[1]+scale[3]
            if (z_w*scale[0] < 300 ) and (z_h*scale[1] < 300 ):
                mapping = np.array([[scale[0], 0, 300-cx],
                            [0, scale[1], 300-cy]]).astype(np.float)
                scale[2] = 300-cx
                scale[3] = 300-cy
                check=1
            else:
                mapping = np.array([[scale[0], 0, 0],
                            [0, scale[1], 0]]).astype(np.float)
                check=0
            
            search_image2 = cv2.warpAffine(search_image, mapping, (600, 600), 
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
            search_box = self._search_gt_box(search_image,search[1],scale,check)
            bbox = search_box
            search_image = search_image2
        
        # save image
        if DEBUG:
            file_name = template[0].split("/")[-1]
            check_image.draw_bbox(search_image, bbox, file_name)
            template_image_name = "template_" + template[0].split("/")[-1]
            check_image.save_image("template", template_image, template_image_name)
            search_image_name = "search_" + search[0].split("/")[-1]
            check_image.save_image("search", search_image, search_image_name)

        # get bounding box
        # 先用 255*255 就好 (跑起來比較快)
        # template_box = self._get_bbox(template_image, template[1], "template")
        # search_box = self._get_bbox(search_image, search[1], "search")

        if DEBUG:
            print(f"search_image shape: {search_image.shape}")
            print(f"search_box type: {type(search_box)}")
            print(f"search_box:\n {search_box}")
        
        # (image, bbox) is the return data type
        """ 
        template_image, _ = self.template_aug(template_image,
                                            template_box,
                                            cfg.TRAIN.EXEMPLAR_SIZE,
                                            gray=gray)

        search_image, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)
        """
        
        ####################################################################
        # Step 4.
        # get the label for training
        ####################################################################
        bbox = np.asarray(bbox)
        bbox = np.transpose(bbox, (1, 0))
        bbox = Corner(bbox[0], bbox[1], bbox[2], bbox[3])

        # # 先試試看單一追蹤
        # random_pick = np.random.randint(low=len(bbox[0]), size=1)
        # bbox = np.asarray(bbox)
        # new_bbox = np.zeros((4, 1))
        # new_bbox[0] = bbox[0][random_pick]
        # new_bbox[1] = bbox[1][random_pick]
        # new_bbox[2] = bbox[2][random_pick]
        # new_bbox[3] = bbox[3][random_pick]
        # bbox = Corner(new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3])
        
        cls, delta, delta_weight, overlap = self.anchor_target(
                bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
        
        template_image = template_image.transpose((2, 0, 1)).astype(np.float32)     # [3, 127, 127]
        search_image = search_image.transpose((2, 0, 1)).astype(np.float32)
        
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
    print(f"Loading dataset...")
    train_dataset = PCBDataset()
    train_dataset.__getitem__(2)
    print(f"Loading dataset has done!")

    # train_loader = DataLoader(train_dataset,
    #                           batch_size=cfg.TRAIN.BATCH_SIZE,
    #                           num_workers=cfg.TRAIN.NUM_WORKERS,
    #                           collate_fn=train_dataset.collate_fn,      # 參考: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/train.py#L72
    #                           pin_memory=True,
    #                           sampler=None)
    # print(len(train_dataset))
    # print(len(train_loader))

    # for data in enumerate(train_loader):
    #     pass

    print("="*20 + " Done!! " + "="*20 + "\n")
