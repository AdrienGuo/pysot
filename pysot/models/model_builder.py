# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time

import cv2
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.backbone import get_backbone
from pysot.models.head import get_mask_head, get_refine_head, get_rpn_head
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.neck import get_neck
from pysot.rpn.anchor_target import AnchorTarget

# debug mode
DEBUG = cfg.DEBUG


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # size
        self.output_size = cfg.TRAIN.OUTPUT_SIZE

        # define rpn
        self.anchor_target = AnchorTarget()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        cls, loc = self.rpn_head(self.zf, xf)
        
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        
        return {
            'cls': cls,
            'loc': loc,
            'mask': mask if cfg.MASK.MASK else None
        }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        Args:
            只需要傳入 gt，
            label_cls, label_loc 在這裡面算
        """
        # start = time.time()
        # template = [torch.from_numpy(template).cuda() for template in data['template_image']]
        # search = [torch.from_numpy(search).cuda() for search in data['search_image']]
        # label_cls = [torch.from_numpy(label_cls).cuda() for label_cls in data['label_cls']]
        # label_loc = [torch.from_numpy(label_loc).cuda() for label_loc in data['label_loc']]
        # label_loc_weight = [torch.from_numpy(label_loc_weight).cuda() for label_loc_weight in data['label_loc_weight']]
        # gt_boxes = [torch.from_numpy(gt_box).cuda() for gt_box in data['gt_boxes']]

        # template = torch.stack(template, dim=0)     # turn to tensor datatype with [b, c, w, h] (not sure about the order of last two dims)
        # search = torch.stack(search, dim=0)
        # label_cls = torch.stack(label_cls, dim=0)
        # label_loc = torch.stack(label_loc, dim=0)
        # label_loc_weight = torch.stack(label_loc_weight, dim=0)

        ####################################################################
        # 拿資料
        ####################################################################
        template = data['template_image'].cuda()
        search = data['search_image'].cuda()
        gt_boxes = data['gt_boxes'].cuda()

        ####################################################################
        # 計算 label_cls, label_loc
        ####################################################################
        label_cls, label_loc, label_loc_weight, _ = self.anchor_target(gt_boxes, self.output_size)

        print(f"label_cls: {label_cls.shape}")
        print(f"label_loc: {label_loc.shape}")

        ipdb.set_trace()

        # template_image = template.cpu().numpy()
        # print(f"template: {template_image.shape}")
        # cv2.imwrite("./image_check/trash/template.jpg", template_image[0].transpose(1, 2, 0))
        # print("save template image")

        # search_image = search.cpu().numpy()
        # print(f"search: {search_image.shape}")
        # cv2.imwrite("./image_check/trash/search.jpg", search_image[0].transpose(1, 2, 0))
        # print("save search image")

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        
        # cls: (b, 10, 25, 25), loc: (b, 20, 25, 25)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        # print(f"cls: {cls.shape}")
        cls = self.log_softmax(cls)     # cls (b, 5, 25, 25, 2)
        # print(f"cls: {cls[0, 0, 0, 0, :]}")
        # print(f"label_cls: {label_cls.shape}")
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss \
                                + cfg.TRAIN.LOC_WEIGHT * loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss

        # end = time.time()
        # print(f"=== forward duration: {end - start:.4f} s")

        return outputs
