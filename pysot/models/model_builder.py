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
            只需要傳入 gt，label_cls, label_loc 在這裡面算
        """
        ####################################################################
        # 拿資料
        ####################################################################
        z_img = data['z_img'].cuda()
        x_img = data['x_img'].cuda()
        gt_boxes_padding = data['gt_boxes_padding'].cuda()
        image_name = data['image_name'][0]
        idx = data['idx'][0]

        ####################################################################
        # get labels of cls, loc, weight
        ####################################################################
        label_cls, label_loc, label_loc_weight, _ = self.anchor_target(
            gt_boxes_padding,
            self.output_size,
            image_name=image_name,
            idx=idx
        )

        ####################################################################
        # get feature maps
        ####################################################################
        # out = [out[i] for i in self.used_layers]
        zf = self.backbone(z_img)
        xf = self.backbone(x_img)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)    # ((b, c, 7, 7), (), ) #list
            xf = self.neck(xf)    # ((b, c, 31, 31), (), ) #list

        ####################################################################
        # get preds of cls, loc
        ####################################################################
        # cls: (b, 10, 25, 25), loc: (b, 20, 25, 25)
        cls, loc = self.rpn_head(zf, xf)

        ####################################################################
        # calculate loss of cls, loc
        ####################################################################
        cls = self.log_softmax(cls)     # cls: (b, 5, 25, 25, 2)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['total_loss'] = (cfg.TRAIN.CLS_WEIGHT * cls_loss) + (cfg.TRAIN.LOC_WEIGHT * loc_loss)

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss

        return outputs
