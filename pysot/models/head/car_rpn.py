# Copyright (c) SenseTime. All Rights Reserved.

# Reference: https://github.com/ohhhyeahhh/SiamCAR/blob/master/pysot/models/model_builder.py

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.xcorr import xcorr_depthwise, xcorr_fast
from pysot.models.init_weight import init_weights


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelRPN(RPN):
    def __init__(self, anchor_num=5, feature_in=256):
        super(UPChannelRPN, self).__init__()

        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self):
        super(DepthwiseXCorr, self).__init__()
        # SiamCAR 這裡好像沒對特徵再經過網路訓練
        # self.conv_kernel = nn.Sequential(
        #     nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
        #     nn.BatchNorm2d(hidden),
        #     nn.ReLU(inplace=True),
        # )
        # self.conv_search = nn.Sequential(
        #     nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
        #     nn.BatchNorm2d(hidden),
        #     nn.ReLU(inplace=True),
        # )
        # 先不要經過 head，所以 out_channels=256
        # self.head = nn.Sequential(
        #         nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
        #         nn.BatchNorm2d(hidden),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(hidden, out_channels, kernel_size=1)
        #         )

    def forward(self, kernel, search):
        # kernel = self.conv_kernel(kernel)
        # search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        # out = self.head(feature)
        return feature


class DepthwiseRPN(nn.Module):
    def __init__(self):
        super(DepthwiseRPN, self).__init__()
        self.xcorr_depthwise = DepthwiseXCorr()

    def forward(self, z_f, x_f):
        feature = self.xcorr_depthwise(z_f, x_f)
        return feature


class CARHead(nn.Module):
    def __init__(self, anchor_num, in_channel, weighted=False):
        super(CARHead, self).__init__()
        self.weighted = weighted
        self.add_module(
            'rpn',
            DepthwiseRPN()
        )
        self.down = nn.ConvTranspose2d(in_channel * 3, in_channel, 1, 1)
        # for i in range(len(in_channels)):
        #     self.add_module('rpn'+str(i+2),
        #                     DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        # if self.weighted:
        #     self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
        #     self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        features = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn')
            feature = rpn(z_f, x_f)
            features.append(feature)
        features = torch.cat(features, 1)    # c=256*3
        features = self.down(features)    # c: 256*3 -> 256



        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)


class Branches(nn.Module):
    def __init__(self, anchor_num, in_channels):
        super(Branches, self).__init__()

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, 2 * anchor_num, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4 * anchor_num, kernel_size=3, stride=1,
            padding=1
        )