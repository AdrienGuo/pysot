# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math

import ipdb
import numpy as np
from pysot.core.config import cfg
from pysot.utils.bbox import center2corner, corner2center


class Anchors:
    """
    This class generates anchors.
    """
    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size

        # self.anchor_num = len(self.scales) * len(self.ratios)
        self.anchor_num = cfg.ANCHOR.ANCHOR_NUM

        self.anchors = None

        self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        generate self.anchors: (anchor_num, 4)
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)

        anchors_wh = np.array([[155.2646798 , 105.87584095],
                               [ 16.47197038,   9.99376587],
                               [ 94.69877538, 174.15738125],
                               [ 39.82304951,  19.7316645 ],
                               [ 49.13396732,  91.50407477],
                               [ 15.558725  ,  38.64495505],
                               [ 26.48748143,  75.01758954],
                               [  3.3549437 ,   4.48305704],
                               [ 75.87577087,  35.93555343],
                               [ 92.6430865 ,  72.51349166],
                               [198.60379193, 186.63266023]])
        self.anchors = np.array([-(anchors_wh[:, 0] * 0.5), -(anchors_wh[:, 1] * 0.5),
                                 (anchors_wh[:, 0] * 0.5), (anchors_wh[:, 1] * 0.5)]).transpose(1, 0)

        # self.anchors = np.array([[-52, -16, 52, 16],
        #                          [-44, -20, 44, 20],
        #                          [-32, -32, 32, 32],
        #                          [-20, -40, 20, 40],
        #                          [-16, -48, 16, 48]])

        # size = self.stride * self.stride
        # count = 0
        # for r in self.ratios:
        #     ws = int(math.sqrt(size*1. / r))
        #     hs = int(ws * r)

        #     for s in self.scales:
        #         w = ws * s
        #         h = hs * s
        #         self.anchors[count][:] = [-w*0.5, -h*0.5, w*0.5, h*0.5][:]
        #         count += 1

        print(f"anchors:\n {self.anchors}")

    def generate_all_anchors(self, im_c, size):
        """
        im_c: image center = search image // 2 = 255 // 2 = 127
        size: feature map after correlation
              = (instance_size // 8) - (exemplar_size // 8) + 1
              這是 correlation 的公式
        """
        if self.image_center == im_c and self.size == size:
            ipdb.set_trace()
            return False
        self.image_center = im_c
        self.size = size

        # 到底為甚麼要加這個??? 害我要整個重新 train
        # a0x = im_c - size // 2 * self.stride
        # ori = np.array([a0x] * 4, dtype=np.float32)
        # zero_anchors = self.anchors + ori

        # x1 = zero_anchors[:, 0]
        # y1 = zero_anchors[:, 1]
        # x2 = zero_anchors[:, 2]
        # y2 = zero_anchors[:, 3]

        x1 = self.anchors[:, 0]
        y1 = self.anchors[:, 1]
        x2 = self.anchors[:, 2]
        y2 = self.anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1),
                             [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),      # (4, 5, 25, 25)
                            np.stack([cx, cy, w, h]).astype(np.float32))       # (4, 5, 25, 25)
        return True
