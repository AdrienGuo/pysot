# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from pysot.core.config import cfg
from pysot.utils.bbox import IoU, corner2center, target_overlaps, target_delta
from pysot.utils.anchor import Anchors

import ipdb
DEBUG = cfg.DEBUG

class AnchorTarget:
    def __init__(self,):
        self.anchors = Anchors(cfg.ANCHOR.STRIDE,
                               cfg.ANCHOR.RATIOS,
                               cfg.ANCHOR.SCALES)

        self.anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE//2,
                                          size=cfg.TRAIN.OUTPUT_SIZE)

    def __call__(self, target, size, neg=False):
        """
        Args:
            target: box (x1, y1, x2, y2)
            size: cfg.TRAIN.OUTPUT_SIZE=25
        Return:
            cls: anchor 的類別 (-1 ignore, 0 negative, 1 positive)
            delta: 
            delta_weight: 
            overlap: 
        """
        anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)

        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)

        if neg:
            # l = size // 2 - 3
            # r = size // 2 + 3 + 1
            # cls[:, l:r, l:r] = 0
            
            # randomly get only "one" target from all targets
            random_pick = np.random.randint(low=len(tcx), size=1)
            tcx = tcx[random_pick]
            tcy = tcy[random_pick]

            cx = size // 2
            cy = size // 2
            cx += int(np.ceil((tcx - cfg.TRAIN.SEARCH_SIZE // 2) /
                      cfg.ANCHOR.STRIDE + 0.5))
            cy += int(np.ceil((tcy - cfg.TRAIN.SEARCH_SIZE // 2) /
                      cfg.ANCHOR.STRIDE + 0.5))
            l = max(0, cx - 3)
            r = min(size, cx + 4)
            u = max(0, cy - 3)
            d = min(size, cy + 4)
            cls[:, u:d, l:r] = 0

            neg, neg_num = select(np.where(cls == 0), cfg.TRAIN.NEG_NUM)
            cls[:] = -1
            cls[neg] = 0

            overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
            return cls, delta, delta_weight, overlap

        anchor_box = self.anchors.all_anchors[0]            # anchor_box: [(x1, y1, x2, y2)=4, anchor_num=5, feature_width=25, feature_height=25]
        anchor_center = self.anchors.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], \
            anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], \
            anchor_center[2], anchor_center[3]

        
        # 把 target 疊起來變成 [4, K]
        target_stack = np.stack((target[0], target[1], target[2], target[3]))
        if DEBUG:
            print(f"anchor_box shape: {anchor_box.shape}")
            print(f"target_stack shape: {target_stack.shape}")

        # 多個 target 的 overlap 算法
        overlaps = target_overlaps(
            np.ascontiguousarray(anchor_box, dtype=np.float32),
            np.ascontiguousarray(target_stack, dtype=np.float32))       # overlaps: [N, K]
        if DEBUG:
            print(f"overlaps shape: {overlaps.shape}")

        # 找 anchor 要對應到哪個 target
        # 參考 https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/rpn/anchor_target_layer.py#L130
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps]
        overlap = np.reshape(max_overlaps, anchor_box.shape[-3:])
        if DEBUG:
            print(f"overlap shape: {overlap.shape}")

        # 遇到多個 target 的問題了
        # 參考 https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/model.py#L1526
        delta = target_delta(anchor_center, target, argmax_overlaps)    # delta: [4, 5, 25, 25]
        if DEBUG:
            print(f"delta shape: {delta.shape}")
        
        # delta[0] = (tcx - cx) / w
        # delta[1] = (tcy - cy) / h
        # delta[2] = np.log(tw / w)
        # delta[3] = np.log(th / h)

        # overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > cfg.TRAIN.THR_HIGH)        # pos (positive): 3維的，就是 anchor_box[-3:] 的維度
        neg = np.where(overlap < cfg.TRAIN.THR_LOW)

        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)           # 最多只會選 16 個 anchors 出來
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        cls[pos] = 1
        delta_weight[pos] = 1. / (pos_num + 1e-6)       # anchor 的數量越少, weight 越高 (why)
        # 這個 delta_weight 讓我找好久... 07/14/2022

        cls[neg] = 0
        return cls, delta, delta_weight, overlap
