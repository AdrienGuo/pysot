# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import time

import ipdb
import numpy as np
from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.utils.bbox import IoU, corner2center, target_delta, target_overlaps
from pysot.utils.check_image import draw_box, save_image

DEBUG = cfg.DEBUG

class AnchorTarget:
    def __init__(self):
        self.anchors = Anchors(cfg.ANCHOR.STRIDE,
                               cfg.ANCHOR.RATIOS,
                               cfg.ANCHOR.SCALES)

        self.anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE // 2,
                                          size=cfg.TRAIN.OUTPUT_SIZE)

        self._allowed_border = cfg.TRAIN.ALLOWED_BORDER

    def __call__(self, target, size, neg=False, index=None):
        """
        Args:
            target: box (x1, y1, x2, y2)
            size: cfg.TRAIN.OUTPUT_SIZE
        Return:
            cls: anchor 的類別 (-1 ignore, 0 negative, 1 positive)
            delta: 
            delta_weight: 
            overlap: 
        """
        # anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        anchor_num = cfg.ANCHOR.ANCHOR_NUM

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

        # anchor_box: [(x1, y1, x2, y2)=4, anchor_num=5, feature_width=25, feature_height=25]
        # anchor_center: [(cx, cy, w, h)=4, anchor_num=5, feature_width=25, feature_height=25]
        # print(f"anchor_box: {anchor_box[:, 0, 0, 0]}")
        # ipdb.set_trace()

        anchor_box = self.anchors.all_anchors[0]
        anchor_center = self.anchors.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], \
            anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], \
            anchor_center[2], anchor_center[3]

        # 把 target 疊起來變成 (4, K)
        target_stack = np.stack((target[0], target[1], target[2], target[3]))

        # TODO: only keep anchors inside the image
        # print(f"anchor_box: {anchor_box.shape}")
        
        # anchor_box_shape = anchor_box.shape[-3:]
        # for i in range(anchor_box.shape[0]):
        #     anchor_box = anchor_box.reshape((4, -1))
        #     anchor_center = anchor_center.reshape((4, -1))
        # idxs_inside = np.where(
        #     (anchor_box[0] >= -self._allowed_border)
        #     & (anchor_box[1] >= -self._allowed_border)
        #     & (anchor_box[2] < cfg.TRAIN.SEARCH_SIZE + self._allowed_border)
        #     & (anchor_box[3] < cfg.TRAIN.SEARCH_SIZE + self._allowed_border)
        # )[0]

        # print(f"idx_inside: {idxs_inside.shape}")
        # anchor_box = anchor_box[:, idxs_inside]
        # anchor_center = anchor_center[:, idxs_inside]

        # for i in range(anchor_box.shape[0]):
        #     anchor_box[i] = anchor_box[i].reshape((4, *anchor_box_shape))
        #     anchor_center[i] = anchor_center[i].reshape((4, *anchor_box_shape))
        
        # 多個 target 的 overlap 算法

        anchortarget_start = time.time()
        overlaps = target_overlaps(
            np.ascontiguousarray(anchor_box, dtype=np.float32),
            np.ascontiguousarray(target_stack, dtype=np.float32))       # overlaps: (N, K)
        anchortarget_end = time.time()

        print(f"=== anchor target duration: {anchortarget_end - anchortarget_start} s ===")

        # 找 anchor 要對應到哪個 target
        # 參考 https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/rpn/anchor_target_layer.py#L130
        # fg label: for each gt, anchor with highest overlap
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        # fg label: for each anchor, gt with the highest overlap
        argmax_overlaps = overlaps.argmax(axis=1)       # argmax_overlaps: (all_anchor_num, )
        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps]

        # 我確定這個 reshape 不會影響排列順序
        max_overlaps = np.reshape(max_overlaps, anchor_box.shape[-3:])
        assert max_overlaps[max_overlaps < 0].size == 0, "max_overlaps has iou smaller than 0!!!"
        assert max_overlaps[max_overlaps > 1].size == 0, "max_overlaps has iou bigger than 1!!!"

        # 遇到多個 target 的問題了
        # 參考 https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/model.py#L1526
        delta = target_delta(anchor_center, target, argmax_overlaps)    # delta: (4, anchor_num, 25, 25)
        
        # delta[0] = (tcx - cx) / w
        # delta[1] = (tcy - cy) / h
        # delta[2] = np.log(tw / w)
        # delta[3] = np.log(th / h)

        # max_overlaps = IoU([x1, y1, x2, y2], target)

        # fg label: above threshold IOU
        pos = np.where(max_overlaps > cfg.TRAIN.THR_HIGH)        # pos (positive): 3維的，就是 anchor_box[-3:] 的維度
        neg = np.where(max_overlaps < cfg.TRAIN.THR_LOW)

        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)           # 最多只會選 16 個 anchors 出來
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        cls = np.reshape(cls, (-1))
        cls[gt_argmax_overlaps] = 1     # 將與 gt 有最大 IoU 的直接判為 fg
        cls = np.reshape(cls, (anchor_num, size, size))
        cls[pos] = 1                    # 將所有的 anchor 與 gt 的 IoU 有大於 threshold 的判為 fg
        delta_weight[pos] = 1. / (pos_num + 1e-6)       # 正樣本的 anchor 數量越少, weight 越高 (why)
        # 這個 delta_weight 讓我找好久... 07/14/2022

        cls[neg] = 0

        # 印出原本的 anchor
        all_anchors = self.anchors.all_anchors[0].copy()
        all_anchors[2] = all_anchors[2] - all_anchors[0]
        all_anchors[3] = all_anchors[3] - all_anchors[1]

        # anchor = all_anchors[:, 0:1, -1, 0]      # 選要哪一個 anchor
        # anchor = np.transpose(anchor, (1, 0))
        # img = np.zeros((cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE, 3))   # 製作黑色底圖
        # anchor_image = draw_box(img, anchor)
        # anchor_dir = "./image_check/train/anchor/"
        # anchor_path = os.path.join(anchor_dir, "anchor.jpg")
        # save_image(anchor_image, anchor_path)
        # print(f"save anchor image to: {anchor_path}")

        # # 把 anchor 印出來
        # pos = np.where(cls == 1)
        # pos_anchors = all_anchors[:, pos[0], pos[1], pos[2]]
        # pos_anchors = np.transpose(pos_anchors, (1, 0))
        # img = np.zeros((cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE, 3))   # 製作黑色底圖
        # pos_image = draw_box(img, pos_anchors)
        # pos_dir = "./image_check/train/pos/"
        # pos_path = os.path.join(pos_dir, f"{index}.jpg")
        # save_image(pos_image, pos_path)
        # print(f"save pos image to: {pos_path}")

        # ipdb.set_trace()

        return cls, delta, delta_weight, max_overlaps
