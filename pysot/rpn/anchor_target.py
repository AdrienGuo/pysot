# Copyright (c) SenseTime. All Rights Reserved.

# Actually, it is me to move this part to train on the gpu card.
# The original code put this part in Dataset, which means it will be processed on the cpu card.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import time

import ipdb
import numpy as np
import torch
import torch.nn as nn
from pysot.core.config import cfg
from pysot.rpn.anchor import Anchors
from pysot.rpn.bbox import bbox_overlaps_batch, bbox_transform_batch
from pysot.utils.check_image import draw_box, save_image

DEBUG = cfg.DEBUG


class AnchorTarget(nn.Module):
    def __init__(self):
        super(AnchorTarget, self).__init__()

        self.anchors = Anchors(cfg.ANCHOR.STRIDE,
                               cfg.ANCHOR.RATIOS,
                               cfg.ANCHOR.SCALES)

        all_anchors = self.anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE // 2,
                                                        size=cfg.TRAIN.OUTPUT_SIZE)
        self.all_anchors = torch.from_numpy(all_anchors).cuda()     # (all_anchor_num, 4) #corner

        # self._allowed_border = cfg.TRAIN.ALLOWED_BORDER

    def forward(self, gt_boxes, size, neg=False, index=None):
        """
        Args:
            gt_boxes: (b, K, 4) #corner
            size: cfg.TRAIN.OUTPUT_SIZE
        Return:
            cls: anchor 的類別 (-1 ignore, 0 negative, 1 positive)
            delta: 
            delta_weight: 
            overlap: 
        """
        # anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        # anchor_num = cfg.ANCHOR.ANCHOR_NUM
        batch_size = gt_boxes.size(0)

        # label: 1 is positive, 0 is negative, -1 is dont care
        # cls = gt_boxes.new(batch_size, self.all_anchors.size(0)).fill_(-1)
        cls = -1 * gt_boxes.new_ones((batch_size, self.all_anchors.size(0)), dtype=torch.int64)    # cls default is -1
        bbox_weights = gt_boxes.new_zeros((batch_size, self.all_anchors.size(0)))
        bbox_inside_weights = gt_boxes.new_zeros((batch_size, self.all_anchors.size(0)))
        bbox_outside_weights = gt_boxes.new_zeros((batch_size, self.all_anchors.size(0)))

        # -1 ignore 0 negative 1 positive
        # cls = -1 * np.ones((anchor_num, size, size), dtype=np.int64)
        # delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        # delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        # tcx, tcy, tw, th = corner2center(gt_boxes)

        # if neg:
        #     # l = size // 2 - 3
        #     # r = size // 2 + 3 + 1
        #     # cls[:, l:r, l:r] = 0
            
        #     # randomly get only "one" gt_boxes from all gt_boxess
        #     random_pick = np.random.randint(low=len(tcx), size=1)
        #     tcx = tcx[random_pick]
        #     tcy = tcy[random_pick]

        #     cx = size // 2
        #     cy = size // 2
        #     cx += int(np.ceil((tcx - cfg.TRAIN.SEARCH_SIZE // 2) /
        #               cfg.ANCHOR.STRIDE + 0.5))
        #     cy += int(np.ceil((tcy - cfg.TRAIN.SEARCH_SIZE // 2) /
        #               cfg.ANCHOR.STRIDE + 0.5))
        #     l = max(0, cx - 3)
        #     r = min(size, cx + 4)
        #     u = max(0, cy - 3)
        #     d = min(size, cy + 4)
        #     cls[:, u:d, l:r] = 0

        #     neg, neg_num = select(np.where(cls == 0), cfg.TRAIN.NEG_NUM)
        #     cls[:] = -1
        #     cls[neg] = 0

        #     overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
        #     return cls, delta, delta_weight, overlap

        # anchor_box: [(x1, y1, x2, y2)=4, anchor_num=5, feature_width=25, feature_height=25]
        # anchor_center: [(cx, cy, w, h)=4, anchor_num=5, feature_width=25, feature_height=25]
        # anchor_box = self.anchors.all_anchors[0]
        # anchor_center = self.anchors.all_anchors[1]
        # x1, y1, x2, y2 = anchor_box[0], anchor_box[1], \
        #     anchor_box[2], anchor_box[3]
        # cx, cy, w, h = anchor_center[0], anchor_center[1], \
        #     anchor_center[2], anchor_center[3]

        # 把 gt_boxes 疊起來變成 (4, K)
        # gt_boxes_stack = np.stack((gt_boxes[0], gt_boxes[1], gt_boxes[2], gt_boxes[3]))

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

        self.all_anchors = self.all_anchors.type_as(gt_boxes)   # move to the same gpu

        ####################################################################
        # 找 anchor 與 target 之間的對應關係
        ####################################################################
        # 參考 https://github.com/jwyang/faster-rcnn.pytorch/blob/f9d984d27b48a067b29792932bcb5321a39c1f09/lib/model/rpn/anchor_target_layer.py#L98
        overlaps = bbox_overlaps_batch(self.all_anchors, gt_boxes)      # overlaps: (b, K, N=100)

        max_overlaps, argmax_overlaps = torch.max(overlaps, dim=2)      # max_overlaps, argmax_overlaps: (b, K)
        gt_max_overlaps, _ = torch.max(overlaps, dim=1)                 # gt_max_overlaps: (b, N)

        ####################################################################
        # 在 faster-rcnn 原本的程式碼裡面，
        # <= THR_LOW 是放在後面，要去 clobber positives，
        # 但是我怕我連最高 IOU 的都沒有超過 THR_LOW，
        # 會變成完全沒有 positives 的情況，所以把這個往前放
        ####################################################################
        # fg label: above threshold IOU
        cls[max_overlaps >= cfg.TRAIN.THR_HIGH] = 1
        cls[max_overlaps <= cfg.TRAIN.THR_LOW] = 0

        ####################################################################
        # 計算 cls
        ####################################################################
        # 看不懂這步在幹嘛，我猜是將與 gt 有最大 IoU 的 anchor 直接判為 fg (對 就是 我直覺頗準呢)
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), dim=2)   # keep (b, K)
        if torch.sum(keep) > 0:
            cls[keep > 0] = 1

        sum_fg = torch.sum((cls == 1).int(), 1)
        sum_bg = torch.sum((cls == 0).int(), 1)

        # 將過多的 pos, neg 刪除
        for i in range(batch_size):
            # subsample positive cls if we have too many
            if sum_fg[i] > cfg.TRAIN.POS_NUM:
                fg_inds = torch.nonzero(cls[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                # rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0) - cfg.TRAIN.POS_NUM]]
                cls[i][disable_inds] = -1

#           num_bg = cfg.TRAIN.BATCH_SIZE - sum_fg[i]
            # num_bg = cfg.TRAIN.BATCH_SIZE - torch.sum((cls == 1).int(), 1)[i]

            # subsample negative cls if we have too many
            if sum_bg[i] > cfg.TRAIN.TOTAL_NUM:
                bg_inds = torch.nonzero(cls[i] == 0).view(-1)
                # rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0) - cfg.TRAIN.TOTAL_NUM]]
                cls[i][disable_inds] = -1

        ####################################################################
        # 計算 delta
        ####################################################################
        # 不懂這個 offset 要幹嘛
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        # bbox_targets (b, K, 4)
        bbox_targets = bbox_transform_batch(self.all_anchors, gt_boxes.view(-1, 4)[argmax_overlaps.view(-1), :].view(batch_size, -1, 4))

        pos_num = torch.sum(cls[i] == 1)
        # if pos_num == 0:
        #     pos_num = torch.tensor(1, dtype=torch.int64)
        bbox_weights[cls == 1] = 1.0 / pos_num.item()       # 正樣本的 anchor 數量越少, weight 越高 (why)

        ####################################################################
        # 換成之後要處理的樣子
        ####################################################################
        # label_cls: torch.Size([32, 11, 17, 17])
        # cls (b, anchor_num, size, size)
        cls = cls.view(batch_size, size, size, cfg.ANCHOR.ANCHOR_NUM).permute(0, 3, 1, 2).contiguous()

        # # 印出原本的 anchor
        # all_anchors = self.all_anchors.view(size, size, cfg.ANCHOR.ANCHOR_NUM, 4).permute(3, 2, 0, 1).cpu().numpy().copy()
        # all_anchors[2] = all_anchors[2] - all_anchors[0]
        # all_anchors[3] = all_anchors[3] - all_anchors[1]

        # anchor = all_anchors[:, :, size//2, size//2]      # 選要哪一個 anchor
        # anchor = np.transpose(anchor, (1, 0))
        # img = np.zeros((cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE, 3))   # 製作黑色底圖
        # anchor_image = draw_box(img, anchor)
        # anchor_dir = "./image_check/train/anchor/"
        # anchor_path = os.path.join(anchor_dir, "anchor.jpg")
        # save_image(anchor_image, anchor_path)
        # print(f"save anchor image to: {anchor_path}")

        # # 把 anchor 印出來
        # cls_check = cls[0].cpu().numpy().copy()
        # pos = np.where(cls_check == 1)
        # pos_anchors = all_anchors[:, pos[0], pos[1], pos[2]]
        # pos_anchors = np.transpose(pos_anchors, (1, 0))
        # img = np.zeros((cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE, 3))   # 製作黑色底圖
        # pos_image = draw_box(img, pos_anchors)
        # pos_dir = "./image_check/train/pos/"
        # pos_path = os.path.join(pos_dir, f"{index}.jpg")
        # save_image(pos_image, pos_path)
        # print(f"save pos image to: {pos_path}")

        # ipdb.set_trace()

        # bbox_targets = bbox_targets.view(batch_size, size, size, cfg.ANCHOR.ANCHOR_NUM * 4).permute(0, 3, 1, 2).contiguous()
        # label_loc: torch.Size([32, 4, 11, 17, 17])
        # bbox_targets (b, 4, anchor_num, size, size)
        bbox_targets = bbox_targets.view(batch_size, size, size, cfg.ANCHOR.ANCHOR_NUM, 4).permute(0, 4, 3, 1, 2).contiguous()
        bbox_weights = bbox_weights.view(batch_size, size, size, cfg.ANCHOR.ANCHOR_NUM).permute(0, 3, 1, 2).contiguous()

        return cls, bbox_targets, bbox_weights, max_overlaps

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
