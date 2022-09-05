# Copyright (c) SenseTime. All Rights Reserved.

# Actually, it is me to move this part to train on the gpu card.
# The original code put this part in Dataset, which means it will be processed on the cpu card.
# So, the original version is insanely slowwwww.

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
        # score_size: correlation 做完後的 size，下面這個是公式
        score_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        all_anchors = self.anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE // 2,
                                                        size=score_size)
        # self.all_anchors: (all_anchor_num, 4) #corner
        self.all_anchors = torch.from_numpy(all_anchors).cuda()

        self._allowed_border = cfg.TRAIN.ALLOWED_BORDER

    def forward(self, gt_boxes, size, neg=False, image_name=None, idx=None):
        """
        Args:
            gt_boxes: (b, K, 4) #corner
            size: cfg.TRAIN.OUTPUT_SIZE
            idx: 存 anchor 在 search image 上的位置的檔名 (會跟 template, search 對應)
                 要用的話 batch size 要等於 1
        Return:
            cls: anchor 的類別 (-1 ignore, 0 negative, 1 positive)
            delta: 
            delta_weight: 
            overlap: 
        """
        # anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        # anchor_num = cfg.ANCHOR.ANCHOR_NUM
        batch_size = gt_boxes.size(0)

        # -1 ignore, 0 negative, 1 positive
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

        self.all_anchors = self.all_anchors.type_as(gt_boxes)   # move to the same gpu

        # Only keep anchors which are "totally" inside the image
        keep = ((self.all_anchors[:, 0] >= -self._allowed_border) &
                (self.all_anchors[:, 1] >= -self._allowed_border) &
                (self.all_anchors[:, 2] < cfg.TRAIN.SEARCH_SIZE + self._allowed_border) &
                (self.all_anchors[:, 3] < cfg.TRAIN.SEARCH_SIZE + self._allowed_border))
        inds_inside = torch.nonzero(keep).view(-1)

        anchors = self.all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        cls = -1 * gt_boxes.new_ones((batch_size, inds_inside.size(0)), dtype=torch.int64)    # cls default is -1
        bbox_weights = gt_boxes.new_zeros((batch_size, inds_inside.size(0)))
        bbox_inside_weights = gt_boxes.new_zeros((batch_size, inds_inside.size(0)))
        bbox_outside_weights = gt_boxes.new_zeros((batch_size, inds_inside.size(0)))

        ####################################################################
        # 找 anchor 與 target 之間的對應關係
        ####################################################################
        # 參考 https://github.com/jwyang/faster-rcnn.pytorch/blob/f9d984d27b48a067b29792932bcb5321a39c1f09/lib/model/rpn/anchor_target_layer.py#L98
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)      # overlaps: (b, K, N=100)

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

        ####################################################################
        # 將過多的 pos, neg 刪除
        ####################################################################
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
        bbox_targets = bbox_transform_batch(anchors, gt_boxes.view(-1, 4)[argmax_overlaps.view(-1), :].view(batch_size, -1, 4))

        pos_num = torch.sum(cls[i] == 1)
        # if pos_num == 0:
        #     pos_num = torch.tensor(1, dtype=torch.int64)
        bbox_weights[cls == 1] = 1.0 / pos_num.item()       # 正樣本的 anchor 數量越少, weight 越高 (why)

        cls = _unmap(cls, self.all_anchors.size(0), inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, self.all_anchors.size(0), inds_inside, batch_size, fill=0)
        bbox_weights = _unmap(bbox_weights, self.all_anchors.size(0), inds_inside, batch_size, fill=0)

        ####################################################################
        # 換成之後要處理的格式
        ####################################################################
        # label_cls: torch.Size([32, 11, 17, 17])
        # cls (b, anchor_num, size, size)
        cls = cls.view(batch_size, size, size, cfg.ANCHOR.ANCHOR_NUM).permute(0, 3, 1, 2).contiguous()

        # bbox_targets = bbox_targets.view(batch_size, size, size, cfg.ANCHOR.ANCHOR_NUM * 4).permute(0, 3, 1, 2).contiguous()
        # label_loc: torch.Size([32, 4, 11, 17, 17])
        # bbox_targets (b, 4, anchor_num, size, size)
        bbox_targets = bbox_targets.view(batch_size, size, size, cfg.ANCHOR.ANCHOR_NUM, 4).permute(0, 4, 3, 1, 2).contiguous()
        bbox_weights = bbox_weights.view(batch_size, size, size, cfg.ANCHOR.ANCHOR_NUM).permute(0, 3, 1, 2).contiguous()

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # all_anchors = self.all_anchors.view(size, size, cfg.ANCHOR.ANCHOR_NUM, 4).permute(3, 2, 0, 1).cpu().numpy().copy()
        # all_anchors[2] = all_anchors[2] - all_anchors[0]    # x2 -> w
        # all_anchors[3] = all_anchors[3] - all_anchors[1]    # y2 -> h

        # # === 畫出在 black bg 的 anchor ===
        # # anchor = all_anchors[:, :, size//2, size//2]      # 選要哪一個 anchor
        # # anchor = np.transpose(anchor, (1, 0))
        # # img = np.zeros((cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE, 3))   # 製作黑色底圖
        # # anchor_image = draw_box(img, anchor)
        # # anchor_dir = "./image_check/train/anchor/"
        # # anchor_path = os.path.join(anchor_dir, "anchor.jpg")
        # # save_image(anchor_image, anchor_path)
        # # print(f"save anchor image to: {anchor_path}")

        # # === 把 pos anchor 印出來 ===
        # cls_check = cls[0].cpu().numpy().copy()
        # pos = np.where(cls_check == 1)
        # pos_anchors = all_anchors[:, pos[0], pos[1], pos[2]]    # (4, n)

        # pos_anchors = np.transpose(pos_anchors, (1, 0))
        # black_image = np.zeros((cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE, 3))   # 製作黑色底圖
        # pos_image = draw_box(black_image, pos_anchors)

        # # sub_dir 是以 "圖片名稱" 命名
        # sub_dir = os.path.join("./image_check/train/", image_name)
        # # 創 sub_dir/pos，裡面存 pos image
        # pos_dir = os.path.join(sub_dir, "pos")
        # if not os.path.isdir(pos_dir):
        #     os.makedirs(pos_dir)
        #     print(f"create dir: {pos_dir}")

        # pos_path = os.path.join(pos_dir, f"{idx}.jpg")
        # save_image(pos_image, pos_path)
        # print(f"save pos image to: {pos_path}")
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        return cls, bbox_targets, bbox_weights, max_overlaps


def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds, :] = data
    return ret
