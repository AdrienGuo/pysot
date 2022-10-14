# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ipdb
import numpy as np
import torch
from pysot.core.config import cfg


def bbox_overlaps_batch(anchors, gt_boxes):
    """ caculate interection over union
        原本的 gt_boxes 只有一個，但因為我們的 gt_boxes 會有很多個，所以這裡需要改寫
        參考 Faster R-CNN: https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/bbox.pyx
    Args:
        anchors: (N, 4) #corner #tensor
        gt_boxes: (b, G, 4) #corner #tensor
    Returns:
        overlaps: (b, N, G) ndarray of overlap between anchors(N) and targets(G)
    """
    # batch > 1 的 overlaps 算法
    # Reference: https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/rpn/bbox_transform.py
    batch_size = gt_boxes.size(0)
    N = anchors.size(0)     # all_anchor_num
    G = gt_boxes.size(1)    # max_num_box
    anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
    gt_boxes = gt_boxes.contiguous()

    # 還是不懂為甚麼要 +1 ??
    anchors_w = anchors[:, :, 2] - anchors[:, :, 0] + 1
    anchors_h = anchors[:, :, 3] - anchors[:, :, 1] + 1
    anchors_area = (anchors_w * anchors_h).view(batch_size, N, 1)

    gt_boxes_w = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1
    gt_boxes_h = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1
    gt_boxes_area = (gt_boxes_w * gt_boxes_h).view(batch_size, 1, G)

    gt_area_zero = (gt_boxes_w == 1) & (gt_boxes_h == 1)
    anchors_area_zero = (anchors_w == 1) & (anchors_h == 1)

    # 這個做法超強!! 完全不用 for loop，也太聰明
    boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, G, 4)
    query_boxes = gt_boxes.view(batch_size, 1, G, 4).expand(batch_size, N, G, 4)

    iw = torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) \
        - torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1
    iw[iw < 0] = 0

    ih = torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) \
        - torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1
    ih[ih < 0] = 0
    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    # 這步應該根本不用做吧? 因為不可能發生阿?
    # mask the overlap here.
    overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, G).expand(batch_size, N, G), 0)
    overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, G), -1)

    return overlaps


def bbox_transform_batch(ex_rois, gt_rois):
    """
    Args:
        ex_rois: (N, 4)
        gt_rois: (b, N, 4)
    """
    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))
    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh), 2)

    return targets
