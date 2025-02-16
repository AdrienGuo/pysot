# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import namedtuple

import ipdb
import numpy as np
from pysot.core.config import cfg

DEBUG = cfg.DEBUG

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
# alias
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def ratio2real(image, box):
    h, w = image.shape[:2]

    box[:, 0] = box[:, 0] * w
    box[:, 1] = box[:, 1] * h
    box[:, 2] = box[:, 2] * w
    box[:, 3] = box[:, 3] * h

    return box


def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
               or np.array (N * 4)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    elif isinstance(center, list):
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2
    elif isinstance(center, np.ndarray):
        center[:, 0] = center[:, 0] - center[:, 2] * 0.5    # x1
        center[:, 1] = center[:, 1] - center[:, 3] * 0.5    # y1
        center[:, 2] = center[:, 0] + center[:, 2]    # x2 = x1 + w
        center[:, 3] = center[:, 1] + center[:, 3]    # y2 = y1 + h
        return center    # (N, 4)


def target_overlaps(anchor, target):
    """ caculate interection over union
        原本的 target 只有一個，但因為我們的 target 會有很多個，所以這裡需要改寫
        參考 Faster R-CNN: https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/bbox.pyx
    Args:
        anchor: (x1, y1, x2, y2), anchor
        target: (), target (can call it bbox?)
    Returns:
        iou: (N, K) ndarray of overlap between anchors(N) and targets(K)
    """
    # 把 anchor 拉成 (4, N), N = 5*25*25
    anchor_flatten = np.reshape(anchor, (4, -1))
    N = len(anchor_flatten[0])  # anchor 的數量
    K = len(target[0])          # target 的數量
    overlaps = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        target_area = (target[2, k] - target[0, k]) * (target[3, k] - target[1, k])
        assert target_area > 0, f"target_area: {target_area}"
        for n in range(N):
            anchor_area = (anchor_flatten[2, n] - anchor_flatten[0, n]) * (anchor_flatten[3, n] - anchor_flatten[1, n])
            intersection_x1 = np.maximum(anchor_flatten[0, n], target[0, k])
            intersection_y1 = np.maximum(anchor_flatten[1, n], target[1, k])
            intersection_x2 = np.minimum(anchor_flatten[2, n], target[2, k])
            intersection_y2 = np.minimum(anchor_flatten[3, n], target[3, k])
            intersection_width = np.maximum(0, intersection_x2 - intersection_x1)
            intersection_height = np.maximum(0, intersection_y2 - intersection_y1)
            intersection_area = intersection_width * intersection_height

            ua = target_area + anchor_area - intersection_area
            assert ua > 0, "wrong" + \
                f"anchor_flatten: {anchor_flatten[:, n]}" \
                f"target: {target[:, k]}"
            overlaps[n, k] = intersection_area / ua

    # 確保 iou 都大於 0
    assert overlaps[overlaps < 0].size == 0, "overlaps has area smaller than 0!!!"
    # 確保 iou 都小於 1
    assert overlaps[overlaps > 1].size == 0, "overlaps has area bigger than 1!!!"

    return overlaps


def target_delta(anchor, target, argmax):
    """ 算每個 anchor 跟對應 target 的 delta
        參考 https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/model.py#L1526
    Args:
        anchor (4(corner), anchor_num, output_size, output_size)
        target (4(center), target_num)
        argmax (all_anchor_num, ): for each anchor, the "idx" of gt with the highest overlap
    Return:
        delta:
    """
    delta = np.zeros((4, argmax.shape[0]), dtype=np.float32)
    anchor_flatten = np.reshape(anchor, (4, -1))
    acx, acy, aw, ah = anchor_flatten[0], anchor_flatten[1], anchor_flatten[2], anchor_flatten[3]
    tcx, tcy, tw, th = corner2center(target)
    for i in range(argmax.shape[0]):
        # Closest target (it might have IoU < threshold)
        # 這裡就一樣照算，因為之後會乘上 delta_weight 所以就算 IoU = 0 也沒關係
        closest_target_idx = argmax[i]
        tcx_closest = tcx[closest_target_idx]
        tcy_closest = tcy[closest_target_idx]
        tw_closest = tw[closest_target_idx]
        th_closest = th[closest_target_idx]

        delta[0] = (tcx_closest - acx) / aw
        delta[1] = (tcy_closest - acy) / ah
        delta[2] = np.log(tw_closest / aw)
        delta[3] = np.log(th_closest / ah)

    delta = np.reshape(delta, anchor.shape)

    return delta


def IoU(rect1, rect2):
    """ caculate interection over union
    Args:
        rect1: (x1, y1, x2, y2)
        rect2: (x1, y1, x2, y2)
    Returns:
        iou
    """
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2-x1) * (y2-y1)
    target_a = (tx2-tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou


def cxy_wh_2_rect(pos, sz):
    """ convert (cx, cy, w, h) to (x1, y1, w, h), 0-index
    """
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])


def rect_2_cxy_wh(rect):
    """ convert (x1, y1, w, h) to (cx, cy, w, h), 0-index
    """
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2]), \
        np.array([rect[2], rect[3]])


def cxy_wh_2_rect1(pos, sz):
    """ convert (cx, cy, w, h) to (x1, y1, w, h), 1-index
    """
    return np.array([pos[0]-sz[0]/2+1, pos[1]-sz[1]/2+1, sz[0], sz[1]])


def rect1_2_cxy_wh(rect):
    """ convert (x1, y1, w, h) to (cx, cy, w, h), 1-index
    """
    return np.array([rect[0]+rect[2]/2-1, rect[1]+rect[3]/2-1]), \
        np.array([rect[2], rect[3]])


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
        這個步驟對我來說超級沒用，應該是為了處理當 nv == 8 的時候 (其實我也不確定，不想研究)
    """
    nv = region.shape[0]
    if nv == 8:     # nv 怎麼可能等於 8 阿??
        assert False, "WHAT, nv == 8!?"
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + (w / 2)
        cy = y + (h / 2)
    return cx, cy, w, h


def get_min_max_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by mim-max box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        w = x2 - x1
        h = y2 - y1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h
