from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import os
from re import template

import cv2
import ipdb
import numpy as np
from pysot.utils.bbox import Center, Corner, center2corner, corner2center
from pysot.utils.check_image import draw_box, save_image

from da


# 為了幫 template 加入背景
# 使用論文的裁切方式才會用到
def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop

# 使用論文的裁切方式才會用到
def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop(image, box, out_sz, padding=(0, 0, 0)):
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]

    ####################################################################
    # 處理 image
    ####################################################################
    # === box 小於輸出的大小，則 template 不做縮放 ===
    if box_w < out_sz and box_h < out_sz:
        template_crop = image[box[1]: box[3], box[0]: box[2]]  # 先高再寬
        r = 1    # 不做縮放
        # 計算移到中心需要的位移
        x = (out_sz / 2) - (box_w / 2)
        y = (out_sz / 2) - (box_h / 2)
        mapping = np.array([[r, 0, x],
                            [0, r, y]]).astype(np.float)
        img_crop = cv2.warpAffine(template_crop, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    # === box 大於輸出的大小，則根據長邊對 template 做縮放 ===
    else:
        template_crop = image[box[1]: box[3], box[0]: box[2]]  # 先高再寬
        if box_w >= box_h:
            r = out_sz / box_w
        else:
            r = out_sz / box_h
        # 計算移到中心需要的位移
        x = (out_sz / 2) - (r * box_w) / 2
        y = (out_sz / 2) - (r * box_h) / 2
        mapping = np.array([[r, 0, x],
                            [0, r, y]]).astype(np.float)
        img_crop = cv2.warpAffine(template_crop, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)

    return img_crop, r, box


def crop_method(image, bbox, context_amount=0.5, exemplar_size=127, instance_size=255, padding=(0, 0, 0)):
    img_h, img_w = image.shape[:2]
    target_center = [(bbox.x2 + bbox.x1) / 2., (bbox.y2 + bbox.y1) / 2.]
    target_size = [(bbox.x2 - bbox.x1), (bbox.y2 - bbox.y1)]    # [w, h]

    # === 老師要的裁切方式 ===
    crop_w = target_size[0] * 2
    crop_h = target_size[1] * 2
    if crop_w > exemplar_size:
        crop_w = exemplar_size
    if crop_h > exemplar_size:
        crop_h = exemplar_size
    # s_z: [x1, y1, x2, y2]
    s_z = [target_center[0] - (crop_w / 2), target_center[1] - (crop_h / 2),
           target_center[0] + (crop_w / 2), target_center[1] + (crop_h / 2)]
    # === 論文原版裁切方式 ===
    # wc_z = target_size[1] + context_amount * sum(target_size)
    # hc_z = target_size[0] + context_amount * sum(target_size)
    # crop_wh = np.sqrt(wc_z * hc_z)
    # s_z = [target_center[0] - (crop_wh / 2), target_center[1] - (crop_wh / 2),
    #        target_center[0] + (crop_wh / 2), target_center[1] + (crop_wh / 2)]

    # 超出原圖範圍的裁掉
    s_z[0] = max(0, s_z[0])
    s_z[1] = max(0, s_z[1])
    s_z[2] = min(img_w, s_z[2])
    s_z[3] = min(img_h, s_z[3])
    s_z = list(map(lambda x: int(x), s_z))

    z = crop(image, s_z, exemplar_size, padding)

    return z


class PCBCropAmy:
    def __init__(self, template_size, search_size, type) -> None:
        self.template_size = template_size
        self.search_size = search_size
        self.type = type

    def _template_crop(self, image, box, bg, context_amount=0, padding=(0, 0, 0)):
        img_h, img_w = image.shape[:2]

        ####################################################################
        # 將 image 縮放成和 search image 一樣的大小 (有 padding)
        ####################################################################
        if img_w >= img_h:
            r = self.search_size / img_w
        else:
            r = self.search_size / img_h
        x = (self.search_size / 2) - (r * img_w) / 2
        y = (self.search_size / 2) - (r * img_h) / 2
        # r 是縮放比例； x, y 是位移
        mapping = np.array([[r, 0, x],
                            [0, r, y]]).astype(np.float)
        image = cv2.warpAffine(image,
                               mapping,
                               (self.search_size, self.search_size),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=padding)

        ####################################################################
        # === 將 template box 調整成在 search image 上的位置 ===
        # 跟下面的 _search_crop 幾乎一模一樣；造成這樣是因為
        # Augmentation 這個 Object，template 和 search 各有一個 (他們之間的參數不能互通)
        # 加上我不想把屬於 template 的東西拿給 search 做 (很亂)，乾脆兩邊都做
        ####################################################################
        box = Corner(*center2corner(box))     # 超剛好前幾天才問啟恩這個用法，感謝啟恩
        box = Corner(img_w * box.x1, img_h * box.y1,
                     img_w * box.x2, img_h * box.y2)
        origin_box = box    # 在 original image 上的 box
        box = np.array(box)
        box = box[:, np.newaxis]    # box: (4, n=1)
        # 做 template box 的縮放&平移
        # [[x1, y1, 1]      [[r, 0, 0]
        #  [x2, y2, 1]]  *   [0, r, 0]
        #                    [x, y, 1]]
        # TODO: 把後面的那個 0, 0, 1 拿掉，我用不到
        ones = np.ones(box.shape[1])
        box = np.array([[box[0], box[1], ones],
                        [box[2], box[3], ones]]).astype(np.float)
        box = np.transpose(box, (2, 0, 1))
        ratio = np.array([[r, 0, 0],
                          [0, r, 0],
                          [x, y, 1]]).astype(np.float)
        box = np.dot(box, ratio)
        box = box[:, :, :-1]
        box = np.transpose(box, (1, 2, 0))
        box = np.concatenate(box, axis=0)         # ((1, 2), (x, y), n) -> ((x1, y1, x2, y2), n)
        box = Corner(box[0], box[1], box[2], box[3])

        ####################################################################
        # 在 search image 上，裁切出 template 的部分 (分成 沒有bg, 有bg)
        ####################################################################
        template_w = box.x2 - box.x1
        template_h = box.y2 - box.y1

        # 沒有 bg (應該是不會做了，效果不好)
        if bg == "nbg":
            # 將長邊變成 255
            # r: 放大or縮小比率
            template_image = image[box.y1: box.y1 + template_h, box.x1: box.x1 + template_w]
            if template_w >= template_h:
                r = self.template_size / template_w
            else:
                r = self.template_size / template_h
            # 計算移到中心需要的位移
            x = (self.template_size / 2) - (r * template_w) / 2
            y = (self.template_size / 2) - (r * template_h) / 2
            mapping = np.array([[r, 0, x],
                                [0, r, y]]).astype(np.float)
            template_image = cv2.warpAffine(template_image, mapping, (self.template_size, self.template_size), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        # 有 bg, 且 context_amount 可以決定要加入多少比例的 bg
        elif bg == "bg":
            template_image, template_ratio, box = crop_method(
                image,
                box,
                context_amount=context_amount,
                exemplar_size=self.template_size,
                instance_size=self.search_size    # 好像用不到
            )

        box = np.array(box)     # corner -> array (??)
        origin_box = np.array(origin_box)

        return template_image, template_ratio, box, origin_box

    def _search_crop(self, image, bbox, ratio, padding=(0, 0, 0)):
        search_image = image
        search_h, search_w = search_image.shape[:2]

        # === 處理 search image ===
        # 先將 image 變成 search size
        if search_w >= search_h:
            r = self.search_size / search_w
        else:
            r = self.search_size / search_h
        # 再將 r 乘上 template 的縮放比例，讓 search 跟 template 做一樣的縮放
        # (因為 template 也是這樣)
        # r = r * ratio

        x = (self.search_size / 2) - (r * search_w) / 2
        y = (self.search_size / 2) - (r * search_h) / 2
        # r 是縮放比例； x, y 是位移
        mapping = np.array([[r, 0, x],
                            [0, r, y]]).astype(np.float)
        search_image = cv2.warpAffine(search_image, mapping,
                                      (self.search_size, self.search_size),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=padding)

        # === 處理 bounding box ===
        bbox = np.transpose(bbox, (1, 0))       # (n, 4) -> (4, n)
        bbox = Corner(*center2corner(bbox))
        bbox = Corner(search_w * bbox.x1, search_h * bbox.y1,
                      search_w * bbox.x2, search_h * bbox.y2)
        bbox = np.array(bbox)
        # 做 bounding box 的縮放&平移
        # [[x1, y1, 1]      [[r, 0, 0]
        #  [x2, y2, 1]]  *   [0, r, 0]
        #                    [x, y, 1]]
        # TODO: 把後面的那個 0, 0, 1 拿掉，我用不到
        ones = np.ones(bbox.shape[1])
        bbox = np.array([[bbox[0], bbox[1], ones],
                         [bbox[2], bbox[3], ones]]).astype(np.float)
        bbox = np.transpose(bbox, (2, 0, 1))
        affine = np.array([[r, 0, 0],
                           [0, r, 0],
                           [x, y, 1]]).astype(np.float)
        bbox = np.dot(bbox, affine)
        bbox = bbox[:, :, :-1]
        bbox = np.transpose(bbox, (1, 2, 0))
        bbox = np.concatenate(bbox, axis=0)         # ((1, 2), (x, y), n) -> ((x1, y1, x2, y2), n)

        spatium = (x, y)    # 為了要將之後的 pred box 轉成 origin image 的位置

        return search_image, bbox, r, spatium

    def __call__(self, image, boxes, ratio=None, bg=None, context_amount=None):
        """
        Returns:
            image: 圖片
            boxes: gt boxes
        """
        if self.type == "template":
            template_image, template_ratio, box, origin_box = self._template_crop(image, boxes, bg, context_amount)
            # box, origin_box: 只有在 testing 畫圖的時候會用
            return template_image, template_ratio, box, origin_box

        if self.type == "search":
            search_image, boxes, r, spatium = self._search_crop(image, boxes, ratio)
            return search_image, boxes, r, spatium
