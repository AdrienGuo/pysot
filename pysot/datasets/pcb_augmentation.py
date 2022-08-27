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


# 這幾個都只是為了幫 template 加入背景
def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop(image, box, out_sz, padding=(0, 0, 0)):
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    # box 小於輸出的大小，則 template 不做縮放
    if box_w < out_sz and box_h < out_sz:
        template_crop = image[box[1]: box[3], box[0]: box[2]]  # 先高再寬
        # 計算移到中心需要的位移
        x = (out_sz / 2) - (box_w / 2)
        y = (out_sz / 2) - (box_h / 2)
        mapping = np.array([[1, 0, x],
                            [0, 1, y]]).astype(np.float)
        crop = cv2.warpAffine(template_crop, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    # box 大於輸出的大小，則根據長邊對 template 做縮放
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
        crop = cv2.warpAffine(template_crop, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instance_size=255, padding=(0, 0, 0)):
    img_h, img_w = image.shape[:2]
    target_center = [(bbox.x2 + bbox.x1) / 2., (bbox.y2 + bbox.y1) / 2.]
    target_size = [(bbox.x2 - bbox.x1), (bbox.y2 - bbox.y1)]    # [w, h]
    # target_center = [img_w * (bbox[2] + bbox[0])/2., img_h * (bbox[3] + bbox[1])/2.]
    # target_size = [img_w * (bbox[2] - bbox[0]), img_h * (bbox[3] - bbox[1])]

    # 老師要的裁切方式
    crop_w = target_size[0] * 2
    crop_h = target_size[1] * 2
    # 論文原版裁切方式
    # wc_z = target_size[1] + context_amount * sum(target_size)
    # hc_z = target_size[0] + context_amount * sum(target_size)
    # s_z = np.sqrt(wc_z * hc_z)

    # 跟 search image 的裁切有關，我用不到
    # scale_z = exemplar_size / s_z
    # d_search = (instance_size - exemplar_size) / 2
    # pad = d_search / scale_z
    # s_x = s_z + 2 * pad

    # s_z: [x1, y1, x2, y2]
    s_z = [target_center[0] - (crop_w / 2), target_center[1] - (crop_h / 2),
           target_center[0] + (crop_w / 2), target_center[1] + (crop_h / 2)]
    # 超出範圍的裁掉
    s_z[0] = max(0, s_z[0])
    s_z[1] = max(0, s_z[1])
    s_z[2] = min(img_w, s_z[2])
    s_z[3] = min(img_h, s_z[3])
    s_z = list(map(lambda x: int(x), s_z))

    # z = crop_hwc(image, pos_s_2_bbox(target_center, s_z), exemplar_size, padding)
    # x = crop_hwc(image, pos_s_2_bbox(target_center, s_x), instance_size, padding)

    z = crop(image, s_z, exemplar_size, padding)

    return z


class Augmentation:
    def __init__(self, template_size, search_size, type) -> None:
        self.template_size = template_size
        self.search_size = search_size
        self.type = type
    
    def _template_crop(self, image, box, bg, context_amount=0, padding=(0, 0, 0)):
        img_h, img_w = image.shape[:2]

        ####################################################################
        # 將 image 縮放成和 search image 一樣的樣子 (有 padding)
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
        image = cv2.warpAffine(image, mapping,
                               (self.search_size, self.search_size),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=padding)
        # x = (self.search_size / 2) - (r * template_w) / 2
        # y = (self.search_size / 2) - (r * template_h) / 2

        # image: 縮放成 search size 大小後的 template image
        # image = cv2.resize(image, (int(r * img_w), int(r * img_h)), interpolation=cv2.INTER_LINEAR)

        ####################################################################
        # === 將 template box 調整成在 search image 上的位置 ===
        # （只在做 testing 得時候會用到，因為我想將 template 的位置秀在 search image 上）
        # 跟下面的 _search_crop 幾乎一模一樣；造成這樣是因為
        # Augmentation 這個 Object，template 和 search 各有一個 (他們之間的參數不能互通)
        # 加上我不想把屬於 template 的東西拿給 search 做 (很亂)，乾脆兩邊都做
        ####################################################################
        box = Corner(*center2corner(box))     # 超剛好前幾天才問啟恩這個用法，感謝啟恩
        box = Corner(img_w * box.x1, img_h * box.y1,
                     img_w * box.x2, img_h * box.y2)
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
        # box = box.squeeze()
        box = Corner(box[0], box[1], box[2], box[3])

        ####################################################################
        # 在 search image 上，裁切出 template 的部分 (分成 有bg, 沒有bg)
        ####################################################################
        template_w = box.x2 - box.x1
        template_h = box.y2 - box.y1

        # 沒有 bg
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
            template_image = crop_like_SiamFC(
                image,
                box,
                context_amount=context_amount,
                exemplar_size=127,
                instance_size=255   # 好像用不到
            )

        ####################################################################
        # === 處理 template box 在 search image 上的位置 ===
        # （只在做 testing 得時候會用到，因為我想將 template 的位置秀在 search image 上）
        # 跟下面的 _search_crop 幾乎一模一樣；造成這樣是因為
        # Augmentation 這個 Object，template 和 search 各有一個 (他們之間的參數不能互通)
        # 加上我不想把屬於 template 的東西拿給 search 做 (很亂)，乾脆兩邊都做
        ####################################################################
        # box = np.transpose(box, (1, 0))       # (n, 4) -> (4, n)
        # box = box[:, np.newaxis]
        # box = Corner(*center2corner(box))
        # box = Corner(img_w * box.x1, img_h * box.y1,
        #               img_w * box.x2, img_h * box.y2)
        # 上面已經處理過 box 了
        box = np.array(box)     # corner -> array (??)
        # if img_w >= img_h:
        #     r = self.search_size / img_w
        # else:
        #     r = self.search_size / img_h
        # x = (self.search_size / 2) - (r * img_w) / 2
        # y = (self.search_size / 2) - (r * img_h) / 2

        # box = np.array(box)
        # box = box[:, np.newaxis]
        # # 做 template box 的縮放&平移
        # # [[x1, y1, 1]      [[r, 0, 0]
        # #  [x2, y2, 1]]  *   [0, r, 0]
        # #                    [x, y, 1]]
        # # TODO: 把後面的那個 0, 0, 1 拿掉，我用不到
        # ones = np.ones(box.shape[1])
        # box = np.array([[box[0], box[1], ones],
        #                  [box[2], box[3], ones]]).astype(np.float)
        # box = np.transpose(box, (2, 0, 1))
        # ratio = np.array([[r, 0, 0],
        #                   [0, r, 0],
        #                   [x, y, 1]]).astype(np.float)
        # box = np.dot(box, ratio)
        # box = box[:, :, :-1]
        # box = np.transpose(box, (1, 2, 0))
        # box = np.concatenate(box, axis=0)         # ((1, 2), (x, y), n) -> ((x1, y1, x2, y2), n)
        # box = box.squeeze()

        return template_image, box

    def _search_crop(self, image, bbox, padding=(0, 0, 0)):
        search_image = image
        search_h, search_w = search_image.shape[:2]
        # === 處理 search image ===
        # 原理上跟 template 在做一樣的事
        if search_w >= search_h:
            r = self.search_size / search_w
        else:
            r = self.search_size / search_h
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
        ratio = np.array([[r, 0, 0],
                          [0, r, 0],
                          [x, y, 1]]).astype(np.float)
        bbox = np.dot(bbox, ratio)
        bbox = bbox[:, :, :-1]
        bbox = np.transpose(bbox, (1, 2, 0))
        bbox = np.concatenate(bbox, axis=0)         # ((1, 2), (x, y), n) -> ((x1, y1, x2, y2), n)
        return search_image, bbox, r

    def __call__(self, image, bbox, bg=None, context_amount=None):
        """
        Returns:
            image: 圖片
            boxes: gt boxes
        """
        if self.type == "template":
            template_image, box = self._template_crop(image, bbox, bg, context_amount)
            # 這個 box 只有在 testing 畫圖的時候會用
            return template_image, box

        if self.type == "search":
            search_image, boxes, r = self._search_crop(image, bbox)
            return search_image, boxes, r
