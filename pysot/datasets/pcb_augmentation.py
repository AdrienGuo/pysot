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
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instance_size=255, padding=(0, 0, 0)):
    # img_h, img_w = image.shape[:2]
    target_pos = [(bbox.x2 + bbox.x1) / 2., (bbox.y2 + bbox.y1) / 2.]
    target_size = [(bbox.x2 - bbox.x1), (bbox.y2 - bbox.y1)]
    # target_pos = [img_w * (bbox[2] + bbox[0])/2., img_h * (bbox[3] + bbox[1])/2.]
    # target_size = [img_w * (bbox[2] - bbox[0]), img_h * (bbox[3] - bbox[1])]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    # scale_z = exemplar_size / s_z
    # d_search = (instance_size - exemplar_size) / 2
    # pad = d_search / scale_z
    # s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    # x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instance_size, padding)
    return z


class Augmentation:
    def __init__(self, template_size, search_size, type) -> None:
        self.template_size = template_size
        self.search_size = search_size
        self.type = type
    
    def _template_crop(self, image, bbox, bg, context_amount=0, padding=(0, 0, 0)):
        img_h, img_w = image.shape[:2]
        bbox = Corner(*center2corner(bbox))     # 超剛好前幾天才問啟恩這個用法，感謝啟恩
        bbox = Corner(img_w * bbox.x1, img_h * bbox.y1,
                      img_w * bbox.x2, img_h * bbox.y2)
        bbox = Corner(*map(lambda x: int(x), bbox))
        template_w = bbox.x2 - bbox.x1
        template_h = bbox.y2 - bbox.y1

        ####################################################################
        # 裁切出 template 的部分 (分成 有 bg, 沒有 bg)
        ####################################################################
        # 沒有 bg
        if bg == "nbg":
            # 將長邊變成 255
            # r: 放大or縮小比率
            template_image = image[bbox.y1: bbox.y1 + template_h, bbox.x1: bbox.x1 + template_w]
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
        # 有 bg, 而 context_amount 可以決定要加入多少比例的 bg
        elif bg == "bg":
            template_image = crop_like_SiamFC(
                image,
                bbox,
                context_amount=context_amount,
                exemplar_size=127,
                instance_size=255
            )

        ####################################################################
        # === 處理 template box 在 search image 上的位置 ===
        # （只在做 testing 得時候會用到，因為我想將 template 的位置秀在 search image 上）
        # 跟下面的 _search_crop 幾乎一模一樣；造成這樣是因為
        # Augmentation 這個 Object，template 和 search 各有一個 (他們之間的參數不能互通)
        # 加上我不想把屬於 template 的東西拿給 search 做 (很亂)，乾脆兩邊都做
        ####################################################################
        # bbox = np.transpose(bbox, (1, 0))       # (n, 4) -> (4, n)
        # bbox = bbox[:, np.newaxis]
        # bbox = Corner(*center2corner(bbox))
        # bbox = Corner(img_w * bbox.x1, img_h * bbox.y1,
        #               img_w * bbox.x2, img_h * bbox.y2)
        # 上面已經處理過 bbox 了
        bbox = np.array(bbox)
        # if img_w >= img_h:
        #     r = self.search_size / img_w
        # else:
        #     r = self.search_size / img_h
        # x = (self.search_size / 2) - (r * img_w) / 2
        # y = (self.search_size / 2) - (r * img_h) / 2

        # bbox = np.array(bbox)
        # bbox = bbox[:, np.newaxis]
        # # 做 template box 的縮放&平移
        # # [[x1, y1, 1]      [[r, 0, 0]
        # #  [x2, y2, 1]]  *   [0, r, 0]
        # #                    [x, y, 1]]
        # # TODO: 把後面的那個 0, 0, 1 拿掉，我用不到
        # ones = np.ones(bbox.shape[1])
        # bbox = np.array([[bbox[0], bbox[1], ones],
        #                  [bbox[2], bbox[3], ones]]).astype(np.float)
        # bbox = np.transpose(bbox, (2, 0, 1))
        # ratio = np.array([[r, 0, 0],
        #                   [0, r, 0],
        #                   [x, y, 1]]).astype(np.float)
        # bbox = np.dot(bbox, ratio)
        # bbox = bbox[:, :, :-1]
        # bbox = np.transpose(bbox, (1, 2, 0))
        # bbox = np.concatenate(bbox, axis=0)         # ((1, 2), (x, y), n) -> ((x1, y1, x2, y2), n)
        # bbox = bbox.squeeze()

        return template_image, bbox

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
        # r 是縮放比例；x, y 是位移
        mapping = np.array([[r, 0, x],
                            [0, r, y]]).astype(np.float)
        search_image = cv2.warpAffine(search_image, mapping, (self.search_size, self.search_size), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)

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
            bbox: bounding box
        """
        if self.type == "template":
            template_image, bbox = self._template_crop(image, bbox, bg, context_amount)
            return template_image, bbox
        
        if self.type == "search":
            search_image, bbox, r = self._search_crop(image, bbox)
            return search_image, bbox, r
