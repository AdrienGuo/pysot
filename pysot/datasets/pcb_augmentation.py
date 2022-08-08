from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import ipdb
import numpy as np
from pysot.utils.bbox import Center, Corner, center2corner, corner2center


class Augmentation:
    def __init__(self, template_size, search_size, type) -> None:
        self.template_size = template_size
        self.search_size = search_size
        self.type = type
    
    def _template_crop(self, image, bbox, padding=(0, 0, 0)):
        img_h, img_w = image.shape[:2]
        bbox = Corner(*center2corner(bbox))     # 超剛好前幾天才問啟恩這個用法，感謝啟恩
        bbox = Corner(img_w * bbox.x1, img_h * bbox.y1,
                      img_w * bbox.x2, img_h * bbox.y2)
        bbox = Corner(*map(lambda x: int(x), bbox))
        template_w = bbox.x2 - bbox.x1
        template_h = bbox.y2 - bbox.y1
        # 裁切出 template 的部分
        # TODO: 多新增一個參數，來決定要加入多少背景 (現在是 0)
        template_image = image[bbox.y1: bbox.y1 + template_h, bbox.x1: bbox.x1 + template_w]

        # 將長邊變成 255
        # r: 放大or縮小比率
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

        ####################################################################
        # === 處理 template box ===
        # 跟下面的 search crop 幾乎一模一樣；造成這樣是因為
        # Augmentation 這個 Object，template 和 search 各有一個 (他們之間的參數不能互通)
        # 加上我不想把屬於 template 的東西拿給 search 做 (很亂)，乾脆兩邊都做
        ####################################################################
        # bbox = np.transpose(bbox, (1, 0))       # (n, 4) -> (4, n)
        # bbox = bbox[:, np.newaxis]
        # bbox = Corner(*center2corner(bbox))
        # bbox = Corner(img_w * bbox.x1, img_h * bbox.y1,
        #               img_w * bbox.x2, img_h * bbox.y2)
        if img_w >= img_h:
            r = self.search_size / img_w
        else:
            r = self.search_size / img_h
        x = (self.search_size / 2) - (r * img_w) / 2
        y = (self.search_size / 2) - (r * img_h) / 2

        bbox = np.array(bbox)
        bbox = bbox[:, np.newaxis]
        # 做 template box 的縮放&平移
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
        bbox = bbox.squeeze()

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

    def __call__(self, image, bbox):
        """
        Returns:
            image: 圖片
            bbox: bounding box
        """
        if self.type == "template":
            template_image, bbox = self._template_crop(image, bbox)
            return template_image, bbox
        
        if self.type == "search":
            search_image, bbox, r = self._search_crop(image, bbox)
            return search_image, bbox, r
