from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from pysot.utils.bbox import corner2center, \
        Center, center2corner, Corner

class Augmentation:
    def __init__(self, type) -> None:
        self.type = type
    
    def _template_crop(image, bbox, size, padding=(0, 0, 0)):
        print(f"aug: {type(image)}")
        img_h, img_w = image.shape[:2]
        bbox = Corner(*center2corner(bbox))     # 超剛好前幾天才問啟恩這個用法，感謝啟恩
        bbox = Corner(img_w * bbox.x1, img_h * bbox.y1,
                      img_w * bbox.x2, img_h * bbox.y2)
        bbox = Corner(*map(lambda x: int(x), bbox))
        template_w = bbox.x2 - bbox.x1
        template_h = bbox.y2 - bbox.y1
        # 裁切出 template 的部分
        # TODO: 多新增一個參數，來決定要加入多少背景 (現在是 0)
        template_image = image[bbox.y1 : bbox.y1 + template_h, bbox.x1 : bbox.x1 + template_w]

        # r: 放大or縮小比率
        if template_w >= template_h:
            r = size / template_w
        else:
            r = size / template_h
        # 計算移到中心需要的位移
        x = (size/2) - (r * template_w)/2
        y = (size/2) - (r * template_h)/2
        mapping = np.array([[r, 0, x],
                            [0, r, y]]).astype(np.float)
        template_image = cv2.warpAffine(template_image, mapping, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return template_image

    def __call__(self, image, bbox, size):
        """
        Returns:
            image: 圖片
            bbox: bounding box
        """
        if self.type == "template":
            print(f"aug: {type(image)}")
            template_image = Augmentation._template_crop(image, bbox, size)
            return template_image, bbox
