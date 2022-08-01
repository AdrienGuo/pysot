from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import numpy as np
from pysot.datasets.check_image import draw_bbox
from pysot.utils.bbox import Center, Corner, center2corner, corner2center


def template_crop(image, bbox, size):
    img_h, img_w = image.shape[:2]

    template_w = bbox[2]
    template_h = bbox[3]

    if img_w >= img_h:
        r = size / img_w
    else:
        r = size / img_h

    template_w = template_w * r * size
    template_h = template_h * r * size

    return template_w, template_h


# def draw(image, bbox, size):
#     search_image = image
#     search_h, search_w = search_image.shape[:2]
#     # === 處理 search image ===
#     # 原理上跟 template 在做一樣的事
#     if search_w >= search_h:
#         r = size / search_w
#     else:
#         r = size / search_h
#     x = (size / 2) - (r * search_w) / 2
#     y = (size / 2) - (r * search_h) / 2
#     mapping = np.array([[r, 0, x],
#                         [0, r, y]]).astype(np.float)
#     search_image = cv2.warpAffine(search_image, mapping, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)

#     # === 處理 bounding box ===
#     bbox = np.transpose(bbox, (1, 0))       # (n, 4) -> (4, n)
#     bbox = Corner(*center2corner(bbox))
#     bbox = Corner(search_w * bbox.x1, search_h * bbox.y1,
#                   search_w * bbox.x2, search_h * bbox.y2)
#     bbox = np.array(bbox)
#     # print(f"bbox: {bbox.shape}")
#     # 做 bounding box 的縮放&平移
#     # [[x1, y1, 1]      [[r, 0, 0]
#     #  [x2, y2, 1]]  *   [0, r, 0]
#     #                    [x, y, 1]]
#     # TODO: 把後面的那個 0, 0, 1 拿掉，我用不到
#     ones = np.ones(bbox.shape[1])
#     # print(f"ones: {ones.shape}")
#     bbox = np.array([[bbox[0], bbox[1], ones],
#                      [bbox[2], bbox[3], ones]]).astype(np.float)
#     bbox = np.transpose(bbox, (2, 0, 1))
#     # print(f"x: {type(x)}")
#     ratio = np.array([[r, 0, 0],
#                       [0, r, 0],
#                       [x, y, 1]]).astype(np.float)
#     bbox = np.dot(bbox, ratio)
#     bbox = bbox[:, :, :-1]
#     bbox = np.transpose(bbox, (1, 2, 0))
#     bbox = np.concatenate(bbox, axis=0)         # ((1, 2), (x, y), n) -> ((x1, y1, x2, y2), n)
#     bbox = np.transpose(bbox, (1, 0))

#     draw_bbox(image, search_image, bbox, "./image_check/train/" + )
