# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import ipdb
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
# from PIL import Image
from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
# from pysot.utils.anchor import Anchors
from pysot.rpn.anchor import Anchors
from torchvision import transforms

from .transform_amy import get_transforms


def search2origin(box, r, s):
    """ 這裡要很小心的是，一定要"先做平移"，才能"再做縮放"
        因為原本 cv2 的乘法順序會是：縮放 -> 平移
        我現在要還原回去，所以是：平移 -> 縮放
    """
    r = 1 / r
    x = -s[0]
    y = -s[1]

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
    # 平移
    ratio = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [x, y, 1]]).astype(np.float)
    box = np.dot(box, ratio)
    # 縮放
    ratio = np.array([[r, 0, 0],
                      [0, r, 0],
                      [0, 0, 1]]).astype(np.float)
    box = np.dot(box, ratio)
    box = box[:, :, :-1]
    box = np.transpose(box, (1, 2, 0))
    box = np.concatenate(box, axis=0)         # ((1, 2), (x, y), n) -> ((x1, y1, x2, y2), n)
    box = box.squeeze()

    return box


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        # score_size: 最後的 feature map 大小
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        # self.score_size = cfg.TRAIN.OUTPUT_SIZE
        self.anchor_num = cfg.ANCHOR.ANCHOR_NUM
        # self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)

        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)     # 外積
        self.window = np.tile(window.flatten(), self.anchor_num)

        # self.anchors: (anchor_num*25*25, 4) #corner
        self.anchors = self.generate_anchor(img_c=cfg.TRACK.INSTANCE_SIZE // 2,
                                            score_size=self.score_size)

        # self.anchors = Anchors(cfg.ANCHOR.STRIDE,
        #                        cfg.ANCHOR.RATIOS,
        #                        cfg.ANCHOR.SCALES)
        # self.anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE // 2,
        #                                   size=cfg.TRAIN.OUTPUT_SIZE)
        # self.anchors = self.anchors.all_anchors[1]    # (4, num_anchor, 25, 25)
        # self.anchors = np.reshape(self.anchors, (4, -1))
        # self.anchors = np.transpose(self.anchors, (1, 0))

        self.model = model
        self.model.eval()

    def generate_anchor(self, img_c, score_size):
        """
        Arg:
            score_size: 最後的 feature map 大小
        Return:
            anchor: (5*25*25, 4)
        """

        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)

        anchor = anchors.anchors    # (5, 4)

        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), axis=1)      # (cx, cy, w, h) = (5, 4)
        stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))      # (5*25*25, 4)

        # 把所有 anchor 以 search image 的中心當作 (0, 0)，(不需要了，我自己算！！)
        # ori = - (cfg.TRAIN.SEARCH_SIZE // 2)

        ####################################################################
        # === 測試要用這個方式產生 anchor！！ ===
        # 因為要和之後算 _convert_bbox() 的時候，“順序” 要一樣！！
        ####################################################################
        # 這個 ori 超級重要
        ori = img_c - score_size // 2 * stride
        # 生成網格座標
        xx, yy = np.meshgrid([ori + stride * dx for dx in range(score_size)],
                             [ori + stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)

        ####################################################################
        # === 使用我自己 train 的 model ===
        # 又是順序問題，一定要先算出 w, h，才能去算 cx, cy
        # 先算 cx, cy 的話，w, h 就不能直接 x2 - x1 了，因為會變成 x2 - cx
        # 所以要寫成 (x2 - cx) * 2
        # 我找超久...
        ####################################################################
        # anchor = anchors.generate_all_anchors(im_c=cfg.TRACK.INSTANCE_SIZE // 2,
        #                                       size=score_size)
        # anchor[:, 0] = (anchor[:, 0] + anchor[:, 2]) * 0.5
        # anchor[:, 1] = (anchor[:, 1] + anchor[:, 3]) * 0.5
        # anchor[:, 2] = (anchor[:, 2] - anchor[:, 0]) * 2
        # anchor[:, 3] = (anchor[:, 3] - anchor[:, 1]) * 2
        ipdb.set_trace()

        return anchor   # (5*25*25, 4), #center

    def _convert_score(self, score):
        # (1, 2*5, 25, 25) -> (2*5, 25, 25, 1) -> contiguous -> (2, 5*25*25) -> (5*25*25, 2)
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)    # (5*25*25, 2)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()        # (5*25*25, )
        return score    # (5*25*25, )

    def _convert_bbox(self, delta, anchor):
        # (1, 4*5, 25, 25) -> (4*5, 25, 25, 1) -> contiguous -> (4, 5*25*25)
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)

        # ...，為甚麼要轉去 cpu 上面運算??
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]     # cx (real)，應該不是誤差了 (?)
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta    # (4, 5*25*25) #center #real

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def nms(self, bbox, scores, iou_threshold):
        """ non maximum suppression
        Args:
            bbox (4, 5*25*25), (cx, cy, w, h (#real))
            scores (5*25*25, )
        """
        cx = bbox[0, :]
        cy = bbox[1, :]
        width = bbox[2, :]
        height = bbox[3, :]

        x1 = cx - width / np.array(2)
        y1 = cy - height / np.array(2)
        x2 = cx + width / np.array(2)
        y2 = cy + height / np.array(2)
        areas = (x2 - x1) * (y2 - y1)

        # 结果列表
        results = []
        index = scores.argsort()[::-1]  # 由高到低排序信心值取得 index
        while index.size > 0:
            results.append(index[0])
            # 計算該框與其他框的 IoU
            x11 = np.maximum(x1[index[0]], x1[index[1:]])
            y11 = np.maximum(y1[index[0]], y1[index[1:]])
            x22 = np.minimum(x2[index[0]], x2[index[1:]])
            y22 = np.minimum(y2[index[0]], y2[index[1:]])
            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)
            overlaps = w * h
            ious = overlaps / (areas[index[0]] + areas[index[1:]] - overlaps)
            # 只保留满足 IoU 閥值的 index
            idx = np.where(ious <= iou_threshold)[0]
            index = index[idx + 1]  # 剩下的框

        return results

    def init(self, z_img, z_box):
        """
        Args:
            img(np.ndarray): BGR image
            z_box: (x1, y1, w, h) z_box
            template_image (batch, w, h, channel), dtype=tensor: template image that had been preprocessed
        """
        self.center_pos = np.array([z_box[0] + (z_box[2] - 1) / 2,
                                    z_box[1] + (z_box[3] - 1) / 2])
        self.size = np.array([z_box[2], z_box[3]])    # 在做 window penalty 的時候會用到

        # calculate z crop size
        # w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        # self.channel_average = np.mean(img, axis=(0, 1))

        # print("average:",self.channel_average)
        # get crop
        # z_crop: (1, 3, 127, 127), dtype=tensor
        # TODO: 改變 z_crop 的切法
        # z_crop = self.get_subwindow(img,
        #                             self.center_pos,
        #                             cfg.TRACK.EXEMPLAR_SIZE,
        #                             s_z,
        #                             self.channel_average)

        '''
        z_crop = img[int(self.center_pos[1])-int(cfg.TRACK.EXEMPLAR_SIZE/2):int(self.center_pos[1])+int(cfg.TRACK.EXEMPLAR_SIZE/2),int(self.center_pos[0])-int(cfg.TRACK.EXEMPLAR_SIZE/2):int(self.center_pos[0])+int(cfg.TRACK.EXEMPLAR_SIZE/2)]
        z_crop = z_crop.transpose(2, 0, 1)
        z_crop = z_crop[np.newaxis, :, :, :]
        z_crop = z_crop.astype(np.float32)
        z_crop = torch.from_numpy(z_crop)
        z_crop = z_crop.cuda()
        to_pil_image = transforms.ToPILImage()
        imgs = to_pil_image(z_crop[0])
        imgs.show()
        #plt.imshow(imgs)
        #plt.show()
        '''

        self.model.template(z_img)

        # if cfg.CUDA:
        #     template_image = template_image.cuda()
        # self.model.template(template_image)

        # template_image = template_image.cpu().numpy()

        return z_img.cpu().numpy().squeeze()

    def track(self, image, x_img, scale_ratio, spatium):
        """
        Args:
            image: original image
            x_img (np.ndarray): preprocessed search image
            scale_ratio (r): ratio of (image -> x_img)
            spatium (x, y): displacement of (image -> x_img)
        Return:
            bbox(list): [x, y, width, height]
        """

        # w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        # s_z = np.sqrt(w_z * h_z)
        # scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        # s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        # self.center_pos = np.array([image.shape[1] / 2, image.shape[0] / 2])

        # x_img = self.get_subwindow(x_img,
        #                             self.center_pos,
        #                             cfg.TRACK.INSTANCE_SIZE,
        #                             round(s_x),
        #                             self.channel_average)

        ''' 
        _,transform_val=get_transforms(600)
        x_img = transform_val(Image.fromarray(x_img))
        x_img = x_img.unsqueeze(0)
        x_img = x_img.cuda()
        
        to_pil_image = transforms.ToPILImage()
        x_imgs = to_pil_image(x_img[0])
        #x_imgs.show()
        plt.imshow(x_imgs)
        plt.show()
        '''

        outputs = self.model.track(x_img)                         # (1, 2*anchor_num, 25, 25)

        scores = self._convert_score(outputs['cls'])                     # (anchor_num*25*25, )
        pred_bboxes = self._convert_bbox(outputs['loc'], self.anchors)    # (4, anchor_num*25*25)

        ####################################################################
        # 用 box 的長寬比例來篩除 box
        # 若是比例與原本的 template 差距太大的話，就會被刪掉 (我猜的)
        ####################################################################
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bboxes[2, :], pred_bboxes[3, :]) /
                     (sz(self.size[0]*scale_ratio, self.size[1]*scale_ratio)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bboxes[2, :]/pred_bboxes[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * scores
        
        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE

        ####################################################################
        # 加 NMS
        # 只保留 score 大於 threshold 的 pred_box
        ####################################################################
        # TODO: 改成 config
        iou_threshold = 0.1
        # nms_bboxes = self.nms(pred_bboxes, scores, iou_threshold)
        nms_bboxes = self.nms(pred_bboxes, pscore, iou_threshold)

        pred_boxes = []
        origin_pred_boxes = []
        top_scores = []
        # TODO: 創一個新的 def block, ex: select_bbox
        for i in range(len(nms_bboxes)):
            pred_box = pred_bboxes[:, nms_bboxes[i]]
            # pred_box = pred_bboxes[:, nms_bboxes[i]]
            # lr = penalty[nms_bboxes[i]] * scores[nms_bboxes[i]] * cfg.TRACK.LR
            score = scores[nms_bboxes[i]]
            if score >= 0.5:
                top_scores.append(score)
                cx = pred_box[0] # s+ self.center_pos[0]
                cy = pred_box[1] # + self.center_pos[1]
                #cx = pred_box[0]*x_img.shape[1]
                #cy = pred_box[1]*x_img.shape[0]
                #print(":",lr)
                width = pred_box[2]    #self.size[0] * (1 - lr) + pred_box[2] * lr
                height = pred_box[3]   #self.size[1] * (1 - lr) + pred_box[3] * lr

                #width = pred_box[2]#*x_img.shape[1]
                #height = pred_box[3]#*x_img.shape[0]

                # clip boundary
                # print(f"cx: {cx}")
                # print(f"cy: {cy}")
                # print(f"width: {width}")
                # print(f"height: {height}")
                # ipdb.set_trace()
                # cx, cy, width, height = self._bbox_clip(cx, cy, width, height, x_img.shape[:2])

                # === pred_box on "search" image
                pred_box = [cx - width / 2,
                            cy - height / 2,
                            width,
                            height]
                pred_boxes.append(pred_box)

                # === pred_box on "original" image
                origin_pred_box = copy.deepcopy(pred_box)
                origin_pred_box[2] = origin_pred_box[0] + origin_pred_box[2]    # w -> x2
                origin_pred_box[3] = origin_pred_box[1] + origin_pred_box[3]    # h -> y2
                origin_pred_box = search2origin(origin_pred_box, scale_ratio, spatium)    # origin_pred_box (x1, y1, x2, y2)
                origin_pred_box = [origin_pred_box[0],
                                   origin_pred_box[1],
                                   origin_pred_box[2] - origin_pred_box[0],
                                   origin_pred_box[3] - origin_pred_box[1]]
                origin_pred_boxes.append(origin_pred_box)
            else:
                continue

        return {
            'x_img': x_img.cpu().numpy().squeeze(),
            'pred_boxes': pred_boxes,
            'origin_pred_boxes': origin_pred_boxes,
            'top_scores': top_scores
        }
