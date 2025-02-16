# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import torch
from .transform_amy import get_transforms

import ipdb

class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        # score_size 就是最後的 feature map 的大小: 25
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        #print("self.score_size:",self.score_size)

        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES) #5 = 5 * 1

        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)     # 外積
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        """
        Arg:
            score_size: 最後的 feature map 大小: 25
        Return:
            anchor: (5*25*25, 4)
        """
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors    # (5, 4)
        
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack(((x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1), axis=1)      # (cx, cy, w, h) = (5, 4)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))      # (5*25*25, 4)
        ori = - (score_size // 2) * total_stride        # 起始點嗎??
        # 生成網格座標
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)

        return anchor

    def _convert_score(self, score):
        # (1, 2*5, 25, 25) -> (2*5, 25, 25, 1) -> contiguous -> (2, 5*25*25) -> (5*25*25, 2)
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)    # (5*25*25, 2)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()        # (5*25*25, )
        return score    # (5*25*25, )

    def _convert_bbox(self, delta, anchor):
        # (1, 4*5, 25, 25) -> (4*5, 25, 25, 1) -> contiguous -> (4, 5*25*25)
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta    # (4, 5*25*25)

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x1, y1, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2]-1)/2,
                                    bbox[1] + (bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
       
        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))
        
        #print("average:",self.channel_average)
        # get crop
        
        z_crop = self.get_subwindow(img,
                                    self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z,
                                    self.channel_average)

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
        self.model.template(z_crop)

        # print(f"z_crop: {z_crop.cpu().numpy().shape}")
        z_crop = z_crop.cpu().numpy()
        return z_crop
        
    def nms(self, bbox, scores, image, iou_threshold):
        # bbox[0] 是 anchor 的 x 的中心點要位移的距離，
        # 所以再加上原本 anchor x 軸的中心點，就變成預測的框框的 x 軸中心點座標 = cx
        # (y 軸也同理)
        cx = bbox[0, :] + self.center_pos[0]
        cy = bbox[1, :] + self.center_pos[1]
        width = bbox[2, :]#*image.shape[1]
        height = bbox[3, :]#*image.shape[0]
        
        x1 = cx - width/np.array(2)
        y1 = cy - height/np.array(2)
        x2 = x1 + width
        y2 = y1 + height
        areas = (x2 - x1) * (y2 - y1)
        
        # 结果列表
        results = []
        index = scores.argsort()[::-1]  # 由高到低排序信心值取得 index
        while index.size > 0:
            results.append(index[0])
            # 計算其他框與該框的 IoU
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
        #print("results:",results)
        
        return results

    def track(self, img):
        """
        Args:
            img(np.ndarray): BGR image
        Return:
            bbox(list): [x, y, width, height]
        """
       
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        
        
        center = (img.shape[1]/2,img.shape[0]/2)
        self.center_pos = np.array([img.shape[1]/2,img.shape[0]/2])
        x_crop = self.get_subwindow(img,
                                    self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x),
                                    self.channel_average)
        
        ''' 
        _,transform_val=get_transforms(600)
        x_crop = transform_val(Image.fromarray(img))
        x_crop = x_crop.unsqueeze(0)
        x_crop = x_crop.cuda()
        
        to_pil_image = transforms.ToPILImage()
        imgs = to_pil_image(x_crop[0])
        #imgs.show()
        plt.imshow(imgs)
        plt.show()
        '''
        
        outputs = self.model.track(x_crop)                              # (1, 2*anchor_num, 25, 25)
        scores = self._convert_score(outputs['cls'])                     # (5*25*25, )
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)    # (4, 5*25*25)
        #print("pred_bbox:",pred_bbox.shape)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * scores
        
        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        #best_idx = np.argmax(pscore)
        
        
        #加nms，計算所有框
        bboxes = []
        top_scores = []
        iou_threshold = 0.1
        results = self.nms(pred_bbox, scores, img, iou_threshold)

        for i in range(len(results)):
            bbox = pred_bbox[:, results[i]] / scale_z
            lr = penalty[results[i]] * scores[results[i]] * cfg.TRACK.LR
            score = scores[results[i]]
            # print(f"score: {score}")
            # ipdb.set_trace()
            top_scores.append(score)
            if score >= 0.5:
                cx = bbox[0] + self.center_pos[0]
                cy = bbox[1] + self.center_pos[1]
                #cx = bbox[0]*img.shape[1]
                #cy = bbox[1]*img.shape[0]
                #print(":",lr)
                width = bbox[2]#self.size[0] * (1 - lr) + bbox[2] * lr
                height = bbox[3]#self.size[1] * (1 - lr) + bbox[3] * lr

                #width = bbox[2]#*img.shape[1]
                #height = bbox[3]#*img.shape[0]

                # clip boundary
                cx, cy, width, height = self._bbox_clip(cx, cy, width,height, img.shape[:2])
                
                bbox = [cx - width / 2,
                        cy - height / 2,
                        width,
                        height]
                bboxes.append(bbox)
            else:
                continue
        '''
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * scores[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        top_scores = scores[best_idx]
        '''
        return {
                'bboxes': bboxes,
                'top_scores': top_scores
               }
