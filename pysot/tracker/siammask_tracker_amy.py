# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np

from pysot.core.config import cfg
from pysot.utils.bbox import cxy_wh_2_rect
from pysot.tracker.siamrpn_tracker import SiamRPNTracker

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

class SiamMaskTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamMaskTracker, self).__init__(model)
        assert hasattr(self.model, 'mask_head'), \
            "SiamMaskTracker must have mask_head"
        assert hasattr(self.model, 'refine_head'), \
            "SiamMaskTracker must have refine_head"

    def _crop_back(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _mask_post_processing(self, mask):
        target_mask = (mask > cfg.TRACK.MASK_THERSHOLD)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(self.center_pos, self.size)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        return rbox_in_img
    
    def nms(self, bbox,scores,image,iou_threshold ):
        cx = bbox[0,:]+self.center_pos[0]
        cy = bbox[1,:]+self.center_pos[1]
        width = bbox[2,:]#*image.shape[1]
        height = bbox[3,:]#*image.shape[0]
       
        x1 = cx - width/np.array(2)
        y1 = cy - height/np.array(2)
        x2 = x1 + width
        y2 = y1 + height
        #print("x1:",x1)
        areas = (x2 - x1)*(y2 - y1)
        
        # 结果列表
        result = []
        index = scores.argsort()[::-1]  # 由高到低排序信心值取得index
        while index.size > 0:
            i = index[0]
            result.append(i)  

            # 計算其他框與該框的IOU
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])
            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)
            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            # 只保留满足IOU閥值的index
            idx = np.where(ious <= iou_threshold)[0]
            index = index[idx + 1]  #剩下的框
        #print("result:",result)
        
        return result
   
    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        # print("self.size:",self.size) 寬、高
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        s_x = round(s_x)

        x_crop = self.get_subwindow(img,
                                    self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    s_x,
                                    self.channel_average)
        crop_box = [self.center_pos[0] - s_x / 2,
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]
        '''
        to_pil_image = transforms.ToPILImage()
        imgs = to_pil_image(x_crop[0])
        #imgs.show()
        plt.imshow(imgs)
        plt.show()
        '''
        
        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        
        
        
        '''
        box =[]
        best_score =[]
        iou_threshold = 0.1
        result=self.nms(pred_bbox,score,img,iou_threshold)
        for i in range(len(result)):
           
            bbox = pred_bbox[:, result[i]]
            scores = score[result[i]]
            best_score.append(scores)
            #print("scores:",scores)
            if scores >=0.5:
                cx = bbox[0]*img.shape[1]
                cy = bbox[1]*img.shape[0]
                

                width = bbox[2]#*img.shape[1]
                height = bbox[3]#*img.shape[0]

                # clip boundary
                cx, cy, width, height = self._bbox_clip(cx, cy, width,height, img.shape[:2])
                bbox = [cx - width / 2,
                        cy - height / 2,
                        width,
                        height]
                box.append(bbox)
            else:
                continue
        bbox = box
        '''
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        
        
        #加nms，計算所有框
        box =[]
        best_score =[]
        iou_threshold = 0.1
        result=self.nms(pred_bbox,score,img,iou_threshold)
        for i in range(len(result)):
           
            bbox = pred_bbox[:, result[i]] / scale_z
            lr = penalty[result[i]] * score[result[i]] * cfg.TRACK.LR
            scores = score[result[i]]
            best_score.append(scores)
            #print("scores:",scores)
            if scores >=0.5:
                cx = bbox[0] + self.center_pos[0]
                cy = bbox[1] + self.center_pos[1]
                #cx = bbox[0]*img.shape[1]
                #cy = bbox[1]*img.shape[0]
                
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
                box.append(bbox)
            else:
                continue
        bbox = box
        '''
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy,
                                                width, height, img.shape[:2])

        # udpate state
        #self.center_pos = np.array([cx, cy])
        #self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
         
        # processing mask
        pos = np.unravel_index(best_idx, (5, self.score_size, self.score_size))
        delta_x, delta_y = pos[2], pos[1]

        mask = self.model.mask_refine((delta_y, delta_x)).sigmoid().squeeze()
        out_size = cfg.TRACK.MASK_OUTPUT_SIZE
        mask = mask.view(out_size, out_size).cpu().data.numpy()

        s = crop_box[2] / cfg.TRACK.INSTANCE_SIZE
        base_size = cfg.TRACK.BASE_SIZE
        stride = cfg.ANCHOR.STRIDE
        sub_box = [crop_box[0] + (delta_x - base_size/2) * stride * s,
                   crop_box[1] + (delta_y - base_size/2) * stride * s,
                   s * cfg.TRACK.EXEMPLAR_SIZE,
                   s * cfg.TRACK.EXEMPLAR_SIZE]
        s = out_size / sub_box[2]

        im_h, im_w = img.shape[:2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w*s, im_h*s]
        mask_in_img = self._crop_back(mask, back_box, (im_w, im_h))
        polygon = self._mask_post_processing(mask_in_img)
        polygon = polygon.flatten().tolist()
       '''
        return {
                'bbox': bbox,
                'best_score': best_score,
                #'mask': mask_in_img,
                #'polygon': polygon,
               }
