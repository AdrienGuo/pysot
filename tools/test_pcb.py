# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, annotations, division, print_function,
                        unicode_literals)

import argparse
import os
import re
from unicodedata import decimal

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from pysot.utils.check_image import draw_box, draw_preds, save_image
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from pysot.core.config import cfg
from pysot.datasets.pcbdataset_test import PCBDatasetTest
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_float2str, vot_overlap
from torch.utils.data import DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--model', default='', type=str, help='model of models to eval')
parser.add_argument('--template_bg', type=str, help='whether crop template with bg')
parser.add_argument('--template_context_amount', type=float, help='how much bg for template')
parser.add_argument('--config', default='', type=str, help='config file')
parser.add_argument('--dataset', type=str, help='datasets')
parser.add_argument('--annotation', type=str, help='annotation for testing')
parser.add_argument('--save_dir', type=str, help='save to which directory')
# parser.add_argument('--video', default='', type=str, help='eval one special video')
# parser.add_argument('--vis', action='store_true', help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)


# def resize(image,size):
#     img_height = image.height
#     img_width = image.width
    
#     if (img_width / img_height) > 1:
#         rate = size / img_width
#     else:
#         rate = size/ img_height
       
#     width=int(img_width*rate)
#     height=int(img_height*rate)
   
#     return F.resize(image, (height,width))


    # for i in range (len(imgs)):
    #     if not Image.open(imgs[i]):
    #         continue
    #     else:
    #         transform = transforms.CenterCrop(255)
    #         im = Image.open(imgs[i])
    #         #im = transform(im)
    #         #im.save("PIL_img.jpg")
    #         # print("name:",imgs[i])
    #         length = int(len(annotation[i])/4)
    #         for j in range (length):
    #             bbox = [annotation[i][0+j*4],annotation[i][1+j*4],annotation[i][2+j*4],annotation[i][3+j*4]]
    #             print('bbox:',bbox)
    #             # 2.获取边框坐标
    #             # 边框格式　bbox = [xl, yl, xr, yr]
    #             #bbox1 =[111,98,25,101]
    #             #bbox1 = [72, 41, 208, 330]
    #             #label1 = 'man'

    #             #bbox2 = [100, 80, 248, 334]
    #             #label2 = 'woman'

    #             # 设置字体格式及大小
    #             #font = ImageFont.truetype(font='./Gemelli.ttf', size=np.floor(1.5e-2 * np.shape(im)[1] + 15).astype('int32'))
    #             #fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf",  size=np.floor(1.5e-2 * np.shape(im)[1] + 15).astype('int32'))
    #             draw = ImageDraw.Draw(im)
    #             # 获取label长宽
    #             #label_size1 = draw.textsize(label1)
    #             #label_size2 = draw.textsize(label2)

    #             # 设置label起点
    #             #text_origin1 = np.array([bbox1[0], bbox1[1]])
    #             #text_origin2 = np.array([bbox2[0], bbox2[1] - label_size2[1]])

    #             # 绘制矩形框，加入label文本
    #             draw.rectangle([bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]],outline='red',width=4)
    #             #draw.rectangle([tuple(text_origin1), tuple(text_origin1 + label_size1)], fill='red')
    #             #draw.text(text_origin1, str(label1), fill=(255, 255, 255),font=fnt)


    #             del draw

    #             # save the result image
    #             if not os.path.exists(save_dir):
    #                 os.makedirs(save_dir)
    #             im.save(save_dir + names[i])
    

# def main():
#     # load config
#     cfg.merge_from_file(args.config)

#     dataset_root = os.path.join('./testing_dataset', args.dataset)

#     # create model
#     model = ModelBuilder()

#     # load model
#     model = load_pretrain(model, args.model).cuda().eval()

#     # build tracker
#     tracker = build_tracker(model)
    
#     model_name = args.model.split('/')[-1].split('.')[0]
#     print(f"model_name: {model_name}")
    
#     gt_bbox_x1y1x2y2 = [45.09031666815281, 123.62768352031708, 123.27953104674816, 376.83301651477814]
#     gt_bbox_xywh = [gt_bbox_x1y1x2y2[0],
#                     gt_bbox_x1y1x2y2[1],
#                     gt_bbox_x1y1x2y2[2] - gt_bbox_x1y1x2y2[0],
#                     gt_bbox_x1y1x2y2[3] - gt_bbox_x1y1x2y2[1]]
#     toc = 0
#     pred_boxes = []
#     scores = []
#     track_times = []
    
#     ####################################################################
#     # Step 1.
#     # get the test data
#     ####################################################################
#     image_path = "./data/train/TemplateMatchingData/train/20200706 (114).bmp"
#     image_name = "20200706 (114).bmp"
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     tic = cv2.getTickCount()
#     cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox_xywh))
#     gt_bbox_xywh = [cx-w/2, cy-h/2, w, h]
#     print(f"gt_bbox: {gt_bbox_xywh}")

#     tracker.init(image, gt_bbox_xywh)
#     pred_box = gt_bbox_xywh
#     pred_boxes.append(pred_box)

#     outputs = tracker.track(image)
#     scores.append(outputs['best_score'])
#     pred_boxes.append(outputs['bbox'])
#     print(f"pred_boxes: {pred_boxes}")

#     toc += cv2.getTickCount() - tic
#     track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
#     toc /= cv2.getTickFrequency()
#     idx = 1
#     print('Time: {:5.1f}s Speed: {:3.1f}fps'.format(toc, idx / toc))

#     # save annotation result
#     model_path = os.path.join('./results', args.dataset, model_name)
#     if not os.path.isdir(model_path):
#         os.makedirs(model_path)
#     result_path = os.path.join(model_path, '{}.txt'.format(image_name))
#     with open(result_path, 'w') as f:
#         for x in pred_boxes:
#             f.write(','.join([str(i) for i in x])+'\n')
#     print(f"save annotation result to: {result_path}")
    
#     # draw result
#     annotation_path = result_path
#     save_dir = "./results/images/"
#     image_dir = "./testing_dataset/2/"
#     draw_preds(image_dir, annotation_path, save_dir)


def test(test_loader, tracker, dir):
    clocks = 0
    for idx, data in enumerate(test_loader):
        # only one data in a batch (batch_size=1)
        image_path = data['image_path'][0]
        template_box = data['template_box'][0]
        origin_template_box = data['origin_template_box'][0]
        template_image = data['template_image'].cuda()
        search_image = data['search_image'].cuda()
        gt_boxes = data['gt_boxes'][0]
        scale_ratio = data['r'][0].cpu().numpy()
        spatium = [x.cpu().item() for x in data['spatium']]

        ipdb.set_trace()

        ####################################################################
        # load image
        ####################################################################
        print(f"load image from: {image_path}")
        image = cv2.imread(image_path)

        ####################################################################
        # creat directories
        ####################################################################
        # 用圖片檔名當作 sub_dir 的名稱
        image_name = image_path.split('/')[-1].split('.')[0]
        sub_dir = os.path.join(dir, image_name)
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)
            print(f"create dir: {sub_dir}")

        # 創 sub_dir/origin，裡面存 original image
        origin_dir = os.path.join(sub_dir, "origin")
        if not os.path.isdir(origin_dir):
            os.makedirs(origin_dir)
            print(f"create dir: {origin_dir}")
        # 創 sub_dir/search，裡面存 search image
        search_dir = os.path.join(sub_dir, "search")
        if not os.path.isdir(search_dir):
            os.makedirs(search_dir)
            print(f"create dir: {search_dir}")
        # 創 sub_dir/template，裡面存 template image
        z_dir = os.path.join(sub_dir, "template")
        if not os.path.isdir(z_dir):
            os.makedirs(z_dir)
            print(f"Create dir: {z_dir}")
        # >>>>>>>>>>>>>>>>>>
        # 創 sub_dir/pred_annotation，裡面存 pred_annotation
        anno_dir = os.path.join(sub_dir, "pred_annotation")
        if not os.path.isdir(anno_dir):
            os.makedirs(anno_dir)
            print(f"Create dir: {anno_dir}")
        # 創 sub_dir/origin_pred_annotation，裡面存 origin_pred_annotation
        origin_anno_dir = os.path.join(sub_dir, "origin_pred_annotation")
        if not os.path.isdir(origin_anno_dir):
            os.makedirs(origin_anno_dir)
            print(f"Create dir: {origin_anno_dir}")
        # <<<<<<<<<<<<<<<<<<
        # >>>>>>>>>>>>>>>>>>
        # 創 sub_dir/pred，裡面存 pred image
        pred_dir = os.path.join(sub_dir, "pred")
        if not os.path.isdir(pred_dir):
            os.makedirs(pred_dir)
            print(f"Create dir: {pred_dir}")
        # 創 sub_dir/origin_pred，裡面存 original pred image
        origin_pred_dir = os.path.join(sub_dir, "origin_pred")
        if not os.path.isdir(origin_pred_dir):
            os.makedirs(origin_pred_dir)
            print(f"Create dir: {origin_pred_dir}")
        # <<<<<<<<<<<<<<<<<<

        ####################################################################
        # save original image
        ####################################################################
        origin_path = os.path.join(origin_dir, f"{image_name}.jpg")
        save_image(image, origin_path)
        print(f"save original image to: {origin_path}")

        ####################################################################
        # save search image
        ####################################################################
        search_image_cpu = search_image[0].cpu().numpy().copy()
        search_image_cpu = search_image_cpu.transpose(1, 2, 0)      # (3, 255, 255) -> (255, 255, 3)
        search_path = os.path.join(search_dir, f"{idx}.jpg")
        save_image(search_image_cpu, search_path)
        print(f"save search image to: {search_path}")

        ####################################################################
        # pred_boxes, scores
        ####################################################################
        pred_boxes = []
        origin_pred_boxes = []
        scores = None
        template_box = template_box.cpu().numpy().squeeze()
        origin_template_box = origin_template_box.cpu().numpy().squeeze()
        gt_boxes = gt_boxes.cpu().numpy()    # gt_boxes: (n, 4) #x1y1wh

        # convert template_box to the original size
        # template_box[0], template_box[2] = template_box[0] * image.shape[1], template_box[2] * image.shape[1]
        # template_box[1], template_box[3] = template_box[1] * image.shape[0], template_box[3] * image.shape[0]

        # template_corner = template[1]
        # # turn to (x1, y1, w, h), in order to match the input of .track()
        # gt_box = [                      # (x1, y1, w, h)
        #     template_corner[0],
        #     template_corner[1],
        #     template_corner[2] - template_corner[0],
        #     template_corner[3] - template_corner[1]
        # ]

        # template_box: (x1, y1, x2, y2) -> (x1, y1, w, h)
        template_box = [
            template_box[0],
            template_box[1],
            template_box[2] - template_box[0],
            template_box[3] - template_box[1]
        ]
        template_box = np.around(template_box, decimals=2)
        pred_boxes.append(template_box)

        # origin_template_box: (x1, y1, x2, y2) -> (x1, y1, w, h)
        origin_template_box = [
            origin_template_box[0],
            origin_template_box[1],
            origin_template_box[2] - origin_template_box[0],
            origin_template_box[3] - origin_template_box[1]
        ]
        origin_template_box = np.around(origin_template_box, decimals=2)
        origin_pred_boxes.append(origin_template_box)

        ####################################################################
        # init tracker
        # save template image to ./results/images/{image_name}/template/{idx}.jpg
        ####################################################################
        tic = cv2.getTickCount()
        # 用 template image 將 tracker 初始化
        z_img = tracker.init(template_image, template_box)
        z_img = np.transpose(z_img, (1, 2, 0))        # (3, 127, 127) -> (127, 127, 3)
        z_path = os.path.join(z_dir, f"{idx}.jpg")
        save_image(z_img, z_path)
        print(f"save z_img image to: {z_path}")

        ####################################################################
        # tracking
        ####################################################################
        # 用 search image 進行 "track" 的動作
        outputs = tracker.track(image, search_image, scale_ratio, spatium)

        scores = np.around(outputs['top_scores'], decimals=2)
        # === pred_boxes on "search" image ===
        for box in outputs['pred_boxes']:
            box = np.around(box, decimals=2)
            pred_boxes.append(box)
        # === pred_boxes on "original" image ===
        for origin_box in outputs['origin_pred_boxes']:
            origin_box = np.around(origin_box, decimals=2)
            origin_pred_boxes.append(origin_box)
        toc = cv2.getTickCount()
        clocks += toc - tic    # 總共有多少個 clocks (clock cycles)

        # save search image
        x_img = outputs['x_img']
        x_img = np.transpose(x_img, (1, 2, 0))
        # x_path = os.path.join(x_dir, f"{idx}.jpg")
        # save_image(x_img, x_path)
        # print(f"save x_img image to: {x_path}")

        ####################################################################
        # save annotation file
        ####################################################################
        # === pred_boxes on "search" image ===
        anno_path = os.path.join(anno_dir, f"{idx}.txt")
        with open(anno_path, 'w') as f:
            f.write(', '.join(map(str, pred_boxes[0])) + '\n')    # template
            for i, x in enumerate(pred_boxes[1:]):
                # format: [x1, y1, w, h, score]
                f.write(', '.join(map(str, x)) + ', ' + str(scores[i]) + '\n')
        print(f"save annotation result to: {anno_path}")
        # === pred_boxes on "original" image ===
        origin_anno_path = os.path.join(origin_anno_dir, f"{idx}.txt")
        with open(origin_anno_path, 'w') as f:
            f.write(', '.join(map(str, origin_pred_boxes[0])) + '\n')    # template
            for i, x in enumerate(origin_pred_boxes[1:]):
                # format: [x1, y1, w, h, score]
                f.write(', '.join(map(str, x)) + ', ' + str(scores[i]) + '\n')
        print(f"save origin annotation result to: {origin_anno_path}")

        ####################################################################
        # draw the gt boxes
        ####################################################################
        # === gt_boxes on "search" image ===
        gt_image = draw_box(search_image_cpu, gt_boxes, type="gt")

        ####################################################################
        # draw the pred boxes
        ####################################################################
        # === pred_boxes on "search" image ===
        pred_path = os.path.join(pred_dir, f"{idx}.jpg")
        pred_image = draw_preds(sub_dir, gt_image, scores, anno_path, idx)
        if pred_image is None:      # 如果沒偵測到物件，存 search image
            save_image(search_image_cpu, pred_path)
        else:
            save_image(pred_image, pred_path)
        print(f"save pred image to: {pred_path}")
        # === pred_boxes on "original" image ===
        origin_pred_path = os.path.join(origin_pred_dir, f"{idx}.jpg")
        origin_pred_image = draw_preds(sub_dir, image, scores, origin_anno_path, idx)
        if origin_pred_image is None:      # 如果沒偵測到物件，存 original image
            save_image(image, origin_pred_path)
        else:
            save_image(origin_pred_image, origin_pred_path)
        print(f"save origin pred image to: {origin_pred_path}")

        print("=" * 20)

        ipdb.set_trace()

    period = clocks / cv2.getTickFrequency()
    fps = idx / period
    print(f"Speed: {fps} fps")


if __name__ == "__main__":
    test_dataset = PCBDatasetTest(args)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,    # 只能設 1
                             num_workers=0)

    cfg.merge_from_file(args.config)        # 不加 ModelBuilder() 會出問題ㄟ??

    # create model
    model = ModelBuilder()
    # load model
    model = load_pretrain(model, args.model).cuda().eval()

    # model_name = args.model.split("/")[2].split(".")[0]
    model_name = args.model.split('/')[-2]
    print(f"model_name: {model_name}")
    print(f"load model from: {args.model}")

    # build tracker
    tracker = build_tracker(model)

    dir = os.path.join(args.save_dir, model_name)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    test(test_loader, tracker, dir)

    print("="*20, "Done!", "="*20, "\n")
