# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import (absolute_import, annotations, division, print_function,
                        unicode_literals)

import argparse
import os
import re
from unittest import TestLoader

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
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
parser.add_argument('--dataset', type=str,
    help='datasets')
parser.add_argument('--annotation', type=str, help='annotation for testing')
parser.add_argument('--save_dir', type=str, help='save to which directory')
parser.add_argument('--config', default='', type=str,
    help='config file')
parser.add_argument('--snapshot', default='', type=str,
    help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
    help='eval one special video')
parser.add_argument('--vis', action='store_true',
    help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)


def resize(image,size):
    img_height = image.height
    img_width = image.width
    
    if (img_width / img_height) > 1:
        rate = size / img_width
    else:
        rate = size/ img_height
       
    width=int(img_width*rate)
    height=int(img_height*rate)
   
    return F.resize(image, (height,width))


def draw_result(sub_dir, annotation_path, idx):
    imgs=[]
    names=[]
    annotation=[]
    
    #讀圖片檔
    search_dir = os.path.join(sub_dir, "search")
    for search in os.listdir(search_dir):
        image_path = os.path.join(search_dir, search)
        #fname =str(fname)
        # if fname.endswith(('.bmp','.jpg')):
        #     filePath = os.path.join(args.dataset, fname)
        #     imgs.append(filePath)
        #     names.append(fname)
        imgs.append(image_path)
    # imgs=sorted(imgs)
    names=sorted(names)
    # print("imgs:",imgs)
    # print("names:",names)
    #for fname in os.listdir(annotation_path):
    #    filePath = os.path.join(annotation_path,fname)
    
    #讀標註檔
    f = open(annotation_path, 'r')
    lines = f.readlines()[1:]
    print(f"lines: {lines}")
    if lines[0] == '\n':
        print(f"there is no predict item in this image")
        print(f"=" * 20)
        return

    for line in lines:
        line=line.strip('\n')
        line = re.sub("\[|\]","",line)
        #print('line',line)
        #line=line.strip("'")
        line=line.split(',')
        #print('line',line)
        line = list(map(float, line));
        annotation.append(line)
    
    #畫圖
    im = Image.open(imgs[0])
    #im = transform(im)
    #im.save("PIL_img.jpg")
    # print("name:",imgs[i])
    length = int(len(annotation[0])/4)
    for j in range (length):
        bbox = [annotation[0][0+j*4],annotation[0][1+j*4],annotation[0][2+j*4],annotation[0][3+j*4]]
        # print('bbox:',bbox)
        # 2.获取边框坐标
        # 边框格式　bbox = [xl, yl, xr, yr]
        #bbox1 =[111,98,25,101]
        #bbox1 = [72, 41, 208, 330]
        #label1 = 'man'

        #bbox2 = [100, 80, 248, 334]
        #label2 = 'woman'

        # 设置字体格式及大小
        #font = ImageFont.truetype(font='./Gemelli.ttf', size=np.floor(1.5e-2 * np.shape(im)[1] + 15).astype('int32'))
        #fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf",  size=np.floor(1.5e-2 * np.shape(im)[1] + 15).astype('int32'))
        draw = ImageDraw.Draw(im)
        # 获取label长宽
        #label_size1 = draw.textsize(label1)
        #label_size2 = draw.textsize(label2)

        # 设置label起点
        #text_origin1 = np.array([bbox1[0], bbox1[1]])
        #text_origin2 = np.array([bbox2[0], bbox2[1] - label_size2[1]])

        # 绘制矩形框，加入label文本
        draw.rectangle([bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]],outline='red',width=6)
        #draw.rectangle([tuple(text_origin1), tuple(text_origin1 + label_size1)], fill='red')
        #draw.text(text_origin1, str(label1), fill=(255, 255, 255),font=fnt)

        del draw

    # save the result image
    save_dir = os.path.join(sub_dir, "predict")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, str(idx) + ".jpg")
    im.save(save_path)
    print(f"save predict image to: {save_path}")
    print("=" * 20)

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
#     model = load_pretrain(model, args.snapshot).cuda().eval()

#     # build tracker
#     tracker = build_tracker(model)
    
#     model_name = args.snapshot.split('/')[-1].split('.')[0]
#     print(f"model_name: {model_name}")
    
#     gt_bbox_x1y1x2y2 = [45.09031666815281, 123.62768352031708, 123.27953104674816, 376.83301651477814]
#     gt_bbox_xywh = [gt_bbox_x1y1x2y2[0],
#                     gt_bbox_x1y1x2y2[1],
#                     gt_bbox_x1y1x2y2[2] - gt_bbox_x1y1x2y2[0],
#                     gt_bbox_x1y1x2y2[3] - gt_bbox_x1y1x2y2[1]]
#     toc = 0
#     pred_bboxes = []
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
#     pred_bbox = gt_bbox_xywh
#     pred_bboxes.append(pred_bbox)

#     outputs = tracker.track(image)
#     scores.append(outputs['best_score'])
#     pred_bboxes.append(outputs['bbox'])
#     print(f"pred_bboxes: {pred_bboxes}")

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
#         for x in pred_bboxes:
#             f.write(','.join([str(i) for i in x])+'\n')
#     print(f"save annotation result to: {result_path}")
    
#     # draw result
#     annotation_path = result_path
#     save_dir = "./results/images/"
#     image_dir = "./testing_dataset/2/"
#     draw_result(image_dir, annotation_path, save_dir)


def test(test_loader, tracker, inner_dir):
    for idx, data in enumerate(test_loader):
        pred_bboxes = []
        scores = []
        
        ####################################################################
        # Step 1.
        # get the test data
        ####################################################################
        # get the image
        print(f"image_path: {data['image_path'][0]}")
        image = cv2.imread(data['image_path'][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_name = data['image_path'][0].split("/")[-1].split(".")[0]
        # 用圖片檔名當作 sub_dir 的名稱
        sub_dir = os.path.join(inner_dir, image_name)
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)
            print(f"create dir: {sub_dir}")
        # 在 sub_dir 裡面再創一個 search_dir，然後存 search image 進去
        search_dir = os.path.join(sub_dir, "search")
        if not os.path.isdir(search_dir):
            os.makedirs(search_dir)
            print(f"create dir: {search_dir}")
        search_path = os.path.join(search_dir, image_name + ".jpg")
        cv2.imwrite(search_path, image)
        print(f"save search image to: {search_path}")

        template_x1y1x2y2 = data['template'].numpy()    # turn tensor to numpy
        # print(f"template: {template_x1y1x2y2}")
        template_x1y1x2y2 = template_x1y1x2y2.transpose(1, 0).squeeze()
        # turn to [x1, y1, w, h], in order to match the input of .track()
        gt_box = [
            template_x1y1x2y2[0],
            template_x1y1x2y2[1],
            template_x1y1x2y2[2] - template_x1y1x2y2[0],
            template_x1y1x2y2[3] - template_x1y1x2y2[1]
        ]
        # print(f"gt_box: {gt_box}")
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_box))
        # print(f"{cx}, {cy}, {w}, {h}")
        gt_bbox_xywh = [cx-w/2, cy-h/2, w, h]
        
        ####################################################################
        # Step 2.
        # start predict
        ####################################################################
        # print(f"image: {type(image)}")
        # print(f"gt_bbox_xywh: {type(gt_bbox_xywh[0])}")
        z_crop = tracker.init(image, gt_bbox_xywh)
        # print(f"z_crop: {z_crop}")
        pred_bbox = gt_bbox_xywh
        pred_bboxes.append(pred_bbox)

        outputs = tracker.track(image)

        scores.append(outputs['best_score'])
        pred_bboxes.append(outputs['bbox'])
        # print(f"pred_bboxes: {pred_bboxes}")

        # save template result to ./results/images/{image_name}/template/{idx}.jpg
        template_dir = os.path.join(sub_dir, "template")
        if not os.path.isdir(template_dir):
            os.makedirs(template_dir)
        z_crop = z_crop[0]
        z_crop = np.transpose(z_crop, (1, 2, 0))        # [3, 127, 127] -> [127, 127, 3]
        template_path = os.path.join(template_dir, f"{idx}.jpg")
        cv2.imwrite(template_path, z_crop)
        print(f"save template to: {template_path}")

        # save annotation result to ./results/images/{image_name}/annotation/{idx}.txt
        anno_dir = os.path.join(sub_dir, "annotation")
        if not os.path.isdir(anno_dir):
            os.makedirs(anno_dir)
        anno_path = os.path.join(anno_dir, f"{idx}.txt")
        with open(anno_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
        print(f"save annotation result to: {anno_path}")

        # draw result
        draw_result(sub_dir, anno_path, idx)


if __name__ == "__main__":
    data_dir = args.dataset.split("/")[-2]
    data_dir = os.path.join(args.save_dir, data_dir)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    test_dataset = PCBDatasetTest(args)
    test_loader = DataLoader(test_dataset,
                             batch_size=1)
    print(len(test_loader.dataset))
    
    cfg.merge_from_file(args.config)        # 不加 ModelBuilder() 會出問題ㄟ??
    
    # create model
    model = ModelBuilder()
    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()
    # model_name = args.snapshot.split("/")[2].split(".")[0]
    # model_name = "amy_model"
    model_name = "my_model"
    print(f"load model from: {args.snapshot}")

    # build tracker
    tracker = build_tracker(model)

    inner_dir = os.path.join(data_dir, model_name)
    test(test_loader, tracker, inner_dir)

    print("="*20, "Done!", "="*20, "\n")
