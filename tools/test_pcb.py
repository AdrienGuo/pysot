# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from PIL import Image , ImageDraw , ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import torchvision.transforms.functional as F
from torchvision import transforms

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

import torchvision.transforms.functional as F

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
    help='datasets')
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

def draw_result(image_dir, annotation_path, save_dir):
    imgs=[]
    names=[]
    annotation=[]
    
    #讀圖片檔
    for fname in os.listdir(image_dir):
        #fname =str(fname)
        if fname.endswith(('.bmp','.jpg')):
            filePath = os.path.join(image_dir, fname)
            imgs.append(filePath)
            names.append(fname)
    imgs=sorted(imgs)
    names=sorted(names)
    # print("imgs:",imgs)
    # print("names:",names)
    #for fname in os.listdir(annotation_path):
    #    filePath = os.path.join(annotation_path,fname)
    
    #讀標註檔
    f = open(annotation_path, 'r')
    lines = f.readlines()[1:]
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
    
    for i in range (len(imgs)):
        if not Image.open(imgs[i]):
            continue
        else:
            transform = transforms.CenterCrop(255)
            im = Image.open(imgs[i])
            #im = transform(im)
            #im.save("PIL_img.jpg")
            # print("name:",imgs[i])
            length = int(len(annotation[i])/4)
            for j in range (length):
                bbox = [annotation[i][0+j*4],annotation[i][1+j*4],annotation[i][2+j*4],annotation[i][3+j*4]]
                print('bbox:',bbox)
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
                draw.rectangle([bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]],outline='red',width=4)
                #draw.rectangle([tuple(text_origin1), tuple(text_origin1 + label_size1)], fill='red')
                #draw.text(text_origin1, str(label1), fill=(255, 255, 255),font=fnt)


                del draw

                # save the result image
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                im.save(save_dir + names[i])
    

def main():
    # load config
    cfg.merge_from_file(args.config)

    dataset_root = os.path.join('./testing_dataset', args.dataset)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)
    
    model_name = args.snapshot.split('/')[-1].split('.')[0]
    print(f"model_name: {model_name}")
    
    gt_bbox_x1y1x2y2 = [45.09031666815281, 123.62768352031708, 123.27953104674816, 376.83301651477814]
    gt_bbox_xywh = [gt_bbox_x1y1x2y2[0],
                    gt_bbox_x1y1x2y2[1],
                    gt_bbox_x1y1x2y2[2] - gt_bbox_x1y1x2y2[0],
                    gt_bbox_x1y1x2y2[3] - gt_bbox_x1y1x2y2[1]]
    toc = 0
    pred_bboxes = []
    scores = []
    track_times = []
    
    ####################################################################
    # Step 1.
    # get the test data
    ####################################################################
    image_path = "./data/train/TemplateMatchingData/train/20200706 (114).bmp"
    image_name = "20200706 (114).bmp"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tic = cv2.getTickCount()
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox_xywh))
    gt_bbox_xywh = [cx-w/2, cy-h/2, w, h]
    tracker.init(image, gt_bbox_xywh)
    pred_bbox = gt_bbox_xywh
    pred_bboxes.append(pred_bbox)

    outputs = tracker.track(image)
    scores.append(outputs['best_score'])
    pred_bboxes.append(outputs['bbox'])
    print(f"pred_bboxes: {pred_bboxes}")

    toc += cv2.getTickCount() - tic
    track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
    toc /= cv2.getTickFrequency()
    idx = 1
    print('Time: {:5.1f}s Speed: {:3.1f}fps'.format(toc, idx / toc))

    # save result
    model_path = os.path.join('./results', args.dataset, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    result_path = os.path.join(model_path, '{}.txt'.format(image_name))
    with open(result_path, 'w') as f:
        for x in pred_bboxes:
            f.write(','.join([str(i) for i in x])+'\n')
    print(f"save annotation result to: {result_path}")
    
    # draw result
    annotation_path = result_path
    save_dir = "./results/images/"
    image_dir = "./testing_dataset/2/"
    draw_result(image_dir, annotation_path, save_dir)


if __name__ == "__main__":
    main()
    