# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
#from pysot_toolkit.trackers.transform import get_transforms
from torch.utils.data.dataloader import DataLoader

import torchvision.transforms.functional as F
from torchvision import transforms

from PIL import Image 



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

def main():
    # load config
    cfg.merge_from_file(args.config)

    #cur_dir = os.path.dirname(os.path.realpath('/tf/pysot/tools/'))
    #dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.model).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    # create dataset
    #dataset = DatasetFactory.create_dataset(name=args.dataset,
    #                                        dataset_root=dataset_root,
    #                                        load_img=False)

    model_name = args.model.split('/')[-1].split('.')[0]
    model_name = "siamrpn_r50_l234_dwxcorr" #siamrpn_r50_l234_dwxcorr siammask_r50_l3
    print("????:",model_name)
    total_lost = 0
    
    #==================================================================
    # load config

    dataset_root = '/tf/TransT/PCB' #Absolute path of the dataset
    net_path = '/tf/TransT/pytracking/networks/transt.pth' #Absolute path of the model

    # create model
    #net = NetWithBackbone(net_path=net_path, use_gpu=True)
    #tracker = Tracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)
    
    # create dataset
    #dataset = DatasetFactory.create_dataset(name=args.dataset,
    #                                        dataset_root=dataset_root,
    #                                        load_img=False)
    
    #model_name = args.name
    total_lost = 0
    idx=1
    # OPE tracking
    print("YES")
    #gt_bbox = [461,265,59,141] #(x1,y1,w,h)
    
    gt_bbox = [210,493,113,143] #(x1,y1,w,h) 840x764
    #gt_bbox = [90,427,113,143] #(x1,y1,w,h) 600x600 UD7_1_LowAngleLight
    #gt_bbox = [339,222,64,141] #600x600 L5_1_SolderLight.jpg
    #gt_bbox = [451,11,118,72] #600x600 U3_3_LowAngleLight.jpg_0617131534_IC_OK_OCVFAIL_4589
    #gt_bbox = [197,154,35,28] #U49_1_LowAngleLight.jpg
    #gt_bbox =[0.786328 ,0.752083, 0.157031 ,0.081944]
    toc = 0
    pred_bboxes = []
    scores = []
    track_times = []
    #_,transform_val = get_transforms(512)
    img_path1 = './tf/pysot/testing_dataset/PCB/UD7_1_LowAngleLight.jpg'#1/Sot23_20210817_uniform_2.bmp'#20200721_159.bmp'
    #img_path2 = '/tf/pysot/testing_dataset/PCB/test/L5_1_SolderLight.jpg0401231012__COMPARE_NG(0.49)__9363.jpg'
    #img_path2 = '/tf/pysot/testing_dataset/PCB/UD13_2_LowAngleLight.jpg'
    img_name ="UD7_1_LowAngleLight"
    img = cv2.imread(img_path1)
    
    w = 600
    h = 600
    idx =1
    #print("??:",img.shape[0])
    
    x = img.shape[1]/2 - w/2
    y = img.shape[0]/2 - h/2

    crop_img = img[int(y):int(y+h), int(x):int(x+w)]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imh ,imw = img.shape[:2]
    #image = Image.fromarray((cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    #origin_size = image.size
    #img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)    
        #tic = cv2.getTickCount()
    ''' 
    gt_bbox[0]=gt_bbox[0]*imw
    gt_bbox[1]=gt_bbox[1]*imh
    gt_bbox[2]=gt_bbox[2]*imw
    gt_bbox[3]=gt_bbox[3]*imh
    x1 = (gt_bbox[0]-gt_bbox[2]/2)
    y1 = (gt_bbox[1]-gt_bbox[3]/2)
    x2 = (gt_bbox[0]+gt_bbox[2]/2)
    y2 = (gt_bbox[1]+gt_bbox[3]/2) 
    gt_bbox[0] = x1
    gt_bbox[1] = y1
    gt_bbox[2] = x2-x1
    gt_bbox[3] = y2-y1
      
    ''' 
        
    tic = cv2.getTickCount()
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    gt_bbox_ = [cx-w/2, cy-h/2, w, h]
    init_info = {'init_bbox':gt_bbox_}
    tracker.init(img, gt_bbox_)
    pred_bbox = gt_bbox_
    scores.append(None)
    pred_bboxes.append(pred_bbox)
        #img=img.cuda()
    # for i in range(1):
    #     if i == 1:
    #         img = cv2.imread(img_path2)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     if i == 2:
    #         img = cv2.imread(img_path3)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    outputs = tracker.track(img)
    


    pred_bbox = outputs['bbox']
    pred_bboxes.append(pred_bbox)
    scores.append(outputs['best_score'])
    toc += cv2.getTickCount() - tic
    track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
        
    model_path = os.path.join('./tf/pysot/results/', args.dataset, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    result_path = os.path.join(model_path, '{}.txt'.format(img_name))
    with open(result_path, 'w') as f:
        for x in pred_bboxes:
            f.write(','.join([str(i) for i in x])+'\n')
    toc /= cv2.getTickFrequency()
    # save results
    print('Time: {:5.1f}s Speed: {:3.1f}fps'.format(toc, idx / toc))

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets',default='PCB')
# parser.add_argument('--config', default='./experiments/siamrpn_r50_l234_dwxcorr/config.yaml', type=str,
#         help='config file')#siamrpn_r50_l234_dwxcorr siammask_r50_l3
parser.add_argument('--config', default='./tf/pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml', type=str,
        help='config file')#siamrpn_r50_l234_dwxcorr siammask_r50_l3
# parser.add_argument('--model', default='./experiments/siamrpn_r50_l234_dwxcorr/model_amy.pth', type=str,
#         help='model of models to eval')
parser.add_argument('--model', default='./tf/pysot/model/siamrpn_r50_l234_dwxcorr/model.pth', type=str,
        help='model of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args([])

torch.set_num_threads(1)
main()

#畫bbox
from PIL import Image , ImageDraw , ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import torchvision.transforms.functional as F
from torchvision import transforms

def pil_draw(img_path,annotation_path):
    imgs=[]
    names=[]
    annotation=[]
    #讀圖片檔
    for fname in os.listdir(img_path):
        #fname =str(fname)
        if fname.endswith(('.bmp','.jpg')):
            filePath = os.path.join(img_path, fname)
            imgs.append(filePath)
            names.append(fname)
    imgs=sorted(imgs)
    names=sorted(names)
    print("imgs:",imgs)
    print("names:",names)
    #for fname in os.listdir(annotation_path):
    #    filePath = os.path.join(annotation_path,fname)
    #讀標註檔
    f = open(annotation_path,'r')
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
            transform = transforms.CenterCrop(600)
            im = Image.open(imgs[i])
            #im = transform(im)
            #im.save("PIL_img.jpg")
            print("name:",imgs[i])
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
                draw.rectangle([bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]],outline='red',width=6)
                #draw.rectangle([tuple(text_origin1), tuple(text_origin1 + label_size1)], fill='red')
                #draw.text(text_origin1, str(label1), fill=(255, 255, 255),font=fnt)


                del draw
                #im.show()
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                im.save(save_path+names[i])
                im2 = Image.open(save_path+names[i])
                plt.imshow(im2)
                plt.show()
        
img_path = './tf/pysot/testing_dataset/PCB/2/'
annotation_path = './tf/pysot/results/PCB/siamrpn_r50_l234_dwxcorr/UD7_1_LowAngleLight.txt'#siammask_r50_l3
save_path = './tf/pysot/result_img/'
pil_draw(img_path,annotation_path)