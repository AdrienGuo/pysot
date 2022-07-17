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
	
	gt_bbox = [123.27983754873276, 46.277657985687256, 502.71278911828995, 453.522319316864]
	toc = 0
	pred_bboxes = []
	scores = []
	track_times = []
	

	image1_path = "./data/train/TemplateMatchingData/train/20200706 (114).bmp"
	image_name = "20200706 (114).bmp"
	image = cv2.imread(image1_path)

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	tic = cv2.getTickCount()
	cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
	gt_bbox = [cx-w/2, cy-h/2, w, h]
	tracker.init(image, gt_bbox)
	pred_bbox = gt_bbox
	pred_bboxes.append(pred_bbox)
	
	outputs = tracker.track(image)
	pred_bboxes.append(pred_bbox)
	scores.append(outputs['best_score'])

	toc += cv2.getTickCount() - tic
	track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())

	model_path = os.path.join('./results', args.dataset, model_name)
	if not os.path.isdir(model_path):
		os.makedirs(model_path)
	result_path = os.path.join(model_path, '{}.txt'.format(image_name))
	with open(result_path, 'w') as f:
		for x in pred_bboxes:
			f.write(','.join([str(i) for i in x])+'\n')
	toc /= cv2.getTickFrequency()
	
	# save results
	idx = 1
	print('Time: {:5.1f}s Speed: {:3.1f}fps'.format(toc, idx / toc))

if __name__ == "__main__":
	main()
	