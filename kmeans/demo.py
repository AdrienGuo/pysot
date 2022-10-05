
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import json
import logging
import math
import os
import random
import time

import ipdb
import numpy as np
import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from pysot.core.config import cfg
from pysot.datasets.pcbdataset_old import PCBDataset

from kmeans.kmean import AnchorKmeans

parser = argparse.ArgumentParser(description='find anchors by k-means')
parser.add_argument('--crop_method', default='', type=str, help='teacher / amy')
parser.add_argument('--bg', type=str, nargs='?', const='', help='background')
parser.add_argument('--anchors', type=int, help='number of anchors')
parser.add_argument('--config', default='', type=str, help='config file')
parser.add_argument('--dataset', type=str, help='datasets')
parser.add_argument('--annotation', type=str, help='annotation for testing')
args = parser.parse_args()


plt.style.use('ggplot')

# k value: number of anchor you want
choose_k = args.anchors

anchor_paper = np.array(
    [[104, 32],
     [88, 40],
     [64, 64],
     [40, 80],
     [32, 96]]
)


####################################################################
# visualizing default anchors
####################################################################
def visualize_anchors(anchors, size, name):
    """
    Parameters:
        anchors: (n, 2)
    """
    anchors = np.round(anchors).astype(int)

    rects = np.empty((anchors.shape[0], 4), dtype=int)
    for i in range(len(anchors)):
        w, h = anchors[i]
        x1, y1 = -(w // 2), -(h // 2)
        rects[i] = [x1, y1, w, h]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    for rect in rects:
        x1, y1, w, h = rect
        rect1 = Rectangle((x1, y1), w, h, color='royalblue', fill=False, linewidth=2)
        ax.add_patch(rect1)
    plt.xlim([-(size // 2), size // 2])
    plt.ylim([-(size // 2), size // 2])
    plt.xlabel("width")
    plt.ylabel("height")
    ax = plt.gca()
    ax.set_aspect(1)    # 讓 x, y 他成正比

    save_path = f"./kmeans/demo/{name}.jpg"
    plt.savefig(save_path)
    print(f"save visualizing-anchors plot to: {save_path}")


####################################################################
# get all the z_box (template) on search image
####################################################################
cfg.merge_from_file("./experiments/siamrpn_r50_l234_dwxcorr/config.yaml")
dataset = PCBDataset(args)

boxes = list()
for i in range(len(dataset)):
    # data: (1, [x1, y1, x2, y2])
    data = dataset[i]
    z_box = data["z_box"].squeeze()
    z_w = z_box[2] - z_box[0]
    z_h = z_box[3] - z_box[1]
    boxes.append([z_w, z_h])
boxes = np.array(boxes)
print(f"Number of boxes: {boxes.shape[0]}")


####################################################################
# Plot scatter of original boxes
####################################################################
print("[INFO] Draw boxes")
plt.figure()
plt.xlabel("width")
plt.ylabel("height")
plt.scatter(boxes[:, 0], boxes[:, 1], c='orange')
ax = plt.gca()
ax.set_aspect(1)    # 讓 x, y 軸成正比
save_path = "./kmeans/demo/boxes_scatter.jpg"
plt.savefig(save_path)
print(f"save boxes scatter plot to: {save_path}")


####################################################################
# Plot average IoU curve
####################################################################
print('[INFO] Run anchor k-means with k = 2,3,...,k')
results = {}
for k in range(2, 21):
    model = AnchorKmeans(k, random_seed=333)
    model.fit(boxes)
    avg_iou = model.avg_iou()
    results[k] = {'anchors': model.anchors_, 'avg_iou': avg_iou}
    print(f"K = {k:<2} | Avg IOU = {avg_iou:.4f}".format(k, avg_iou))

print('[INFO] Plot average IOU curve')
plt.figure()
plt.plot(range(2, 21), [results[k]["avg_iou"] for k in range(2, 21)], "o-")
plt.ylabel("Avg IOU")
plt.xlabel("K (#anchors)")
plt.xticks(range(2, 21, 1))

save_path = "./kmeans/demo/k-iou_line.jpg"
plt.savefig(save_path)
print(f"save k-iou plot to {save_path}")

print('[INFO] The result anchors:')
anchors = results[choose_k]['anchors']
print(anchors)


####################################################################
# save defined anchors with the original boxes in scatter plot
####################################################################
print("[INFO] Draw boxes")
plt.figure()
plt.xlabel("width")
plt.ylabel("height")
plt.scatter(boxes[:, 0], boxes[:, 1], c="orange")
plt.scatter(anchors[:, 0], anchors[:, 1], c="blue")
ax = plt.gca()
ax.set_aspect(1)    # 讓 x, y 成正比

save_path = f"./kmeans/demo/boxes_kmeans{choose_k}_scatter.jpg"
plt.savefig(save_path)
print(f"save boxes-kmeans{choose_k} scatter plot to: {save_path}")

####################################################################
# Visualize paper & kmeans anchors
####################################################################
print('[INFO] Visualizing paper anchors')
visualize_anchors(anchor_paper, 255, "anchor_paper")

print('[INFO] Visualizing kmeans anchors')
visualize_anchors(anchors, cfg.TRAIN.SEARCH_SIZE, "anchor_kmeans")

print("=" * 20 + " Done!! " + "=" * 20 + "\n")
