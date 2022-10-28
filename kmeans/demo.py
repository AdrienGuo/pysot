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
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from pysot.core.config import cfg
from pysot.datasets.pcbdataset_tri import PCBDataset
from pysot.utils.check_image import create_dir
from torch.utils.data import DataLoader

from kmeans.kmean import AnchorKmeans

parser = argparse.ArgumentParser(description='find anchors by k-means')
parser.add_argument('--crop_method', default='', type=str, help='teacher / amy')
parser.add_argument('--bg', type=str, default=1, nargs='?', const='', help='background')
parser.add_argument('--neg', type=float, default=0, help='useless')
parser.add_argument('--anchors', type=int, help='number of anchors')
parser.add_argument('--config', default='', type=str, help='config file')
parser.add_argument('--part', type=str, help='train / test')
parser.add_argument('--dataset_path', type=str, help='datasets path')
parser.add_argument('--dataset_name', type=str, help='datasets name')
parser.add_argument('--criteria', type=str, help='sample criteria for dataset')
args = parser.parse_args()

plt.style.use('ggplot')
cfg.merge_from_file("./experiments/siamrpn_r50_l234_dwxcorr/config.yaml")

# k value: number of anchor you want
choose_k = args.anchors
save_dir = f"./kmeans/demo/{args.part}/{args.dataset_name}/{args.criteria}"
create_dir(save_dir)

anchor_official = np.array(
    [[104, 32],
     [88, 40],
     [64, 64],
     [40, 80],
     [32, 96]]
)


##################################################
# Visualize anchors
##################################################
def visualize_anchors(anchors, size, name=None):
    """
    Args:
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
    ax.set_aspect(1)    # 讓 x, y 軸成正比

    save_path = os.path.join(save_dir, f"k{choose_k}_anchor.jpg")
    plt.savefig(save_path)
    print(f"Save visualizing-anchors plot to: {save_path}")


##################################################
# Get all the z_box (template) on search image
##################################################
dataset = PCBDataset(args, "val")
data_loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=16
)
print("--- Build dataset done ---")

print("Processing data...")
boxes = list()
for idx, data in enumerate(data_loader):
    # data["z_box"]: (1, [x1, y1, x2, y2])
    z_box = data["z_box"].squeeze()
    z_w = z_box[2] - z_box[0]
    z_h = z_box[3] - z_box[1]
    boxes.append([z_w, z_h])
boxes = np.array(boxes)
boxes_num = boxes.shape[0]
print(f"Number of boxes: {boxes_num}")


##################################################
# Count width & height values
##################################################
wh_below64 = boxes[(boxes[:, 0] < 64) & (boxes[:, 1] < 64)]
wh_below64_num = wh_below64.shape[0]
below_ratio = round(wh_below64_num / boxes_num, 2) * 100

wh_above64 = boxes[(boxes[:, 0] > 64) & (boxes[:, 1] > 64)]
wh_above64_num = wh_above64.shape[0]
above_ratio = round(wh_above64_num / boxes_num, 2) * 100

print(f"number: {boxes_num}")
print(f"below 64 number: {wh_below64_num}")
print(f"below 64 ratio: {below_ratio}%")
print(f"above 64 number: {wh_above64_num}")
print(f"above 64 ratio: {above_ratio}%")
print("-" * 10)


##################################################
# Plot scatter graph of original boxes
##################################################
print("[INFO] Draw boxes")
# plt.figure()
# plt.title(f"{args.dataset_name}_{args.crop_method}")
# plt.xlabel("width")
# plt.ylabel("height")
# plt.scatter(boxes[:, 0], boxes[:, 1], c='orange')
# # plt.text(1, 0, f"box num: {boxes_num}", fontsize=12)
# # plt.text(1, 1, f"w & h below 64: {wh_below64_num}", fontsize=12)
# ax = plt.gca()
# ax.set_aspect(1)    # 讓 x, y 軸成正比
# save_path = f"./kmeans/demo/{args.part}/boxes_scatter.jpg"
# plt.savefig(save_path)
# print(f"Save boxes scatter plot to: {save_path}")

fig, ax = plt.subplots()
ax.scatter(x=boxes[:, 0], y=boxes[:, 1], c="orange")
ax.set_xlim([0, cfg.TRAIN.SEARCH_SIZE])
ax.set_ylim([0, cfg.TRAIN.SEARCH_SIZE])
ax.set_xlabel("width")
ax.set_ylabel("height")
ax.set_aspect(1)
ax.hlines(y=64, xmin=0, xmax=64, linewidth=1, color='k')
ax.vlines(x=64, ymin=0, ymax=64, linewidth=1, color='k')
ax.text(200, 255, f"Total: {boxes_num}", fontsize=12)
ax.text(200, 235, f"below 64: {wh_below64_num}", fontsize=12)
ax.text(200, 220, f"below ratio: {below_ratio}%", fontsize=12)
ax.text(200, 200, f"above 64: {wh_above64_num}", fontsize=12)
ax.text(200, 185, f"above ratio: {above_ratio}%", fontsize=12)
save_path = os.path.join(save_dir, "scatter.jpg")
plt.savefig(save_path)
print(f"Save boxes scatter plot to: {save_path}")


##################################################
# Plot average IoU curve
##################################################
print('[INFO] Run anchor k-means with k = 2, 3, ..., 21')
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

save_path = os.path.join(save_dir, "k-iou_line.jpg")
plt.savefig(save_path)
print(f"Save k-iou plot to {save_path}")


##################################################
# Print anchors
##################################################
print('[INFO] The result anchors:')
for k in range(2, 21):
    print(f"Anchor: {k}")
    anchors = results[k]['anchors']
    anchors = np.around(anchors, decimals=3)
    print(anchors)
    print("-" * 20 + "\n")


##################################################
# Save defined anchors with the original boxes in scatter graph
##################################################
choose_anchor = results[choose_k]['anchors']
iou = results[choose_k]["avg_iou"]

print("[INFO] Draw boxes")
fig, ax = plt.subplots()
ax.scatter(x=boxes[:, 0], y=boxes[:, 1], c="orange")
ax.scatter(x=choose_anchor[:, 0], y=choose_anchor[:, 1], c="blue")
ax.set_xlim([0, cfg.TRAIN.SEARCH_SIZE])
ax.set_ylim([0, cfg.TRAIN.SEARCH_SIZE])
ax.set_xlabel("width")
ax.set_ylabel("height")
ax.set_aspect(1)
ax.hlines(y=64, xmin=0, xmax=64, linewidth=1, color='k')
ax.vlines(x=64, ymin=0, ymax=64, linewidth=1, color='k')
ax.text(200, 255, f"Total: {boxes_num}", fontsize=12)
ax.text(200, 235, f"below 64: {wh_below64_num}", fontsize=12)
ax.text(200, 220, f"below ratio: {below_ratio}%", fontsize=12)
ax.text(200, 200, f"above 64: {wh_above64_num}", fontsize=12)
ax.text(200, 185, f"above ratio: {above_ratio}%", fontsize=12)
ax.text(200, 165, f"Avg IOU: {round(iou, 3)}", fontsize=12)

# plt.figure()
# plt.xlabel("width")
# plt.ylabel("height")
# plt.scatter(boxes[:, 0], boxes[:, 1], c="orange")
# plt.scatter(anchors[:, 0], anchors[:, 1], c="blue")
# ax = plt.gca()
# ax.set_aspect(1)    # 讓 x, y 成正比

save_path = os.path.join(save_dir, f"k{choose_k}_scatter.jpg")
plt.savefig(save_path)
print(f"Save boxes-kmeans{choose_k} scatter plot to: {save_path}")


##################################################
# Visualize paper & kmeans anchors
##################################################
# print('[INFO] Visualizing paper anchors')
# visualize_anchors(anchor_official, 255, "official")

print('[INFO] Visualizing kmeans anchors')
visualize_anchors(choose_anchor, cfg.TRAIN.SEARCH_SIZE)

print("=" * 20 + " Done!! " + "=" * 20 + "\n")
