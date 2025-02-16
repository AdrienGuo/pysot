# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

# debug mode
__C.DEBUG = False

__C.META_ARC = "siamrpn_r50_l234_dwxcorr"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
# Positive anchor threshold
__C.TRAIN.THR_HIGH = 0.6
# Negative anchor threshold
__C.TRAIN.THR_LOW = 0.3

# Number of negative
__C.TRAIN.NEG_NUM = 16
# __C.TRAIN.NEG_NUM = 35
# Number of positive
__C.TRAIN.POS_NUM = 16
# Number of anchors per images
__C.TRAIN.TOTAL_NUM = 64
# The edge allow anchors to sit over the image
__C.TRAIN.ALLOWED_BORDER = 0

__C.TRAIN.EXEMPLAR_SIZE = 127
__C.TRAIN.SEARCH_SIZE = 255
__C.TRAIN.BASE_SIZE = 8

# 這邊都要手動調整，真的超蠢...
# 而且因為還會在 neck.py 裏面做裁切，要全部都自動計算又更麻煩了
# 他們原始的 code 真的是很難用誒
__C.TRAIN.OUTPUT_SIZE = 25    # template: 127, search: 255, crop
# __C.TRAIN.OUTPUT_SIZE = 68    # template: 127, search: 600, crop

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''
# __C.TRAIN.PRETRAINED = "./experiments/siamrpn_r50_l234_dwxcorr/model.pth"

__C.TRAIN.LOG_DIR = './logs'
# __C.TRAIN.MODEL_DIR = './save_models'
__C.TRAIN.SNAPSHOT_DIR = './save_models'
__C.TRAIN.NUM_WORKERS = 0
__C.TRAIN.SAVE_MODEL_FREQ = 50

__C.TRAIN.START_EPOCH = 0
__C.TRAIN.EPOCH = 100
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0
# __C.TRAIN.CLS_WEIGHT = 2.0
__C.TRAIN.LOC_WEIGHT = 1.2
__C.TRAIN.MASK_WEIGHT = 1

__C.TRAIN.PRINT_FREQ = 20
__C.TRAIN.EVAL_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# validation_split ratio
__C.DATASET.VALIDATION_SPLIT = 0.0

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
# __C.DATASET.TEMPLATE.SHIFT = 4
__C.DATASET.TEMPLATE.SHIFT = 0
# __C.DATASET.TEMPLATE.SCALE = 0.05
__C.DATASET.TEMPLATE.SCALE = 0
__C.DATASET.TEMPLATE.BLUR = 0.0
__C.DATASET.TEMPLATE.FLIP = 0.0
# __C.DATASET.TEMPLATE.COLOR = 1.0
__C.DATASET.TEMPLATE.COLOR = 0

__C.DATASET.SEARCH = CN()
__C.DATASET.SEARCH.SHIFT = 64
__C.DATASET.SEARCH.SCALE = 0.18
__C.DATASET.SEARCH.BLUR = 0.0
__C.DATASET.SEARCH.FLIP = 0.0
__C.DATASET.SEARCH.COLOR = 1.0

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048) for detail discussion
__C.DATASET.NEG = True

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

# __C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB', 'CUSTOM')
__C.DATASET.NAMES = ('COCO',)

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'training_dataset/vid/crop511'
__C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = 'training_dataset/coco/crop511'
__C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = 'training_dataset/det/crop511'
__C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = -1

__C.DATASET.CUSTOM = CN()
__C.DATASET.CUSTOM.ROOT = 'data/train/TemplateMatchingData/train'
__C.DATASET.CUSTOM.ANNO = 'data/train/TemplateMatchingData/train'
__C.DATASET.CUSTOM.FRAME_RANGE = 1
__C.DATASET.CUSTOM.NUM_USE = -1

__C.DATASET.VIDEOS_PER_EPOCH = 600000
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'
# __C.BACKBONE.TYPE = 'res18'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# --- Pretrained backbone weights ---
# (沒有 pretrained backbone 完全 train 不起來)
# __C.BACKBONE.PRETRAINED = ''
__C.BACKBONE.PRETRAINED = './pretrained_models/resnet50.model'
# __C.BACKBONE.PRETRAINED = './pretrained_models/resnet18.pth'

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 200

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.RPN = CN()

# RPN type
__C.RPN.TYPE = 'MultiRPN'

__C.RPN.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# mask options
# ------------------------------------------------------------------------ #
__C.MASK = CN()

# Whether to use mask generate segmentation
__C.MASK.MASK = False

# Mask type
__C.MASK.TYPE = "MaskCorr"

__C.MASK.KWARGS = CN(new_allowed=True)

__C.REFINE = CN()

# Mask refine
__C.REFINE.REFINE = False

# Refine type
__C.REFINE.TYPE = "Refine"

# ------------------------------------------------------------------------ #
# Anchor options
# ------------------------------------------------------------------------ #
__C.ANCHOR = CN()
# Anchor stride
# stride 應該要設定成最後的 feature map 是原圖縮小幾倍 (?)
__C.ANCHOR.STRIDE = 8
# Anchor ratios
__C.ANCHOR.RATIOS = [0.33, 0.5, 1, 2, 3]
# Anchor scales
__C.ANCHOR.SCALES = [8]
# Anchor number
# __C.ANCHOR.ANCHOR_NUM = len(__C.ANCHOR.RATIOS) * len(__C.ANCHOR.SCALES)
__C.ANCHOR.ANCHOR_NUM = 11


# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamRPNTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127
# Instance size
__C.TRACK.INSTANCE_SIZE = 255
# __C.TRACK.INSTANCE_SIZE = 600

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

# Long term lost search size
__C.TRACK.LOST_INSTANCE_SIZE = 831

# Long term confidence low
__C.TRACK.CONFIDENCE_LOW = 0.85

# Long term confidence high
__C.TRACK.CONFIDENCE_HIGH = 0.998

# Mask threshold
__C.TRACK.MASK_THERSHOLD = 0.30

# Mask output size
__C.TRACK.MASK_OUTPUT_SIZE = 127
