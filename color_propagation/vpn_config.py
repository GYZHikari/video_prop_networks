#!/usr/bin/env python
'''
    File name: vpn_config.py
    Author: Guangyu Zhong (guangyuzhonghikari@gmail.com)
    Date: 11/03/2017
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks (adaptive)
#----------------------------------------------------------------------------
# Copyright 2017 Guangyu Zhong
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------


import numpy as np
from easydict import EasyDict as edict

config = edict()
config.MAX_INPUT_POINTS = 500000 # or 300,000?
config.NUM_PREV_FRAMES = 1
config.RAND_SEED = 2345
config.MAX_K_SCALE = 30
config.FEATURE_SCALES = [0.2, 0.04, 0.04, 0.04]
config.DATASET = 'davis'
config.K_SCALE = [gap for gap in range(5, config.MAX_K_SCALE, 5)]
config.STAGE_ID = 0

if config.DATASET == 'action':
    config.SEQ_LIST_FILE = '/tmp5/sliu32_data/frame_color/action_frames_labels/task1/testlist.txt'
    config.IMAGE_FOLDER = '/tmp5/sliu32_data/frame_color/action_frames_release/'
    config.GT_FOLDER = '/tmp5/sliu32_data/frame_color/action_frames_release/'
    config.FEATURE_FOLDER = '../data/XLACTION/color_feature_folder/'
    config.RESULT_FOLDER = '../data/XLACTION/color_results/'
    config.MODEL_FOLDER = '../data/color_models/'
else:
    config.SEQ_LIST_FILE = '../data/fold_list/main_val.txt'
    config.IMAGE_FOLDER = '../data/DAVIS/JPEGImages/480p/'
    config.GT_FOLDER = '../data/DAVIS/Annotations/480p/'
    config.FEATURE_FOLDER = '../data/DAVIS_RES/color_feature_folder/'
    config.RESULT_FOLDER = '../data/DAVIS_RES/color_results/'
    config.MODEL_FOLDER = '../data/color_models/'
