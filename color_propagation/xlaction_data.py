#!/usr/bin/env python

'''
    File name: xlaction_data.py
    Author: Guangyu Zhong
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Guangyu Zhong
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

MAX_INPUT_POINTS = 300000
NUM_PREV_FRAMES = 3
MAX_FRAMES = 25
RAND_SEED = 2345

SEQ_PATH = '/home/chrisz/../../purescratch/sliu/frame_color/'
SEQ_LIST_FILE = '../data/fold_list/all_seqs_action.txt' #specific video
IMAGE_FOLDER = SEQ_PATH + 'action_frames_release/'
#GT_FOLDER = '../data/DAVIS/Annotations/480p/'
IMAGESET_FOLDER = '../data/fold_list/'
FEATURE_FOLDER = '../data/color_feature_folder/'

MAIN_TRAIN_SEQ = '../data/fold_list/main_train_action.txt'
MAIN_VAL_SEQ = '../data/fold_list/main_val_action.txt'

RESULT_FOLDER = '../data/color_results/'
MODEL_FOLDER = '../data/color_models/'
