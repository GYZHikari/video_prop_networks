#!/usr/bin/env python

'''
    File name: fetch_and_transform_data_adaptive.py
    Author: Guangyu Zhong (guangyuzhonghikari@gmail.com)
    Date: 11/03/2017
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks (adaptive)
#----------------------------------------------------------------------------
# Copyright 2017 Guangyu Zhong
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import sys
import numpy as np
import scipy.io as sio
from PIL import Image
import cv2
from davis_data import *
from init_caffe import *

# For YXYT features
def transform_and_get_features(features, height, width):
    input_color = np.zeros((height, width, 2))
    input_color[:, :, :] = features[:, 0 : width, 1:3]

    num_out_frames = features.shape[1] / width
    out_features = np.zeros((height, width * num_out_frames, 4))
    out_features[:, :, 0] = features[:, :, 0]
    out_features[:, :, 1:] = features[:, :, 3:6]

    input_color = np.transpose(input_color, (2,0,1))[None,:,:,:] / 255.0 - 0.5
    out_features = np.transpose(out_features, (2,0,1))[None,:,:,:]

    return [input_color, out_features, num_out_frames]

def fetch_and_transform_data(features, height, width):

    inputs = {}
    [inputs['input_color'], inputs['out_features'], num_out_frames] = transform_and_get_features(features, height, width)

    return [inputs, num_out_frames]
