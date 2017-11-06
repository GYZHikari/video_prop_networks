#!/usr/bin/env python

'''
    File name: demo_vpn_test_gaps.py
    Author: Guangyu Zhong
    Date: 11/03/2017
'''
# ---------------------------------------------------------------------------
# Video Propagation Networks (adaptive)
#----------------------------------------------------------------------------
# Copyright 2017 Guangyu Zhong
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import sys
import copy
import random
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from PIL import Image
from skimage import color
from init_caffe import *
from create_online_net_adaptive import *
from vpn_config import *
from utils import *
from fetch_and_transform_data_adaptive import fetch_and_transform_data
gc.enable()

def extract_frame_features(seq_path, seq_frames, seq_feature_path):
# extract feature for each video clip and save into folders
    frame_no = 0
    for img_name in seq_frames:
        if not img_name.lower().endswith(('jpg')):
            print(img_name + 'NOT JPG!!')
            continue
        img_file = seq_path + img_name
        # Extract features and save
        if not os.path.exists(seq_feature_path):
            os.makedirs(seq_feature_path)
        feat_path = seq_feature_path + img_name[0:-4] + '.npy'
        if not os.path.exists(feat_path):
            img = Image.open(img_file)
            ycbcr = img.convert('YCbCr')
            I = np.ndarray((img.size[1], img.size[0], 3),
                           'u1', ycbcr.tobytes())
            features = extract_features(I, frame_no)
            np.save(feat_path, features)
        frame_no += 1
    return frame_no


def extract_features(I,t):
    xvalues, yvalues = np.meshgrid(np.arange(I.shape[1]), np.arange(I.shape[0]))
    tvalues = np.ones((I.shape[0], I.shape[1])) * t

    feat = np.append(I, xvalues[:, :, None], axis=2)
    feat = np.append(feat, yvalues[:, :, None], axis=2)
    feat = np.append(feat, tvalues[:, :, None], axis=2)
    return feat

def generate_subsequence_index(frame_no, k):
    sub_seq_num = frame_no / k
    if sub_seq_num == 0:
        sub_seq_start_end = np.zeros((1, 2))
        sub_seq_start_end[0] = [0, frame_no]
    else:
        sub_seq_start = [id*k for id in range(0, sub_seq_num)]
        sub_seq_end = [(id+1)*k for id in range(0, sub_seq_num)]
        if frame_no % k == 0:
            sub_seq_start_end = np.zeros((sub_seq_num, 2))
            sub_seq_start_end[:, 0] = sub_seq_start
            sub_seq_start_end[:, 1] = sub_seq_end
        else:
            sub_seq_start_end = np.zeros((sub_seq_num + 1, 2))
            sub_seq_start_end[:-1, 0] = sub_seq_start
            sub_seq_start_end[:-1, 1] = sub_seq_end
            sub_seq_start_end[-1, 0] = sub_seq_end[-1]
            sub_seq_start_end[-1, 1] = frame_no
    return sub_seq_num, sub_seq_start_end

def fetch_subsequence_feature(start_id, end_id, seq_feature_path, feats_names):
    all_frame_features = None
    height = None
    width = None
    for tmp in range(start_id, end_id):
        # load
        load_feat_name = seq_feature_path + feats_names[tmp]
        features = np.load(load_feat_name)
        if all_frame_features is None:
            all_frame_features = features
        else:
            all_frame_features = np.append(all_frame_features, features, axis=1)
    height = features.shape[0]
    width = features.shape[1]
    return all_frame_features, height, width

def propagation_subsequence(all_frame_features, height, width, out_folder, prev_color_file, result_folder, feats_names, max_input_points=100000):
    [inputs, num_out_frames] = fetch_and_transform_data(all_frame_features, height, width)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if stage_id > 0:
        prev_color_result = np.load(prev_color_file)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # ---------------------------------------------------------------------
    # save start frame
    # -------------------------------------------c--------------------------
    color_result = (np.transpose(np.squeeze(inputs['input_color']), (1, 2, 0)) + 0.5) * 255.0
    gray_result = np.squeeze(inputs['out_features'][:, 0, :, 0:width])[:,:,None]
    full_result = np.append(gray_result, color_result, axis = 2)
    rgb_result = convert_to_rgb(full_result)
    misc.imsave(result_folder + '/' + feats_names[start_id][0:-4] + '.png',
                rgb_result)
    all_frames_color_result = inputs['input_color']
    prev_frame_result = None
    net_inputs = {}
    net_inputs['input_color'] = inputs['input_color']
    net_inputs['scales'] = np.ones((1, 4, 1, 1))

    for tmp_scale in range(0, 4):
        net_inputs['scales'][0, tmp_scale, 0, 0] = config.FEATURE_SCALES[tmp_scale]
    f_value = end_id - start_id - 1
    ignore_feat_value = -1000
    if stage_id == 0:
        standard_net = load_bnn_deploy_net(max_input_points, height, width)
    else:
        caffe_model = config.MODEL_FOLDER + 'COLOR_STAGE1.caffemodel'
        standard_net = load_bnn_cnn_deploy_net(max_input_points, height, width)
        standard_net.copy_from(caffe_model)

    for t in range(1, end_id - start_id):

        if t < f_value:
            net_inputs['input_color'] = copy.copy(inputs['input_color'])
            net_inputs['in_features'] = copy.copy(inputs['out_features'][:, :, :, 0: width*t])
            net_inputs['out_features'] = copy.copy(inputs['out_features'][:, :, :, width * t : width * (t+1)])
        else:
            net_inputs['input_color'] = copy.copy(inputs['input_color'][:, :, :, width*(t-f_value): width*t])
            net_inputs['in_features'] = copy.copy(inputs['out_features'][:, :, :, width*(t-f_value): width*t])
            net_inputs['out_features'] = copy.copy(inputs['out_features'][:, :, :, width * t : width * (t+1)])

        height1 = net_inputs['in_features'].shape[2]
        width1 = net_inputs['in_features'].shape[3]
        num_input_points = height1 * width1
        # Random sampling input points
        if num_input_points > max_input_points:
            sampled_indices = random.sample(xrange(num_input_points), max_input_points)
        else:
            sampled_indices = random.sample(xrange(num_input_points), num_input_points)

        h_indices = (np.array(sampled_indices) / width1).tolist()
        w_indices = (np.array(sampled_indices) % width1).tolist()
        net_inputs['input_color'] = net_inputs['input_color'][:, :, h_indices, w_indices]
        net_inputs['input_color'] = net_inputs['input_color'][:, :, np.newaxis, :]
        net_inputs['in_features'] = net_inputs['in_features'][:, :, h_indices, w_indices]
        net_inputs['in_features'] = net_inputs['in_features'][:, :, np.newaxis, :]
        if num_input_points > max_input_points:
            prev_frame_result = standard_net.forward_all(**net_inputs)['out_color_result']
        if num_input_points < max_input_points:
            if stage_id == 0:
                net = load_bnn_deploy_net(num_input_points, height, width)
            else:
                caffe_model = config.MODEL_FOLDER + 'COLOR_STAGE1.caffemodel'
                net = load_bnn_cnn_deploy_net(num_input_points, height, width)
                net.copy_from(caffe_model)
            prev_frame_result = net.forward_all(**net_inputs)['out_color_result']

        # import pdb; pdb.set_trace()
        result = np.squeeze(prev_frame_result)
        color_result = (np.transpose(result, (1, 2, 0)) + 0.5) * 255.0
        gray_result = np.squeeze(inputs['out_features'][:, 0, :, width * t : width * (t+1)])[:,:,None]
        full_result = np.append(gray_result, color_result, axis = 2)
        rgb_result = convert_to_rgb(full_result)
        misc.imsave(result_folder + '/' + feats_names[t + start_id][0:-4] + '.png',
                    rgb_result)
        #print('saving' + result_folder + '/' + feats_names[t + start_id][0:-4] + '.png')
        all_frames_color_result = np.append(all_frames_color_result, prev_frame_result,
                                            axis=3)

        if stage_id > 0:
            prev_frame_result = prev_color_result[:, :, :, width * t : width * (t+1)]

        inputs['input_color'] = np.append(inputs['input_color'],
                                          prev_frame_result,
                                          axis=3)
        gc.collect()
    return all_frames_color_result
# =============================================
# prepare for testing data for each sequence
# =============================================

seq_fid = open(config.SEQ_LIST_FILE, 'r')
k_scale = config.K_SCALE
stage_id = config.STAGE_ID

print('=============================================')
print('calculating based on these scales: ')
print(k_scale)
print('stage: ')
print(str(stage_id))
print('=============================================')

stage_id = 0

for seq_line in seq_fid.readlines():
    line = seq_line.strip('\n')
    cur_seq = line.split(' ')[0] + '/'
    # ======================================
    # generate features for each sequence
    # ======================================
    seq_path = config.IMAGE_FOLDER + cur_seq
    seq_frames = os.listdir(seq_path)
    seq_feature_path = config.FEATURE_FOLDER + cur_seq
    frame_no = extract_frame_features(seq_path, seq_frames, seq_feature_path)
    # =============================================
    # generate k sub-sequences for each sequence
    # =============================================
    feats_names = [s for s in os.listdir(config.FEATURE_FOLDER + cur_seq) if s.endswith('.npy')]
    feats_names.sort() # Note: need to be sorted otherwise the fetched results will be incorrect.

    for k in k_scale:
    	print('calculating gap scale: ' + str(k))
        sub_seq_num, sub_seq_start_end = generate_subsequence_index(frame_no, k)
        # ================================================
        # fetch and calculate results for each subsequence
        # ================================================
        for sub_seq_id in range(sub_seq_start_end.shape[0]):

            start_id = int(sub_seq_start_end[sub_seq_id,0]) #start from
            end_id = int(sub_seq_start_end[sub_seq_id,1]) #end here
            # ---------------------------------------------------------------------
            # start propagation for the given batch features
            # ---------------------------------------------------------------------
            all_frame_features, height, width = fetch_subsequence_feature(start_id, end_id, seq_feature_path, feats_names)
            out_folder = config.RESULT_FOLDER + '/' + str(k) + '/STAGE' + str(stage_id) + '_RESULT/'
            prev_color_file = config.RESULT_FOLDER + '/' + str(k) + '/STAGE' + str(stage_id-1) + '_RESULT/' + cur_seq + '/all_frame_color_result' + '-' + str(start_id) + '-' + str(end_id) + '.npy'
            result_folder = out_folder + '/' + cur_seq + '/'
            all_frames_color_result = propagation_subsequence(all_frame_features, height, width, out_folder, prev_color_file, result_folder, feats_names, config.MAX_INPUT_POINTS)
            # Save the all frames color result
            out_file = result_folder + '/all_frame_color_result' + '-' + str(start_id) + '-' + str(end_id) + '.npy'
            np.save(out_file, all_frames_color_result)
