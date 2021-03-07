# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import sys

import numpy as np
import random
import imageio

from natsort import natsort
from tqdm import tqdm
import pickle

def get_img_paths(dir_path, wildcard='*.png'):
    return natsort.natsorted(glob.glob(dir_path + '/' + wildcard))

def create_all_dirs(path):
    if "." in path.split("/")[-1]:
        dirs = os.path.dirname(path)
    else:
        dirs = path
    os.makedirs(dirs, exist_ok=True)

def to_pklv4(obj, path, vebose=False):
    create_all_dirs(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    if vebose:
        print("Wrote {}".format(path))


from NTIRE21_Learning_SR_Space import imresize

def random_crop(img, size):
    h, w, c = img.shape

    h_start = np.random.randint(0, h - size)
    h_end = h_start + size

    w_start = np.random.randint(0, w - size)
    w_end = w_start + size

    return img[h_start:h_end, w_start:w_end]


def imread(img_path):
    img = imageio.imread(img_path)
    if len(img.shape) == 2:
        img = np.stack([img, ] * 3, axis=2)
    return img


def to_pklv4_1pct(obj, path, vebose):
    n = int(round(len(obj) * 0.01))
    print('\t>>n: ', n)
    print('\t>>path: ', path)

    path = path.replace(".pklv4", "_1pct.pklv4")
    print('\t>>@path: ', path)
    to_pklv4(obj[:n], path, vebose=True)

def dump_save(hrs, lqsX4, lqsX8, save_dir, save_tag): 
    shuffle_combined_3(hrs, lqsX4, lqsX8)
    print('hrs: ', np.shape(hrs))
    print('lqsX4: ', np.shape(lqsX4))
    print('lqsX8: ', np.shape(lqsX8))

    # HR
    hrs_path = get_path_withtag(save_dir, save_tag)
    print('hrs_path: ', hrs_path)
    to_pklv4(hrs, hrs_path, vebose=True)
    to_pklv4_1pct(hrs, hrs_path, vebose=True)

    # LR_X4
    lqsX4_path = get_path_withtag(save_dir, save_tag+'_X4')
    print('lqs_path: ', lqsX4_path)
    to_pklv4(lqsX4, lqsX4_path, vebose=True)
    to_pklv4_1pct(lqsX4, lqsX4_path, vebose=True)

    # LR_X8
    lqsX8_path = get_path_withtag(save_dir, save_tag+'_X8')
    print('lqs_path: ', lqsX8_path)
    to_pklv4(lqsX8, lqsX8_path, vebose=True)
    to_pklv4_1pct(lqsX8, lqsX8_path, vebose=True)

def main(dir_path, patch_size, save_dir, save_tag):
    hrs = []
    lqsX4 = []
    lqsX8 = []
    #patch_size = 512
    save_tag = '%s_patch%d'%(save_tag, patch_size)
  
    img_paths = get_img_paths(dir_path)
    print('img_paths: ', len(img_paths))
    for img_path in tqdm(img_paths):
        img = imread(img_path) 

        for i in range(16):
            crop = random_crop(img, 512)
            cropX4 = imresize(crop, scalar_scale=0.25)
            cropX8 = imresize(crop, scalar_scale=0.125)
            hrs.append(crop)
            lqsX4.append(cropX4)
            lqsX8.append(cropX8)

    # save HR, LR_X4, LR_X8
    dump_save(hrs, lqsX4, lqsX8, save_dir, save_tag)


def get_hrs_path(dir_path):
    base_dir = os.path.dirname(dir_path)
    name = os.path.basename(dir_path)
    hrs_path = os.path.join(base_dir, 'pkls', tag + '%s.pklv4')
    return hrs_path


def get_lqs_path(dir_path):
    base_dir = os.path.dirname(dir_path)
    name = os.path.basename(dir_path)
    hrs_path = os.path.join(base_dir, 'pkls', name + '_X4.pklv4')
    return hrs_path

def get_path_withtag(save_dir, save_tag): 
    path = os.path.join(save_dir, '%s.pklv4'%save_tag)
    return path


def shuffle_combined(hrs, lqs):
    combined = list(zip(hrs, lqs))
    random.shuffle(combined)
    hrs[:], lqs[:] = zip(*combined)

def shuffle_combined_3(hrs, lqsX4, lqsX8):
    combined = list(zip(hrs, lqsX4, lqsX8))
    random.shuffle(combined)
    hrs[:], lqsX4[:], lqsX8[:] = zip(*combined)


if __name__ == "__main__":
    dir_path = sys.argv[1]
    patch_size = int(sys.argv[2])
    save_dir = sys.argv[3]
    save_tag = sys.argv[4]
    assert os.path.isdir(dir_path)
    main(dir_path, patch_size, save_dir, save_tag)
