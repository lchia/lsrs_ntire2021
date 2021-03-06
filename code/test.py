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
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE


import glob
import sys
from collections import OrderedDict

from natsort import natsort

import options.options as option
from Measure import Measure, psnr
from imresize import imresize
from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get('rlt', visuals.get("SR"))


def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


def main():
    conf_path = sys.argv[1]
    conf = conf_path.split('/')[-1].replace('.yml', '')
    model, opt = load_model(conf_path)

    lr_dir = opt['dataroot_LR']
    hr_dir = opt['dataroot_GT']

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    hr_paths = fiFindByWildcard(os.path.join(hr_dir, '*.png'))

    this_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(this_dir, '..', 'results', conf)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    print(f"Out dir: {test_dir}")

    measure = Measure(use_gpu=False)

    fname = f'measure_full.csv'
    fname_tmp = fname + "_"
    path_out_measures = os.path.join(test_dir, fname_tmp)
    path_out_measures_final = os.path.join(test_dir, fname)

    if os.path.isfile(path_out_measures_final):
        df = pd.read_csv(path_out_measures_final)
    elif os.path.isfile(path_out_measures):
        df = pd.read_csv(path_out_measures)
    else:
        df = None

    scale = opt['scale']

    pad_factor = 2

    for lr_path, hr_path, idx_test in zip(lr_paths, hr_paths, range(len(lr_paths))):

        print('>>process %s'%(lr_path))

        lr = imread(lr_path)
        hr = imread(hr_path)
        print('\tlr: ', lr.shape)
        print('\thr: ', hr.shape)
 
        _, img_name = os.path.split(lr_path)

        # Pad image to be % 2
        h, w, c = lr.shape
        if h >= 338:
            print('\tskip ...')
            continue

        lq_orig = lr.copy()
        lr = impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                   right=int(np.ceil(w / pad_factor) * pad_factor - w))
        print('\tlr: ', lr.shape) 

        lr_t = t(lr)
        print('\tlr_t: ', lr_t.shape, type(lr_t)) 

        # heat = opt['heat']
        for heat in np.linspace(0, 1, num=11):
            # Sample a super-resolution for a low-resolution image
            print('\theat: ', heat)

            if df is not None and len(df[(df['heat'] == heat) & (df['name'] == idx_test)]) == 1:
                continue

            sr_t = model.get_sr(lq=lr_t, heat=heat)
            print('\tsr_t: ', sr_t.shape, type(sr_t)) 

            sr = rgb(torch.clamp(sr_t, 0, 1))
            sr = sr[:h * scale, :w * scale]
            print('\tsr: ', sr.shape, type(sr)) 

            save_name = '%s_%s.png'%(img_name.replace('.png', ''), str(heat))
            # path_out_sr = os.path.join(test_dir, "{:0.2f}".format(heat).replace('.', ''), save_name)
            path_out_sr = os.path.join(test_dir, save_name)
            print('\tsave to %s'%(path_out_sr))
            imwrite(path_out_sr, sr)
     
            # add compare
            h,w,c = np.shape(hr)
            lr_resize = cv2.resize(lr, (w,h))
            print('\tlr_resize: ', lr_resize.shape) 

            # TypeError: Expected Ptr<cv::UMat> for argument 'img'
            sr = np.array(sr).astype('uint8')
            sr = cv2.cvtColor(np.array(sr), cv2.COLOR_RGB2BGR)
            print('\tsr: ', sr.shape, type(sr), np.min(sr[:]), np.max(sr[:]))

            hr_copy = np.array(hr).astype('uint8')
            hr_copy = cv2.cvtColor(np.array(hr_copy), cv2.COLOR_RGB2BGR)
            print('\thr: ', sr.shape, type(hr_copy), np.min(hr_copy[:]), np.max(hr_copy[:]))

            start_x = 100
            start_y = 100
            font_scale =5
            thickness = 6
            #lr = lr_resize[:,:,[2,1,0]]
            lr_copy = np.array(lr_resize).astype('uint8')
            lr_copy = cv2.cvtColor(np.array(lr_copy), cv2.COLOR_RGB2BGR)
            color = (0,0,255)
            cv2.putText(lr_copy, 'lr', (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.putText(sr,      'sr', (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.putText(hr_copy, 'hr', (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            save_name = 'compare_%s_%s.png'%(img_name.replace('.png', ''), str(heat))
            # path_out_sr_compare = os.path.join(test_dir, "{:0.2f}".format(heat).replace('.', ''), save_name)
            path_out_sr_compare = os.path.join(test_dir, save_name)
            compare = np.hstack([lr_copy, sr])
            compare = np.hstack([compare, hr_copy])
            cv2.imwrite(path_out_sr_compare, compare)
            print('\tsave to %s'%(path_out_sr_compare))

            meas = OrderedDict(conf=conf, heat=heat, name=idx_test)
            meas['PSNR'], meas['SSIM'], meas['LPIPS'] = measure.measure(sr, hr)

            lr_reconstruct_rgb = imresize(sr, 1 / opt['scale'])
            meas['LRC PSNR'] = psnr(lq_orig, lr_reconstruct_rgb)

            str_out = format_measurements(meas)
            print(str_out)

            df = pd.DataFrame([meas]) if df is None else pd.concat([pd.DataFrame([meas]), df])

            df.to_csv(path_out_measures + "_", index=False)
            os.rename(path_out_measures + "_", path_out_measures)

    df.to_csv(path_out_measures, index=False)
    os.rename(path_out_measures, path_out_measures_final)

    str_out = format_measurements(df.mean())
    print(f"Results in: {path_out_measures_final}")
    print(f'Mean: ' + str_out)


def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.2f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


if __name__ == "__main__":
    main()
