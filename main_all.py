# -*- coding: utf-8 -*-
"""
@author: lsl
E-mail: cug_lsl@cug.edu.cn
"""
# import sys
# sys.path.append('/home/aistudio/work/code')
import numpy as np
import pandas as pd
import os
import time

# from function import write_img_gdal
from metrics import ref_evaluate
from save_img_jiaxing import generate_data, crop_data
from RHDN_main.Model_train import main as RHDN
from Pgnet.Pgnet import Pgnet


if __name__ == '__main__':

    path = 'G:/jiaxing/JiaXing.npy'  # HSI data path
    path_srf = 'G:/SRF/srf_pan.npy'
    hs_image, hrpan_image, label = generate_data(path, path_srf)
    ms_crop, pan_crop, label_crop = crop_data(hs_image, hrpan_image, label, ratio=16, training_size=4)

    train_ratio = 0.8
    train_num = int(pan_crop.shape[0] * train_ratio)
    test_num = pan_crop.shape[0] - train_num

    index = np.arange(ms_crop.shape[0])

    np.random.seed(1000)
    np.random.shuffle(index)
    # print(index[:50])
    train_ms_image = ms_crop[index[:train_num], :, :, :]
    train_pan_image = pan_crop[index[:train_num], :, :, :]
    train_label = label_crop[index[:train_num], :, :, :]

    # index2 = 45
    test_ms_image = ms_crop[index[-test_num:], :, :, :]
    test_pan_image = pan_crop[index[-test_num:], :, :, :]
    test_label = label_crop[index[-test_num:], :, :, :]

    ratio = int(test_pan_image.shape[2] / test_ms_image.shape[2])
    print(ratio)

    '''setting save parameters'''
    save_num = 3  # the number of testing images to save
    save_images = False
    save_dir = []
    for i7 in range(save_num):
        save_dir.append('/home/aistudio/result/results' + str(i7) + '/')
        if save_images and (not os.path.isdir(save_dir[i7])):
            os.makedirs(save_dir[i7])

    '''定义度量指标和度量函数'''
    ref_results = {}
    ref_results.update({'metrics: ': '  PSNR,    SSIM,   SAM,    ERGAS,   SCC,    Q,     RMSE'})  # 记得更新下面数组长度
    len_ref_metrics = 7

    result = []
    result_diff = []
    metrics_result = []  # 存储测试影像指标

    train_Pgnet = False

    if train_Pgnet:
        # 35.3327, 0.908, 0.0766, 0.7511, 0.8592, 0.6385, 0.0173
        fused_image = Pgnet(
            train_ms_image, train_pan_image, train_label,
            test_ms_image, test_pan_image, test_label, ratio=ratio)

        del train_ms_image
        del train_pan_image
        del train_label
        del test_ms_image
        del test_pan_image

        fused_image = fused_image.cpu().numpy()
        ref_results_all = []
        for i5 in range(test_label.shape[0]):
            temp_ref_results = ref_evaluate(fused_image[i5, :, :, :].transpose([1, 2, 0]),
                                            test_label[i5, :, :, :].transpose([1, 2, 0]), scale=ratio)

            ref_results_all.append(np.expand_dims(temp_ref_results, axis=0))

        ref_results_all = np.concatenate(ref_results_all, axis=0)
        ref_results.update({'Pf       ': np.mean(ref_results_all, axis=0)})

        if save_images:
            for j5 in range(save_num):
                np.save(save_dir[j5] + 'Pf_result' + str(j5) + '.npy',
                                fused_image[j5, :, :, :])
                np.save(save_dir[j5] + 'Pf_true' + str(j5) + '.npy',
                                test_label[j5, :, :, :])

        metrics_result.append(np.mean(ref_results_all, axis=0, keepdims=True))


    print('################## reference comparision #######################')
    for index1, i in enumerate(ref_results):
        if index1 == 0:
            print(i, ref_results[i])
        else:
            print(i, [round(j, 4) for j in ref_results[i]])
    print('################## reference comparision #######################')
