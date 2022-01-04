# -*- coding: utf-8 -*-
"""
@author: lsl
E-mail: cug_lsl@cug.edu.cn
"""
import numpy as np
import pandas as pd
import os
import time

from function import write_img_gdal
from metrics import ref_evaluate
from Pgnet import Pgnet
from save_img_jiaxing import generate_JiaXing_data

path = r'G:\\1research_data\\JiaXing.npy'       # HSI data path
path_srf = r'G:\\1research_data\\srf_pan.npy'   # SRF path
train_hs_image, train_hrpan_image, train_label,\
    test_hs_image, test_hrpan_image, test_label = \
    generate_JiaXing_data(path, path_srf, train_ratio=0.8)

ratio = int(test_hrpan_image.shape[2] / test_hs_image.shape[2])
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

if __name__ == '__main__':

    fused_image = Pgnet(
        train_hs_image, train_hrpan_image, train_label,
        test_hs_image, test_hrpan_image, test_label, ratio=ratio)

    fused_image = fused_image.cpu().numpy()
    ref_results_all = []
    for i5 in range(test_hrpan_image.shape[0]):
        temp_ref_results = ref_evaluate(fused_image[i5, :, :, :].transpose([1, 2, 0]),
                                        test_label[i5, :, :, :].transpose([1, 2, 0]), scale=ratio)

        ref_results_all.append(np.expand_dims(temp_ref_results, axis=0))

    ref_results_all = np.concatenate(ref_results_all, axis=0)
    ref_results.update({'Pf       ': np.mean(ref_results_all, axis=0)})

    if save_images:
        for j5 in range(save_num):
            write_img_gdal(save_dir[j5] + 'Pf_result' + str(j5) + '.tif',
                            fused_image[j5, :, :, :])
            write_img_gdal(save_dir[j5] + 'Pf_true' + str(j5) + '.tif',
                            test_label[j5, :, :, :])

    metrics_result.append(np.mean(ref_results_all, axis=0, keepdims=True))

    print('################## reference comparision #######################')
    for index1, i in enumerate(ref_results):
        if index1 == 0:
            print(i, ref_results[i])
        else:
            print(i, [round(j, 4) for j in ref_results[i]])
    print('################## reference comparision #######################')
