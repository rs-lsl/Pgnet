import numpy as np
import pandas as pd
import os
import random
import time
import cv2
from scipy import signal
import torch 
import torch.nn as nn
import torch.nn.functional as F
import gdal

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def write_img_gdal(filename, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    # dataset.SetGeoTransform(im_geotrans)    #写入仿射变换参数
    # dataset.SetProjection(im_proj)          #写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

def all_valid(data, axis=0):
    # axis == channel dimension
    if axis == 0:
        length = data.shape[1] * data.shape[2]
    elif axis == 2:
        length = data.shape[0] * data.shape[1]
    sum_data = np.sum(data, axis=axis)
    if len(np.where(sum_data > 0)[0]) == length:
        return True
    else:
        return False

def Crop_traindata(image_hs, image_hrpan, label, size, ratio=16, test=False):
    image_hs_all = []
    image_hrpan_all = []
    # image_lrms_all = []
    label_all = []

    """crop images"""
    temp_name = 'test' if test else 'train'
    print('croping ' + temp_name + ' images...')
    print(image_hs.shape)
    print(image_hrpan.shape)
    print(label.shape)

    for j in range(0, image_hs.shape[1] - int(size / (ratio)), int(size / (ratio))):
        for k in range(0, image_hs.shape[2] - int(size / (ratio)), int(size / ratio)):
            temp_image_hs = image_hs[:, j:j + int(size / ratio), k:k + int(size / ratio)]
            temp_image_hrpan = image_hrpan[:, j * ratio:j * ratio + size, k * ratio:k * ratio + size]
            temp_label = label[:, j * ratio:j * ratio + size, k * ratio:k * ratio + size]

            if all_valid(temp_image_hrpan, axis=0):
                image_hs_all.append(temp_image_hs)
                image_hrpan_all.append(temp_image_hrpan)
                label_all.append(temp_label)

    image_hs_all = np.array(image_hs_all, dtype='float32')
    image_hrpan_all = np.array(image_hrpan_all, dtype='float32')
    label_all = np.array(label_all, dtype='float32')

    if test == False:
        print("size of train_hrpanimage:{} train_lrhsimage:{} train_label:{} ".
              format(image_hrpan_all.shape, image_hs_all.shape, label_all.shape))
    else:
        print("size of test_hrpanimage:{} test_lrhsimage:{} test_label:{} ".
              format(image_hrpan_all.shape, image_hs_all.shape, label_all.shape))

    return image_hs_all, image_hrpan_all, label_all

def generate_cropdata(hs_data, hrpan, true_label, ratio=4, band=4, training_size=128, test=False, step_facter=1):
    train_image_hs_data = []
    train_image_hrpan = []
    train_label = []
    for i4 in range(len(hrpan)):
        temp_image_hs_data, temp_image_hrpan, temp_label = \
            Crop_traindata(hs_data[i4], hrpan[i4], true_label[i4],
                           training_size, ratio=ratio, step=training_size, test=test, step_facter=step_facter)
        train_image_hs_data.append(temp_image_hs_data)
        train_image_hrpan.append(temp_image_hrpan)
        train_label.append(temp_label)

    train_image_hs_data = np.concatenate(train_image_hs_data, axis=0)
    train_image_hrpan = np.concatenate(train_image_hrpan, axis=0)
    train_label = np.concatenate(train_label, axis=0)

    assert train_image_hrpan.shape[0] == train_label.shape[0]
    return train_image_hs_data, train_image_hrpan, train_label

def log(base, x):
    return np.log(x) / np.log(base)

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()

    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

class Downsampler(nn.Module):
    '''
        http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    '''
    def __init__(self, factor, kernel_size=9, padding=0, n_planes=1, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
        super(Downsampler, self).__init__()
        self.ratio = factor
        self.padding = padding
        kernel_size = factor
        
        sig = (1 / (2 * 2.7725887 / 16 ** 2)) ** 0.5
        kernel = np.multiply(cv2.getGaussianKernel(kernel_size, sig),
                            cv2.getGaussianKernel(kernel_size, sig).T)
        self.kernel = torch.tensor(np.expand_dims(np.expand_dims(kernel, axis=0), axis=0), device=device)
        
    def forward(self, input):
        input = torch.tensor(input).to(device)

        row = input.shape[2]
        col = input.shape[3]

        result = torch.zeros([input.shape[0], input.shape[1], 
                            int(row/self.ratio), int(col/self.ratio)]).to(device)
        for j in range(input.shape[0]):
            for i in range(input.shape[1]):
                result[j, i, :, :] = F.conv2d(torch.reshape(input[j, i, :, :], (1, 1, row, col)),
                                    self.kernel, stride=self.ratio, padding=self.padding)
        return result
        
def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']
    
    # factor  = float(factor)
    if phase == 0.5 and kernel_type != 'box': 
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])
    
        
    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1./(kernel_width * kernel_width)
        
    elif kernel_type == 'gauss': 
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'
        
        center = (kernel_width + 1.)/2.
        print(center, kernel_width)
        sigma_sq =  sigma * sigma
        
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center)/2.
                dj = (j - center)/2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj)/(2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1]/(2. * np.pi * sigma_sq)
    elif kernel_type == 'lanczos': 
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor  
                    dj = abs(j + 0.5 - center) / factor 
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor
                
                
                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)
                
                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)
                
                kernel[i - 1][j - 1] = val
            
        
    else:
        assert False, 'wrong method name'
    
    kernel /= kernel.sum()
    
    return kernel