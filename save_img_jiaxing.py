import numpy as np
import torch

from function import Downsampler
from function import Crop_traindata

def generate_data(path, path_srf):
    ratio_hs = 16

    # spectral response function of worldview2
    srf_pan = np.expand_dims(np.expand_dims(np.load(path_srf), axis=-1), axis=-1)

    noise_mean = 0.0
    noise_var = 0.0001

    dowmsample16 = Downsampler(ratio_hs)

    original_msi = np.float32(np.load(path))    # load HSI data, range:[0, 1]
    # original_msi = np.random.rand(126, 1000, 1000)
    band, row, col = original_msi.shape

    original_msi = original_msi[:, :(row - row%(ratio_hs*4)), :(col - col%(ratio_hs*4))]
    print('original_msi.shape:', original_msi.shape)
    print('max value:', np.max(original_msi))

    # ratio 16
    temp_blur = dowmsample16(np.expand_dims(original_msi, 0))
    print(temp_blur.shape)
    temp_blur = np.squeeze(temp_blur.cpu().numpy())
    _, rows, cols = temp_blur.shape
    blur_data = []
    for i3 in range(temp_blur.shape[0]):
        blur_data.append(np.expand_dims(temp_blur[i3, :, :] +
                                        np.random.normal(noise_mean, noise_var ** 0.5, [rows, cols]), axis=0))
    blur_data = np.concatenate(blur_data, axis=0)
    print('blur_data.shape:' + str(blur_data.shape))

    # simulated pan image
    temp_pan = np.expand_dims(np.sum(original_msi * srf_pan, axis=0) / np.sum(srf_pan), axis=0)
    print('temp_pan.shape:' + str(temp_pan.shape))

    return original_msi, temp_blur, temp_pan

def crop_data(hrhs, lrhs, pan, ratio=16, train_ratio=0.8):
    training_size = 64  # training patch size
    testing_size = 256  # testing patch size

    idx = int(lrhs.shape[2] * train_ratio)

    '''产生训练和测试数据'''
    train_hs_image, train_hrpan_image, train_label = \
        Crop_traindata(lrhs[:, :, :idx],
                        pan[:, :, :idx*ratio],
                        hrhs[:, :, :idx*ratio],
                        ratio=ratio, size=training_size,
                        test=False)
    test_hs_image, test_hrpan_image, test_label = \
        Crop_traindata(lrhs[:, :, idx:],
                        pan[:, :, idx*ratio:],
                        hrhs[:, :, idx*ratio:],
                        ratio=ratio, size=testing_size,
                        test=True)

    print('train size:' + str(train_hrpan_image.shape))
    print('test size:' + str(test_hrpan_image.shape))

    return train_hs_image, train_hrpan_image, train_label, test_hs_image, test_hrpan_image, test_label

def generate_JiaXing_data(path, path_srf, train_ratio=0.8):
    hrhs, lrhs, pan = generate_data(path, path_srf)
    return crop_data(hrhs, lrhs, pan, ratio=16, train_ratio=train_ratio)

if __name__ == '__main__':

    pass
    # hrhs, lrhs, pan = generate_data()
    # crop_data(hrhs, lrhs, pan, ratio=16)
