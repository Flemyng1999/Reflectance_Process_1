# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:37:27 2022

@author: Flemyng
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy import ndarray
from tqdm import tqdm

import tiff_tool as tt


def find_nearest_index(target: float, arr: np.ndarray) -> ndarray[int]:
    """
    寻找目标值在数组中的最近值，并返回其索引

    Args:
        target: 目标值
        arr: 数组

    Returns:
        最近值的索引
    """
    nearest_values = np.abs(target - arr)
    index = np.argmin(nearest_values)
    return index


def ndvi_segment(ndvi_array: np.ndarray, bins_num: int = 500, show_plot: bool = False, n: float = 1.5) -> np.ndarray:
    """
    根据NDVI数组，计算出阈值，并将数组转换为二值图像

    Args:
        ndvi_array: NDVI数组
        bins_num: 直方图的箱数，默认为500
        show_plot: 是否显示绘制的图像，默认为False
        n: 阈值计算中的参数n，默认为1.5.数字越大阈值越低

    Returns:
        二值图像数组
    """
    ndvi_number, ndvi_value = np.histogram(ndvi_array[ndvi_array > 0], bins=bins_num)
    cycle, ndvi_number = sm.tsa.filters.hpfilter(ndvi_number, 500)
    ndvi_value = (ndvi_value[1:] + ndvi_value[:-1]) / 2

    threshold_index = find_nearest_index(0.4, ndvi_value)
    ndvi_number_max = np.max(ndvi_number[threshold_index:])
    ndvi_number_max_index = find_nearest_index(ndvi_number_max, ndvi_number)
    half_distance = int(bins_num - ndvi_number_max_index)
    down_limit = ndvi_number_max_index - half_distance

    sample = ndvi_array[ndvi_value[down_limit] < ndvi_array]
    mean = np.mean(sample)
    std = np.std(sample)
    target = mean - n * std
    target_index = find_nearest_index(target, ndvi_value)

    ndvi_array = np.where(ndvi_array >= target, 1, 0)

    if show_plot:
        fig, axes = plt.subplots(2, constrained_layout=1, figsize=(7, 9), dpi=200)
        axes[0].plot(ndvi_value, ndvi_number)
        axes[0].scatter(ndvi_value[target_index], ndvi_number[target_index],
                        c='r', zorder=5)
        axes[0].set_xlabel('NDVI')
        axes[0].set_ylabel('Count')
        axes[1].imshow(ndvi_array, cmap='gray_r')
        axes[1].set_xlabel('NDVI')
        axes[1].axis('off')
        plt.show()
        plt.close()
    else:
        pass
    return ndvi_array


# 使用反射率数据计算NDVI分离植被
def ref_segment(red: int, nir: int, reflectance_array: np.ndarray, if_show: bool = False, n: float = 1.5) -> np.ndarray:
    """
    根据红波段和近红外波段的反射率数据计算NDVI分离植被

    Args:
        red: 红波段的波段序号
        nir: 近红外波段的波段序号
        reflectance_array: 反射率数组
        if_show: 是否显示绘制的图像，默认为False
        n: 阈值计算中的参数n，默认为1.5.数字越大阈值越低

    Returns:
        二值图像数组
    """
    red_band = (reflectance_array[red - 2] + reflectance_array[red - 1] + reflectance_array[red]) / 3
    nir_band = (reflectance_array[nir - 2] + reflectance_array[nir - 1] + reflectance_array[nir]) / 3
    red_band, nir_band = red_band.astype(np.float64), nir_band.astype(np.float64)
    sub = nir_band - red_band
    add = nir_band + red_band
    ndvi_ = np.divide(sub, add, out=np.zeros_like(sub), where=add != 0)  # 防止除数为零

    ndvi_mask = ndvi_segment(ndvi_, show_plot=if_show, n=n)
    return ndvi_mask


def main(path_):

    vi = ["ndvi", "nirv", "fcvi", "sif", "APAR"]

    im_data, geo, proj = tt.read_tif(os.path.join(path_, "5ref", "ref.bip"))  # type: ignore # 读取地表反射率文件
    mask = ref_segment(60, 100, im_data)

    ref = im_data * mask
    tt.write_tif(os.path.join(path_, "5ref", "vegetation_pixels.tif"), ref, geo, proj)  # type: ignore # 保存地表反射率文件

    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.rc('font', family='Times New Roman', size=10)
    fig, ax = plt.subplots(constrained_layout=1, figsize=(4, 2.5), dpi=300)
    shape = ref.shape
    rgb = np.zeros([shape[1], shape[2], 3])
    rgb[:, :, 0] = ref[61]
    rgb[:, :, 1] = ref[39]
    rgb[:, :, 2] = ref[17]
    image = np.power(rgb, 0.18)
    ax.imshow(image)
    plt.show()

    for i in vi:
        print('Cutting ' + i + ' ...')
        VI = tt.read_tif_array(os.path.join(path_, "VIs", i+".tif"))
        new_vi = VI * mask
        tt.write_tif(os.path.join(path_, "VIs", i+"_in_vege.tif"), new_vi, geo, proj)  # type: ignore # 保存地表反射率文件


# 主函数
if __name__ == '__main__':
    dir_path = r'C:\2022_HSI'
    # paths = ["2022_7_5_sunny", ]
    paths = ["2022_7_5_sunny", "2022_7_9_cloudy", "2022_7_12_sunny",
             "2022_7_13_cloudy", "2022_7_16_sunny", "2022_7_20_sunny",
             "2022_7_23_sunny", "2022_7_27_sunny", "2022_8_2_sunny",
             "2022_8_9_cloudy", "2022_8_13_cloudy", "2022_8_14_sunny",
             "2022_8_16_sunny", "2022_8_20_sunny", "2022_8_24_cloudy"]

    for i in tqdm(range(len(paths))):
        path = os.path.join(dir_path, paths[i])
        ref = tt.read_tif_array(os.path.join(path, '5ref', 'ref.bip'))
        ref_segment(60, 100, ref, True)
