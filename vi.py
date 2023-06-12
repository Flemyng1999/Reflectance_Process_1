# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 10:31:42 2022

@author: Flemyng
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import tiff_tool as tt
from tqdm import tqdm


# 打开并简单取读靶标布[反射率]
def reflect(file):
    with open(file, "r") as f:
        file_content = f.read()
    data = file_content.split()  # 单词拆分
    # 提取Mean_Rad
    mean = np.ones(150)
    index_ = np.arange(1, 999, 2)
    for i in range(150):
        mean[i] = data[index_[i]]
    # 画图
    # x = np.linspace(393.899994, 1026.187012, 150)
    # plt.figure()
    # plt.plot(x, mean)
    # plt.xlabel('Wavelength(nm)')
    # plt.ylabel('Target Reflectance')
    # plt.show()
    return mean


# 打开并简单读取靶标布[辐射]
def target_rad(file):
    with open(file, "r") as f:
        file_content = f.read()
    data = file_content.split()  # 单词拆分
    # 去掉表头
    for i in range(999):
        data.pop(0)
        if data[0] == "Stdev":
            break
    # 提取Mean_Rad
    mean = np.ones(150)
    index = np.arange(5, 9999, 6)
    for i in range(150):
        mean[i] = data[index[i]]
    return mean


# 计算入射辐射
def irrad(file1, file2):
    radiance = target_rad(file1)
    reflectance = reflect(file2)
    irr = np.divide(radiance, reflectance)

    x = np.linspace(393.899994, 1026.187012, 150)
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.rc('font', family='Times New Roman', size=10)
    fig, ax = plt.subplots(constrained_layout=1, figsize=(5, 3), dpi=300)
    ax.plot(x, irr, label='Down-welling radiance')  # 画图
    ax.plot(x, radiance, label='Up-welling radiance')
    ax.set_xlabel('Wavelength(nm)')
    ax.set_ylabel('Radiance')
    plt.legend(loc='best')

    spi = open(file1.strip('target_rad.txt') + r"irr.txt", "w")  # 输出irr.txt文件到源文件夹
    for i in range(150):
        spi.write(str(irr[i]) + '\n')
    spi.close()

    return irr


# 计算NDVI
def ndvi(red, nir, img_data):
    # print('\n' + 'Calculating NDVI ...')
    red_band = (img_data[red - 2] + img_data[red - 1] + img_data[red]) / 3
    nir_band = (img_data[nir - 2] + img_data[nir - 1] + img_data[nir]) / 3
    red_band, nir_band = red_band.astype(np.float64), nir_band.astype(np.float64)
    sub = nir_band - red_band
    add = nir_band + red_band
    ndvi_ = np.divide(sub, add, out=np.zeros_like(sub), where=add != 0)  # 防止除数为零
    return ndvi_


# 计算NIRv
def nirv(red, nir, img_data):
    # print('\n' + 'Calculating NIRv ...')
    nir_band = (img_data[nir - 2] + img_data[nir - 1] + img_data[nir]) / 3
    ndvi_ = ndvi(red, nir, img_data)
    nirv_ = ndvi_ * nir_band
    return nirv_


# 计算FCVI
def fcvi(nir1, nir2, vis1, vis2, img_data):
    # print('\n' + 'Calculating FCVI ...')
    sum_1 = img_data[nir1 - 1]
    sum_2 = img_data[vis1 - 1]
    for i in range(nir1, nir2 + 1):
        sum_1 = sum_1 + img_data[i]
    for i in range(vis1, vis2 + 1):
        sum_2 = sum_2 + img_data[i]
    sum_1, sum_2 = sum_1.astype(np.float64), sum_2.astype(np.float64)
    sub = sum_1 / (nir2 - nir1 + 1)
    add = sum_2 / (vis2 - vis1 + 1)
    fcvi_ = np.divide(sub, add, out=np.zeros_like(sub), where=add != 0)  # 防止除数为零
    return fcvi_


# 计算SIF
# def sif(target, reflectance, radiance):
#     print('\n' + 'Calculating SIF...')
#     # 寻找3个波段
#     irr = \
#         (target, reflectance)
#     list_irr = irr.tolist()  # 转为List
#     list_min = min(list_irr[30:100])  # 返回最小值
#     s_in = list_irr.index(list_min)  # 返回最小值的索引
#     max1 = max(list_irr[s_in - 10:s_in])
#     s_l = list_irr.index(max1)
#     max2 = max(list_irr[s_in:s_in + 10])
#     s_r = list_irr.index(max2)
#     # 参数计算
#     w_l = np.divide(s_in - s_l, s_r - s_l)
#     w_r = np.divide(s_r - s_in, s_r - s_l)
#     a = np.divide(irr[s_in], w_l * irr[s_l] + w_r * irr[s_r])
#     # 获取地表辐射
#     img_data = tt.readTiffArray(radiance, np.uint16)
#     sif_ = np.divide(img_data[s_in] - a * w_l * img_data[s_l] - a * w_r * img_data[s_r], 1 - a)
#     return sif_
def sif(target_rad_, target_ref, R, rad_):
    # print('\n' + 'Calculating SIF...')
    # 寻找3个波段
    irr = irrad(target_rad_, target_ref)
    list_irr = irr.tolist()  # 转为List
    E_min = min(list_irr[30:100])  # 返回最小值
    s_in = list_irr.index(E_min)  # 返回最小值的索引
    max1, max2 = max(list_irr[s_in - 10:s_in]), max(list_irr[s_in:s_in + 10])
    s_1, s_2 = list_irr.index(max1), list_irr.index(max2)

    L = tt.read_tif_array(rad_)  # 读取地表上行辐射文件
    R_in = R[s_in, :, :]
    L_in = L[s_in, :, :]
    R_out1 = R[s_1, :, :]
    R_out2 = R[s_2, :, :]
    a1 = (s_in - s_1) / (s_2 - s_1)  # 差分系数
    a2 = (s_2 - s_in) / (s_2 - s_1)
    F = L_in * (R_in - a2 * R_out1 - a1 * R_out2)
    return F


# 计算APAR
def APAR(target_rad_path, target_reflect_path, rad_path):
    # print('\n' + 'Calculating APAR...')
    radiance = target_rad(target_rad_path)
    reflectance = reflect(target_reflect_path)
    par_in = np.divide(radiance, reflectance)  # 计算入射辐射PARin

    par_out = tt.read_tif_array(rad_path)  # 计算地表反射辐射PARout
    index = np.argwhere(par_out[100] != 0)  # 获得非背景像素的索引

    par_in = par_in[0:75]
    par_out = par_out[0:75, :, :]
    shape = par_out.shape

    apar = np.zeros(shape, dtype=np.int64)
    for i in index:
        apar[:, i[0], i[1]] = (par_in - par_out[:, i[0], i[1]])
    apar = np.mean(apar, axis=0)  # 对75个波段取平均
    return apar


def main(path_):
    target_radiance = os.path.join(path_, "4rad", "rad_target.txt")  # 靶标布辐射文件
    target_reflect = os.path.join("docs", "resample50178.txt")  # 靶标布反射率文件(长期固定)
    rad_path = os.path.join(path_, "4rad", "rad_corr.tif")  # 地面辐射文件(校正后)
    refl_path = os.path.join(path_, "5ref", "ref.bip")  # 地面反射率文件

    # 检测目标文件夹是否存在
    dst_folder = os.path.join(path_, 'vi')
    if not os.path.exists(dst_folder):
        # 如果子文件夹不存在，创建它
        os.makedirs(dst_folder)

    im_data, geo, proj = tt.read_tif(refl_path)  # 读取地表反射率文件

    apar_data = APAR(target_radiance, target_reflect, rad_path)
    tt.write_tif(os.path.join(path_, "vi", "APAR.tif"), apar_data, geo, proj)

    ndvi_data = ndvi(60, 100, im_data)
    tt.write_tif(os.path.join(path_, "vi", "ndvi.tif"), ndvi_data, geo, proj)

    nirv_data = nirv(60, 100, im_data)
    tt.write_tif(os.path.join(path_, "vi", "nirv.tif"), nirv_data, geo, proj)

    fcvi_data = fcvi(93, 113, 1, 76, im_data)
    tt.write_tif(os.path.join(path_, "vi", "fcvi.tif"), fcvi_data, geo, proj)

    sif_data = sif(target_radiance, target_reflect, im_data, rad_path)
    tt.write_tif(os.path.join(path_, "vi", "sif.tif"), sif_data, geo, proj)


if __name__ == '__main__':
    if sys.platform == "win32":
        disk1 = 'D:'
        disk2 = 'E:'
    elif sys.platform == "darwin":
        disk1 = os.path.join('/Volumes', 'HyperSpec')
        disk2 = os.path.join('/Volumes', 'HyperSpecII')
    else:  # 默认为 Linux
        disk1 = os.path.join('/Volumes', 'HyperSpec')
        disk2 = os.path.join('/Volumes', 'HyperSpecII')
    paths = ["2022_7_5_sunny", "2022_7_9_cloudy", "2022_7_12_sunny",]
    # paths = ["2022_7_5_sunny", "2022_7_9_cloudy", "2022_7_12_sunny",
    #          "2022_7_13_cloudy", "2022_7_16_sunny", "2022_7_20_sunny",
    #          "2022_7_23_sunny", "2022_7_27_sunny", "2022_8_2_sunny",
    #          "2022_8_9_cloudy", "2022_8_13_cloudy", "2022_8_14_sunny",
    #          "2022_8_16_sunny", "2022_8_20_sunny", "2022_8_24_cloudy"]

    for i in tqdm(range(len(paths))):
        if i < 9:
            main(os.path.join(disk1, paths[i]))
        else:
            main(os.path.join(disk2, paths[i]))
