# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 10:31:42 2022

@author: Flemyng
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import tiff_tool as tt


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
def read_target(txt_path):
    li = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for x, line in enumerate(f):
            if x < 4:
                continue
            if x > 153:
                break
            result = line.split('\t')
            li.append(result[3])
    array = np.array(li, dtype=np.float32)
    return array

# 计算入射辐射
def irrad(file1, file2):
    radiance = read_target(file1)
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
    save_path = file1.replace('rad_target.txt', 'I&E.png')
    fig.savefig(save_path)
    plt.show()

    spi = open(file1.strip('rad_target.txt') + r"irr.txt", "w")  # 输出irr.txt文件到源文件夹
    for i in range(150):
        spi.write(str(irr[i]) + '\n')
    spi.close()

    return irr


# 计算NDVI
def ndvi(red, nir, img_data):
    print('\n' + 'Calculating NDVI ...')
    red_band = (img_data[red - 2] + img_data[red - 1] + img_data[red]) / 3
    nir_band = (img_data[nir - 2] + img_data[nir - 1] + img_data[nir]) / 3
    red_band, nir_band = red_band.astype(np.float64), nir_band.astype(np.float64)
    sub = nir_band - red_band
    add = nir_band + red_band
    ndvi_ = np.divide(sub, add, out=np.zeros_like(sub), where=add != 0)  # 防止除数为零
    return ndvi_


# 计算NIRv
def nirv(red, nir, img_data):
    print('\n' + 'Calculating NIRv ...')
    nir_band = (img_data[nir - 2] + img_data[nir - 1] + img_data[nir]) / 3
    ndvi_ = ndvi(red, nir, img_data)
    nirv_ = ndvi_ * nir_band
    return nirv_


# 计算FCVI
def fcvi(nir1, nir2, vis1, vis2, img_data):
    print('\n' + 'Calculating FCVI ...')
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
    print('\n' + 'Calculating SIF...')
    # 寻找3个波段
    irr = irrad(target_rad_, target_ref)
    list_irr = irr.tolist()  # 转为List
    E_min = min(list_irr[30:100])  # 返回最小值
    s_in = list_irr.index(E_min)  # 返回最小值的索引
    max1, max2 = max(list_irr[s_in - 10:s_in]), max(list_irr[s_in:s_in + 10])
    s_1, s_2 = list_irr.index(max1), list_irr.index(max2)

    L = tt.readTiffArray(rad_)  # 读取地表上行辐射文件
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
    print('\n' + 'Calculating APAR...')
    radiance = read_target(target_rad_path)
    reflectance = reflect(target_reflect_path)
    par_in = np.divide(radiance, reflectance)  # 计算入射辐射PARin

    par_out = tt.readTiffArray(rad_path, np.int64)  # 计算地表反射辐射PARout
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
    target_reflect = r"C:\Users\imFle\OneDrive\resample50178.txt"  # 靶标布反射率文件(长期固定)
    rad = os.path.join(path_, "4rad", "rad_corr.tif")  # 地面辐射文件(校正后)
    refl = os.path.join(path_, "5ref", "ref.bip")  # 地面反射率文件

    dataset, im_data = tt.readTiff(refl)  # 读取地表反射率文件

    apar_data = APAR(target_radiance, target_reflect, rad)
    tt.writeTiff(dataset, apar_data, os.path.join(path_, "VIs", "APAR.tif"))

    ndvi_data = ndvi(60, 100, im_data)
    tt.writeTiff(dataset, ndvi_data, os.path.join(path_, "VIs", "ndvi.tif"))

    nirv_data = nirv(60, 100, im_data)
    tt.writeTiff(dataset, nirv_data, os.path.join(path_, "VIs", "nirv.tif"))

    fcvi_data = fcvi(93, 113, 1, 76, im_data)
    tt.writeTiff(dataset, fcvi_data, os.path.join(path_, "VIs", "fcvi.tif"))

    sif_data = sif(target_radiance, target_reflect, im_data, rad)
    tt.writeTiff(dataset, sif_data, os.path.join(path_, "VIs", "sif.tif"))


if __name__ == '__main__':
    path = r'F:\2022_8_20_sunny'
    main(path)
