import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm

import tiff_tool


def find_nearest_index(target, arr):
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


def find_threshold(target, arr):
    """
    根据目标值和数组，计算出阈值

    Args:
        target: 目标值
        arr: 数组

    Returns:
        阈值
    """
    filtered_arr = arr[arr > target]
    min_four = np.partition(filtered_arr, 3)[:4]
    threshold = min_four[2] + target
    return threshold


def ndvi_threshold(ndvi_array, show_plot=False, n=5):
    """
    根据NDVI数组，计算出阈值，并将数组转换为二值图像

    Args:
        ndvi_array: NDVI数组
        show_plot: 是否显示绘制的图像，默认为False
        n: 阈值计算中的参数n，默认为3.数字越大阈值越低

    Returns:
        二值图像数组
    """
    bins_num = 500
    arr = ndvi_array[ndvi_array > 0]
    y, x = np.histogram(arr, bins=bins_num)
    cycle, y = sm.tsa.filters.hpfilter(y, 500)
    x = (x[1:] + x[:-1]) / 2

    a1 = np.max(y[int(bins_num / 2):])
    a1_i = np.argwhere(y == a1)
    a2 = np.min(y[int(bins_num / 5):int(bins_num * 4 / 5)])
    a2_i = np.argwhere(y == a2)
    a3 = int((a1 + (n - 1) * a2) / n)

    up_limit = None
    if a1_i.size > 1:
        for i_ in range(a1_i.size):
            candy = int(a1_i[i_])
            if a2 < candy < a1:
                up_limit = candy
    else:
        up_limit = int(a1_i[0])

    a4 = find_threshold(a3, y[int(a2_i[0]):up_limit])
    th = x[find_nearest_index(a4, y)]

    ndvi_array[ndvi_array >= th] = 1
    ndvi_array[ndvi_array < th] = 0

    if show_plot:
        fig, axes = plt.subplots(2, constrained_layout=1, figsize=(7, 9), dpi=200)
        axes[0].plot(x, y)
        axes[0].scatter(th, y[find_nearest_index(th, x)], c='r', zorder=5)
        axes[0].set_xlabel('NDVI')
        axes[0].set_ylabel('Count')
        axes[1].imshow(ndvi_array, cmap='gray_r')
        axes[1].set_xlabel('NDVI')
        axes[1].axis('off')
        plt.show()
    else:
        plt.close()

    return ndvi_array


if __name__ == '__main__':
    dir_path = r'C:\Users\Lenovo\Desktop\data\2022_HSI'
    paths = ["2022_7_5_sunny", ]
    # paths = ["2022_7_5_sunny", "2022_7_9_cloudy", "2022_7_12_sunny",
    #          "2022_7_13_cloudy", "2022_7_16_sunny", "2022_7_20_sunny",
    #          "2022_7_23_sunny", "2022_7_27_sunny", "2022_8_2_sunny",
    #          "2022_8_9_cloudy", "2022_8_13_cloudy", "2022_8_14_sunny",
    #          "2022_8_16_sunny", "2022_8_20_sunny", "2022_8_24_cloudy"]

    for i in tqdm(range(len(paths))):
        path = os.path.join(dir_path, paths[i])
        ndvi = tiff_tool.read_tif_array(os.path.join(path, 'vi', 'ndvi.tif'))
        ndvi_threshold(ndvi, True)