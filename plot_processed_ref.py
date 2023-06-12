import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from tqdm import tqdm


def test_plot(path: str, data: np.ndarray, reconstruct: np.ndarray, wavelength1: np.ndarray, wavelength2: np.ndarray):
    """
    Args:
        - data: 一个numpy数组, 包含原始数据
        - reconstruct: 一个numpy数组, 包含处理后的数据
        - wavelength1: 一个numpy数组, 包含原始数据的波长, 1D
        - wavelength2: 一个numpy数组, 包含处理后的数据的波长, 1D
    功能用法：
        该函数将原始数据和处理后的数据绘制在同一张图上，并展示出来。
    """
    plt.rc('font', family='Times New Roman', )
    fig, axs = plt.subplots(figsize=(8, 4), constrained_layout=1, dpi=100)

    axs.plot(wavelength1, data[2], label='Raw data',
             linewidth=6, alpha=.4, solid_capstyle='round')
    axs.plot(wavelength2, reconstruct[2], label='Processed data',
             linewidth=1.5, alpha=1., solid_capstyle='round', color='r')
    axs.legend()
    plt.title('Raw data and denoised & resampled data')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')

    plt.savefig(os.path.join(path, 'test.png'), dpi=300)
    # plt.show()
    # plt.close()


def main(path: str):
    df = pd.read_csv(os.path.join(path, '5ref', 'vege_ref_in_roi.csv'), encoding='utf-8', index_col=0)
    resample = pd.read_csv(os.path.join(path, '5ref', 'resampled_vege_ref.csv'), encoding='utf-8', index_col=0)

    wl_1 = np.loadtxt(os.path.join('/Volumes', 'HyperSpec', '50_target_resample.txt'))[:, 0]
    wl_2 = np.linspace(400, 1000, 601)

    test_plot(os.path.join(path, '5ref'), df.values.T, resample.values.T, wl_1, wl_2)


if __name__ == '__main__':
    if sys.platform == "win32":
        disk1 = 'D:'
        disk2 = 'E:'
    elif sys.platform == "darwin":
        disk1 = os.path.join('/Volumes', 'HyperSpec')
        disk2 = os.path.join('/Volumes', 'HyperSpecII')
    else:
        disk1 = None
        disk2 = None

    # paths = ["2022_7_5_sunny", ]
    paths = ["2022_7_5_sunny", "2022_7_9_cloudy", "2022_7_12_sunny",
             "2022_7_13_cloudy", "2022_7_16_sunny", "2022_7_20_sunny",
             "2022_7_23_sunny", "2022_7_27_sunny", "2022_8_2_sunny",
             "2022_8_9_cloudy", "2022_8_13_cloudy", "2022_8_14_sunny",
             "2022_8_16_sunny", "2022_8_20_sunny", "2022_8_24_cloudy"]

    for i in tqdm(range(len(paths))):
        if i < 9:
            main(os.path.join(disk1, paths[i]))
        else:
            main(os.path.join(disk2, paths[i]))
