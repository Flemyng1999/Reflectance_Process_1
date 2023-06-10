import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def test_plot(data: np.ndarray, reconstruct: np.ndarray):
    # Plot the results
    plt.rc('font', family='Times New Roman', )
    fig, axs = plt.subplots(figsize=(8, 4), constrained_layout=1, dpi=300)

    axs.plot(data[2], label='raw data',
             linewidth=2, alpha=.5, solid_capstyle='round')
    axs.plot(reconstruct[2], label='processed data',
             linewidth=2, alpha=.5, solid_capstyle='round')
    axs.legend()

    plt.show()


def resample(path: str, wavelength: np.ndarray, start_wl: float = 400, end_wl: float = 990):
    """
    重新采样函数
    Args:
    path: str, 数据文件夹路径
    wavelength: np.ndarray, 原始数据的光谱波长
    start_wl: float, 重采样后的起始波长，默认为400
    end_wl: float, 重采样后的结束波长，默认为990

    """
    # load data
    df = pd.read_csv(os.path.join(path, '5ref', 'denoised_vege_ref.csv'), encoding='utf-8', index_col=0)
    ref = df.values.T
    name_list = df.columns.values
    old_wl = wavelength

    wl_number = int(end_wl - start_wl + 1.0)
    new_wl = np.linspace(start_wl, end_wl, wl_number)

    # 初始化一个字典，用于存储重采样后的信号
    resample_ref_dict: Dict[str, np.ndarray] = {}

    for i_ in range(ref.shape[0]):
        # 对当前信号进行重采样
        result = np.interp(new_wl, old_wl, ref[i_])
        resample_ref_dict[name_list[i_]] = result

    df = pd.DataFrame(resample_ref_dict)
    df.to_csv(os.path.join(path, '5ref', 'resampled_vege_ref.csv'))


if __name__ == '__main__':
    if sys.platform == "win32":
        disk1 = 'D:'
        disk2 = 'E:'
    elif sys.platform == "darwin":
        disk1 = os.path.join('/Volumes', 'HyperSpec')
        disk2 = os.path.join('/Volumes', 'HyperSpecII')
    else:  # 默认为 Linux
        disk1 = None
        disk2 = None

    wl = np.loadtxt(os.path.join('/Volumes', 'HyperSpec', '50_target_resample.txt'))[:, 0]
    # paths = ["2022_7_5_sunny", ]
    paths = ["2022_7_5_sunny", "2022_7_9_cloudy", "2022_7_12_sunny",
             "2022_7_13_cloudy", "2022_7_16_sunny", "2022_7_20_sunny",
             "2022_7_23_sunny", "2022_7_27_sunny", "2022_8_2_sunny",
             "2022_8_9_cloudy", "2022_8_13_cloudy", "2022_8_14_sunny",
             "2022_8_16_sunny", "2022_8_20_sunny", "2022_8_24_cloudy"]

    for i in tqdm(range(len(paths))):
        if i < 9:
            resample(os.path.join(disk1, paths[i]), wl)
        else:
            resample(os.path.join(disk2, paths[i]), wl)