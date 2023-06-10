import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from tqdm import tqdm


def test_plot(data: np.ndarray, reconstruct: np.ndarray):
    # Plot the results
    plt.rc('font', family='Times New Roman', )
    fig, axs = plt.subplots(figsize=(8, 4), constrained_layout=1, dpi=300)

    axs.plot(data[2], label='Raw data',
             linewidth=2, alpha=.7, solid_capstyle='round')
    axs.plot(reconstruct[2], label='Processed data',
             linewidth=2, alpha=.7, solid_capstyle='round')
    axs.legend()

    plt.show()


def main(path):
    # Load data
    begin_wl = 86
    df = pd.read_csv(os.path.join(path, '5ref', 'vege_ref_in_roi.csv'), encoding='utf-8', index_col=0)
    ref = df.values.T[:, begin_wl:]

    # 对每组数据进行HP滤波
    filtered_data = np.zeros_like(ref)
    for item in range(ref.shape[0]):
        cycle, filtered_data[item] = sm.tsa.filters.hpfilter(ref[item], 2)
    # test_plot(ref, filtered_data)

    df.values[begin_wl:, :] = filtered_data.T
    df.to_csv(os.path.join(path, '5ref', 'denoised_vege_ref.csv'))


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
