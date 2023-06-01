import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from tqdm import tqdm

import tiff_tool as tt
import numpy as np


# 去除异常值
def clean_arr(arr, n=3):
    mu, sigma = np.mean(arr), np.std(arr)
    arr[np.abs(arr - mu) > n * sigma] = np.nan
    return arr


def map_array(array, extreme, my_color, label_, save_path_):
    min_, max_ = np.min(extreme), np.max(extreme)
    ticks = np.linspace(min_, max_, 5)
    array[0, 0], array[0, 1] = max_, min_

    map_color = colors.LinearSegmentedColormap.from_list('my_list', my_color)

    # 把背景值设为白色
    N = 2560
    my_colors = plt.get_cmap(map_color, N)
    new_colors = my_colors(np.linspace(0, 1, N))
    white = np.array([256 / 256, 256 / 256, 256 / 256, 1])
    num = int(N * (0 - min_) / (max_ - min_))
    new_colors[num, :] = white
    new_cmp = ListedColormap(new_colors)

    fig, ax = plt.subplots(constrained_layout=1)
    im = ax.imshow(array, cmap=new_cmp)
    fig.colorbar(im, label=label_, orientation="horizontal", pad=0.03, ticks=ticks)
    ax.axis('off')  # 设置不显示边框和刻度值
    fig.savefig(save_path_)
    plt.show()
    plt.close()


def map_one_vi(path_, vi_name, my_color, label_):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # plt.rc('font', family='Times New Roman', size=10)
    arr = tt.read_tif_array(os.path.join(path_, 'vi', vi_name+'.tif'))
    min_, max_ = np.min(arr), np.max(arr)
    ticks = np.linspace(min_, max_, 5)

    map_color = colors.LinearSegmentedColormap.from_list('my_list', my_color)

    # 把背景值设为白色
    N = 2560
    my_colors = plt.get_cmap(map_color, N)
    new_colors = my_colors(np.linspace(0, 1, N))
    num = int(N * (0 - min_) / (max_ - min_))
    new_colors[num, :] = [1, 1, 1, 1]
    new_cmp = ListedColormap(new_colors)

    fig, ax = plt.subplots(constrained_layout=1, figsize=(5, 2.5), dpi=300)
    im = ax.imshow(arr, cmap=new_cmp)
    fig.colorbar(im, label=label_, pad=0.03, ticks=ticks)
    ax.axis('off')
    fig.savefig(os.path.join(path_, 'vi', vi_name), dpi=300)
    plt.show()
    plt.close()


def map_multi_vi(path_, vi_name_list, label_, colormap_):
    for i_ in range(len(vi_name_list)):
        my_color = list(colormap_.values())[i_]
        map_one_vi(path_, vi_name_list[i_], my_color, label_[i_])


def main(path_):
    vi_names = ['ndvi', 'nirv', 'fcvi', 'sif', 'APAR']
    label = ['NDVI', 'NIRv', 'FCVI', 'SIF', 'APAR']

    # 自定义ColorMap
    comp = {'NDVI': ['white', '#f7f7f7', '#ededed', '#e6e6e6', '#dbdbdb', '#d4d4d4', '#cccccc', '#c2c2c2', '#b9b9b9', '#b9bf45', '#64a83d', '#167f39', '#044c29', '#00261c'],
            'NIRV': ['white', '#e9e9e9', '#d1d1d1', '#b9b9b9', '#b9bf45', '#64a83d', '#167f39', '#044c29', '#00261c'],
            'FCVI': ['white', '#b9b9b9', '#b9bf45', '#64a83d', '#167f39', '#044c29', '#00261c', 'black'],
            'SIF': ['white', 'white', 'white', '#f6f6f6', '#ececec', '#e2e2e2', '#d8d8d8', '#cecece', '#b9bf45', '#64a83d', '#167f39', '#044c29', '#00261c', 'black', 'black', 'black', 'black', 'black', 'black', 'black'],
            'APAR': ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', '#5e7987', '#c3d7df', '#cdceb6', '#e5b715', '#a61b29', '#47040a']}
    map_multi_vi(path_, vi_names, label, comp)


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
        main(path)
