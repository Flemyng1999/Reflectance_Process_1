import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import tiff_tool as tt
import numpy as np
import Clean_data as cd


# function to convert to superscript
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)


# 去除异常值
def clean_arr(arr, n=3):
    mu = np.mean(arr)
    sigma = np.std(arr)
    lim_top = mu + n * sigma  # 上界限 µ + n*σ
    lim_bot = mu - n * sigma  # 下界限 µ - n*σ
    arr[arr > lim_top] = np.median(arr)
    arr[arr < lim_bot] = np.median(arr)
    return arr


# def map_array(array, extreme, my_color, label_, save_path_):
#     cd.clean_arr(array, 3)  # 去除异常值
#     # arr[arr == 0] = np.nan
#     min_, max_ = np.min(extreme), np.max(extreme)
#     ticks = list(np.linspace(min_, max_, 5))
#     array[0, 0], array[0, 1] = max_, min_
#
#     map_color = colors.LinearSegmentedColormap.from_list('my_list', my_color)
#
#     # 把背景值设为白色
#     N = 2560
#     my_colors = cm.get_cmap(map_color, N)
#     new_colors = my_colors(np.linspace(0, 1, N))
#     white = np.array([256 / 256, 256 / 256, 256 / 256, 1])
#     num = int(N * (0 - min_) / (max_ - min_))
#     new_colors[num, :] = white
#     new_cmp = ListedColormap(new_colors)
#
#     im = ax.imshow(array, cmap=new_cmp)
#     fig.colorbar(im,
#                  label=label_,
#                  orientation="horizontal",
#                  pad=0.03,
#                  ticks=ticks
#                  )
#     # plt.axis('off')  # 设置不显示边框
#     ax = plt.gca()  # 设置显示边框,不显示刻度值
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
#     fig.savefig(save_path_)
#     plt.show()


def map_one_vi(path_, tif_path_, my_color, PNG_path_, label_):
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.rc('font', family='Times New Roman', size=10)
    fig, ax = plt.subplots(constrained_layout=1, figsize=(5, 2.5), dpi=300)
    arr = tt.readTiffArray(path_ + tif_path_)
    cd.clean_arr(arr, 3)  # 将阈值设置为5σ
    # arr[arr == 0] = np.nan
    min_, max_ = np.min(arr), np.max(arr)
    ticks = list(np.linspace(min_, max_, 5))

    map_color = colors.LinearSegmentedColormap.from_list('my_list', my_color)

    # 把背景值设为白色
    N = 2560
    my_colors = matplotlib.colormaps.get_cmap(map_color)
    new_colors = my_colors(np.linspace(0, 1, N))
    white = np.array([256 / 256, 256 / 256, 256 / 256, 1])
    num = int(N * (0 - min_) / (max_ - min_))
    new_colors[num, :] = white
    new_cmp = ListedColormap(new_colors)

    im = ax.imshow(arr, cmap=new_cmp)
    fig.colorbar(im,
                 label=label_,
                 pad=0.03,
                 ticks=ticks
                 )
    # plt.axis('off')  # 设置不显示边框
    ax = plt.gca()  # 设置显示边框,不显示刻度值
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    fig.savefig(path_ + PNG_path_)
    plt.show()


def map_multi_vi(path_, tif_path_, PNG_path_, label_, colormap_):
    for i in range(len(tif_path_)):
        my_color = list(colormap_.values())[i]
        map_one_vi(path_, tif_path_[i], my_color, PNG_path_[i], label_[i])


def main(path_):
    name = ["ndvi.tif", "nirv.tif",
            "fcvi.tif", "sif.tif",
            "APAR.tif"]
    tif_path = []
    PNG_path = []
    for i in range(len(tif_path)):
        tif_path.append(os.path.join("VIs", name[i] + ".tif"))
    for i in range(len(tif_path)):
        PNG_path.append(os.path.join("VIs", name[i] + ".png"))

    label = ['NDVI', 'NIRv', 'FCVI',
             'SIF(W·m' + get_super('-2') + '·sr' + get_super('-1') + '·um' + get_super('-1') + ')',
             'APAR(W·m' + get_super('-2') + ')']

    # 自定义ColorMap
    comp = {'NDVI': ['white', '#f7f7f7', '#ededed', '#e6e6e6', '#dbdbdb', '#d4d4d4', '#cccccc', '#c2c2c2', '#b9b9b9',
                     '#b9bf45', '#64a83d', '#167f39', '#044c29', '#00261c'],
            'NIRV': ['white', '#e9e9e9', '#d1d1d1', '#b9b9b9', '#b9bf45', '#64a83d', '#167f39', '#044c29', '#00261c'],
            'FCVI': ['white', '#b9b9b9', '#b9bf45', '#64a83d', '#167f39', '#044c29', '#00261c', 'black'],
            'SIF': ['white', 'white', 'white', '#f6f6f6', '#ececec', '#e2e2e2', '#d8d8d8', '#cecece', '#b9bf45',
                    '#64a83d', '#167f39', '#044c29', '#00261c', 'black', 'black', 'black', 'black', 'black', 'black',
                    'black'],
            'APAR': ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                     'white', 'white', 'white', '#5e7987', '#c3d7df', '#cdceb6', '#e5b715', '#a61b29', '#47040a']}
    map_multi_vi(path_, tif_path, PNG_path, label, comp)


if __name__ == '__main__':
    path = r"F:\2022_8_20_sunny"
    main(path)
