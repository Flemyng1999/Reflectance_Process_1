import os
import time
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import tiff_tool as tt


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("函数 %s 执行时间为 %f 秒" % (func.__name__, end_time - start_time))
        return result
    return wrapper


# 产生感兴趣区的蒙版
def roi(vi, geotransform, projection, roi_limit, save_path):
    w1s1_ = np.zeros((4, 2), dtype=int)
    w1s1_[:, 0] = (roi_limit[:, 0] - geotransform[3]) / geotransform[5]
    w1s1_[:, 1] = (roi_limit[:, 1] - geotransform[0]) / geotransform[1]

    index = np.argwhere(vi)
    shape = index.shape
    index1 = np.zeros(shape)
    x, y = index[:, 0], index[:, 1]
    xx, yy = w1s1_[:, 0], w1s1_[:, 1]

    arr1 = (y - yy[1]) / (yy[0] - yy[1]) - (x - xx[1]) / (xx[0] - xx[1])
    arr1[arr1 > 0] = 1
    arr1[arr1 < 0] = 0
    arr2 = (y - yy[2]) / (yy[1] - yy[2]) - (x - xx[2]) / (xx[1] - xx[2])
    arr2[arr2 > 0] = 0
    arr2[arr2 < 0] = 1
    arr3 = (y - yy[3]) / (yy[2] - yy[3]) - (x - xx[3]) / (xx[2] - xx[3])
    arr3[arr3 > 0] = 1
    arr3[arr3 < 0] = 0
    arr4 = (y - yy[0]) / (yy[3] - yy[0]) - (x - xx[0]) / (xx[3] - xx[0])
    arr4[arr4 > 0] = 0
    arr4[arr4 < 0] = 1

    arr_ = arr1 * arr2 * arr3 * arr4
    index1[:, 0] = x * arr_
    index1[:, 1] = y * arr_
    shape1 = vi.shape
    vi1 = np.zeros(shape1)
    for i_ in index1:
        vi1[int(i_[0]), int(i_[1])] = 1
    vi1[0, 0] = 0
    tt.write_tif(save_path, vi1, geotransform, projection)
    return vi1


def normalize(array: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D numpy array to [0,1].

    Args:
        array (np.ndarray): A 1D numpy array to be normalized.

    Returns:
        np.ndarray: A normalized 1D numpy array.

    Raises:
        AssertionError: If the input is not a 1D numpy array.

    """
    assert isinstance(array, np.ndarray), "Input must be a numpy array"
    assert len(array.shape) == 1, "Input must be a 1D numpy array"

    max_val = np.max(array)
    min_val = np.min(array)

    new_arr = (array - min_val) / (max_val - min_val)
    return new_arr


# 裁切所有的感兴趣区并且输出效果图
def roi_cut(path_date_weather, csv_path):
    limit_value = pd.read_csv(csv_path, index_col=0).values

    roi_names = ['w{}s{}.tif'.format(w, s) for w in range(1, 5) for s in range(1, 6)]

    # 检测目标文件夹是否存在
    dst_folder = os.path.join(path_date_weather, 'roi')
    if not os.path.exists(dst_folder):
        # 如果子文件夹不存在，创建它
        os.makedirs(dst_folder)

    ndvi, geo, proj = tt.read_tif(os.path.join(path_date_weather, "vi", "ndvi.tif"))
    # 定义一个进程池
    cup_count = os.cpu_count()
    with Pool(processes=cup_count) as pool:
        # 对每个区域都启动一个进程进行处理
        for i_ in range(20):
            area = limit_value[i_, :].reshape(2, 4).T
            save_path = os.path.join(path_date_weather, 'roi', roi_names[i_])
            # 使用 partial 函数固定部分参数，方便进程池调用
            func = partial(roi, ndvi, geo, proj, area, save_path)
            pool.apply_async(func)

        # 等待所有进程完成
        pool.close()
        pool.join()


# 检测roi的位置是否正确
def check_roi_site(dir_path_):
    roi_files = ['w{}s{}.tif'.format(w, s) for w in range(1, 5) for s in range(1, 6)]

    roi_masks = []
    # 逐个读取 ROI 图像
    for roi_file in roi_files:
        roi_path = os.path.join(dir_path_, 'roi', roi_file)
        roi_mask = tt.read_tif_array(roi_path)
        roi_masks.append(roi_mask)

    # 将所有 ROI 图像叠加
    roi_mask = np.sum(roi_masks, axis=0)

    # 检测目标文件是否存在
    dst_file = os.path.join(dir_path_, '5ref', 'rgb.tif')
    if not os.path.exists(dst_file):
        # 如果文件不存在，创建它
        ref, geo, proj = tt.read_tif(os.path.join(dir_path_, "5ref", "ref.bip"))
        r, g, b = ref[58, :, :], ref[36, :, :], ref[18, :, :]
        del ref
        rgb = np.stack((r, g, b), axis=-1)
        rgb_image = normal(rgb)  # rgb归一化
        rgb_image[rgb_image == 0] = 1
        tt.write_tif(dst_file, rgb_image, geo, proj)
    else:
        rgb_image, geo, proj = tt.read_tif(dst_file)

    roi_mask = roi_mask * 0.4
    roi_mask = np.stack((roi_mask, roi_mask, roi_mask), axis=-1)
    rgb_image = rgb_image + roi_mask
    rgb_image = normal(rgb_image)

    # plot
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=1)
    ax.imshow(rgb_image)
    ax.set_axis_off()
    plt.savefig(os.path.join(dir_path_, 'roi', 'roi.png'), dpi=300)
    plt.show()
    plt.close()


# 将ROI中的数据保存到csv文件中
def data_in_roi(vi_name, date_weather_path):
    roi_names = ['w{}s{}'.format(w, s) for w in range(1, 5) for s in range(1, 6)]
    VI_dict = {}

    vi = tt.read_tif_array(os.path.join(date_weather_path, 'vi', vi_name+'.tif'))
    max_ = 0
    for i_ in range(20):
        mask = tt.read_tif_array(os.path.join(date_weather_path, 'roi', roi_names[i_]+'.tif'))
        vi_in_ROI = vi * mask
        vi_in_ROI = vi_in_ROI[vi_in_ROI != 0]
        vi_in_ROI = vi_in_ROI[~np.isnan(vi_in_ROI)]
        if len(vi_in_ROI) >= max_:
            max_ = len(vi_in_ROI)

    for i_ in range(20):
        mask = tt.read_tif_array(os.path.join(date_weather_path, 'roi', roi_names[i_]+'.tif'))
        vi_in_ROI = vi * mask
        vi_in_ROI = vi_in_ROI[vi_in_ROI != 0]
        vi_in_ROI = vi_in_ROI[~np.isnan(vi_in_ROI)]
        arr = np.empty(max_-len(vi_in_ROI))
        arr[:] = np.nan
        arr1 = np.hstack((vi_in_ROI, arr))
        VI_dict[roi_names[i_]] = arr1

    df = pd.DataFrame(VI_dict)
    df.to_csv(os.path.join(date_weather_path, 'vi', vi_name+'.csv'))


def main(path_):
    roi_limit_path = r"C:\Users\Lenovo\OneDrive\Projects\HSI\Interest_Area.csv"
    vi = ["ndvi", "nirv", "fcvi", "sif", "APAR"]

    # 如果ROI文件已经存在，可节省时间
    roi_img_path = os.path.join(path_, 'roi', 'roi.png')
    if not os.path.exists(roi_img_path):
        roi_cut(path_, roi_limit_path)
        check_roi_site(path_)
    else:
        pass

    for i_ in range(len(vi)):
        # print('Collecting Data in ' + vi[i])
        data_in_roi(vi[i_], path_)


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
    # paths = ["2022_7_5_sunny", ]
    paths = ["2022_7_5_sunny", "2022_7_9_cloudy", "2022_7_12_sunny",
             "2022_7_13_cloudy", "2022_7_16_sunny", "2022_7_20_sunny",
             "2022_7_23_sunny", "2022_7_27_sunny", "2022_8_2_sunny",
             "2022_8_9_cloudy", "2022_8_13_cloudy", "2022_8_14_sunny",
             "2022_8_16_sunny", "2022_8_20_sunny", "2022_8_24_cloudy"]

    for i in tqdm(range(len(paths))):
        if i < 9:
            main()
