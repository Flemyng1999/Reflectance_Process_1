import numpy as np
import os
import imageio.v2 as imageio
import tiff_tool as tt
import ROI_Cutting as rc
import Tiff2ColorMap as tc
import Clean_data as cd


def check_dir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # 创建文件时如果路径不存在会创建这个路径


def delete_zero(array):
    array = array[~(array == 0).all(1)]  # 删除全0行
    array = array.T
    array = array[~(array == 0).all(1)]  # 删除全0列
    array = array.T
    return array


def generate_gif(png_path, weather=0):
    all_name = os.listdir(png_path)
    png_name = []
    for i in all_name:
        if '.png' in i:
            png_name.append(i)
    if weather is 1:
        # 去掉阴天
        for i in png_name:
            if 'cloudy' in i:
                png_name.remove(i)

    gif_images = []
    for i in png_name:
        gif_images.append(imageio.imread(png_path + '\\' + i))  # 读取多张图片
    imageio.mimsave(png_path + r"\sunny.gif", gif_images, fps=1)  # 转化为gif动画


def roi_mask(date_, VI):
    save_path = r"F:\VI" + r"\\" + VI + r"\\" + r"tif"
    check_dir(save_path)
    # 蒙版边界四个角坐标（N，E）
    limits = [4496920.955, 4496931.958, 4496945.98, 4496934.501,
              517436.8242, 517464.1132, 517457.9166, 517430.9056]
    roi_points = np.array(limits).reshape([2, 4]).T
    dataset, vi_ = tt.readTiff(date_ + r"\VIs" + r"\\" + VI + r".tif")
    roi_ = rc.ROI(vi_, dataset, roi_points, save_path + date_[2:] + r".tif")
    return roi_


def main(vi_):
    disk1 = r'E:'
    disk2 = r'F:'
    date = [disk1 + r"\2022_7_5_sunny", disk1 + r"\2022_7_9_cloudy", disk1 + r"\2022_7_12_sunny",
            disk1 + r"\2022_7_13_cloudy", disk1 + r"\2022_7_16_sunny", disk1 + r"\2022_7_20_sunny",
            disk1 + r"\2022_7_23_sunny", disk1 + r"\2022_7_27_sunny", disk1 + r"\2022_8_2_sunny",
            disk2 + r"\2022_8_9_cloudy", disk2 + r"\2022_8_13_cloudy", disk2 + r"\2022_8_14_sunny",
            disk2 + r"\2022_8_16_sunny", disk2 + r"\2022_8_20_sunny", disk2 + r"\2022_8_24_cloudy"]

    max_, min_, extreme = np.array([]), np.array([]), np.array([])  # 创建空数组
    for day in date:
        print('Getting extreme of ' + day[3:])
        mask = roi_mask(day, vi_)
        arr = tt.readTiffArray(day + r"\VIs" + r"\\" + vi_ + r".tif")
        roi_data = cd.clean_arr(arr * mask)  # 套用蒙版并去除异常值
        max_, min_ = np.append(max_, np.max(roi_data)), np.append(min_, np.min(roi_data))  # 将极值都放入数组中，最后再求极值的极值
    extreme = np.append((np.append(extreme, np.max(max_))), np.min(min_))

    for day in date:
        mask = roi_mask(day, vi_)
        arr = tt.readTiffArray(day + r"\VIs" + r"\\" + vi_ + r".tif")
        roi = delete_zero(arr * mask)  # 套用蒙版并去除空白行、列
        color = ['white', '#e9e9e9', '#d1d1d1', '#b9b9b9', '#b9bf45', '#64a83d', '#167f39', '#044c29', '#00261c']
        label = day[8:]
        tc.map_array(roi, extreme, color, label, r"F:\VI" + r"\\" + vi_ + r"\\" + day[3:] + vi_ + r".png")

    generate_gif(r"F:\VI" + r'\\' + vi_)


if __name__ == '__main__':
    vi = r"nirv_in_vege"
    main(vi)
