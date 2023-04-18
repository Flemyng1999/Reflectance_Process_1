import os

import tiff_tool as tt
import numpy as np
import pandas as pd
from PIL import Image


# 产生感兴趣区的蒙版
def ROI(VI, dataset, roi, Save_path):
    geotransform = dataset.GetGeoTransform()
    w1s1_ = np.zeros((4, 2), dtype=int)
    w1s1_[:, 0] = (roi[:, 0] - geotransform[3]) / geotransform[5]
    w1s1_[:, 1] = (roi[:, 1] - geotransform[0]) / geotransform[1]

    index = np.argwhere(VI)
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
    shape1 = VI.shape
    vi1 = np.zeros(shape1)
    for i in index1:
        vi1[int(i[0]), int(i[1])] = 1
    vi1[0, 0] = 0
    tt.writeTiff(dataset, vi1, Save_path)
    return vi1


# 裁切所有的感兴趣区并且输出效果图
def ROI_cut(path_date_weather, csv_path):
    geo = pd.read_csv(csv_path, index_col=0)
    b = geo.values
    data_set, VI_ = tt.readTiff(os.path.join(path_date_weather, "VIs", "ndvi.tif"))
    li = ['w1s1.tif', 'w1s2.tif', 'w1s3.tif', 'w1s4.tif', 'w1s5.tif',
          'w2s1.tif', 'w2s2.tif', 'w2s3.tif', 'w2s4.tif', 'w2s5.tif',
          'w3s1.tif', 'w3s2.tif', 'w3s3.tif', 'w3s4.tif', 'w3s5.tif',
          'w4s1.tif', 'w4s2.tif', 'w4s3.tif', 'w4s4.tif', 'w4s5.tif']
    shape = VI_.shape
    xxx = np.zeros(shape)
    for i in range(20):
        print('Calculating ' + li[i] + ' ...', end='\n')
        area = b[i, :].reshape(2, 4).T
        xxx = xxx + ROI(VI_, data_set, area, os.path.join(path_date_weather, "ROI", li[i]))
    xxx = 90 * xxx + 160 * (VI_ - np.min(VI_)) / (np.max(VI_) - np.min(VI_))
    im = Image.fromarray(xxx)
    try:
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(os.path.join(path_date_weather, "VIs", "All_ROI.png"))
        print('\n' + 'ROI picture successfully saved at ' + path_date_weather + r'\ROI\All_ROI.png' + "\n")
    except IOError:
        print('\n' + 'An IOError happened...' + '\n')
    im.show()


# 将ROI中的数据保存到csv文件中
def data_in_ROI(VI_path, Date_weather_path, Save_path):
    li = ['w1s1', 'w1s2', 'w1s3', 'w1s4', 'w1s5',
          'w2s1', 'w2s2', 'w2s3', 'w2s4', 'w2s5',
          'w3s1', 'w3s2', 'w3s3', 'w3s4', 'w3s5',
          'w4s1', 'w4s2', 'w4s3', 'w4s4', 'w4s5']
    VI_dict = {}
    name_li = ['w1s1.tif', 'w1s2.tif', 'w1s3.tif', 'w1s4.tif', 'w1s5.tif',
               'w2s1.tif', 'w2s2.tif', 'w2s3.tif', 'w2s4.tif', 'w2s5.tif',
               'w3s1.tif', 'w3s2.tif', 'w3s3.tif', 'w3s4.tif', 'w3s5.tif',
               'w4s1.tif', 'w4s2.tif', 'w4s3.tif', 'w4s4.tif', 'w4s5.tif']
    data_set, vi_path = tt.readTiff(os.path.join(Date_weather_path, VI_path))
    max_ = 0
    for i in range(20):
        mask = tt.readTiffArray(os.path.join(Date_weather_path, "ROI", name_li[i]))
        vi_in_ROI = vi_path * mask
        vi_in_ROI = vi_in_ROI[vi_in_ROI != 0]
        vi_in_ROI = vi_in_ROI[~np.isnan(vi_in_ROI)]
        if len(vi_in_ROI) >= max_:
            max_ = len(vi_in_ROI)

    for i in range(20):
        mask = tt.readTiffArray(os.path.join(Date_weather_path, "ROI", name_li[i]))
        vi_in_ROI = vi_path * mask
        vi_in_ROI = vi_in_ROI[vi_in_ROI != 0]
        vi_in_ROI = vi_in_ROI[~np.isnan(vi_in_ROI)]
        arr = np.empty(max_ - len(vi_in_ROI))
        arr[:] = np.nan
        arr1 = np.hstack((vi_in_ROI, arr))
        VI_dict[li[i]] = arr1

    df = pd.DataFrame(VI_dict)
    df.to_csv(os.path.join(Date_weather_path, Save_path))


def main(path_):
    path1 = "Interest_Area.csv"
    vi = ["ndvi_in_vege.tif", "nirv_in_vege.tif",
          "fcvi_in_vege.tif", "sif_in_vege.tif",
          "APAR_in_vege.tif"]
    save_path = ["ndvi_in_ROI.csv", "nirv_in_ROI.csv",
                 "fcvi_in_ROI.csv", "sif_in_ROI.csv",
                 "APAR_in_ROI.csv"]
    ROI_cut(path_, path1)  # 如果ROI文件已经存在，可注释此行以节省时间
    for i in range(len(vi)):
        print('Collecting Data in ' + vi[i])
        data_in_ROI(os.path.join("VIs", vi[i]), path_, os.path.join("VIs", save_path[i]))


if __name__ == '__main__':
    path = r'F:\2022_8_20_sunny'
    main(path)
