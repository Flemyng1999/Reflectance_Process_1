import os

import numpy as np
import pandas as pd


def clean_arr(arr, n=3):
    arr_ = arr[arr != 0]
    mu = np.mean(arr_)
    sigma = np.std(arr_)
    lim_top = mu + n * sigma  # 上界限 µ + n*σ
    lim_bot = mu - n * sigma  # 下界限 µ - n*σ
    arr[((arr > lim_top) & (arr != 0)).all()] = np.median(arr_)
    arr[((arr < lim_bot) & (arr != 0)).all()] = np.median(arr_)
    return arr


def clean_dataframe(df_path, n=3):
    df = pd.read_csv(df_path, index_col=0)
    li = ['w1s1', 'w1s2', 'w1s3', 'w1s4', 'w1s5',
          'w2s1', 'w2s2', 'w2s3', 'w2s4', 'w2s5',
          'w3s1', 'w3s2', 'w3s3', 'w3s4', 'w3s5',
          'w4s1', 'w4s2', 'w4s3', 'w4s4', 'w4s5']
    VI_dict = {}
    for i in range(20):
        arr0 = np.array(df.iloc[:, i])
        length = len(arr0)
        arr1 = arr0[~np.isnan(arr0)]  # 去掉数组内NAN空值
        arr1 = clean_arr(arr1, n)
        arr2 = np.empty(length - len(arr1))
        arr2[:] = np.nan
        arr3 = np.hstack((arr1, arr2))
        VI_dict[li[i]] = arr3
    df1 = pd.DataFrame(VI_dict)
    df1.to_csv(df_path)


def main(path_):
    save_path = ["ndvi_in_ROI.csv", "nirv_in_ROI.csv",
                 "fcvi_in_ROI.csv", "sif_in_ROI.csv",
                 "APAR_in_ROI.csv"]
    for i in range(len(save_path)):
        print('Cleaning Data in ' + save_path[i])
        clean_dataframe(os.path.join(path_, "VIs", save_path[i]), 3)


if __name__ == '__main__':
    path = r'D:\2022_8_20_sunny'
    main(path)
