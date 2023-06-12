import os
import sys
import time

from tqdm import tqdm
import numpy as np

import vi
import roi_cut as rc
import hp_filter as hp
from resample import resample
import plot_processed_ref as test


def main(path):
    # 记录开始时间
    start_time = time.time()
    
    # 步骤1
    step1_start_time = time.time()
    vi.main(path)
    step1_end_time = time.time()
    step1_elapsed_time = step1_end_time - step1_start_time
    print("步骤1执行时间: ", step1_elapsed_time, "秒")

    # 步骤2
    step2_start_time = time.time()
    rc.main(path)
    step2_end_time = time.time()
    step2_elapsed_time = step2_end_time - step2_start_time
    print("步骤2执行时间: ", step2_elapsed_time, "秒")

    # 步骤3
    step3_start_time = time.time()
    hp.main(path)
    step3_end_time = time.time()
    step3_elapsed_time = step3_end_time - step3_start_time
    print("步骤3执行时间: ", step3_elapsed_time, "秒")

    # 步骤4
    step4_start_time = time.time()
    wl = np.loadtxt(os.path.join('/Volumes', 'HyperSpec', '50_target_resample.txt'))[:, 0]
    resample(path, wl)
    step4_end_time = time.time()
    step4_elapsed_time = step4_end_time - step4_start_time
    print("步骤4执行时间: ", step4_elapsed_time, "秒")

    # 步骤5
    step5_start_time = time.time()
    test.main(path)
    step5_end_time = time.time()
    step5_elapsed_time = step5_end_time - step5_start_time
    print("步骤5执行时间: ", step5_elapsed_time, "秒")

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time # 计算整个函数执行时间
    print("函数执行时间: ", elapsed_time, "秒")


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

    paths = ["2022_7_5_sunny", ]
    # paths = ["2022_7_5_sunny", "2022_7_9_cloudy", "2022_7_12_sunny",
    #          "2022_7_13_cloudy", "2022_7_16_sunny", "2022_7_20_sunny",
    #          "2022_7_23_sunny", "2022_7_27_sunny", "2022_8_2_sunny",
    #          "2022_8_9_cloudy", "2022_8_13_cloudy", "2022_8_14_sunny",
    #          "2022_8_16_sunny", "2022_8_20_sunny", "2022_8_24_cloudy"]

    for i in tqdm(range(len(paths))):
        if i < 9:
            main(os.path.join(disk1, paths[i]))
        else:
            main(os.path.join(disk2, paths[i]))