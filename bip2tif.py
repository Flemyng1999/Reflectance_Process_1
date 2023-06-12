import os
import sys

from tqdm import tqdm

import tiff_tool as tt


def main(path: str):
    ref, geo, trans = tt.read_tif(os.path.join(path, '5ref', 'ref.bip'))
    tt.write_tif(os.path.join(path, '5ref', 'ref.tif'), ref, geo, trans)
    del ref, geo, trans


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