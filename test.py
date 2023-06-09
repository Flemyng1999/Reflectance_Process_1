import os

import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import tiff_tool as tt


def calculate_ref_mean_in_roi(path: str, ref: np.ndarray, roi_names: List[str]) -> Dict[str, np.ndarray]:
    """
    计算ROI中每个波段上的参考数据平均值
    Args:
        path(str): 存放ROI和csv文件的路径
        ref(np.ndarray): 参考数据
        roi_names(List[str]): ROI名称列表
    Returns:
        ref_mean_dict(Dict[str, np.ndarray]): ROI中每个波段上的参考数据平均值
    """
    ref_mean_dict: Dict[str, np.ndarray] = {}

    for roi_name in tqdm(roi_names, desc='ROI进度'):
        roi_path = os.path.join(path, 'roi', roi_name + '.tif')

        mask = tt.read_tif_array(roi_path)
        ref_ = ref * mask
        flat_arr = np.moveaxis(ref_, 0, -1).reshape((-1, 150))

        nonzero_idx = np.nonzero(flat_arr)
        nonzero_wave_idx, nonzero_pixel_idx = nonzero_idx[1], nonzero_idx[0]

        nonzero_value = flat_arr[nonzero_pixel_idx]
        result = np.nanmean(nonzero_value, axis=0)
        ref_mean_dict[roi_name] = result

    return ref_mean_dict


path = os.path.join('/Volumes', 'HyperSpec', '2022_7_5_sunny')
ref = np.load(os.path.join(path, '5ref', 'ref_in_vege.npy'))

roi_names = ['w{}s{}'.format(w, s) for w in range(1, 5) for s in range(1, 6)]

ref_mean_dict = calculate_ref_mean_in_roi(path, ref, roi_names)

df = pd.DataFrame(ref_mean_dict)
df.to_csv(os.path.join(path, '5ref', 'ref_in_vege.csv'))