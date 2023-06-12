import os

import numpy as np
import matplotlib.pyplot as plt

import tiff_tool as tt

path = os.path.join('/Volumes', 'HyperSpec', '2022_7_5_sunny', 'roi', 'roi.tif')
roi = tt.read_tif_array(path)
roi_sum = np.sum(roi, axis=0)

# 画图
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=1, dpi=200)

ax.imshow(roi_sum, cmap='gray')

plt.show()
plt.close()

print(roi.shape)