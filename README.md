# Reflectance_Process_1

对ref.bip进行一些初步分析

**以下所有计算都是基于TOC反射率，未排除BRDF和冠层结构等因素的影响：**

1. 计算各种VIs
2. 分割出植被（阴阳叶也可以）
3. 制作出ROI蒙版，用于下面的ROI分析
4. 数据初步预处理，排除3$\sigma$之外的数据
5. 分析ROI内的数据分布
6. 将VI的空间分布展示出来