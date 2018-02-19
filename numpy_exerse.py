# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

# numpy的基础：数组与矢量计算
    # NumPy的ndarray：一种多维数组对象
        # 创建ndarray:
arr1 = np.array([[12,243,231,1,89,3],[1,2,3,4,5,6]])
arr2 = np.arange(6)
arr4 = np.eye(3)
        # ndarray的数据类型:
            # int8,unit8,16,32,64
            # float,16,32,64,128
            # complex64,128,256
            # bool
arr5 = arr1.astype(np.float64)
arr5.dtype
        # 数组与标量计算：
arr1+arr1, arr1*12, arr1*arr1, arr1/12, 1/arr1
            # 基本的索引与切片：
arr2[3]  # 索引
arr2[2:6]  # 切片
arr2[3:4] = 12  # 切片添加，切片为源数组的视图，视图的变化直接反映到源数组上
arr1[1][2]  # 二维索引
arr1[1:, :3]  # 二维切片
            # 布尔型索引：
words = np.array(['kik', 'asd', 'chicken', 'duck', 'name', 'die', 'chocalate', 'pig', 'FFF'])
arr4[arr4 < 2] = 11
            # 花式索引：(花式索引是Numpy的一个术语，是利用整数进行索引,花式索引与切片不同，它是将数据复制到新数组中)
arr3 = np.empty((9, 9))
for i in range(9):
    arr3[i] = i

arr3[[4, 3, 3, 0, 5]]
            # 数组对换与轴对换：
arr2.T
        # 通用函数：快速的元素级数组函数
    # 利用数组进行数据处理：

points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs**2 + ys**2)
plt.imshow(z, cmap=plt.cm.Blues); plt.colorbar()
plt.title('What do u think of?')
        # 将条件逻辑表述为数组计算：
xarr = np.array([1.1, 1.2, 1.3, 1.2, 1.7, 1.9])
yarr = np.array([2.8, 11, -9.6, 2323, -987541, 76.678678])
cond = np.array([True, False, True, False, False, True])
result = [(x if c else y)
            for x, y, c in zip(xarr, yarr, cond)]
# 等同于：
result = np.where(cond, xarr, yarr)

arr = randn(4, 4)
arr.sum(), arr.mean(), arr.std(), arr.var(), arr.min(), arr.max()
arr.argmax(), arr.argmin(), arr.cumsum(), arr.cumprod()
            # 排序：
arr.sort()  # 从小到大排序，可传入轴编号
        # 用于数组的文件输入输出（可读入二进制文件和文本文件）：
np.save('numpy_ex', arr=arr)        # 数组以未压缩的原始二进制格式保存到原始的*.npy文件中
np.load('numpy_ex.npy')  # 路径查找此文件
        # 存取文本文件
arr = np.loadtxt('array_ex.txt', delimiter=',')  # 数组从txt文件读出
np.savetxt('array_ex.txt', arr)  # 数组写入txt文件
    # 线代
    # 随机数的生成及随机漫步

