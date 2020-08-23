# -*- coding:gb2312 -*-
# -*- coding:UTF-8 -*-
# @Time     :2019 11 2019/11/19 17:18
# @Author   :千乘

import matplotlib.pylab as plt
import numpy as np


# 构建函数：检测特征对回归器的贡献
def plot_feature_importances(feature_importances, feature_names, color='red', title=None):
    # 重要性值标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # 得分高低排序
    index_sorted = np.flipud(np.argsort(feature_importances))
    # x轴标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # 画条形图
    plt.figure(figsize=(20, 6))
    plt.bar(pos, feature_importances[index_sorted], align='center', color=color)
    plt.xticks(pos, feature_names[index_sorted])
    plt.xticks(rotation=-15)  # 设置x轴标签旋转角度
    plt.ylabel("Relative Importance")
    plt.title(title)
    plt.show()


# 构建混淆矩阵图像函数
def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Paired)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# 构建二维分类图像函数
def plot_classifier(classifier, X, y):
    # 定义图形的取值范围，在最大值，最小值的基础上增加余量(buffer)
    X_min, X_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 0]) + 1.0

    # 设置网格(grid)数据画出边界
    # 设置网格步长
    step_size = 0.01

    # 定义网格
    X_values, y_values = np.meshgrid(np.arange(X_min, X_max, step_size), np.arange(y_min, y_max, step_size))

    # 计算分类器的输出结果
    mesh_output = classifier.predict(np.c_[X_values.ravel(), y_values.ravel()])

    # 数组维度变形
    mesh_output = mesh_output.reshape(X_values.shape)

    # 作图
    plt.figure()
    plt.pcolormesh(X_values, y_values, mesh_output, cmap=plt.cm.gray)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    plt.xlim(X_values.min(), X_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # 设置X轴，Y轴
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))

    plt.show()


# 绘制学习/验证曲线
def plot_curve(parameter_grid, train_scores):
    plt.figure()
    plt.plot(parameter_grid, 100 * np.average(train_scores, axis=1))
    plt.title('Curve')
    plt.xlabel('Numbers of Training Samples')
    plt.ylabel('Accuracy')
    plt.show()


def plot_cluster(cluster, data):
    # 定义图形的取值范围，在最大值，最小值的基础上增加余量(buffer)
    X_min, X_max = min(data[:, 0]) - 1.0, max(data[:, 0]) + 1.0
    y_min, y_max = min(data[:, 1]) - 1.0, max(data[:, 0]) + 1.0

    # 设置网格(grid)数据画出边界
    # 设置网格步长
    step_size = 0.01

    # 定义网格
    X_values, y_values = np.meshgrid(np.arange(X_min, X_max, step_size), np.arange(y_min, y_max, step_size))

    # 预测网格中所有数据点的标记
    mesh_output = cluster.predict(np.c_[X_values.ravel(), y_values.ravel()])

    # 数组维度变形
    mesh_output = mesh_output.reshape(X_values.shape)

    # 作图描绘网格
    plt.figure()
    plt.clf()
    plt.imshow(mesh_output, interpolation='nearest',
               extent=(X_values.min(), X_values.max(), y_values.min(), y_values.max())
               , cmap=plt.cm.Paired, aspect='auto', origin='lower')
    plt.scatter(data[:, 0], data[:, 1], marker='o', facecolors='none', edgecolors='k', s=30)
    # 作图描绘点
    centroids = cluster.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, color='k', zorder=10, linewidth=1)

    # 设置X轴，Y轴
    plt.xlim(X_min, X_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
