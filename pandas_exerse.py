# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
# Series(系列)是 带有索引的一维数组 的生成器,特点是自动对齐索引与数值
obj = Series([213, 1231, 645, 34, 123, 7564])
print obj.index, obj.values

index = ['stand', 'up', 'sit', 'down', 'dog', 'cat']
values = [12, 23, 345, 56, 1, 2]
obj = Series(values, index)
print obj
data = {'sad': 122, 'happy': 1234, 'dog': 9897, 'cat': 1999}
obj = pd.Series(data)
print obj

# DataFrame是形成表格式数组的生成器，可自动对齐索引与数值，生成横向索引
# DataFrame的常见输入形式及索引
data = {'state': ['Hubei', 'Hubei', 'Zhejiang', 'Jiangsu', 'Henan'],
        'year': [2001, 2002, 2001, 2001, 2001],
        'pop': [3.4, 3.6, 12.7, 16.6, 83.4]
        }
frame = DataFrame(data, index=['one', 'two', 'three', 'four', 'five'])
print frame, frame['pop'], frame.ix('two')


frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['北京', '上海', '深圳'])
# frame的name属性
frame.name = 'population'
frame.columns.name = '城市'
frame.index.name = '房价:百万元/每平'
# frame的reindex修改columns（列）,index（索引）,method(插值：ffill,bfill)
# reindex函数的参数：index,method,fill_value,limit,level,copy
frame = frame.reindex(index=['一环', '二环', '三环','四环'])
frame['北京'] = 8.9
frame['上海'] = 7.4
frame['深圳'] = 8.4
print frame, frame.T
frame['北京'] = range(4, 8)
frame['上海'] = range(3, 7)
frame["深圳"] = range(5, 9)
frame2 = frame.reindex(columns=['北京', '上海', '深圳', '武汉', '广州'])
print frame2
# 丢弃指定轴上的项
frame2 = frame2.drop(['三环', '武汉'], axis=1)
# 索引，选取，过滤