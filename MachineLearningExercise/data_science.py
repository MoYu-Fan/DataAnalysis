#!/usr/bin/env python
# coding: utf-8

# # 本代码用于学习python的数据分析，数据挖掘基础知识，是对学习的一种记录，初次学习内容为实验楼免费课。

# In[1]:


# -*- coding: utf-8 -*-

# 数据分析基本库
import numpy as np
import pandas as pd
# 绘图库
import matplotlib.pylab as plt
import seaborn as sns
import pydotplus
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas_datareader.data as pdr
import datetime
import yfinance as yf

'''
  一个好用的数据csv https://labfile.oss.aliyuncs.com/courses/906/los_census.csv"
 人口普查数据 'https://labfile.oss.aliyuncs.com/courses/1283/adult.data.csv'
 电信用户离网数据 'https://labfile.oss.aliyuncs.com/courses/1283/telecom_churn.csv'
'''

# # pandas去省略
# '''
# #显示所有列
# pd.set_option('display.max_columns', None)
# #显示所有行
# pd.set_option('display.max_rows', None)
# #设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth',100)
# '''

# # 尝试一：pandas数据探索

# In[2]:


df1 = pd.read_csv("https://labfile.oss.aliyuncs.com/courses/906/los_census.csv")
print(df1)

# In[3]:

print(df1.describe())
# In[4]:

# 透视表的制作

df1 = pd.read_csv("https://labfile.oss.aliyuncs.com/courses/906/los_census.csv")
print(df1)
table = pd.pivot_table(df1, index=['Zip Code'])
table = table.T
# In[5]:


# 可视化尝试
plt.bar(np.arange(319), table.iloc[2])
plt.bar(np.arange(319), table.iloc[5], color='Blue')

# # 数据预览及预处理

# In[6]:


df2 = pd.read_csv("https://labfile.oss.aliyuncs.com/courses/1283/adult.data.csv")
df2.describe()

# In[7]:


df2.info()

# In[8]:


# 问题：数据集中有多少男性和女性？
df2['sex'].value_counts()

# In[9]:


# 问题：数据集中女性的平均年龄是多少？
df2[df2['sex'] == 'Female']['age'].mean()

# In[10]:


# 问题:数据集中德国公民的比例是多少？
print(float((df2['native-country'] == 'Germany').sum()) / df2.shape[0])
df2['native-country'].value_counts(normalize=True)

# In[11]:


# 使用 groupby 和 describe 统计不同种族和性别人群的年龄分布数据
df2.groupby(by=['sex', 'race'])['age'].describe()

# In[12]:


# 问题：计算各国超过和低于 50K 人群各自的平均周工作时长。
df2.groupby(by=['salary', 'native-country'])['hours-per-week'].mean()

# 尝试二：学习数据可视化


df3 = pd.read_csv('https://labfile.oss.aliyuncs.com/courses/1283/telecom_churn.csv')
print(df3)

sns.distplot(df3['Total day charge'])  # 同时显示直方图和密度分布图

feature = ['Total eve calls', 'Total night calls']
df3[feature].hist(figsize=(10, 4))  # 直方图

df3[feature].plot(kind='density', subplots=True, layout=(1, 2),
                  sharex=False, figsize=(10, 4), legend=False, title=feature)  # 密度分布图

# 箱型图 ,疑惑？？？箱形图显示了单独样本的特定统计数据
sns.boxplot(x='Total intl calls', data=df3)

# 提琴型图？
_, axes = plt.subplots(1, 2, figsize=(6, 4))
sns.boxplot(data=df3['Total intl calls'], ax=axes[0])
sns.violinplot(data=df3['Total intl calls'], ax=axes[1])
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
sns.countplot(x='Churn', data=df3, ax=axes[0])
sns.countplot(x='Customer service calls', data=df3, ax=axes[1])

# 多变量可视化 不理解 ！！！T-T 流泪

plt.scatter(df3['Total day minutes'], df3['Total night minutes'])
sns.jointplot(x='Total day minutes', y='Total night minutes', data=df3, kind='scatter')
sns.jointplot('Total day minutes', 'Total night minutes', data=df3, kind='kde', color='g')

numerical = list(set(df3.columns) - {'State', 'International plan', 'Voice mail plan', 'Area code', 'Churn',
                                     'Customer service calls'})
# 计算和绘图
corr_matrix = df3[numerical].corr()
sns.heatmap(corr_matrix)
rical = list(set(numerical) - {'Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge'})
sns.pairplot(df3[numerical])
sns.lmplot('Total day minutes', 'Total night minutes', data=df3, hue='Churn', fit_reg=False)
plt.show()

# 机器学习


data = pd.read_csv('https://labfile.oss.aliyuncs.com/courses/1283/telecom_churn.csv')
print(data.head())
data['International plan'] = pd.factorize(data['International plan'])[0]
data['Voice mail plan'] = pd.factorize(data['Voice mail plan'])[0]
data['Churn'] = data['Churn'].astype('int')
states = data['State']
y = data['Churn']
data.drop(['State', 'Churn'], axis=1, inplace=True)
print(data.describe())

X_train, X_holdout, y_train, y_holdout = train_test_split(data.values, y, test_size=0.3, random_state=25)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_holdout)
accuracy_score(y_holdout, knn_pred)
yf.pdr_override()

# 载入数据
start = datetime.datetime(2006, 10, 1)
end = datetime.datetime(2012, 1, 1)
pdr.get_data_yahoo('AAPL', start, end)
