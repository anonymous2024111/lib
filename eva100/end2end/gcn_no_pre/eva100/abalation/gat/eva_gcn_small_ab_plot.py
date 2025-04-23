import torch
import numpy as np
from scipy.sparse import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('eva100/abalation/gcn/')
from eva100.abalation.gat.mtest_gat import *
from eva100.abalation.gat.mdataset_gat import *
import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
def list_gen() :
    # 声明一个空的3维列表
    my_list = []
    # 定义列表的维度大小
    dim1 = 3  # 第一维度大小
    # 使用嵌套循环创建空的三维列表
    for i in range(dim1):
        # 创建第一维度的空列表
        sublist1 = []
        my_list.append(sublist1)
    return my_list
dataset = ['Reddit2', 'ovcar', 'amazon','amazon0505',
            'yelp', 'sw620', 'dd',
            'HR_NO', 'HU_NO', 'ell', 'GitHub',
            'artist', 'comamazon', 
            'yeast', 'blog', 'DGraphFin', 'reddit', 'ogb', 'AmazonProducts', 'IGB_medium']
subdataset = ['blog', 'amazon','reddit', 'IGB_medium','AmazonProducts']
hugedataset = ['reddit', 'IGB_medium','AmazonProducts']
hidden = [64, 128, 256, 512]
subhidden = ['256', '512']
result = dict()
result['dataset'] = []
result['time'] = []
result['type'] = []

temp = list_gen()

dim = dict()
cur_row=0
with open('./eva100/abalation/gcn/fp16-H100.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for row in reader:
        if row[0] in subdataset and row[1] in subhidden:
            for i in range(3):
                result['dataset'].append(row[0]+'-'+str(row[1]))
                if row[0] in hugedataset:
                    result['time'].append(float(row[2+i])/20)
                else: result['time'].append(float(row[2+i]))
                result['type'].append(i)
        cur_row+=1


df = pd.DataFrame(result)
# print(df)
sns.set(rc={'figure.figsize':(20, 6)})
# 使用Seaborn绘制带有hue参数的条形图
g = sns.barplot(x='dataset', y='time', hue='type', data=df, palette="Blues_d", linewidth=1, legend=False)
#在每个柱子上显示值
#a 为数据集的个数 ， b为hidden的个数， c为每组3个柱子
a = 5
b = 2
c=3
for i, p in enumerate(g.patches):
    if i<(a*b)*c and (i//(a*b))!=0:
        if (i//(a*b))==1 :
            pre = g.patches[i-(a*b)].get_height()
            g.annotate(format(p.get_height()-pre, '.2f'), (p.get_x(), p.get_height()), ha = 'center', va = 'center', xytext = (0, 8), textcoords = 'offset points', fontsize=8)
        if (i//(a*b))==2 :
            pre = g.patches[i-(a*b)].get_height()
            g.annotate(format(p.get_height()-pre, '.2f'), (p.get_x(), p.get_height()), ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points', fontsize=8)

plt.xticks(rotation=20)
g.tick_params(labelsize=8)


plt.savefig('./eva100/abalation/gcn/fp16-H100.png', dpi=800)
