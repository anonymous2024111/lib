import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv


dataset_I = ['Coauthor_Physics',  'FacebookPagePage', 'email-Enron', 'loc-Brightkite', 'soc-Epinions1', 
                'HR_NO', 'HU_NO', 'GitHub', 'artist', 'blog']

dataset_II = ['ell',  'com-DBLP', 'Reddit2', 'amazon', 'amazon0505', 
                'dd', 'yelp', 'comamazon', 'roadNet-CA', 'roadNet-PA', 
                'roadNet-TX', 'yeast', 'DGraphFin', ]
dataset_III = ['reddit', 'ogb', 'AmazonProducts']

dataset = dataset_I + dataset_II

speedup = dict()
speedup['layer-hidden'] = []
speedup['baseline'] = []
speedup['speedup'] = []
speedup['dataset'] = []

# 首先读DGL的结果，用于计算加速比
dgl = dict()
for data in dataset:
    dgl[data] = dict()


fp16_data = set()
tf32_data = set()

# DGL
with open('./eva100/end2end/gat_no_pre/result/dgl-v1.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        dgl[row[0]][row[1]] = float(row[2])


# PYG
with open('./eva100/end2end/gat_no_pre/result/pyg-v1.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('pyg')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
        if float(row[2]) < dgl[row[0]][row[1]] : 
            fp16_data.add(row[0])
            
# 开始读取每个baseline的文件，然后计算加速比
# MGCNtf32
with open('./eva100/end2end/gat_no_pre/result/mgcn32-v1.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('mgcn32')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
        if float(row[2]) < dgl[row[0]][row[1]] : 
            tf32_data.add(row[0])

    
# MGCNfp16
with open('./eva100/end2end/gat_no_pre/result/mgcn16-v1.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('mgcn16')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
        if float(row[2]) < dgl[row[0]][row[1]] : 
            fp16_data.add(row[0])


df = pd.DataFrame(speedup)

mycolor = {'advisor':'coral', 'tcgnn':'limegreen', 
           'mgcn32':'cornflowerblue', 'mgcn16':'royalblue', 'pyg':'moccasin'}

# 对数据按照 'dim' 列进行升序排序
sns.set_style("darkgrid")
# 设置背景色为灰白色

g = sns.boxplot(x='layer-hidden', y='speedup', hue='baseline', data=df, 
                palette=mycolor, linewidth=0.5, legend=False, gap=0.2, width=0.5)
plt.axhline(y=1, color='blue', linestyle='--')
g.set_ylabel('')
sns.set_style("white")

# 显示图形
plt.savefig('./eva100/end2end/gat_no_pre/result/gat_end_800.png', dpi=800)
# 清空图形
plt.clf()


