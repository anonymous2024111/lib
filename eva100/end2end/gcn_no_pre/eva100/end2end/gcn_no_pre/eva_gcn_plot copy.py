import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv

dataset_I = ['pubmed',  'Coauthor_Physics',  'Amazon_Computers',
            'DeezerEurope', 'FacebookPagePage']

dataset_II = ['roadNet-CA', 'roadNet-PA', 'roadNet-TX', 'com-DBLP', 'email-Enron',          
                'loc-Brightkite','soc-Epinions1','amazon', 'amazon0505', 'dd', 'ell', 
                'GitHub', 'artist', 'comamazon', 'yeast', 'blog', 'DGraphFin', 'HR_NO', 'HU_NO']
dataset_III = ['reddit', 'ogb', 'AmazonProducts']


dataset = dataset_I+dataset_II+dataset_III
# dataset_3 = ['dd', 'yelp']
# dataset_6 = ['dd', 'yelp']
hidden = [64, 128, 256]

speedup = dict()
speedup['layer-hidden'] = []
speedup['baseline'] = []
speedup['speedup'] = []
speedup['dataset'] = []

# 首先读DGL的结果，用于计算加速比
dgl = dict()
for data in dataset:
    dgl[data] = dict()

# DGL
with open('./eva100/end2end/gcn_no_pre/result/dgl-v2.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        dgl[row[0]][row[1]] = float(row[2])

# 开始读取每个baseline的文件，然后计算加速比



# Advisor
with open('./eva100/end2end/gcn_no_pre/result/advisor-v2.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('advisor')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])

#PYG
with open('./eva100/end2end/gcn_no_pre/result/pyg-v2.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('pyg')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
                
#TC-GNN
with open('./eva100/end2end/gcn_no_pre/result/tcgnn-v2.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('tcgnn')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
        
# MGCNtf32
with open('./eva100/end2end/gcn_no_pre/result/mgcn32-v2.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('mgcn32')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
    
# MGCNfp16
with open('./eva100/end2end/gcn_no_pre/result/mgcn16-v2.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('mgcn16')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
        

df = pd.DataFrame(speedup)

mycolor = {'advisor':'cornflowerblue', 'tcgnn':'lightgreen', 
           'mgcn32':'gold', 'mgcn16':'orange', 'pyg':'tomato'}
sns.set(rc={'figure.figsize':(12, 5)})
# sns.set(rc={"lines.linewidth": 0.5})
# 对数据按照 'dim' 列进行升序排序
sns.set_style("darkgrid")
# 设置背景色为灰白色

g = sns.boxplot(x='layer-hidden', y='speedup', hue='baseline', data=df, 
                palette=mycolor, linewidth=1, legend=False, gap=0.1, width=0.8)
plt.axhline(y=1, color='blue', linestyle='--', linewidth = 1.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.despine(left=True, right=True, top=True)
g.set_ylabel('')
g.set_xlabel('')

# 显示图形
plt.savefig('./eva100/end2end/gcn_no_pre/result/gcn_kernel_100.png', dpi=800)
# 清空图形
plt.clf()


