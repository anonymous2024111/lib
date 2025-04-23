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




hidden = [64, 128, 256]

speedup = dict()
speedup['layer-hidden'] = []
speedup['baseline'] = []
speedup['speedup'] = []
speedup['dataset'] = []
dataset = dataset_I + dataset_II+dataset_III
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

# PYG
with open('./eva100/end2end/gcn_no_pre/result/pyg-v2.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('pyg')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
                
# TC-GNN
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
mgcn32 = df.loc[df['baseline'] == 'mgcn32', 'speedup'].mean()
mgcn16 = df.loc[df['baseline'] == 'mgcn16', 'speedup'].mean()
mgcn32_max = df.loc[df['baseline'] == 'mgcn32', 'speedup'].max()
mgcn16_max = df.loc[df['baseline'] == 'mgcn16', 'speedup'].max()
print("mgcn32:" + str(round((mgcn32),2)))
print("mgcn16:" + str(round((mgcn16),2)))
print("mgcn32-max:" + str(round((mgcn32_max),2)))
print("mgcn16-max:" + str(round((mgcn16_max),2)))
print("avg: " + str(round(((mgcn32+mgcn16)/2),2)))
