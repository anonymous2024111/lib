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
dataset = dataset_I + dataset_II + dataset_III
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
with open('./eva100/end2end/agnn_no_pre/result/dgl-v1.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        dgl[row[0]][row[1]] = float(row[2])



# # pyg
with open('./eva100/end2end/agnn_no_pre/result/pyg-v1.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        if row[1] == '6-256':
            continue
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('pyg')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
        
# # tcgnn
with open('./eva100/end2end/agnn_no_pre/result/tcgnn-v1.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        if row[1] == '6-256':
            continue
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('tcgnn')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
        
        
# 开始读取每个baseline的文件，然后计算加速比
# MGCNtf32
with open('./eva100/end2end/agnn_no_pre/result/mgcn32-v1-1.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        if row[1] == '6-256':
            continue
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('mgcn32')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
    
# MGCNfp16
with open('./eva100/end2end/agnn_no_pre/result/mgcn16-v1-1.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)  # 跳过第一行
    for cur, row in enumerate(reader, start=0):
        if row[1] == '6-256':
            continue
        speedup['layer-hidden'].append(row[1])
        speedup['baseline'].append('mgcn16')
        speedup['speedup'].append(round( (dgl[row[0]][row[1]]/float(row[2])) , 2) )
        speedup['dataset'].append(row[0])
        


        

df = pd.DataFrame(speedup)
mgcn32 = df.loc[df['baseline'] == 'mgcn32', 'speedup'].mean()
mgcn16 = df.loc[df['baseline'] == 'mgcn16', 'speedup'].mean()
mgcn32_max = df.loc[df['baseline'] == 'mgcn32', 'speedup'].max()
mgcn16_max = df.loc[df['baseline'] == 'mgcn16', 'speedup'].max()
print("magnn32:" + str(round((mgcn32),2)))
print("magnn16:" + str(round((mgcn16),2)))
print("magnn32-max:" + str(round((mgcn32_max),2)))
print("magnn16-max:" + str(round((mgcn16_max),2)))
print("avg: " + str(round(((mgcn32+mgcn16)/2),2)))


