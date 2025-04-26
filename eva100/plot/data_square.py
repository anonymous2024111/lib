import csv
import pandas as pd
import time
import os
import numpy as np
#读取csv遍历每个数据集
df = pd.read_csv('/home/shijinliang/module/Libra/data.csv')
#用于存储结果
file_name = '/home/shijinliang/module/Libra/eva100/plot/data_square.csv'
head = ['dataSet', 'num_nodes', 'num_edges']
with open(file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(head)
for index, row in df.iterrows():
    data = row.iloc[0]   
    graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/sp_matrix/'+ data +'.npz')
    num_nodes_ori =  graph['num_nodes_src']-0
    num_nodes_dst =  graph['num_nodes_dst']-0
    num_edges = graph['num_edges']-0
    temp = []
    if num_nodes_ori==num_nodes_dst and num_edges>20000:
        temp.append(data)
        temp.append(num_nodes_ori)
        temp.append(num_edges)
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(temp)
    