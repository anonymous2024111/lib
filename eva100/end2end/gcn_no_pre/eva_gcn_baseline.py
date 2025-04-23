import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('./eva100/end2end/gcn_no_pre')
from advisor import test_advisor
from mydgl import test_dgl
from mypyg import test_pyg
from mgcn import test_mgcn
from tcgnn import test_tcgnn
from mgcn32 import test_mgcn32
#DGL
def dglGCN(data, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_dgl.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-dgl-' + '-' + str(spmm))
    return spmm
    
#Tcgnn
def tcgnn(data, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_tcgnn.test(data, epoches, num_layers, featuredim, hidden, classes)
    return spmm

    
#MGPYG
def pygGCN(data, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_pyg.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-pyg-' + '-' + str(spmm))
    return spmm


if __name__ == "__main__":


    dataset_I = ['Coauthor_Physics',  'FacebookPagePage', 'email-Enron', 'loc-Brightkite', 'soc-Epinions1', 
                 'HR_NO', 'HU_NO', 'GitHub', 'artist', 'blog']

    dataset_II = ['ell',  'com-DBLP', 'Reddit2', 'amazon', 'amazon0505', 
                  'dd', 'yelp', 'comamazon', 'roadNet-CA', 'roadNet-PA', 
                  'roadNet-TX', 'yeast', 'DGraphFin', ]
    dataset_III = ['reddit', 'ogb', 'AmazonProducts']
    
    dataset =['IGB_small', 'reddit', 'ogb',  'GitHub', 'artist', 'blog', 'yeast', 'pubmed', 'ppi']
    dataset = ['pubmed', 'Reddit', 'IGB_small']

    layer = 3
    hidden = 128
    epoches = 100
    featuredim = 512
    classes = 10

    #用于存储结果
    file_name = '/home/shijinliang/module/Libra/eva100/end2end/gcn_no_pre/result/baseline.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'dgl', 'tcgnn', 'pyg']
    # 写入数据到 CSV 文件
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)
        
    for data in dataset:
        res = []
        #DGL
        dgl_time = dglGCN(data, epoches, layer, featuredim, hidden, classes)
        res.append(dgl_time)


        #TCGNN
        tcgnn_time = tcgnn(data, epoches, layer, featuredim, hidden, classes)
        res.append(tcgnn_time)


        #PYG
        pyg_time = pygGCN(data, epoches, layer, featuredim, hidden, classes)
        res.append(pyg_time)

        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res)
        print(data + 'is success')