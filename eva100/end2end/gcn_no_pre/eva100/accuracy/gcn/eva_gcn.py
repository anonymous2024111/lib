import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('./eva100/accuracy/gcn')
from mydgl import test_dgl
from mypyg import test_pyg
from mgcn import test_mgcn
from mgcn32 import test_mgcn32
    
#PYG
def pygGCN(data, file, epoches,  num_layers, hidden):
    res = [data]
    exe = test_pyg.test(data, epoches,num_layers, hidden)
    res.append('pyg')
    res.append(str(exe))
    with open(file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)       
#DGL
def dglGCN(data, file, epoches,  num_layers, hidden):
    res = [data]
    exe = test_dgl.test(data, epoches,num_layers, hidden)
    res.append('dgl')
    res.append(str(exe))
    with open(file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)    

#MGCNtf32
def mGCN32(data, file, epoches, num_layers, hidden):
    res = [data]
    exe = test_mgcn32.test(data, epoches,num_layers, hidden)
    res.append('Magicsphere-tf32')
    res.append(str(exe))
    with open(file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)     
#MGCN
def mGCN(data, file, epoches, num_layers, hidden):
    res = [data]
    exe = test_mgcn.test(data, epoches,num_layers, hidden)
    res.append('Magicsphere-fp16')
    res.append(str(exe))
    with open(file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)    
    

    



if __name__ == "__main__":
    
    #用于统计4种算法的block数
    spmm = dict()
    base_spmm = dict()
    dataset = [ 'cora', 'ell', 'min', 'pubmed', 'question', 'texas', 'wiki']
    load = False
    epoches = 300
    num_layers = 5
    hidden = 100
    filename  = './eva100/accuracy/gcn/result.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    #循环读入每个数据集
    for data in dataset:
        base_spmm.clear()
        path = './dgl_dataset/accuracy/' + data + '.npz'
        graph_obj = np.load(path)
        #PYG
        pygGCN(data, filename, epoches, num_layers, hidden)
        #DGL
        dglGCN(data, filename, epoches, num_layers, hidden)
        #MGCN
        mGCN(data, filename, epoches, num_layers, hidden)
        #MGCN32
        mGCN32(data, filename, epoches, num_layers, hidden)


    # plt.savefig('./eva100/accuracy/gcn/gcn-acc.png', dpi=800)
    print("success")
    
