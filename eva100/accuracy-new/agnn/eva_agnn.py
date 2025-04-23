import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import csv
import sys
import os
from mydgl import test_dgl
from mypyg import test_pyg
from fsagnn16 import test_magnn
from fsagnn32 import test_magnn32
    
#PYG
def pygGCN(data, data_path, epoches,  num_layers, hidden):
    acc = test_pyg.test( data_path, epoches,num_layers, hidden)
    return acc
    
#DGL
def dglGCN(data, data_path, epoches,  num_layers, hidden):
    acc = test_dgl.test( data_path, epoches,num_layers, hidden)
    return acc

#GCNtf32
def GCN32(data, data_path, epoches, num_layers, hidden):
    acc = test_magnn32.test( data_path, epoches,num_layers, hidden)
    return acc
   
        
#GCNfp16
def GCN(data, data_path, epoches, num_layers, hidden):
    acc = test_magnn.test( data_path, epoches,num_layers, hidden)
    return acc

if __name__ == "__main__":

    spmm = dict()
    base_spmm = dict()
    dataset = [ 'cora', 'pubmed', 'question', 'min', 'texas', 'wiki', 'wis']
    dataset = [ 'pubmed']
    load = False
    epoches = 300
    num_layers = 5
    hidden = 100
    current_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    file_name = project_dir + '/result/Baseline/gcn/accuracy.csv'
    head = ['dataSet', 'PyG', 'DGL', 'FlashSparse-FP16', 'FlashSparse-TF32']

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)

    for data in dataset:
        
        res_acc = []
        res_acc.append(data)
        data_path =  project_dir + '/dataset/' + data +'.npz'
        
        # # #PYG
        # acc = pygGCN(data, data_path, epoches, num_layers, hidden)
        # res_acc.append(acc)
        
        # # #DGL
        # acc = dglGCN(data, data_path, epoches, num_layers, hidden)
        # res_acc.append(acc)
        
        #FlashSparse GCN-fp16
        acc = GCN(data, data_path, epoches, num_layers, hidden)
        res_acc.append(acc)
                
        # #FlashSparse GCN-tf32
        # acc = GCN32(data, data_path, epoches, num_layers, hidden)
        # res_acc.append(acc)

        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_acc)
        print(data + "success")
        print()
    
