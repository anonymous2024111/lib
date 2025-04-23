import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('./eva100/accuracy/gat')
from mydgl import test_dgl
from mypyg import test_pyg
from mgat import test_mgat
from mgat32 import test_mgat32
    
#PYG
def pygGCN(data, file, epoches, heads, hidden):
    res = [data]
    exe = test_pyg.test(data, epoches, heads, hidden)
    res.append('pyg')
    res.append(str(exe))
    with open(file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)       
#DGL
def dglGCN(data, file, epoches, heads, hidden):
    res = [data]
    exe = test_dgl.test(data, epoches, heads, hidden)
    res.append('dgl')
    res.append(str(exe))
    with open(file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)          

#MGAT32
def mGAT32(data, file, epoches, heads, hidden):
    res = [data]
    exe = test_mgat32.test(data, epoches, heads, hidden)
    res.append('Magicsphere-tf32')
    res.append(str(exe))
    with open(file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)       

#MGAT
def mGAT(data, file, epoches, heads, hidden):
    res = [data]
    exe = test_mgat.test(data, epoches, heads, hidden)
    res.append('Magicsphere-fp16')
    res.append(str(exe))
    with open(file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)          
   



if __name__ == "__main__":
    
    dataset = ['cite', 'cora', 'cornell', 'ell', 'min', 'pubmed', 'question', 'texas']

    load = False
    epoches = 100
    filename  = './eva100/accuracy/gat/result.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for data in dataset:
        path = './dgl_dataset/accuracy/' + data + '.npz'
        graph_obj = np.load(path)
        head = 6
        # if data in smallData :
        #     head = 1
        #PYG
        pygGCN(data, filename, epoches, head, 16)
        #DGL
        dglGCN(data, filename, epoches, head, 16)
        #MGAT
        mGAT(data, filename, epoches, head, 16)
        #MGAT32
        mGAT32(data, filename, epoches, head, 16)
    print("success")