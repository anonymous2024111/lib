import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import sys
sys.path.append('./eva100/end2end/gat_no_pre')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from mydgl import test_dgl
from mypyg import test_pyg
from mgat_csr import test_mgat_csr
from mgat32_csr import test_mgat32_csr
#DGL
def dglGCN(data, csv_file, epoches, head, num_layers, featuredim, hidden, classes):
    spmm = test_dgl.test(data, epoches, head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-dgl-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)   
    
    
#MGCN
def mGCN16(data, csv_file, epoches,head, num_layers, featuredim, hidden, classes):
    spmm = test_mgat_csr.test(data, epoches, head,num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgat-fp16-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  

#MGCN-tf32
def mGCN32(data, csv_file, epoches, head, num_layers, featuredim, hidden, classes):
    spmm= test_mgat32_csr.test(data, epoches, head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgat-tf32-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  
    
#MGPYG
def pygGCN(data, csv_file, epoches,head, num_layers, featuredim, hidden, classes):
    spmm = test_pyg.test(data, epoches,head, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-pyg-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  



if __name__ == "__main__":


    dataset_I = ['Coauthor_Physics',  'FacebookPagePage', 'email-Enron', 'loc-Brightkite', 'soc-Epinions1', 
                 'HR_NO', 'HU_NO', 'GitHub', 'artist', 'blog']

    dataset_II = ['ell',  'com-DBLP', 'Reddit2', 'amazon', 'amazon0505', 
                  'dd', 'yelp', 'comamazon', 'roadNet-CA', 'roadNet-PA', 
                  'roadNet-TX', 'yeast']
    dataset_III = ['reddit', 'ogb', 'AmazonProducts']

    dataset = dataset_I + dataset_II

    dataset_II_pyg =['ell',  'com-DBLP', 'amazon0505', 
                  'dd',  'comamazon', 'roadNet-PA']
    dataset_pyg = dataset_I + dataset_II_pyg
    epoches = 300
    layer = [2, 4]
    hidden = [64, 128, 256]
    layer = [4]
    hidden = [64, 128, 256]

    head = 2
    
    
    
    featuredim = 512
    classes = 10

 
    # #DGL
    # filename = './eva100/end2end/gat_no_pre/result/dgl-v1.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    # for layer_num in layer:
    #     for hidden_num in hidden:
    #         for data in dataset:
    #             dglGCN(data, filename, epoches, head, layer_num, featuredim, hidden_num, classes)
    # print('DGL-' + 'success')
                
    # #MGCN-fp16
    # filename = './eva100/end2end/gat_no_pre/result/mgcn16-v1.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    # for layer_num in layer:
    #     for hidden_num in hidden:
    #         for data in dataset:
    #             mGCN16(data, filename, epoches, head, layer_num, featuredim, hidden_num, classes)
    # print('MGCN-fp16-' + 'success')            
 
    # #MGCN-tf32   
    # filename = './eva100/end2end/gat_no_pre/result/mgcn32-v1.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    # for layer_num in layer:
    #     for hidden_num in hidden:
    #         for data in dataset:
    #             mGCN32(data, filename, epoches, head, layer_num, featuredim, hidden_num, classes)
    # print('MGCN-tf32-' + 'success')
 
    # #PYG
    filename = './eva100/end2end/gat_no_pre/result/pyg-v1.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            if layer_num==4 and hidden_num==256:
                for data in dataset_pyg:
                    pygGCN(data, filename, epoches, head, layer_num, featuredim, hidden_num, classes)  
            elif layer_num==2 and hidden_num==256:
                for data in dataset:
                    if data == 'roadNet-CA':
                        continue
                    pygGCN(data, filename, epoches, head, layer_num, featuredim, hidden_num, classes)  
            else:
                for data in dataset:
                    if data == 'Reddit2' or data == 'yelp':
                        continue
                    pygGCN(data, filename, epoches, head, layer_num, featuredim, hidden_num, classes)
    print('Pyg-' + 'success')

    # # print('MGAT_small_all success')