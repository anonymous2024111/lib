import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import sys
sys.path.append('./eva100/end2end/agnn_no_pre')

from mydgl import test_dgl
from mypyg import test_pyg
from magnn import test_magnn
from tcgnn import test_tcgnn
from magnn32 import test_magnn32
#DGL
def dglGCN(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_dgl.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-dgl-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)   
    
#Tcgnn
def tcgnn(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_tcgnn.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-tcgnn-' + '-' + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  
    
#MGCN
def mGCN16(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_magnn.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-magnn-fp16-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  

#MGCN-tf32
def mGCN32(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm= test_magnn32.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-magnn-tf32-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  
    
#MGPYG
def pygGCN(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_pyg.test(data, epoches, num_layers, featuredim, hidden, classes)
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
                  'roadNet-TX', 'yeast', 'DGraphFin']
    dataset_III = ['reddit', 'ogb']
    dataset = dataset_I + dataset_II + dataset_III
    dataset_6 = dataset_I + dataset_II
    dataset_pyg_outofmemory = ['blog', 'amazon0505', 'roadNet-CA', 'roadNet-PA', 
                  'roadNet-TX', 'yeast', 'DGraphFin']
    epoches = 300
    layer = [6]
    hidden = [64, 128, 256]
    

    # layer = [6]
    # hidden = [256]
    # dataset = ['ell',  'com-DBLP', 'Reddit2', 'amazon', 'amazon0505', 
    #               'dd', 'yelp', 'comamazon', 'roadNet-CA', 'roadNet-PA', 
    #               'roadNet-TX', 'yeast', 'DGraphFin']
    
    featuredim = 512
    classes = 10
    
    #DGL
    # filename = './eva100/end2end/agnn_no_pre/result/dgl-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    # for layer_num in layer:
    #     if layer_num == 6:
    #         for hidden_num in hidden:
    #             for data in dataset_6:
    #                 dglGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    #     else:
    #         for hidden_num in hidden:
    #             for data in dataset:
    #                 dglGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    # print('DGL-' + 'success')
                
    #MGCN-fp16
    # filename = './eva100/end2end/agnn_no_pre/result/mgcn16-v2-1.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    # for layer_num in layer:
    #     if layer_num == 6:
    #         for hidden_num in hidden:
    #             for data in dataset_6:
    #                 mGCN16(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    #     else:
    #         for hidden_num in hidden:
    #             for data in dataset:
    #                 mGCN16(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    # print('MAGNN-fp16-' + 'success')            
 
    #MGCN-tf32   
    filename = './eva100/end2end/agnn_no_pre/result/mgcn32-v2-1.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    for layer_num in layer:
        if layer_num == 6:
            for hidden_num in hidden:
                for data in dataset_6:
                    if data=='roadNet-CA':
                        continue
                    if data=='DGraphFin' and hidden_num==256:
                        continue
                    mGCN32(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
        else:
            for hidden_num in hidden:
                for data in dataset:
                    mGCN32(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('MAGNN-tf32-' + 'success')

    #TCGNN
    # filename = './eva100/end2end/agnn_no_pre/result/tcgnn-v2.csv'
    # # with open(filename, 'w') as file:
    # #     file.write('H100 : \n')
    # for layer_num in layer:
    #     if layer_num == 6:
    #         for hidden_num in hidden:
    #             for data in dataset_6:
    #                 if data=='AmazonProducts' or data=='reddit' or data=='DGraphFin':
    #                     continue
    #                 tcgnn(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    #     else:
    #         for hidden_num in hidden:
    #             for data in dataset:
    #                 if data in ['AmazonProducts', 'reddit', 'soc-Epinions1']:
    #                     continue
    #                 tcgnn(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    # print('TCGNN-' + 'success')
    
    ##PYG
    #filename = './eva100/end2end/agnn_no_pre/result/pyg-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    # for layer_num in layer:
    #     if layer_num == 6:
    #         for hidden_num in hidden:
    #             for data in dataset_6:
    #                 if data in ['Reddit2', 'yelp', 'DGraphFin', 'roadNet-CA', 'amazon']:
    #                     continue
    #                 if data in dataset_pyg_outofmemory and hidden_num==256:
    #                     continue
    #                 pygGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    #     else:
    #         for hidden_num in hidden:
    #             for data in dataset_6:
    #                 if data in ['Reddit2', 'yelp', 'DGraphFin', 'roadNet-CA', 'amazon']:
    #                     continue
    #                 pygGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    # print('Pyg-' + 'success')

    print('MAGNN_all success')