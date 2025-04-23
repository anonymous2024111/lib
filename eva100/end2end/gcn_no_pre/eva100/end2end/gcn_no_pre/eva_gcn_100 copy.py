import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import sys
sys.path.append('./eva100/end2end/gcn_no_pre')
from advisor import test_advisor
from mydgl import test_dgl
from mypyg import test_pyg
from mgcn import test_mgcn
from tcgnn import test_tcgnn
from mgcn32 import test_mgcn32
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
    
#Advisor
def advisor(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_advisor.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-advisor-' + '-' + str(spmm))
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
    spmm = test_mgcn.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgcn-fp16-'  + str(spmm))
    res = []
    res.append(data)
    res.append(str(num_layers) + '-' + str(hidden))
    res.append(spmm)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(res)  

#MGCN-tf32
def mGCN32(data, csv_file, epoches, num_layers, featuredim, hidden, classes):
    spmm= test_mgcn32.test(data, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgcn-tf32-'  + str(spmm))
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


    dataset_I = ['pubmed',  'Coauthor_Physics',  'Amazon_Computers', 'DeezerEurope','FacebookPagePage',
                 'email-Enron', 'loc-Brightkite', 'soc-Epinions1', 'HR_NO', 'HU_NO',
                 'GitHub', 'artist', 'blog']

    dataset_II = ['roadNet-CA', 'roadNet-PA', 'roadNet-TX', 'com-DBLP','amazon', 'amazon0505', 'dd', 'ell', 
                 'comamazon', 'yeast', 'DGraphFin', ]
    dataset_III = ['reddit', 'ogb', 'AmazonProducts']
    
    # dataset_3 = ['Reddit2', 'ovcar', 'amazon','amazon0505',
    #         'yelp', 'sw620', 'dd',
    #         'HR_NO', 'HU_NO', 'ell', 'GitHub',
    #         'artist', 'comamazon', 
    #         'yeast', 'blog', 'DGraphFin', 'reddit', 'ogb', 'AmazonProducts']
    
    # dataset_6 = ['Reddit2', 'ovcar', 'amazon','amazon0505',
    #         'yelp', 'sw620', 'dd',
    #         'HR_NO', 'HU_NO', 'ell', 'GitHub',
    #         'artist', 'comamazon', 
    #         'yeast', 'blog']
    dataset = dataset_II+dataset_III
    dataset_6 = dataset_II
    dataset = dataset_I
    dataset_6 = dataset_I
    layer = [3,6]
    hidden = [64, 128, 256]
    # dataset = ['cora', 'cite']
    # layer = [3]
    # hidden = [512]

    epoches = 300
    featuredim = 512
    classes = 10
    
    #DGL
    filename = './eva100/end2end/gcn_no_pre/result/dgl-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    for layer_num in layer:
        if layer_num == 6:
            for hidden_num in hidden:
                for data in dataset_6:
                    dglGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
        else:
            for hidden_num in hidden:
                for data in dataset:
                    dglGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('DGL-' + 'success')
                
    #MGCN-fp16
    filename = './eva100/end2end/gcn_no_pre/result/mgcn16-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    for layer_num in layer:
        if layer_num == 6:
            for hidden_num in hidden:
                for data in dataset_6:
                    if data=='DGraphFin':
                        continue
                    mGCN16(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
        else:
            for hidden_num in hidden:
                for data in dataset:
                    mGCN16(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('MGCN-fp16-' + 'success')            
 
    #MGCN-tf32   
    filename = './eva100/end2end/gcn_no_pre/result/mgcn32-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    for layer_num in layer:
        if layer_num == 6:
            for hidden_num in hidden:
                for data in dataset_6:
                    if data=='DGraphFin':
                        continue
                    mGCN32(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
        else:
            for hidden_num in hidden:
                for data in dataset:
                    mGCN32(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('MGCN-tf32-' + 'success')
    
    #Advisor
    filename = './eva100/end2end/gcn_no_pre/result/advisor-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    for layer_num in layer:
        if layer_num == 6:
            for hidden_num in hidden:
                for data in dataset_6:
                    if data=='DGraphFin':
                        continue
                    advisor(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
        else:
            for hidden_num in hidden:
                for data in dataset:
                    advisor(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('Advisor-' + 'success')

    #TCGNN
    filename = './eva100/end2end/gcn_no_pre/result/tcgnn-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    for layer_num in layer:
        if layer_num == 6:
            for hidden_num in hidden:
                for data in dataset_6:
                    if data=='DGraphFin':
                        continue
                    tcgnn(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
        else:
            for hidden_num in hidden:
                for data in dataset:
                    if data=='AmazonProducts':
                        continue
                    tcgnn(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('TCGNN-' + 'success')
    
    #PYG
    filename = './eva100/end2end/gcn_no_pre/result/pyg-v2.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    for layer_num in layer:
        if layer_num == 6:
            for hidden_num in hidden:
                for data in dataset_6:
                    if data=='DGraphFin':
                        continue
                    pygGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
        else:
            for hidden_num in hidden:
                for data in dataset_6:
                    pygGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)

    print('Pyg-' + 'success')

    print('MGCN_all success')