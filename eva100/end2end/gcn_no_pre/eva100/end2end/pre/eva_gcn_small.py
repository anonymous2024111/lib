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
    
    # dataset = ['cite', 'cora', 'cornell', 'min', 
    #            'pubmed', 'ro', 'tol', 'wiki',
    #            'wis', 'actor', 'artist', 'flickr',
    #            'Amazon_Computers', 'Amazon_Photo', 
    #            'Amazon-ratings','CitationFull_DBLP', 'Coauthor_Physics', 'DeezerEurope',
    #            'FacebookPagePage', 'GitHub', 'HR_NO', 'HU_NO',
    #            'Twitch_EN',  'RO_NO', 'Twitch_ES',
    #            'Tolokers', 'Twitch_FR', 'Twitch_PT', 'Twitch_RU']

    dataset = ['cite', 'cora', 'cornell', 'min', 
               'pubmed', 'ro', 'tol', 'wiki',
               'wis', 'actor', 'artist', 'flickr',
               'Amazon_Computers', 'Amazon_Photo', 
               'Amazon-ratings','CitationFull_DBLP', 'Coauthor_Physics', 'DeezerEurope',
               'FacebookPagePage', 'GitHub', 'HR_NO', 'HU_NO',
               'Twitch_EN',  'RO_NO', 'Twitch_ES',
               'Tolokers', 'Twitch_FR', 'Twitch_PT', 'Twitch_RU']

    layer = [3,6]
    hidden = [64, 128, 256]
    # dataset = ['cora', 'cite']
    # layer = [3]
    # hidden = [512]

    epoches = 300
    featuredim = 512
    classes = 10
    
    #DGL
    filename = './eva100/end2end/gcn_no_pre/result_small/dgl.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            for data in dataset:
                dglGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('DGL-' + 'success')
                
    #MGCN-fp16
    filename = './eva100/end2end/gcn_no_pre/result_small/mgcn16.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            for data in dataset:
                mGCN16(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('MGCN-fp16-' + 'success')            
 
    #MGCN-tf32   
    filename = './eva100/end2end/gcn_no_pre/result_small/mgcn32.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            for data in dataset:
                mGCN32(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('MGCN-tf32-' + 'success')
    
    #Advisor
    filename = './eva100/end2end/gcn_no_pre/result_small/advisor.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            for data in dataset:
                advisor(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('Advisor-' + 'success')

    #TCGNN
    filename = './eva100/end2end/gcn_no_pre/result_small/tcgnn.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            for data in dataset:
                tcgnn(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('TCGNN-' + 'success')
    
    #PYG
    filename = './eva100/end2end/gcn_no_pre/result_small/pyg.csv'
    with open(filename, 'w') as file:
        file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            for data in dataset:
                pygGCN(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('Pyg-' + 'success')

    print('MGCN-small_all success')