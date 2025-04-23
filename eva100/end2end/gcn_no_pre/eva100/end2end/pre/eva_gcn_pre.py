import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import sys
sys.path.append('./eva100/end2end/gcn_no_pre')

from mgcn import test_mgcn
from tcgnn import test_tcgnn
from mgcn32 import test_mgcn32

    
#MGCN
def mGCN16(data, file_name, epoches, num_layers, featuredim, hidden, classes):
    rabbit, partition, total = test_mgcn.test(data, epoches, num_layers, featuredim, hidden, classes)
    # print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgcn-fp16-'  + str(spmm))
    rabbit_percen = round(((rabbit/total)*100), 2)
    partition_percen = round(((partition/total)*100), 2)
    train = round((100 - rabbit_percen - partition_percen), 2)
    res = data
    res = res + ' & ' + str(rabbit_percen) + ' & ' + str(partition_percen) + ' & ' + str(train)
    with open(file_name, 'a') as file:
        file.write(res + '\n')
    print(res)

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



if __name__ == "__main__":
    
    # dataset = ['cite', 'cora', 'cornell', 'min', 
    #            'pubmed', 'ro', 'tol', 'wiki',
    #            'wis', 'actor', 'artist', 'flickr',
    #            'Amazon_Computers', 'Amazon_Photo', 
    #            'Amazon-ratings','CitationFull_DBLP', 'Coauthor_Physics', 'DeezerEurope',
    #            'FacebookPagePage', 'GitHub', 'HR_NO', 'HU_NO',
    #            'Twitch_EN',  'RO_NO', 'Twitch_ES',
    #            'Tolokers', 'Twitch_FR', 'Twitch_PT', 'Twitch_RU']

    dataset_3 = ['Reddit2', 'ovcar', 'amazon','amazon0505',
            'yelp', 'sw620', 'dd',
            'HR_NO', 'HU_NO', 'ell', 'GitHub',
            'artist', 'comamazon', 
            'yeast', 'blog', 'DGraphFin', 'reddit', 'ogb', 'AmazonProducts']
    dataset_3 = ['cite', 'cora', 'cornell', 'min', 
               'pubmed', 'ro', 'tol', 'wiki',
               'wis', 'actor', 'artist', 'flickr',
               'Amazon_Computers', 'Amazon_Photo', 
               'Amazon-ratings','CitationFull_DBLP', 'Coauthor_Physics', 'DeezerEurope',
               'FacebookPagePage', 'GitHub', 'HR_NO', 'HU_NO',
               'Twitch_EN',  'RO_NO', 'Twitch_ES',
               'Tolokers', 'Twitch_FR', 'Twitch_PT', 'Twitch_RU']
    # dataset_3 = ['Reddit2', 'ovcar', 'amazon','amazon0505']
    dataset = dataset_3
    layer = [6]
    hidden = [256]

    epoches = 300
    featuredim = 512
    classes = 10
                
    #MGCN-fp16
    filename = './eva100/end2end/pre/result/mgcn16.txt'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    for layer_num in layer:
        for hidden_num in hidden:
            for data in dataset:
                mGCN16(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    print('MGCN-fp16-' + 'success')            
 
    # #MGCN-tf32   
    # filename = './eva100/end2end/gcn_no_pre/result/mgcn32.csv'
    # with open(filename, 'w') as file:
    #     file.write('H100 : \n')
    # for layer_num in layer:
    #     if layer_num == 6:
    #         for hidden_num in hidden:
    #             for data in dataset_6:
    #                 mGCN32(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    #     else:
    #         for hidden_num in hidden:
    #             for data in dataset:
    #                 mGCN32(data, filename, epoches, layer_num, featuredim, hidden_num, classes)
    # print('MGCN-tf32-' + 'success')
