import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('./eva100/end2end/gcn_no_pre')
from mgcn_bcrs import test_mgcn


#MGCN
def mGCN16(data, epoches, num_layers, featuredim, hidden, classes, density, partsize_t, partsize_c, window, wide):
    spmm = test_mgcn.test_libra_tcu_cuda(data, epoches, num_layers, featuredim, hidden, classes, density, partsize_t, partsize_c, window, wide)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgcn-fp16-'  + str(spmm))
    return spmm



if __name__ == "__main__":


    dataset = {}
    # dataset['MOOC'] = 1
    dataset['pubmed'] = 1
    dataset['pubmed'] = 6
    dataset['IGB_small'] = 6
    # dataset['LastFM'] = 5
    # dataset['twitter'] = 1
    dataset['reddit'] = 3
    # dataset['yeast'] = 1
    # dataset['comamazon'] = 3
    # dataset['roadNet-CA'] = 3
    # dataset['roadNet-PA'] = 3
    # dataset['roadNet-TX'] = 3
    # dataset['DGraphFin'] = 3
    
    hidden = 128
    layer = 5
    epoches = 300
    featuredim = 512
    classes = 16
     
     

    density = [1, 2, 3, 4, 5, 6, 7, 8]
    
    partsize_t = 32
    partsize_c = 32
    shortsize = 3
    #MGCN-fp16
    file_name = '/home/shijinliang/module/Libra/eva100/end2end/gcn_no_pre/libra_gcn_new' + str(hidden) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'spmm_tcu_fp16', 'spmm_tcu_tf32']
    # 写入数据到 CSV 文件
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)
        
    for data, density in dataset.items():
        graph = np.load('/public/home/shijinliang/gnns/' + data +'.npz')
        res_temp = []
        src_li = graph['src_li']
        num_nodes  = graph['num_nodes']
        num_edges = len(src_li)
        print(num_edges)
        res_temp = []
        res_temp.append(data)
        res_temp.append(num_nodes)
        res_temp.append(num_edges)
        
        
        spmm_fp16 = mGCN16(data, epoches, layer, featuredim, hidden, classes, density, partsize_t, partsize_c, 8, 8)
        res_temp.append(spmm_fp16)

        # spmm_tf32 = mGCN32(data, epoches, layer, featuredim, hidden, classes, density, partsize_t, partsize_c, 8, 4)
        # res_temp.append(spmm_tf32)
            
        
        # with open(file_name, 'a', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(res_temp)
        #     print(data + 'is success')    

