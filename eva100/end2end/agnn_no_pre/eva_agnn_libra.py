import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('./eva100/end2end/gcn_no_pre')
from magnn import test_magnn
from magnn32 import test_magnn32

#MGCN
def mAGNN16(data, epoches, num_layers, featuredim, hidden, classes, density, partsize_t, partsize_c, window, wide):
    spmm = test_magnn.test(data, epoches, num_layers, featuredim, hidden, classes, density, partsize_t, partsize_c, window, wide)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgcn-fp16-'  + str(spmm))
    return spmm

#MGCN-tf32
def mAGNN32(data, epoches, num_layers, featuredim, hidden, classes, density, partsize_t, partsize_c, window, wide):
    spmm= test_magnn32.test(data, epoches, num_layers, featuredim, hidden, classes, density, partsize_t, partsize_c, window, wide)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-mgcn-tf32-'  + str(spmm))
    return spmm
    



if __name__ == "__main__":


    dataset = {}
    dataset['IGB_small'] = 1
    dataset['reddit'] = 1
    dataset['amazon'] = 1
    # dataset['reddit'] = 1
    # dataset['yeast'] = 1


    layer = 3
    hidden = 128
    epoches = 300
    featuredim = 128
    classes = 10
     
     
    window = 8
    wide = 16
    
    partsize_t = 32
    partsize_c = 32
    shortsize = 3
    #MGCN-fp16
    file_name = '/home/shijinliang/module/Libra/eva100/end2end/agnn_no_pre/libra_agnn' + str(hidden) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'spmm_tcu_fp16', 'spmm_tcu_tf32']
    # 写入数据到 CSV 文件
    # with open(file_name, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(head)
        
    for data, density in dataset.items():
        graph = np.load('dgl_dataset/mythroughput/' + data +'.npz')
        res_temp = []
        src_li = graph['src_li']
        num_nodes  = graph['num_nodes']
        num_edges = len(src_li)
        
        res_temp = []
        res_temp.append(data)
        res_temp.append(num_nodes)
        res_temp.append(num_edges)
        
        
        spmm_fp16 = mAGNN16(data, epoches, layer, featuredim, hidden, classes, density, partsize_t, partsize_c, window, wide)
        res_temp.append(spmm_fp16)

        spmm_tf32 = mAGNN32(data, epoches, layer, featuredim, hidden, classes, density, partsize_t, partsize_c, window, wide)
        res_temp.append(spmm_tf32)
            
        
        # with open(file_name, 'a', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(res_temp)
        #     print(data + 'is success')    

