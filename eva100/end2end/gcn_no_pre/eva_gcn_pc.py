import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from pcgcn import test_mgcn


#GCN-fp32
def mGCN32(data, data_path, epoches, num_layers, featuredim, hidden, classes):
    spmm= test_mgcn.test(data_path, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-gcn-tf32-'  + str(spmm))
    return spmm
    


if __name__ == "__main__":

    dataset =['pubmed', 'reddit', 'IGB_small']
    hidden_list = [128]

    current_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    for hidden in hidden_list:
        layer = 5
        epoches = 300
        featuredim = 512
        classes = 16
        
        #result path
        # file_name = project_dir + '/result/FlashSparse/gcn/pc_gcn_' + str(hidden) + '.csv'
        # head = ['dataSet', 'num_nodes', 'num_edges', 'fs_fp16', 'fs_tf32']
        # with open(file_name, 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(head)
            
        for data in dataset:
            data_path =  '/public/home/shijinliang/gnns/' + data +'.npz'
            graph = np.load(data_path)
            src_li = graph['src_li']
            num_nodes  = graph['num_nodes']-0
            num_edges = len(src_li)
            res_temp = []
            res_temp.append(data)
            res_temp.append(num_nodes)
            res_temp.append(num_edges)
            

            spmm_fp32 = mGCN32(data, data_path, epoches, layer, featuredim, hidden, classes)
            res_temp.append(spmm_fp32)
                
            
            # with open(file_name, 'a', newline='') as csvfile:
            #     csv_writer = csv.writer(csvfile)
            #     csv_writer.writerow(res_temp)
            #     print(data + 'is success')    

