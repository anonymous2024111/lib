import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import sys
# sys.path.append('./eva100/end2end/gcn')
from advisor.mdataset import *
from advisor.advgnn_conv import *
from advisor.gcn_adv import *
from advisor.param import *
import GNNAdvisor as GNNA  

def test(data, epoches, layers, featuredim, hidden, classes):
    # 记录程序开始时间
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # start_time = time.time()
    dataset = GCN_dataset(data, featuredim, classes)
    partSize = 32
    dimWorker = 32
    warpPerBlock = 4
    sharedMem = 100
    column_index = dataset.column_index
    row_pointers = dataset.row_pointers
    degrees = dataset.degrees
    inputInfo = inputProperty(row_pointers, column_index, degrees, 
                            partSize, dimWorker, warpPerBlock, sharedMem,
                            hiddenDim=hidden,dataset_obj=dataset)
    inputInfo.decider()
    inputInfo = inputInfo.set_input()
    inputInfo = inputInfo.set_hidden()
    partPtr, part2Node = GNNA.build_part(inputInfo.partSize, inputInfo.row_pointers)
 
    inputInfo.row_pointers  = inputInfo.row_pointers.to(device)
    inputInfo.column_index  = inputInfo.column_index.to(device)
    inputInfo.partPtr = partPtr.int().to(device)
    inputInfo.part2Node  = part2Node.int().to(device)
    
    model= Net(dataset.num_features, hidden, dataset.num_classes, layers, 0.5)
    model, dataset =model.to(device), dataset.to(device)
   
    train(model, inputInfo, dataset, 10) 
    torch.cuda.synchronize()    
    start_time = time.time()  
    train(model, inputInfo, dataset, epoches)
    # 记录程序结束时间
    torch.cuda.synchronize()
    end_time = time.time()
    # 计算程序执行时间（按秒算）
    execution_time = end_time - start_time
    # print(round(execution_time,4))
    return round(execution_time,4)

# if __name__ == "__main__":
#     dataset = 'amazon'
#     test(dataset, 100, 5, 512)
   