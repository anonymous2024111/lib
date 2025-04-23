import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import sys
# sys.path.append('./eva100/end2end/gcn_no_pre')
from mgcn32.mdataset_tf32 import *
from mgcn32.mgcn_conv import *
from mgcn32.gcn_mgnn import *
from torch.optim import Adam
import time


def test_libra_tcu_cuda_tf32(data, epoches, layers, featuredim, hidden, classes,  density, partsize_t, partsize_c, window, wide):
    # 记录程序开始时间
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #start_time = time.time()
    inputInfo = MGCN_dataset_tcu_cuda(data, featuredim, classes, density,  window, wide)
    # start_time = time.time()  
    inputInfo.to(device)
    model= Net_tcu(inputInfo.num_features, hidden, inputInfo.num_classes, layers, 0.5).to(device)

    train(model, inputInfo, 10)
    torch.cuda.synchronize()
    start_time = time.time()
    train(model, inputInfo, epoches)
    # 记录程序结束时间
    torch.cuda.synchronize()
    end_time = time.time()
    # 计算程序执行时间（按秒算）
    execution_time = end_time - start_time
    # print(round(execution_time,4))
    return round(execution_time,4)



# def test_libra_tcu_tf32(data, epoches, layers, featuredim, hidden, classes,  density, partsize_t, partsize_c, window, wide):
#     # 记录程序开始时间
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     #start_time = time.time()
#     inputInfo = MGCN_dataset_tcu(data, featuredim, classes, density,  window, wide)
#     # start_time = time.time()  
#     inputInfo.to(device)
#     model= Net_tcu(inputInfo.num_features, hidden, inputInfo.num_classes, layers, 0.5).to(device)

#     train(model, inputInfo, 10)
#     torch.cuda.synchronize()
#     start_time = time.time()
#     train(model, inputInfo, epoches)
#     # 记录程序结束时间
#     torch.cuda.synchronize()
#     end_time = time.time()
#     # 计算程序执行时间（按秒算）
#     execution_time = end_time - start_time
#     # print(round(execution_time,4))
#     return round(execution_time,4)