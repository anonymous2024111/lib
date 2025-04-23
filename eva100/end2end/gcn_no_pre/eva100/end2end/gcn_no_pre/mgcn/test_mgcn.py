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
from mgcn.mdataset_fp16 import *
from mgcn.mgcn_conv import *
from mgcn.gcn_mgnn import *
from torch.optim import Adam
import time


def test(data, epoches, layers, featuredim, hidden, classes):
    # 记录程序开始时间
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #start_time = time.time()
    inputInfo = MGCN_dataset(data, featuredim, classes)
    # start_time = time.time()  
    inputInfo.to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes, layers, 0.5).to(device)

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

# if __name__ == "__main__":
#     dataset = 'blog'
#     test(dataset, 100, 3, 512, 128, 10)