
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import sys
# sys.path.append('./eva100/end2end/gcn')
from mydgl.mdataset import *
from mydgl.gcn_dgl import GCN, train
import time

    
def test(data, epoches, layers, featuredim, hidden, classes):
    # 记录程序开始时间
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # start_time = time.time()
    inputInfo = MGCN_dataset(data, featuredim, classes)  
    inputInfo.to(device)
    edge = (inputInfo.src_li, inputInfo.dst_li)
    g = dgl.graph(edge)
    g = dgl.add_self_loop(g)
    g = g.int().to(device)     
    model = GCN(inputInfo.num_features, hidden, inputInfo.num_classes, layers, 0.5).to(device)
    

    train(g, inputInfo,model, 10)
    torch.cuda.synchronize()
    start_time = time.time()
    train(g, inputInfo,model, epoches)
    # 记录程序结束时间
    torch.cuda.synchronize()
    end_time = time.time()
    execution_time = end_time - start_time
    # 计算程序执行时间（按秒算）
    # print(round(execution_time,4))
    return round(execution_time,4)

# if __name__ == "__main__":
#     dataset = 'cora'
#     test(dataset, 100, 5, 512)
   