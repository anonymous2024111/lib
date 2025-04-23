import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import sys
sys.path.append('eva100/end2end/gat_no_pre')
from mydgl.mdataset import *
from mydgl.gat_dgl import GAT, train
import time
    
def test(data, epoches, heads, layers, featuredim, hidden, classes):
    # start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data, featuredim, classes)
    # print(inputInfo.num_nodes)
    # print(inputInfo.num_edges)
    edge = (inputInfo.src_li, inputInfo.dst_li)
    g = dgl.graph(edge)
    g = dgl.add_self_loop(g)
    inputInfo.to(device)
    g = g.int().to(device)
    model = GAT(inputInfo.num_features, hidden, inputInfo.num_classes, heads, layers).to(device)
    
    train(g, inputInfo.x, inputInfo.y, model, 10)
    torch.cuda.synchronize()
    start_time = time.time()
    train(g, inputInfo.x, inputInfo.y, model, epoches)
    # 记录程序结束时间
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算程序执行时间（按秒算）
    execution_time = end_time - start_time
    # print(execution_time)
    return round(execution_time,4)

# if __name__ == "__main__":
#     dataset = 'IGB_medium'
#     test(dataset, 100, 1, 3, 512, 64, 10)
