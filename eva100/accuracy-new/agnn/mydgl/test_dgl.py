import sys
sys.path.append('./eva/accuracy/agnn')
from mydgl.agnn_dgl import AGNN, train
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from mydgl.mdataset import *

    
def test(data, epoches, layers, hidden):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    inputInfo = MGCN_dataset(data).to(device)
    edge = (inputInfo.src_li, inputInfo.dst_li)
    g = dgl.graph(edge)
    g = dgl.add_self_loop(g)
    inputInfo.to(device)
    g = g.int().to(device)
    model = AGNN(inputInfo.num_features, hidden, inputInfo.num_classes, layers).to(device)
    
    train(g, inputInfo.x, inputInfo.y, inputInfo.train_mask, inputInfo.val_mask,model, epoches)
   
if __name__ == "__main__":
    dataset = ['pubmed']

    test('/home/shijinliang/module/git-flashsprase-ae2/dataset/pubmed.npz', 300, 5, 100)