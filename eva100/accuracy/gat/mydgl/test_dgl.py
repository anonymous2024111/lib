from mydgl.gat_dgl import GAT, train, evaluate
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from mydgl.mdataset import *

    
def test(data, epoches, heads, hidden):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data).to(device)
    edge = (inputInfo.src_li, inputInfo.dst_li)
    g = dgl.graph(edge)
    g = g.int().to(device)
    model = GAT(inputInfo.num_features, hidden, inputInfo.num_classes, heads).to(device)
    train(g, inputInfo.x, inputInfo.y, inputInfo.train_mask, inputInfo.val_mask,model, epoches)
    acc = evaluate(g, inputInfo.x, inputInfo.y, inputInfo.test_mask, model)
    acc = round(acc*100, 2)
    print(str(data) + ' DGL '": test_accuracy {:.2f}".format(acc))
    return acc
