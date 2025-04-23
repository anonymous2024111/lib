from mypyg.gat_pyg import GAT, train, evaluate
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from mypyg.mdataset import *

    
def test(data, epoches, heads, hidden):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        inputInfo = MGCN_dataset(data).to(device)
      
        model = GAT(inputInfo.num_features, hidden, inputInfo.num_classes,  num_heads=heads).to(device)
        train(inputInfo.edge_index, inputInfo.x, inputInfo.y, inputInfo.train_mask, inputInfo.val_mask,model, epoches)
        acc = evaluate(inputInfo.edge_index, inputInfo.x, inputInfo.y, inputInfo.test_mask, model)
        acc = round(acc*100, 2)
        print(str(data) + ' PYG '": test_accuracy {:.2f}".format(acc))
        return acc
    
# dataset = ['texas']
# test(dataset)