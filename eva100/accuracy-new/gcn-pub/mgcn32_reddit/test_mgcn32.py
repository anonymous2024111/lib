import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import sys
sys.path.append('./eva/accuracy/gcn')
from mgcn32_reddit.mdataset_fp32 import *
from mgcn32_reddit.mgcn_conv import *
from mgcn32_reddit.gcn_mgnn import *

def test(data, epoches, num_layers, hidden):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = Reddit_dataset().to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes, num_layers, 0.5).to(device)
    train(model, inputInfo, epoches)
    acc = evaluate(model, inputInfo, inputInfo.test_mask)
    acc = round(acc*100, 2)
    print(str(data) + ' FlashSparse-GCN-TF32 '": test_accuracy {:.2f}".format(acc))
    return acc

if __name__ == "__main__":
    dataset = 'question'
    test(dataset, 100, 5,  128)