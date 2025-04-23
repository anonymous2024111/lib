import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import sys
sys.path.append('./eva4090/accuracy/gcn')
from libra8.mdataset_fp32 import *
from libra8.mgcn_conv import *
from libra8.gcn_mgnn import *
from torch.optim import Adam

def test(data, epoches, num_layers, hidden):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data, 4, 0.8, 0.7).to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes, num_layers, 0.5).to(device)
    train(model, inputInfo, epoches)
    acc = evaluate(model, inputInfo, inputInfo.test_mask)
    acc = round(acc*100, 2)
    print(str(data) + ' libra '": test_accuracy {:.2f}".format(acc))
    return acc

if __name__ == "__main__":
    dataset = 'cora'
    test(dataset, 100, 3, 128)
   