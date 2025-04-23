import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from tcgnn.mdataset_tf32 import *
from tcgnn.tcgnn_conv import *
from tcgnn.gcn_tc import *
from torch.optim import Adam
import TCGNN


def test1(data, epoches, num_layers, hidden):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data).to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes, num_layers, 0.5).to(device)
    epoches = 100
    train(model, inputInfo, epoches)
    acc = test(model, inputInfo)
    print(str(data) + ": test_accuracy {:.4f}".format(acc))

# if __name__ == "__main__":
#     dataset = ['question']
#     test1(dataset)
   