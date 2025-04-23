import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import sys
sys.path.append('./eva100/accuracy/gat')
from mgat.mdataset_fp16 import *
from mgat.mgat_conv import *
from mgat.gat_mgnn import *

# print("-------------------------")
# print("    Welcome to M-GAT     ")
# print("-------------------------")
parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=2, help="num layers")
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--head', type=int, default=1, help='Alpha for the leaky_relu.')
args = parser.parse_args()
# print(args)

def test(data, epoches, heads, hidden):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGAT_dataset(data).to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes, args.num_layers, args.dropout, args.alpha, heads).to(device)
    train(model, inputInfo, epoches)
    acc = test1(model, inputInfo)
    acc = round(acc*100, 2)
    print(str(data) + ' MGAT '": test_accuracy {:.2f}".format(acc))
    return acc

# if __name__ == "__main__":
#     dataset = ['cite']
#     for i in range(5) :
#         test('cora', 100, 1, 16)
   