import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from mgat32.mdataset_fp32 import *
from mgat32.mgat_conv import *
from torch.optim import Adam


#########################################
## Build GCN and AGNN Model
#########################################

class Net(torch.nn.Module):
    def __init__(self,in_feats, hidden_feats, out_feats, num_layers, dropout, alpha, head):
        super(Net, self).__init__()
        self.dropout = dropout
        self.attentions = [GATConv(in_feats, hidden_feats, dropout, alpha) for _ in range(head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers -  2):
            self.hidden_layers.append(GATConv(hidden_feats*head, hidden_feats*head, dropout, alpha))
        
        self.conv2 = GATConv(hidden_feats*head, out_feats, dropout, alpha)

    def forward(self, inputInfo):
        x = torch.cat([att(inputInfo.x, inputInfo) for att in self.attentions], dim=1)
        x = F.elu(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        # for Gconv in self.hidden_layers:
        #     x = Gconv(x, inputInfo)
        #     x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, inputInfo)
        # scale_factor = x.abs().max()  # 计算张量的最大绝对值
        # scaled_x = x / scale_factor  # 将张量进行数值范围缩放
        # res = F.log_softmax(scaled_x, dim=1)
        res = F.log_softmax(x, dim=1)
        return res

def evaluate(model, inputInfo):
    model.eval()
    with torch.no_grad():
        logits = model(inputInfo)
        
        logits = logits[inputInfo.val_mask]
        labels = inputInfo.y[inputInfo.val_mask]

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
def test1(model, inputInfo):
    model.eval()
    with torch.no_grad():
        logits = model(inputInfo)
        
        logits = logits[inputInfo.test_mask]
        labels = inputInfo.y[inputInfo.test_mask]

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
# Training 
def train(model, inputInfo, epoches):
    loss_fcn = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name) 
    for epoch in range(epoches):
        model.train()
        logits = model(inputInfo)
        # if torch.isnan(logits).any().item() :
        loss =  F.nll_loss(logits[inputInfo.train_mask], inputInfo.y[inputInfo.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # logits = logits[inputInfo.train_mask]
        # labels = inputInfo.y[inputInfo.train_mask]
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # train_acc = correct.item() * 1.0 / len(labels)
        
        
        # acc = evaluate(model, inputInfo)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} | Train_acc {:.4f} | Val_acc {:.4f}".format(
        #         epoch, loss.item(), train_acc, acc
        #     )
        # )