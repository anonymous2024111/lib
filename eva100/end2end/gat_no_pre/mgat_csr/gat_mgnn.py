import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from mgat_csr.mgat_conv import *
from torch.optim import Adam


#########################################
## Build GCN and AGNN Model
#########################################

class Net(torch.nn.Module):
    def __init__(self,in_feats, hidden_feats, out_feats, dropout, alpha, heads, num_layers):
        super(Net, self).__init__()
        self.dropout = dropout
        self.attentions = [GATConv(in_feats, hidden_feats, dropout, alpha) for _ in range(heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GATConv(hidden_feats*heads, hidden_feats*heads, dropout, alpha))
        
        self.conv2 = GATConv(hidden_feats*heads, out_feats, dropout, alpha)

    def forward(self, inputInfo):
        x = F.elu(torch.cat([att(inputInfo.x, inputInfo) for att in self.attentions], dim=1))
        for Gconv in self.hidden_layers:
            x = F.relu(Gconv(x, inputInfo))
            x = F.dropout(x, self.dropout, training=self.training)
        # scale_factor = x.abs().max()  # 计算张量的最大绝对值
        # scaled_x = x / scale_factor  # 将张量进行数值范围缩放
        # res = F.log_softmax(scaled_x, dim=1)
        x = self.conv2(x, inputInfo)
        res = F.log_softmax(x, dim=1)
        return res
    
# Training 
def train(model, inputInfo, epoches):
    # loss_fcn = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)  
    
    for epoch in range(epoches):
        model.train()
        logits = model(inputInfo)
        # if torch.isnan(logits).any().item() :
        loss =  F.nll_loss(logits, inputInfo.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()