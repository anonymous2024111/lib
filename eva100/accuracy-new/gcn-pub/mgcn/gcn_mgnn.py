import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from mgcn.mdataset_fp16 import *
from mgcn.mgcn_conv import *
from torch.optim import Adam


#########################################
## Build GCN and AGNN Model
#########################################

class Net(torch.nn.Module):
    def __init__(self,in_feats, hidden_feats, out_feats, num_layers, dropout):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers -  2):
            self.hidden_layers.append(GCNConv(hidden_feats, hidden_feats))
        
        self.conv2 = GCNConv(hidden_feats, out_feats)
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, inputInfo):
        x = inputInfo.x
        x = F.relu(self.conv1(x, inputInfo))
        x = F.dropout(x, self.dropout, training=self.training)
        for Gconv in self.hidden_layers:
            x = Gconv(x, inputInfo)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, inputInfo).half()
        res=F.log_softmax(x, dim=1)
        return res

def evaluate(model, inputInfo, mask):
    model.eval()
    with torch.no_grad():
        logits = model(inputInfo)
        
        logits = logits[mask]
        labels = inputInfo.y[mask]

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
def test(model, inputInfo):
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
    # loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name) 
    with open("/home/shijinliang/module/git-flashsprase-ae2/eva/accuracy/gcn/flash-pubmed-fp16.txt", "w", encoding="utf-8") as file:
        file.write("")
    for epoch in range(epoches):
        model.train()
        # 在训练过程中应用混合精度
        logits = model(inputInfo)
        loss = F.nll_loss(logits[inputInfo.train_mask], inputInfo.y[inputInfo.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # logits = logits[inputInfo.train_mask]
        # labels = inputInfo.y[inputInfo.train_mask]
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # train_acc = correct.item() * 1.0 / len(labels)
        acc = evaluate(model, inputInfo, inputInfo.val_mask)
        print(
            "Epoch {:05d} | Loss {:.4f} | Val_acc {:.4f}".format(
                epoch, loss.item(), acc
            )
        )
        data = str(acc)
        # 打开文件，写入模式 ('w' 表示写入，如果文件不存在会创建文件)
        with open("/home/shijinliang/module/git-flashsprase-ae2/eva/accuracy/gcn/flash-pubmed-fp16.txt", "a", encoding="utf-8") as file:
            file.write(data + "\n")