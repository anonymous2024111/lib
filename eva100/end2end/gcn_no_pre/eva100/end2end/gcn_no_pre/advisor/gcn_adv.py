import os.path as osp
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from advisor.advgnn_conv import *


#########################################
## Build GCN and AGNN Model
#########################################

class Net(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, dropout):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers -  2):
            self.hidden_layers.append(GCNConv(hidden_feats, hidden_feats))
        
        self.conv2 = GCNConv(hidden_feats, out_feats)
        self.dropout = dropout
        # self.relu = nn.ReLU()

    def forward(self,inputInfo, dataset):
        x = dataset.x
        x = F.relu(self.conv1(x, inputInfo.set_input()))
        x = F.dropout(x, self.dropout, training=self.training)
        for Gconv  in self.hidden_layers:
            x = F.relu(Gconv(x, inputInfo.set_input()))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, inputInfo.set_input())
        return F.log_softmax(x, dim=1)

def evaluate(model, inputInfo):
    model.eval()
    with torch.no_grad():
        logits = model(inputInfo)
        
        logits = logits
        labels = inputInfo.y

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
def test1(model, inputInfo, dataset):
    model.eval()
    with torch.no_grad():
        logits = model(inputInfo, dataset)
        
        logits = logits
        labels = dataset.y

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
# Training 
def train(model, inputInfo, dataset, epoches):
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    for epoch in range(epoches):
        model.train()
        # 在训练过程中应用混合精度
        logits = model(inputInfo, dataset)
        loss = F.nll_loss(logits, dataset.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(epoch)
        # logits = logits
        # labels = dataset.y
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # train_acc = correct.item() * 1.0 / len(labels)
        # acc = test1(model, inputInfo, dataset)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} | Train_acc {:.4f} | Val_acc {:.4f}".format(
        #         epoch, loss.item(), train_acc, acc
        #     )
        # )
