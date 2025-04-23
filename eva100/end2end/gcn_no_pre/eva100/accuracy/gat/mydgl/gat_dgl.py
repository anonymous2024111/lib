import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.nn.pytorch import GATConv
from dgl import AddSelfLoop

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, head):
        super().__init__()
        self.conv1 = GATConv(in_size, hid_size, head)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(2 - 2):
            self.hidden_layers.append(GATConv(hid_size*head, hid_size, head))
        
        self.conv2 = GATConv(hid_size*head, out_size, head)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        h=self.conv1(g,h).flatten(1)
        h = F.relu(h)
        for layer in self.hidden_layers:
            h = F.relu(layer(g,h).flatten(1))
        h = self.conv2(g,h).mean(1)
        h = F.log_softmax(h,dim=1)
        return h

#输入依次为图，结点特征，标签，验证集或测试集的mask，模型
#注意根据代码逻辑，图和结点特征和标签应该输入所有结点的数据，而不能只输入验证集的数据
def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        
        logits = logits[mask]
        labels = labels[mask]
        #probabilities = F.softmax(logits, dim=1) 
        #print(probabilities)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
#输入依次为图，结点特征，标签，训练、验证、测试的masks，模型，epoches
#注意根据代码逻辑，图和结点特征和标签应该输入所有结点的数据，而不能只输入验证集的数据
def train(g, features, labels, train_mask, val_mask, model,epoches):
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(epoches):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # acc = evaluate(g, features, labels, val_mask, model)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
        #         epoch, loss.item(), acc
        #     )
        # )
