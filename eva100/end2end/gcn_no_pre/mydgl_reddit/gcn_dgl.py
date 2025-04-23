import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import ActorDataset
class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, layers, dropout):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_size, hid_size,bias=False,norm="none")
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(layers - 2):
            self.hidden_layers.append(dglnn.GraphConv(hid_size, hid_size,bias=False,norm="both"))
        
        self.conv2 = dglnn.GraphConv(hid_size, out_size)
        self.dropout = dropout

    def forward(self, g, features):
        h = features
        h=F.relu(self.conv1(g,h))
        h = F.dropout(h, self.dropout, training=self.training)
        for layer in self.hidden_layers:
            h = F.relu(layer(g,h))
            h = F.dropout(h, self.dropout, training=self.training)
        h = self.conv2(g,h)
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
def train(g, inputInfo, model,epoches):
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(epoches):
        model.train()
        logits = model(g, inputInfo.x)
        loss = loss_fcn(logits, inputInfo.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # acc = evaluate(g, features, labels, val_mask, model)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
        #         epoch, loss.item(), acc
        #     )
        # )
