import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_size, hid_size, heads=num_heads)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(2 - 2):
            self.hidden_layers.append(GATConv(hid_size*num_heads, hid_size, heads=num_heads))
        
        self.conv2 = GATConv(hid_size*num_heads, out_size, heads=num_heads)
        self.dropout = nn.Dropout(0.5)

    def forward(self, edge, features):
        h = features
        h = torch.relu(self.conv1(h, edge))
        for layer in self.hidden_layers:
            h = torch.relu(layer(h,edge))
        h = self.conv2(h,edge)
        h = F.log_softmax(h,dim=1)
        return h

#输入依次为图，结点特征，标签，验证集或测试集的mask，模型
#注意根据代码逻辑，图和结点特征和标签应该输入所有结点的数据，而不能只输入验证集的数据
def evaluate(edge, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(edge, features)
        
        logits = logits[mask]
        labels = labels[mask]
        #probabilities = F.softmax(logits, dim=1) 
        #print(probabilities)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
def train(edge, features, labels, train_mask, val_mask, model,epoches):
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(epoches):
        model.train()
        logits = model(edge, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # acc = evaluate(edge, features, labels, val_mask, model)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
        #         epoch, loss.item(), acc
        #     )
        # )
