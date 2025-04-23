import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_heads, num_layers):
        super(GAT, self).__init__()
        self.dropout = 0.5
        self.conv1 = GATConv(in_size, hid_size, heads=num_heads)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GATConv(hid_size*num_heads, hid_size*num_heads, heads=1))
        
        self.conv2 = GATConv(hid_size*num_heads, out_size, 1)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, edge, features):
        h = F.relu(self.conv1(features, edge))
        for Gconv in self.hidden_layers:
            h = F.relu(Gconv(h, edge))
            h = F.dropout(h, self.dropout, training=self.training)
        h = self.conv2(h,edge)
        h = F.log_softmax(h,dim=1)
        return h

def train(edge, features, labels, model,epoches):
    # loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(epoches):
        model.train()
        logits = model(edge, features)
        loss = F.nll_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
