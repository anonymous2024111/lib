import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, head, num_layers):
        super().__init__()
        self.dropout = 0.5
        self.conv1 = GATConv(in_size, hid_size, head)   
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GATConv(hid_size*head, hid_size*head, 1))
                
        self.conv2 = GATConv(hid_size*head, out_size, 1)

    def forward(self, g, features):
        h=F.relu(self.conv1(g,features).flatten(1))
        for Gconv in self.hidden_layers:
            h = F.relu(Gconv(g,h).mean(1))
            h = F.dropout(h, self.dropout, training=self.training)
        h = self.conv2(g,h).mean(1)
        h = F.log_softmax(h,dim=1)
        return h

def train(g, features, labels, model,epoches):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(epoches):
        model.train()
        logits = model(g, features)
        loss = F.nll_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   
