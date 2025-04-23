import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv
class AGNN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super(AGNN, self).__init__()
        self.lin1 = torch.nn.Linear(in_size, hid_size)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(AGNNConv(requires_grad=True))
        self.lin2 = torch.nn.Linear(hid_size, out_size)

    def forward(self, edge, features):
        h = features
  
        h = F.relu(self.lin1(h))
        for conv in self.convs:
            h = F.relu(conv(h, edge))
        h = self.lin2(h)
        return F.log_softmax(h, dim=1)
    

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
    # loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    with open("/home/shijinliang/module/git-flashsprase-ae2/eva/accuracy/agnn/flash-pubmed.txt", "w", encoding="utf-8") as file:
        file.write("")
    # training loop
    for epoch in range(epoches):
        model.train()
        logits = model(edge, features)
        loss = F.nll_loss(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(edge,features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        data = str(acc)
        # 打开文件，写入模式 ('w' 表示写入，如果文件不存在会创建文件)
        with open("/home/shijinliang/module/git-flashsprase-ae2/eva/accuracy/agnn/pyg-pubmed.txt", "a", encoding="utf-8") as file:
            file.write(data + "\n")

        
