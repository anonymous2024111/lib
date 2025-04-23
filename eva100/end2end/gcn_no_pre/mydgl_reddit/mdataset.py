#!/usr/bin/env python3
import torch
import numpy as np
import torch
from torch_geometric.datasets import Reddit
from torch_geometric.data import DataLoader
from scipy.sparse import *

class MGCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self):
        super(MGCN_dataset, self).__init__()

        # self.graph = np.load('dgl_dataset/mythroughput/' + data +'.npz')
        # 加载 Reddit 数据集
        dataset = Reddit(root='/home/shijinliang/module/Libra/eva100/end2end/loadData/reddit')
        # 查看数据集中的第一个图（Reddit 数据集是一个图集合）
        self.graph = dataset[0]
        
        self.num_features = self.graph.x.shape[1]
        self.num_classes = torch.unique(self.graph.y).size(0)

        self.init_edges()
        self.init_embedding()
        self.init_labels()

        # self.train_mask = torch.from_numpy(self.graph['train_mask'])
        # self.val_mask = torch.from_numpy(self.graph['val_mask'])
        # self.test_mask = torch.from_numpy(self.graph['test_mask'])

    def init_edges(self):
        # loading from a .npz graph file
        self.src_li=self.graph.edge_index[0]
        self.dst_li=self.graph.edge_index[1]
        self.num_nodes = self.graph.x.shape[0]
        self.num_edges = len(self.src_li)
        self.edge_index = torch.from_numpy(np.stack([self.src_li, self.dst_li]))

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = self.graph.x
    
    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = self.graph.y


    def to(self, device):
        
        # self.train_mask =  self.train_mask.to(device)
        # self.val_mask =  self.val_mask.to(device)
        # self.test_mask =  self.test_mask.to(device)
        self.edge_index =  self.edge_index.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        return self
