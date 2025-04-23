#!/usr/bin/env python3
import torch
import numpy as np
import time
import FS_Block
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix

from scipy.sparse import *

class MGCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, featuredim, classes):
        super(MGCN_dataset, self).__init__()

        self.graph = np.load(data)
        self.num_features = featuredim
        self.num_classes = classes

        self.init_edges(8, 8)
        self.init_embedding()
        self.init_labels()
        self.init_others()


    def init_edges(self, window, wide):
        # loading from a .npz graph file
        self.num_nodes_ori =  self.graph['num_nodes']-0
        self.num_nodes_dst =  self.graph['num_nodes']-0
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        self.num_edges = len(src_li)    
        # self.edge_index = self.graph['edge_index_new']
        self.edge_index = np.stack([src_li, dst_li])
        # print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        values = torch.tensor(val, dtype=torch.float32)
        # 创建 scipy COO 格式的稀疏矩阵
        self.sparse_matrix = torch.sparse_coo_tensor(self.edge_index, values, (self.num_nodes_ori, self.num_nodes_dst))

    
    
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        # 打印归一化后的特征
        self.x = torch.randn(self.num_nodes_ori, self.num_features).to(dtype=torch.float32)
 
       
    
    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.randint(low=0, high=self.num_classes, size=(self.num_nodes_ori,))

        
    def init_others(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.ones = torch.ones(size=(self.num_nodes_ori,1), dtype=torch.float32)

    def to(self, device):
        self.sparse_matrix = self.sparse_matrix.to(device)

        # self.train_mask =  self.train_mask.to(device)
        # self.val_mask =  self.val_mask.to(device)
        # self.test_mask =  self.test_mask.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        self.ones =  self.ones.to(device)
        return self
    
  