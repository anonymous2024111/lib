#!/usr/bin/env python3
import torch
import numpy as np
import time
import MagicsphereBlock_cmake
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix

import MagicsphereMRabbit_cmake
from scipy.sparse import *

class MGCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, featuredim, classes):
        super(MGCN_dataset, self).__init__()

        self.graph = np.load('./dgl_dataset/block/' + data +'-tf32-8-1-mr.npz')
        self.num_features = featuredim
        self.num_classes = classes

        self.init_edges()
        self.init_embedding()
        self.init_labels()
        self.init_others()


    def init_edges(self):
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        # 打印归一化后的特征
        self.x = torch.randn(self.num_nodes_ori, self.num_features)
 
       
    
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
        self.ones = torch.ones(size=(self.num_nodes_ori,1), dtype=torch.float16)

    def to(self, device):
        self.row_pointers =  self.row_pointers.to(device)
        self.column_index =  self.column_index.to(device)
        self.degrees =  self.degrees.to(device)

        # self.train_mask =  self.train_mask.to(device)
        # self.val_mask =  self.val_mask.to(device)
        # self.test_mask =  self.test_mask.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        self.ones =  self.ones.to(device)
        return self
    
  