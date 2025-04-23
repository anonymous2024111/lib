#!/usr/bin/env python3
import torch
import numpy as np
import time
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix

from scipy.sparse import *

def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    return (sparse_matrix != transposed_matrix).nnz == 0
class MGAT_dataset_csr(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, featuredim, classes):
        super(MGAT_dataset_csr, self).__init__()
        #self.graph = np.load('dgl_dataset/mythroughput/' + data +'.npz')
        self.graph = np.load('./dgl_dataset/block/' + data +'-fp16-8-16-mr-csr.npz')
        self.num_features = featuredim
        self.num_classes = classes

        self.init_edges()
        self.init_embedding()
        self.init_labels()
        self.init_others()

        # self.train_mask = torch.from_numpy(self.graph['train_mask'])
        # self.val_mask = torch.from_numpy(self.graph['val_mask'])
        # self.test_mask = torch.from_numpy(self.graph['test_mask'])
        
        # self.x = torch.index_select(self.x, 0, self.permNew)
        # self.y = torch.index_select(self.y, 0, self.permNew)
        # self.train_mask = torch.index_select(self.train_mask, 0, self.permNew)
        # self.val_mask = torch.index_select(self.val_mask, 0, self.permNew)
        # self.test_mask = torch.index_select(self.test_mask, 0, self.permNew)
        # print()


    def init_edges(self):
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.row_pointers1 = torch.tensor(self.graph['row_pointers1'])
        self.column_index1 = torch.tensor(self.graph['column_index1'])
        self.values = torch.tensor(self.graph['degrees'])
        self.values_templete = torch.tensor(self.graph['templete_8_4'])
        self.values_templete_t = torch.tensor(self.graph['templete_t_8_4'])
        result = self.row_pointers[2::2] - self.row_pointers[:-2:2]
        self.max=max(result)
        self.indices = torch.nonzero(self.values_templete).squeeze()
        
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
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
        self.row_pointers =  self.row_pointers.to(device)
        self.column_index =  self.column_index.to(device)
        self.row_pointers1 =  self.row_pointers1.to(device)
        self.column_index1 =  self.column_index1.to(device)
        self.values =  self.values.to(device)
        self.values_templete = self.values_templete.to(device)
        self.values_templete_t = self.values_templete_t.to(device)
        self.indices = self.indices.to(device)
        
        # self.train_mask =  self.train_mask.to(device)
        # self.val_mask =  self.val_mask.to(device)
        # self.test_mask =  self.test_mask.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        self.ones =  self.ones.to(device)
        return self
