#!/usr/bin/env python3
import torch
import numpy as np
import time
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import FS_Block
from scipy.sparse import *

def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    return (sparse_matrix != transposed_matrix).nnz == 0
class MAGNN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data):
        super(MAGNN_dataset, self).__init__()
        self.graph = np.load(data)

        self.num_features = self.graph['in_size'].item()
        self.num_classes = self.graph['out_size'].item()

        self.init_edges(32, 8, 16)
        self.init_embedding()
        self.init_labels()
        self.init_others()
        
        self.train_mask = torch.from_numpy(self.graph['train_mask'])
        self.val_mask = torch.from_numpy(self.graph['val_mask'])
        self.test_mask = torch.from_numpy(self.graph['test_mask'])
    def init_edges(self, partSize, window, wide):
        src_li=self.graph['src_li']
        dst_li=self.graph['dst_li']
        
        self.num_nodes_ori = self.graph['num_nodes']
        self.num_nodes=self.graph['num_nodes']+16-(self.graph['num_nodes']%16)
        self.num_edges = len(src_li)
        self.edge_index = np.stack([src_li, dst_li])

        #self-loop
        adj = sp.coo_matrix((np.ones(len(src_li)), self.edge_index),
                        shape=(self.num_nodes,self.num_nodes),
                        dtype=np.float32)
        is_sym = is_symmetric(adj)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        is_sym = is_symmetric(adj)
        
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges).half()
        
        self.row_pointers, \
        self.column_index, \
        self.degrees, \
        self.t_window_rowTensor, \
        self.t_atomicTensor    = FS_Block.blockProcess_sddmm_balance_gnn(self.row_pointers, self.column_index, window, wide, partSize)

        max_vectors = torch.max(self.row_pointers[1:]- self.row_pointers[:-1])
        if max_vectors%wide > 0 :
            max_vectors += (wide - (max_vectors%wide))
        self.max = max_vectors / wide
        
        if self.max % 4 > 0 :
            self.max += 4 - self.max%4
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.from_numpy(self.graph['features']).to(torch.float16) 
    
    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.from_numpy(self.graph['labels'])
        
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
        self.t_window_rowTensor =  self.t_window_rowTensor.to(device)
        self.t_atomicTensor =  self.t_atomicTensor.to(device)
        
        self.train_mask =  self.train_mask.to(device)
        self.val_mask =  self.val_mask.to(device)
        self.test_mask =  self.test_mask.to(device)
        
        self.x =  self.x.to(device)
        self.length =  torch.norm(self.x, dim=1)
        self.y =  self.y.to(device)
        self.ones =  self.ones.to(device)
        return self
