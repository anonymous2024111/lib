#!/usr/bin/env python3
import torch
import numpy as np
import time
import MagicsphereBlock
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix

import MagicsphereMRabbit_cmake
from scipy.sparse import *

def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    return (sparse_matrix != transposed_matrix).nnz == 0
class MGAT_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data):
        super(MGAT_dataset, self).__init__()

        self.graph = np.load('./dgl_dataset/accuracy/' + data +'.npz')
        # print(self.graph)
        self.num_features = self.graph['in_size'].item()
        self.num_classes = self.graph['out_size'].item()

        self.init_edges()
        self.init_embedding()
        self.init_labels()
        self.init_others()

        self.train_mask = torch.from_numpy(self.graph['train_mask'])
        self.val_mask = torch.from_numpy(self.graph['val_mask'])
        self.test_mask = torch.from_numpy(self.graph['test_mask'])
        
        self.x = torch.index_select(self.x, 0, self.permNew)
        self.y = torch.index_select(self.y, 0, self.permNew)
        self.train_mask = torch.index_select(self.train_mask, 0, self.permNew)
        self.val_mask = torch.index_select(self.val_mask, 0, self.permNew)
        self.test_mask = torch.index_select(self.test_mask, 0, self.permNew)
        # print()


    def init_edges(self):
        # loading from a .npz graph file
        src_li=self.graph['src_li']
        dst_li=self.graph['dst_li']
        
        self.num_nodes_ori = self.graph['num_nodes']
        self.num_nodes=self.graph['num_nodes']+8-(self.graph['num_nodes']%8)
        self.num_edges = len(src_li)
        self.edge_index = np.stack([src_li, dst_li])

        #Rabbit
        _,_,self.edge_index, self.permNew,_= MagicsphereMRabbit_cmake.reorder(torch.IntTensor(self.edge_index),self.num_nodes_ori,6)

        # #self-loop
        adj = sp.coo_matrix((np.ones(len(src_li)), self.edge_index),
                        shape=(self.num_nodes,self.num_nodes),
                        dtype=np.float32)
        is_sym = is_symmetric(adj)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        is_sym = is_symmetric(adj)
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        
        # val = [1] * self.num_edges
        # scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        # scipy_csr = scipy_coo.tocsr()
        # self.column_index = torch.IntTensor(scipy_csr.indices)
        # self.row_pointers = torch.IntTensor(scipy_csr.indptr)
         
        #bcsr
        row = self.row_pointers
        col = self.column_index
        self.row_pointers, self.column_index, self.values=MagicsphereBlock.blockProcess8_16(self.row_pointers, self.column_index)
        result = self.row_pointers[2::2] - self.row_pointers[:-2:2]
        self.max=max(result)
        _, _ ,self.values_templete = MagicsphereBlock.blockProcess_output_8_8(row, col)
        self.indices = torch.nonzero(self.values_templete).squeeze()
        
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.from_numpy(self.graph['features']).to(dtype=torch.float16)
    
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
        self.values =  self.values.to(device)
        self.values_templete = self.values_templete.to(device)
        self.indices = self.indices.to(device)
        
        self.train_mask =  self.train_mask.to(device)
        self.val_mask =  self.val_mask.to(device)
        self.test_mask =  self.test_mask.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        self.ones =  self.ones.to(device)
        return self
