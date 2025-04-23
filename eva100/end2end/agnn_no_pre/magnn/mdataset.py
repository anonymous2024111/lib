#!/usr/bin/env python3
import torch
import numpy as np
import time
import MagicsphereBlock_cmake
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import Libra5Block
import MagicsphereMRabbit_cmake
from scipy.sparse import *

def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    return (sparse_matrix != transposed_matrix).nnz == 0
class MAGNN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, num_features, num_classes, density, window, wide):
        super(MAGNN_dataset, self).__init__()
        self.graph = np.load('dgl_dataset/mythroughput/' + data +'.npz')

        self.num_features = num_features
        self.num_classes = num_classes

        self.init_edges(density, window, wide)
        self.init_embedding()
        self.init_labels()
        self.init_others()

    def init_edges(self,density,  window, wide):
        self.src_li=self.graph['src_li']
        self.dst_li=self.graph['dst_li']
        self.num_nodes_ori  = self.graph['num_nodes']
        self.num_nodes  = self.graph['num_nodes']
        if self.num_nodes_ori%window !=0 :
            self.num_nodes = self.num_nodes_ori + window - self.num_nodes_ori%window 
        self.num_edges = len(self.src_li)
        self.edge_index = torch.from_numpy(np.stack([self.src_li, self.dst_li]))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        adj = scipy_coo.tocsr()


        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        self.partsize_t =32
        self.partsize_c = 32
        self.short_size = 3
        
        
        self.t_rowNew_offsetTensor, \
        self.t_blockTensor, \
        self.t_columnTensor, \
        self.t_window_rowTensor, \
        self.t_atomicTensor, \
        self.t_binaryTensor, \
        self.c_row_offsetTensor, \
        self.c_rowTensor, \
        self.c_colTensor, \
        self.c_atomicTensor = Libra5Block.block_tf32_tcu_cuda_binary_sddmm(self.row_pointers, self.column_index, self.partsize_t ,self.partsize_c, density,  window,  wide)
        
        self.nnz = self.t_blockTensor[-1]
        self.maxPart = torch.max(self.t_rowNew_offsetTensor[1:]-self.t_rowNew_offsetTensor[:-1])
        self.parts_t = self.t_atomicTensor.shape[0]
        self.parts_c = self.c_atomicTensor.shape[0]
        self.nnz_c = self.c_colTensor.shape[0]
        self.c_rowTensor.shape[0]
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_ori, self.num_features).to(dtype=torch.float16)
    
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
        self.t_rowNew_offsetTensor_=self.t_rowNew_offsetTensor.to(device)
        self.t_blockTensor_=self.t_blockTensor.to(device)
        self.t_columnTensor_=self.t_columnTensor.to(device)
        self.t_window_rowTensor_=self.t_window_rowTensor.to(device)
        self.t_atomicTensor_=self.t_atomicTensor.to(device)
        self.t_binaryTensor_=self.t_binaryTensor.to(device)
                
        self.c_row_offsetTensor_=self.c_row_offsetTensor.to(device)
        self.c_rowTensor_=self.c_rowTensor.to(device)
        self.c_atomicTensor_=self.c_atomicTensor.to(device)
        self.c_colTensor_=self.c_colTensor.to(device)
        
        
        self.x =  self.x.to(device)
        self.length =  torch.norm(self.x, dim=1)
        self.y =  self.y.to(device)
        self.ones =  self.ones.to(device)
        return self
