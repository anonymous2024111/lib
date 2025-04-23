#!/usr/bin/env python3
import torch
import numpy as np
import time
import MagicsphereBlock_cmake
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix

import MagicsphereMRabbit_cmake
import Libra5Block
from scipy.sparse import *
    
class MGCN_dataset_tcu_cuda(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, featuredim, classes, density,  window, wide):
        super(MGCN_dataset_tcu_cuda, self).__init__()

        self.graph = np.load('dgl_dataset/mythroughput/' + data +'.npz')
        self.num_features = featuredim
        self.num_classes = classes

        self.init_edges(density, window, wide)
        self.init_embedding()
        self.init_labels()
        self.init_others()


    def init_edges(self,density, window, wide):
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
        self.t_valueTensor, \
        self.t_window_rowTensor, \
        self.t_atomicTensor, \
        self.t_binaryTensor, \
        self.c_row_offsetTensor, \
        self.c_rowTensor, \
        self.c_atomicTensor, \
        self.c_colTensor, \
        self.c_valueTensor, \
        self.c_row_offsetTensor_short, \
        self.c_rowTensor_short, \
        self.c_atomicTensor_short, \
        self.c_colTensor_short, \
        self.c_valueTensor_short = Libra5Block.block_tf32_tcu_cuda_binary(self.row_pointers, self.column_index,self.degrees, self.partsize_t ,self.partsize_c ,self.short_size, density,  window,  wide)

        # self.t_rowNew_offsetTensor, \
        # self.t_blockTensor, \
        # self.t_columnTensor, \
        # self.t_valueTensor, \
        # self.t_window_rowTensor, \
        # self.t_atomicTensor, \
        # self.t_binaryTensor = Libra5Block.block_csr_binary_tcu_part(self.row_pointers, self.column_index,self.degrees, self.partsize_t,  window,  wide)

        self.parts_t = self.t_atomicTensor.shape[0]
        self.parts_c = self.c_atomicTensor.shape[0]
        self.parts_c_short = self.c_atomicTensor_short.shape[0]
        
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
        self.t_rowNew_offsetTensor_=self.t_rowNew_offsetTensor.to(device)
        self.t_blockTensor_=self.t_blockTensor.to(device)
        self.t_columnTensor_=self.t_columnTensor.to(device)
        self.t_valueTensor_=self.t_valueTensor.to(device)
        self.t_window_rowTensor_=self.t_window_rowTensor.to(device)
        self.t_atomicTensor_=self.t_atomicTensor.to(device)
        self.t_binaryTensor_=self.t_binaryTensor.int().to(device)
                
        self.c_row_offsetTensor_=self.c_row_offsetTensor.to(device)
        self.c_rowTensor_=self.c_rowTensor.to(device)
        self.c_atomicTensor_=self.c_atomicTensor.to(device)
        self.c_colTensor_=self.c_colTensor.to(device)
        self.c_valueTensor_=self.c_valueTensor.to(device)

        self.c_row_offsetTensor_short_=self.c_row_offsetTensor_short.to(device)
        self.c_rowTensor_short_=self.c_rowTensor_short.to(device)
        self.c_atomicTensor_short_=self.c_atomicTensor_short.to(device)
        self.c_colTensor_short_=self.c_colTensor_short.to(device)
        self.c_valueTensor_short_=self.c_valueTensor_short.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        self.ones =  self.ones.to(device)
        return self