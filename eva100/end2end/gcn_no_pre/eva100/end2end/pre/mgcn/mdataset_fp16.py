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

        self.graph = np.load('./dgl_dataset/mythroughput/' + data +'.npz')
        self.num_features = featuredim
        self.num_classes = classes

        self.init_edges()
        self.init_embedding()
        self.init_labels()
        self.init_others()


    def init_edges(self):
        src_li=self.graph['src_li']
        dst_li=self.graph['dst_li']
        
        self.num_nodes_ori = self.graph['num_nodes']
        self.num_nodes=self.graph['num_nodes']+8-(self.graph['num_nodes']%8)
        self.num_edges = len(src_li)
        self.edge_index = np.stack([src_li, dst_li])

        start_time = time.time()
        self.edge_index, self.permNew= MagicsphereMRabbit_cmake.reorder_m(torch.IntTensor(self.edge_index),self.num_nodes_ori,3)
        rabbit_time = time.time()  
        self.rabbit =  round((rabbit_time - start_time), 4)
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        dd = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        dd=torch.tensor(dd, dtype=torch.float32) 
        dd= torch.rsqrt(dd).to(torch.float16)  
        start_time = time.time()
        self.row_pointers, self.column_index, self.degrees=MagicsphereBlock_cmake.blockProcess8_8(self.row_pointers, self.column_index,dd)
        partition_time = time.time()  
        

        self.partition = round((partition_time - start_time), 4)

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        # 打印归一化后的特征
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
    
  