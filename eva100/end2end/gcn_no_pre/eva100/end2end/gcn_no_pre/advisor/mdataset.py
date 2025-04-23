#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import Rabbit_cmake

from scipy.sparse import *

def func(x):
    '''
    node degrees function
    '''
    if x > 0:
        return x
    else:
        return 1
class GCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, featuredim, classes):
        super(GCN_dataset, self).__init__()

        self.graph = np.load('./dgl_dataset/best/' + data +'.npz')
        # print(self.graph)
        self.num_features = featuredim
        self.num_classes = classes
        
        self.avg_degree = -1
        self.avg_edgeSpan = -1

        self.init_edges()
        self.init_embedding()
        self.init_labels()

        # self.train_mask = torch.from_numpy(self.graph['train_mask'])
        # self.val_mask = torch.from_numpy(self.graph['val_mask'])
        # self.test_mask = torch.from_numpy(self.graph['test_mask'])
        
        
    def init_edges(self):
        # loading from a .npz graph file
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.edge_index = self.graph['edge_index_new']
        # self.perm_new = self.graph['perm_new']

        # src_li=self.graph['src_li']
        # dst_li=self.graph['dst_li']
        
        # self.num_nodes = self.graph['num_nodes']
        # # self.num_nodes=self.graph['num_nodes']+8-(self.graph['num_nodes']%8)
        # self.num_edges = len(src_li)
        # self.edge_index = np.stack([src_li, dst_li])
        #self.edge_index,_ = Rabbit_cmake.reorder(torch.IntTensor(self.edge_index), self.num_nodes)
        
        self.avg_degree = self.num_edges / self.num_nodes
        # self.avg_edgeSpan = np.mean(np.abs(np.subtract(src_li, dst_li)))
        
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        # Get degrees array.
        degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes, self.num_features)
    
    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.randint(low=0, high=self.num_classes, size=(self.num_nodes,))

    
    # def rabbit_reorder(self):
    #     '''
    #     If the decider set this reorder flag,
    #     then reorder and rebuild a graph CSR.
    #     otherwise skipped this reorder routine.
    #     Called from external
    #     '''
    #     self.edge_index = Rabbit.reorder(torch.IntTensor(self.edge_index))

    #     # Rebuild a new graph CSR according to the updated edge_index
    #     val = [1] * self.num_edges
    #     scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
    #     scipy_csr = scipy_coo.tocsr()
    #     self.column_index = torch.IntTensor(scipy_csr.indices)
    #     self.row_pointers = torch.IntTensor(scipy_csr.indptr)

    #     # Re-generate degrees array.
    #     degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
    #     self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()
        
    def to(self, device):
        # self.column_index = self.column_index.cuda()
        # self.row_pointers = self.row_pointers.cuda()
        # self.blockPartition = self.blockPartition.cuda()
        # self.edgeToColumn = self.edgeToColumn.cuda()
        # self.edgeToRow = self.edgeToRow.cuda()
        # self.train_mask =  self.train_mask.to(device)
        # self.val_mask =  self.val_mask.to(device)
        # self.test_mask =  self.test_mask.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        return self
