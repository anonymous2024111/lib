#!/usr/bin/env python3
import torch
import numpy as np
import time
import FS_Block
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from torch_geometric.datasets import Reddit
from torch_geometric.data import DataLoader
from scipy.sparse import *

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                            dtype=np.int32)
    return labels_onehot
    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    return (sparse_matrix != transposed_matrix).nnz == 0
class Reddit_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self):
        super(Reddit_dataset, self).__init__()

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

        self.train_mask = self.graph.train_mask
        self.val_mask = self.graph.val_mask
        self.test_mask = self.graph.test_mask


    def init_edges(self):
        self.src_li=self.graph.edge_index[0]
        self.dst_li=self.graph.edge_index[1]
        self.num_nodes = self.graph.x.shape[0]
        self.num_nodes_ori = self.graph.x.shape[0]
        self.num_edges = len(self.src_li)
        self.edge_index = torch.from_numpy(np.stack([self.src_li, self.dst_li]))

        #Rabbit
        # self.edge_index_new, self.perm_new, self.m_edge_index_new, self.m_perm_new, self.l_comesNew= MagicsphereMRabbit_cmake.reorder(torch.IntTensor(self.edge_index),self.num_nodes_ori,1)

        #self-loop
        adj = sp.coo_matrix((np.ones(len(self.src_li)), self.edge_index),
                        shape=(self.num_nodes,self.num_nodes),
                        dtype=np.float32)
        is_sym = is_symmetric(adj)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        is_sym = is_symmetric(adj)
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        dd = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        dd=torch.tensor(dd, dtype=torch.float32) 
        dd= torch.rsqrt(dd) 
        #bcsr
        self.row_pointers, \
        self.column_index, \
        self.degrees, \
        self.t_window_rowTensor, \
        self.t_atomicTensor = FS_Block.blockProcess_tf32_balance(self.row_pointers, self.column_index, dd, 8, 4, 32)
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        # 打印归一化后的特征
        # x = normalize(sp.csr_matrix(self.graph['features'],dtype=np.float32))
        # self.x = torch.from_numpy(np.array(x.todense())).to(torch.float16) 
        self.x = self.graph.x
       
    
    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        # y =  encode_onehot(self.graph['labels'])
        # self.y = torch.from_numpy(np.where(y)[1])
        self.y = self.graph.y
        
    def init_others(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.ones = torch.ones(size=(self.num_nodes_ori,1), dtype=torch.float32)

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
        self.y =  self.y.to(device)
        return self
    
  