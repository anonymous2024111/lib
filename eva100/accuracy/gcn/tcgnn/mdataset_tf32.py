#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from config import *
import TCGNN

from scipy.sparse import *

def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    return (sparse_matrix != transposed_matrix).nnz == 0
class MGCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data):
        super(MGCN_dataset, self).__init__()

        self.graph = np.load('/home/shijinliang/module/MGNN-final-v1/dgl_dataset/accuracy/' + data +'.npz')
        # self.graph = np.load('/home/shijinliang/module/MGNN-final-v1/dgl_dataset/xuyouxuan/npz_have_mask/HeterophilousGraphDataset_Amazon-ratings.npz')
        # print(self.graph)
        self.num_features = self.graph['in_size'].item()
        self.num_classes = self.graph['out_size'].item()

        self.init_edges()
        self.init_embedding()
        self.init_labels()

        self.train_mask = torch.from_numpy(self.graph['train_mask'])
        self.val_mask = torch.from_numpy(self.graph['val_mask'])
        self.test_mask = torch.from_numpy(self.graph['test_mask'])
        
        self.init_tcgnn()
        
        # print()

    def init_tcgnn(self):

        #########################################
        ## Compute TC-GNN related graph MetaData.
        #########################################
        self.num_row_windows = (self.num_nodes + BLK_H - 1) // BLK_H
        self.edgeToColumn = torch.zeros(self.num_edges, dtype=torch.int)
        self.edgeToRow = torch.zeros(self.num_edges, dtype=torch.int)
        self.blockPartition = torch.zeros(self.num_row_windows, dtype=torch.int)
        
        TCGNN.preprocess(self.column_index, self.row_pointers, self.num_nodes,  \
                BLK_H,	BLK_W, self.blockPartition, self.edgeToColumn, self.edgeToRow)
        
    def init_edges(self):
        # loading from a .npz graph file
        src_li=self.graph['src_li']
        dst_li=self.graph['dst_li']
        
        self.num_nodes_ori = self.graph['num_nodes']
        self.num_nodes=self.graph['num_nodes']+8-(self.graph['num_nodes']%8)
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
        # Get degrees array.
        degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        # 打印归一化后的特征
        self.x = torch.from_numpy(self.graph['features']).cuda()
       
    
    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        # y =  encode_onehot(self.graph['labels'])
        # self.y = torch.from_numpy(np.where(y)[1])
        self.y = torch.from_numpy(self.graph['labels']).cuda()

    
    def to(self, device):
        self.column_index = self.column_index.cuda()
        self.row_pointers = self.row_pointers.cuda()
        self.blockPartition = self.blockPartition.cuda()
        self.edgeToColumn = self.edgeToColumn.cuda()
        self.edgeToRow = self.edgeToRow.cuda()
        self.train_mask =  self.train_mask.to(device)
        self.val_mask =  self.val_mask.to(device)
        self.test_mask =  self.test_mask.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        return self