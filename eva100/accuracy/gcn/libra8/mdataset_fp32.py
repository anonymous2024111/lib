#!/usr/bin/env python3
import torch
import numpy as np
import time
import Libra3Block
import Libra3Rabbit
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import MagicsphereMRabbit
import Rabbit
import MagicsphereBlock
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
class MGCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, boundary, percen, miniTcuPercen):
        super(MGCN_dataset, self).__init__()

        self.graph = np.load('./dgl_dataset/accuracy/' + data +'.npz')
        # self.graph = np.load('/home/shijinliang/module/MGNN-final-v1/dgl_dataset/xuyouxuan/npz_have_mask/HeterophilousGraphDataset_Amazon-ratings.npz')
        # print(self.graph)
        self.num_features = self.graph['in_size'].item()
        self.num_classes = self.graph['out_size'].item()

        self.init_edges(boundary, percen, miniTcuPercen)
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


    def init_edges(self, boundary, percen, miniTcuPercen):
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
        
        # val = [1] * self.num_edges
        # scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        # scipy_csr = scipy_coo.tocsr()
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.column_index = torch.IntTensor(adj.indices)
        dd = (self.row_pointers[1:] - self.row_pointers[:-1]).to(torch.float)
        self.dd= torch.rsqrt(dd) 
        
        #找到小于5的节点的索引
        # indexes_less_than_5 = torch.nonzero(torch.lt(dd, 5), as_tuple=False).squeeze()
        # boundary = 7
        result_tensor = (dd.unsqueeze(1) > boundary).int()
        self.rowboundary = torch.sum(dd > boundary).item()
        
        #如果rowboundary超过了TCU计算的miniTcuPercen,则需要砍掉一部分
        if(self.rowboundary>(self.num_nodes_ori*miniTcuPercen)):
            self.rowboundary = self.num_nodes_ori*miniTcuPercen
        if (self.rowboundary%8)!=0:
            self.rowboundary=(int(self.rowboundary/8)+1)*8
        
        #print("TCU percentage: " + str(round(((self.rowboundary/self.num_nodes)*100),4))+'%')
            
        #Rabbit
        edge_index_new, self.permNew= Libra3Rabbit.reorder(torch.IntTensor(self.edge_index), self.num_nodes_ori, result_tensor)
        #edge_index_new, self.permNew= Rabbit.reorder(torch.IntTensor(self.edge_index),self.num_nodes_ori)

        # comsize = comsize1.tolist()
        #根据Rabbit的结构构成新的邻接矩阵
        # scipy_coo_new = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        adj_new = sp.coo_matrix((np.ones(len(src_li)), edge_index_new),
                        shape=(self.num_nodes,self.num_nodes),
                        dtype=np.float32)
        adj_new = adj_new + adj_new.T.multiply(adj_new.T > adj_new) - adj_new.multiply(adj_new.T > adj_new)
        self.column_index_new = torch.IntTensor(adj_new.indices)
        self.row_pointers_new = torch.IntTensor(adj_new.indptr)
        dd_new = (self.row_pointers_new[1:] - self.row_pointers_new[:-1]).to(torch.float32)
        dd_new= torch.rsqrt(dd_new)  
        
        #划分TCU和CUDA的数据
        self.c_rowTensor, self.c_colTensor, self.c_valueTensor, self.t_rowTensor, self.t_colTensor, self.t_valueTensor = Libra3Block.blockProcess8v2(self.row_pointers_new, self.column_index_new, dd_new, percen, self.rowboundary)
        #self.c_rowTensor1, self.c_colTensor1, self.c_valueTensor1, self.t_rowTensor1, self.t_colTensor1, self.t_valueTensor1 = Libra3Block.blockProcess8(self.row_pointers, self.column_index,self.dd,0.6,10)

        # #Rabbit
        # #self.edge_index, self.permNew= MagicsphereMRabbit.reorder(torch.IntTensor(self.edge_index),self.num_nodes_ori,6)
        # self.edge_index, self.permNew= Rabbit.reorder(torch.IntTensor(self.edge_index),self.num_nodes_ori)
        # val = [1] * self.num_edges
        # scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        # adj = scipy_coo.tocsr()
        
        # self.column_index = torch.IntTensor(adj.indices)
        # self.row_pointers = torch.IntTensor(adj.indptr)
        # dd = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        # self.dd=torch.tensor(dd, dtype=torch.float32) 
        # # self.dd=torch.ones(self.num_nodes, dtype=torch.float32) 
        # # self.dd= torch.rsqrt(dd)  
        #self.row_pointers1, self.column_index1, self.degrees=MagicsphereBlock.blockProcess8_4(self.row_pointers, self.column_index,self.dd)
        # # self.row_pointers1 = self.row_pointers
        # # self.column_index1 = self.column_index
        
        #self.c_rowTensor, self.c_colTensor, self.c_valueTensor, self.t_rowTensor, self.t_colTensor, self.t_valueTensor=Libra3Block.blockProcess8(self.row_pointers, self.column_index,self.dd,0.6,8)
        # # print(self.row_pointers[-1]/8)
        # # print((self.row_pointers[-1]/8)*64)
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        # 打印归一化后的特征
        # x = normalize(sp.csr_matrix(self.graph['features'],dtype=np.float32))
        # self.x = torch.from_numpy(np.array(x.todense())).to(torch.float16) 
        self.x = torch.from_numpy(self.graph['features'])
       
    
    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        # y =  encode_onehot(self.graph['labels'])
        # self.y = torch.from_numpy(np.where(y)[1])
        self.y = torch.from_numpy(self.graph['labels'])
        
    def init_others(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.ones = torch.ones(size=(self.num_nodes_ori,1), dtype=torch.float32)

    def to(self, device):
        self.c_rowTensor =  self.c_rowTensor.to(device)
        self.c_colTensor =  self.c_colTensor.to(device)     
        self.c_valueTensor = self.c_valueTensor.to(device)  
        self.t_rowTensor =  self.t_rowTensor.to(device)
        self.t_colTensor =  self.t_colTensor.to(device)     
        self.t_valueTensor = self.t_valueTensor.to(device)  
        
        # self.c_rowTensor1 =  self.c_rowTensor1.to(device)
        # self.c_colTensor1 =  self.c_colTensor1.to(device)     
        # self.c_valueTensor1 = self.c_valueTensor1.to(device)  
        # self.t_rowTensor1 =  self.t_rowTensor1.to(device)
        # self.t_colTensor1 =  self.t_colTensor1.to(device)     
        # self.t_valueTensor1 = self.t_valueTensor1.to(device)  
        
        # self.row_pointers1 =  self.row_pointers1.to(device)
        # self.column_index1 =  self.column_index1.to(device)
        # self.degrees =  self.degrees.to(device)

        self.train_mask =  self.train_mask.to(device)
        self.val_mask =  self.val_mask.to(device)
        self.test_mask =  self.test_mask.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        self.ones =  self.ones.to(device)
        return self
    
  