#!/usr/bin/env python3
import torch
import sys
import math
import time 
import torch.nn as nn
from torch.nn.parameter import Parameter
from tqdm.std import tqdm
import Libra3GCN
import LibraGCN
import MagicsphereGCN

def gen_test_tensor(X_prime):
    n_rows = X_prime.size(0)
    n_cols = X_prime.size(1)
    
    X_new = []
    for i in range(n_rows):
        tmp = [i] * n_cols
        X_new.append(tmp)

    X_new = torch.FloatTensor(X_new).cuda()
    return X_new



class MGCNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, inputInfo):
        # X = torch.sparse.mm(edge_coo, X)
        ctx.save_for_backward(X, weights)
        ctx.inputInfo = inputInfo

        # GEMM node update
        X_prime = torch.mm(X, weights)

        # SpMM: Neighbor AggreAGNNion.
        X_prime = Libra3GCN.forward_tf32_v10(inputInfo.c_rowTensor,inputInfo.c_colTensor,inputInfo.c_valueTensor,
                                         inputInfo.t_rowTensor,inputInfo.t_colTensor,inputInfo.t_valueTensor, X_prime,
                                         inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori, inputInfo.rowboundary)
 
        # X_prime= MagicsphereGCN.forward_tf32(inputInfo.row_pointers1, inputInfo.column_index1, inputInfo.degrees, 
        #                        X_prime, inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)
        # print("==========After Aggreation=========")
        # print(X_prime)
        
        
        # X_prime = Libra3GCN.forward_tf32_v6(inputInfo.c_rowTensor1,inputInfo.c_colTensor1,inputInfo.c_valueTensor1,
        #                                  inputInfo.t_rowTensor1,inputInfo.t_colTensor1,inputInfo.t_valueTensor1, X_prime,
        #                                  inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)
        # sys.exit(0)

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        # print("back")
        X, weights = ctx.saved_tensors
        inputInfo = ctx.inputInfo
        # SPMM backward propaAGNNion.
        
        # d_input_prime = Libra3GCN.forward_tf32_v6(inputInfo.c_rowTensor1,inputInfo.c_colTensor1,inputInfo.c_valueTensor1,
        #                                  inputInfo.t_rowTensor1,inputInfo.t_colTensor1,inputInfo.t_valueTensor1, d_output,
        #                                  inputInfo.num_nodes, d_output.size(1), inputInfo.num_nodes_ori)
        
        # nan_indices = torch.nonzero(torch.isnan(d_input_prime))
        # print(nan_indices.shape)
        # x = nan_indices[0][0]
        # y = nan_indices[0][1]
        # print(d_input_prime[x][y])
        # print(d_output[:,y])
        d_input_prime = Libra3GCN.forward_tf32_v10(inputInfo.c_rowTensor,inputInfo.c_colTensor,inputInfo.c_valueTensor,
                                         inputInfo.t_rowTensor,inputInfo.t_colTensor,inputInfo.t_valueTensor, d_output,
                                         inputInfo.num_nodes, d_output.size(1), inputInfo.num_nodes_ori, inputInfo.rowboundary)
 
        # print( torch.sum(torch.isinf(d_output)).item())
        # d_input_prime = MagicsphereGCN.forward_tf32(inputInfo.row_pointers1, inputInfo.column_index1, inputInfo.degrees, 
        #                        d_output, inputInfo.num_nodes, d_output.size(1), inputInfo.num_nodes_ori)
        # print( torch.sum(torch.isnan(d_input_prime)).item())

        # GEMM backward propaAGNNion.
        d_input = torch.mm(d_input_prime, weights.t())
        d_weights = torch.mm(X.t(), d_input_prime)
        return d_input, d_weights, None


class dropout_gat:
    def __init__(self) :
        # 构造函数，用于初始化对象的属性
        self.ones = torch.ones(10, 2, dtype=torch.float32)
###################################
# Definition of each conv layers
###################################

class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNConv, self).__init__()
        #self.weights = Parameter(torch.ones(input_dim, output_dim))
        self.weights = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        # if self.weights is not None:
        #     nn.init.xavier_uniform_(self.weights)
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)



    def forward(self, X, inputInfo):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        return MGCNFunction.apply(X, self.weights, inputInfo)
