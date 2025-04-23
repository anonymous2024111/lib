#!/usr/bin/env python3
import torch
import math
from torch.nn.parameter import Parameter
import Libra6SpMM



class MGCNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, inputInfo):
        # X = torch.sparse.mm(edge_coo, X)
        ctx.save_for_backward(X, weights)
        ctx.inputInfo = inputInfo

        # GEMM node update
        X_prime = torch.mm(X, weights)

        # SpMM: Neighbor AggreAGNNion.
        
        X_prime = Libra6SpMM.forward_fp16(
        inputInfo.t_rowNew_offsetTensor_, 
        inputInfo.t_columnTensor_, 
        inputInfo.t_valueTensor_, 
        inputInfo.t_window_rowTensor_,
        inputInfo.t_atomicTensor_,

        inputInfo.c_row_offsetTensor_,
        inputInfo.c_rowTensor_, 
        inputInfo.c_atomicTensor_,
        inputInfo.c_colTensor_,
        inputInfo.c_valueTensor_, 

        inputInfo.c_row_offsetTensor_short_,
        inputInfo.c_rowTensor_short_, 
        inputInfo.c_atomicTensor_short_,
        inputInfo.c_colTensor_short_,
        inputInfo.c_valueTensor_short_, 

        X_prime, 
        inputInfo.parts_t, 
        inputInfo.parts_c, 
        inputInfo.partsize_c,
        inputInfo.parts_c_short, 
        X_prime.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_ori)[0]

        return X_prime.half()

    @staticmethod
    def backward(ctx, d_output):
        X, weights = ctx.saved_tensors
        inputInfo = ctx.inputInfo
        # SPMM backward propaAGNNion.
        d_input_prime =Libra6SpMM.forward_fp16(
        inputInfo.t_rowNew_offsetTensor_, 
        inputInfo.t_columnTensor_, 
        inputInfo.t_valueTensor_, 
        inputInfo.t_window_rowTensor_,
        inputInfo.t_atomicTensor_,

        inputInfo.c_row_offsetTensor_,
        inputInfo.c_rowTensor_, 
        inputInfo.c_atomicTensor_,
        inputInfo.c_colTensor_,
        inputInfo.c_valueTensor_, 

        inputInfo.c_row_offsetTensor_short_,
        inputInfo.c_rowTensor_short_, 
        inputInfo.c_atomicTensor_short_,
        inputInfo.c_colTensor_short_,
        inputInfo.c_valueTensor_short_, 

        d_output, 
        inputInfo.parts_t, 
        inputInfo.parts_c, 
        inputInfo.partsize_c,
        inputInfo.parts_c_short, 
        d_output.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_ori)[0]
        
        # GEMM backward propaAGNNion.
        d_input = torch.mm(d_input_prime.half(), weights.t())
        d_weights = torch.mm(X.t(), d_input_prime.half())
        return d_input, d_weights, None


class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNConv, self).__init__()
        self.weights = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
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
        return MGCNFunction.apply(X, self.weights.half(), inputInfo)


