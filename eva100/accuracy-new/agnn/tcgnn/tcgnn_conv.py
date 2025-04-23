#!/usr/bin/env python3
import torch
import sys
import math
import time 

from tqdm.std import tqdm
import TCGNN

n_heads = 1
n_output = 8

def gen_test_tensor(X_prime):
    n_rows = X_prime.size(0)
    n_cols = X_prime.size(1)
    
    X_new = []
    for i in range(n_rows):
        tmp = [i] * n_cols
        X_new.append(tmp)

    X_new = torch.FloatTensor(X_new).cuda()
    return X_new




class TCGNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, inputInfo):
        # X = torch.sparse.mm(edge_coo, X)
        ctx.save_for_backward(X, weights)
        ctx.inputInfo = inputInfo

        # GEMM node update
        X_prime = torch.mm(X, weights)
        
        # X_prime_t = torch.ones_like(X_prime)
        # X_prime_t = gen_test_tensor(X_prime)
        # print("=========Before AggreAGNNion========")
        # print(X_prime_t)
        # sys.exit(0)

        # SpMM: Neighbor AggreAGNNion.
        X_prime = TCGNN.forward(X_prime, inputInfo.row_pointers, inputInfo.column_index, inputInfo.blockPartition, inputInfo.edgeToColumn, inputInfo.edgeToRow)[0]
        # print("==========After Aggreation=========")
        # print(X_prime)
        # sys.exit(0)

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weights = ctx.saved_tensors
        inputInfo = ctx.inputInfo
        # SPMM backward propaAGNNion.
        d_input_prime = TCGNN.forward(d_output, inputInfo.row_pointers, inputInfo.column_index, inputInfo.blockPartition, inputInfo.edgeToColumn, inputInfo.edgeToRow)[0]

        # GEMM backward propaAGNNion.
        d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        return d_input, d_weights, None



class TCGNNFunction_AGNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, attention_w, inputInfo):

        ctx.save_for_backward(X, weights)
        ctx.inputInfo = inputInfo

        # GEMM node update
        X_prime = torch.mm(X, weights)
        
        # SDDMM: edge feature computation. 
        edge_feature = TCGNN.forward_ef(X_prime, inputInfo.row_pointers, inputInfo.column_index, inputInfo.blockPartition, inputInfo.edgeToColumn, inputInfo.edgeToRow)[0]

        # Edge Attention Generation: [n_e, n_head]       
        edge_attentions = torch.mm(edge_feature.unsqueeze(-1), attention_w).transpose(0,1).contiguous()
        # print(edge_attentions.size())
        ctx.edge_attentions = edge_attentions
        
        # SpMM_AGNN: Neighbor AggreAGNNion.
        X_prime = TCGNN.forward_AGNN(X_prime, inputInfo.row_pointers, inputInfo.column_index, edge_attentions, inputInfo.blockPartition, inputInfo.edgeToColumn, inputInfo.edgeToRow)[0]

        # ctx.save_for_backward(X, weights, inputInfo.row_pointers, inputInfo.column_index, inputInfo.edge_attentions, inputInfo.blockPartition, inputInfo.edgeToColumn, inputInfo.edgeToRow)
        # print("==========After Aggreation=========")
        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weights = ctx.saved_tensors
        inputInfo = ctx.inputInfo
        edge_attentions = ctx.edge_attentions
        # SPMM backward propaAGNNion.
        d_input_prime = TCGNN.forward_AGNN(d_output, inputInfo.row_pointers, inputInfo.column_index, edge_attentions, inputInfo.blockPartition, inputInfo.edgeToColumn, inputInfo.edgeToRow)[0]

        # GEMM backward propaAGNNion.
        d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)

        # attention weight back propaAGNNion.
        d_attention = TCGNN.forward_ef(d_output, inputInfo.row_pointers, inputInfo.column_index, inputInfo.blockPartition, inputInfo.edgeToColumn, inputInfo.edgeToRow)[0]
        # print(d_attention.size())
        d_attention_exp = d_attention[None, :].expand(8, -1)
        # print(d_attention_exp.size())

        d_attention_w = torch.mm(d_attention_exp, inputInfo.column_index[:, None].float()).transpose(0,1)
        # print(d_attention_w.size())

        return d_input, d_weights, d_attention_w, None



class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
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
        return TCGNNFunction.apply(X, self.weights, inputInfo)


class AGNNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AGNNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.attention_w = torch.nn.Parameter(torch.randn(1, n_heads))
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
        return TCGNNFunction_AGNN.apply(X, self.weights, self.attention_w,inputInfo)
