#!/usr/bin/env python3
import torch
import sys
import math
import time 
import torch.nn as nn
import torch.nn.functional as F
from tqdm.std import tqdm
import MagicsphereGAT_cmake
import MagicsphereGCN_cmake
import numpy as np


class MGATFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_prime, a0, a1, inputInfo):
        ctx.inputInfo = inputInfo    
        ctx.X_prime = X_prime
        
        output_a0, output_a1=MagicsphereGAT_cmake.tf32_a_feature(X_prime.size(1), inputInfo.num_nodes, X_prime, a0, a1)
        att = MagicsphereGAT_cmake.tf32_csr_v2(X_prime.size(1), inputInfo.num_nodes, inputInfo.row_pointers1, inputInfo.column_index1, output_a0, output_a1,inputInfo.num_edges)
        # att = MagicsphereGAT_cmake.tf32_csr(X_prime.size(1), inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, inputInfo.values, output_a0, output_a1, inputInfo.max, inputInfo.num_edges)

        return att

    @staticmethod
    def backward(ctx, att_grad):
        inputInfo = ctx.inputInfo
        X_prime = ctx.X_prime
        '''
        求a0,a1梯度
        '''
        #att_grad求a0梯度
        a0_tmp= MagicsphereGCN_cmake.forward_tf32_v2_csr(inputInfo.row_pointers, inputInfo.column_index, inputInfo.values_templete, att_grad, X_prime, inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)
        #按列求和，求出a0
        a0_grad = torch.mm(inputInfo.ones.t(), a0_tmp)
        
        #att_grad求a1梯度
        a1_tmp= MagicsphereGCN_cmake.forward_tf32_v2_csr(inputInfo.row_pointers, inputInfo.column_index, inputInfo.values_templete, att_grad, inputInfo.ones, inputInfo.num_nodes, inputInfo.ones.size(1), inputInfo.num_nodes_ori)
        #a1_tmp= MagicsphereGCN_cmake.forward_v2(inputInfo.row_pointers, inputInfo.column_index, att_grad, inputInfo.ones, inputInfo.num_nodes, inputInfo.ones.size(1), inputInfo.num_nodes_ori)
        a1_grad = torch.mm(a1_tmp.t(), X_prime)
        return None, a0_grad, a1_grad, None


class MGATSpmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att, X_prime, inputInfo):
        ctx.att = att
        ctx.X_prime = X_prime
        ctx.inputInfo = inputInfo

        X_prime = MagicsphereGCN_cmake.forward_tf32_v2_csr(inputInfo.row_pointers, inputInfo.column_index, inputInfo.values_templete, att, X_prime, inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)

        return X_prime

    @staticmethod
    def backward(ctx, X_prime_grad):
        X_prime = ctx.X_prime
        inputInfo = ctx.inputInfo
        att = ctx.att

        # 构建csr

        d_X_prime= MagicsphereGCN_cmake.forward_tf32_v2_csr(inputInfo.row_pointers, inputInfo.column_index, inputInfo.values_templete_t, att, X_prime_grad, inputInfo.num_nodes, X_prime_grad.size(1), inputInfo.num_nodes_ori)
        
        #根据X_prime，通过SDDMM反向传播求att_grad梯度
        d_att = MagicsphereGAT_cmake.tf32_sddmm_csr(X_prime.size(1), inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, inputInfo.values, X_prime_grad, X_prime, inputInfo.max, inputInfo.num_edges)

        return d_att, d_X_prime, None

class MGATSpmm1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att, X_prime, inputInfo):
        X_prime = MagicsphereGCN_cmake.forward_tf32_v2_csr(inputInfo.row_pointers, inputInfo.column_index,  inputInfo.values_templete, att, X_prime, inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)
        return X_prime

    @staticmethod
    def backward(ctx, X_prime_grad):
        return None, None, None

class GATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha):
        super(GATConv, self).__init__()
        self.alpha = alpha
        gain1 = nn.init.calculate_gain("relu")
        self.weights = torch.nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_normal_(self.weights.data, gain=gain1)
        self.a0 = torch.nn.Parameter(torch.zeros(size=(1, output_dim)))
        nn.init.xavier_normal_(self.a0.data, gain=gain1)
        self.a1 = torch.nn.Parameter(torch.zeros(size=(1, output_dim)))
        nn.init.xavier_normal_(self.a1.data, gain=gain1)

        self.output_dim = output_dim 
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, X, inputInfo):
        #特征降维
        X_prime = torch.mm(X, self.weights)
        #求att和用于反向传播的att_trans
        att = MGATFunction.apply(X_prime, self.a0, self.a1, inputInfo)
        #leakrelu
        att = self.leakyrelu(att)
        #exp
        max_value= torch.max(att)
        min_value= torch.min(att)
        att = (att - min_value) / (max_value - min_value)
        # sparse_mat = torch.sparse_csr_tensor(inputInfo.row_pointers1, inputInfo.column_index1, att, dtype=torch.float32)
        # # 将稀疏矩阵转换为密集矩阵
        # dense_mat = sparse_mat.to_dense()
        # # 对每行求最大值
        # max_values, _ = torch.max(dense_mat, dim=1, keepdim=True)
        # # 将每行的值减去对应的最大值
        # att = dense_mat - max_values
        
        att = torch.exp(att)
        rows_sum = MGATSpmm1.apply(att, inputInfo.ones, inputInfo)
        #dropout
        att = self.dropout(att)
        #特征更新
        h_prime = MGATSpmm.apply(att, X_prime, inputInfo)
        #softmax
        h_prime = h_prime.div(rows_sum)
        # print(rows_sum[0])
        return h_prime


