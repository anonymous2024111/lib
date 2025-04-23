#!/usr/bin/env python3
import torch
import sys
import math
import time 
import torch.nn as nn
import torch.nn.functional as F
# from tqdm.std import tqdm
import MagicsphereGAT_cmake
import MagicsphereGCN_cmake
import numpy as np


class MGATFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_prime, a0, a1, inputInfo):
        ctx.inputInfo = inputInfo    
        ctx.X_prime = X_prime
        # ctx.a0 = a0
        # ctx.a1 = a1
        
        att= MagicsphereGAT_cmake.forward_tf32(X_prime.size(1), inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, inputInfo.values, X_prime, a0, a1,inputInfo.max)
        #过滤掉0元素
        # att = att[inputInfo.indices]
        return att

    @staticmethod
    def backward(ctx, att_grad):
        inputInfo = ctx.inputInfo
        X_prime = ctx.X_prime
        # a0 = ctx.a0
        # a1 = ctx.a1
        #复原att
        # temp = inputInfo.values_templete.clone()
        # temp[temp!=0] = att_grad
        # att_grad = temp
        '''
        求a0,a1梯度
        '''
        #att_grad求a0梯度
        a0_tmp= MagicsphereGCN_cmake.forward_tf32_filter(inputInfo.row_pointers, inputInfo.column_index, att_grad, inputInfo.values_templete, X_prime, inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)
        #按列求和，求出a0
        a0_grad = torch.mm(inputInfo.ones.t(), a0_tmp)
        #att_grad求a1梯度
        a1_tmp= MagicsphereGCN_cmake.forward_tf32_filter(inputInfo.row_pointers, inputInfo.column_index, att_grad, inputInfo.values_templete, inputInfo.ones, inputInfo.num_nodes, inputInfo.ones.size(1), inputInfo.num_nodes_ori)
        a1_grad = torch.mm(a1_tmp.t(), X_prime)
        
        '''
        求X_prime梯度
        '''
        #a0的扩充
        # a0_matrix = a0.expand(inputInfo.num_nodes_ori, -1)
        # a0_matrix = a0_matrix*1;
        # att_grad_trans = mGAT.trans_gat(inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, att_grad, inputInfo.values_templete, inputInfo.max)
        # d0_X_prime = mGCN.forward(inputInfo.row_pointers, inputInfo.column_index, att_grad_trans, a0_matrix, inputInfo.num_nodes, a0_matrix.size(1), inputInfo.num_nodes_ori)
        # d1_X_prime = torch.mm(a1_tmp, a1)
        # d_X_prime =  d0_X_prime + d1_X_prime
        # if torch.isnan(d_X_prime).any().item()==True or torch.isnan(a0_grad).any().item()==True or torch.isnan(a1_grad).any().item()==True :
        #     print(torch.isnan(att_grad).any().item())
            # print(torch.isnan(d_X_prime).any().item())
            # print(torch.isnan(a0_grad).any().item())
            # print(torch.isnan(a1_grad).any().item())

        return None, a0_grad, a1_grad, None


class MGATSpmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att, X_prime, inputInfo):
        #复原att
        # temp = inputInfo.values_templete.clone()
        # temp[temp!=0] = att
        # att = temp
        ctx.att = att
        ctx.X_prime = X_prime
        ctx.inputInfo = inputInfo
        # GEMM node update
        # SpMM: Neighbor AggreAGNNion.
        X_prime = MagicsphereGCN_cmake.forward_tf32_filter(inputInfo.row_pointers, inputInfo.column_index, att,inputInfo.values_templete, X_prime, inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)
        return X_prime

    @staticmethod
    def backward(ctx, X_prime_grad):
        X_prime = ctx.X_prime
        inputInfo = ctx.inputInfo
        att = ctx.att
        att_trans = MagicsphereGAT_cmake.trans_gat_tf32(inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, att, inputInfo.values_templete, inputInfo.max)
        #求X_prime_grad
        d_X_prime= MagicsphereGCN_cmake.forward_tf32(inputInfo.row_pointers, inputInfo.column_index, att_trans, X_prime_grad, inputInfo.num_nodes, X_prime_grad.size(1), inputInfo.num_nodes_ori)
        #根据X_prime，通过SDDMM反向传播求att_grad梯度
        d_att = MagicsphereGAT_cmake.forward_gen_tf32(X_prime.size(1), inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, inputInfo.values, X_prime_grad, X_prime, inputInfo.max)
        #过滤掉0元素
        # d_att = d_att[inputInfo.indices]
        return d_att, d_X_prime, None
class MGATSpmm1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att, X_prime, inputInfo):
        #复原att
        # temp = inputInfo.values_templete.clone()
        # temp[temp!=0] = att
        # att = temp
        ctx.att = att
        ctx.X_prime = X_prime
        ctx.inputInfo = inputInfo
        # GEMM node update
        # SpMM: Neighbor AggreAGNNion.
        X_prime = MagicsphereGCN_cmake.forward_tf32_filter(inputInfo.row_pointers, inputInfo.column_index, att,inputInfo.values_templete, X_prime, inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)
        return X_prime

    @staticmethod
    def backward(ctx, X_prime_grad):
            return None, None, None
# class MGATSoftmax(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, att, inputInfo, rows_sum):
#         ctx.inputInfo = inputInfo
#         ctx.rows_sum = rows_sum
#         ctx.att = att
#         # GEMM node update
#         # SpMM: Neighbor AggreAGNNion.
#         att=mGATtf32.softmax_gat(inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, att, rows_sum, inputInfo.max, inputInfo.num_nodes_ori)    
#         return att

#     @staticmethod
#     def backward(ctx, att_grad):
#         inputInfo = ctx.inputInfo
#         rows_sum = ctx.rows_sum
#         att = ctx.att
#         d_att=mGATtf32.softmax_gat(inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, att_grad, rows_sum, inputInfo.max, inputInfo.num_nodes_ori)    
#         d_rows_sum = mGATtf32.softmax_gat(inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, att_grad, att, inputInfo.max, inputInfo.num_nodes_ori)    
#         print(torch.isnan(d_att).any().item())
#         return d_att, None, None

###################################
# Definition of each conv layers
###################################

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
        att = torch.exp(att)
        #temp = att
        # att = torch.exp(att/max_value)
        rows_sum = MGATSpmm1.apply(att, inputInfo.ones, inputInfo)
        # rows_sum = torch.where(rows_sum == 0, torch.ones_like(rows_sum), rows_sum)
        #dropout
        att = self.dropout(att)
        #特征更新
        h_prime = MGATSpmm.apply(att, X_prime, inputInfo)
        #softmax
        h_prime = h_prime.div(rows_sum)
        # print(rows_sum[0])
        return h_prime
        

         