# #!/usr/bin/env python3
# import torch
# import sys
# import math
# import time 
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm.std import tqdm
# import MagicsphereGAT_cmake
# import MagicsphereGCN_cmake
# import numpy as np


# class MAGNNFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, X_prime, weights, attention_w, inputInfo):
#         ctx.inputInfo = inputInfo    
#         ctx.X_prime = X_prime
#         ctx.weights = weights

#         # GEMM node update
#         X_prime = torch.mm(X_prime, weights)
#         ctx.X_prime2 = X_prime
        

#         # SDDMM: edge feature computation. 
#         edge_feature = MagicsphereGAT_cmake.forward_gen(X_prime.size(1), inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, inputInfo.values, 
#                             inputInfo.x, inputInfo.x, inputInfo.max)
#         edge_feature = edge_feature * attention_w
#         ctx.edge_feature = edge_feature
#         # SpMM_AGNN: Neighbor AggreAGNNion.
#         X_prime =  MagicsphereGCN_cmake.forward_v2(
#         inputInfo.row_pointers, 
#         inputInfo.column_index, 
#         edge_feature, 
#         X_prime, 
#         inputInfo.num_nodes, 
#         X_prime.size(1), 
#         inputInfo.num_nodes_ori)


#         return X_prime
#     @staticmethod
#     def backward(ctx, d_output):
#         inputInfo = ctx.inputInfo
#         X_prime = ctx.X_prime
#         X_prime2 = ctx.X_prime2
#         weights = ctx.weights
#         edge_feature = ctx.edge_feature
        
#         # SPMM backward propaAGNNion.
#         d_input_prime =  MagicsphereGCN_cmake.forward_v2(
#         inputInfo.row_pointers, 
#         inputInfo.column_index, 
#         edge_feature, 
#         d_output, 
#         inputInfo.num_nodes, 
#         d_output.size(1), 
#         inputInfo.num_nodes_ori)
        
#         # GEMM backward propaAGNNion.
#         d_X_prime = torch.mm(d_input_prime, weights.transpose(0,1))
#         d_weights = torch.mm(X_prime.transpose(0,1), d_input_prime)
        
#         # d_attention = MagicsphereGAT_cmake.forward_gen_tf32(X_prime2.size(1), inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, inputInfo.values, 
#         #                     d_output, X_prime2.transpose(0,1), inputInfo.max)
#         d_attention = MagicsphereGAT_cmake.forward_gen(X_prime2.size(1), inputInfo.num_nodes, inputInfo.row_pointers, inputInfo.column_index, inputInfo.values, 
#                             d_output, d_output, inputInfo.max)
#         d_attention_w = torch.sum(d_attention).view(1)
#         return d_X_prime, d_weights, d_attention_w, None



# class AGNNConv(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(AGNNConv, self).__init__()
#         # gain1 = nn.init.calculate_gain("relu")
#         self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))

#         self.attention_w = torch.nn.Parameter(torch.randn(1))

#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weights.size(1))
#         self.weights.data.uniform_(-stdv, stdv)

        
#     def forward(self, X, inputInfo):

#         return MAGNNFunction.apply(X, self.weights.half(), self.attention_w.half(), inputInfo)


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
import LibraAGNN_new
class MAGNNSpmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att, X_prime, inputInfo):
        ctx.inputInfo = inputInfo    
        ctx.X_prime = X_prime
        ctx.att = att
        
        X_prime = LibraAGNN_new.forward_tf32_spmm(
        inputInfo.t_rowNew_offsetTensor_, 
        inputInfo.t_blockTensor_,
        inputInfo.t_columnTensor_, 
        inputInfo.t_window_rowTensor_,
        inputInfo.t_binaryTensor_,
        inputInfo.t_atomicTensor_,
        att[:inputInfo.nnz_c],

        inputInfo.c_row_offsetTensor_,
        inputInfo.c_rowTensor_, 
        inputInfo.c_colTensor_,
        inputInfo.c_atomicTensor_ ,
        att[-inputInfo.nnz_c :],

        X_prime, 
        
        inputInfo.parts_t,
        inputInfo.parts_c,
        X_prime.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_ori)[0]
        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        inputInfo = ctx.inputInfo
        X_prime = ctx.X_prime
        att = ctx.att
        
        #SDMM 求att的梯度
        d_attention = LibraAGNN_new.forward_tf32(
        inputInfo.t_rowNew_offsetTensor_, 
        inputInfo.t_blockTensor_,
        inputInfo.t_columnTensor_, 
        inputInfo.t_window_rowTensor_,
        inputInfo.t_binaryTensor_,

        inputInfo.c_row_offsetTensor_,
        inputInfo.c_rowTensor_, 
        inputInfo.c_colTensor_,

        d_output,
        X_prime.t(),
        
        inputInfo.nnz, 
        inputInfo.maxPart, 
        inputInfo.parts_t,

        d_output.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_ori,
        inputInfo.parts_c)[0]
        
        # SPMM backward propaAGNNion.
        d_input_prime = LibraAGNN_new.forward_tf32_spmm(
        inputInfo.t_rowNew_offsetTensor_, 
        inputInfo.t_blockTensor_,
        inputInfo.t_columnTensor_, 
        inputInfo.t_window_rowTensor_,
        inputInfo.t_binaryTensor_,
        inputInfo.t_atomicTensor_,
        att[:inputInfo.nnz_c],

        inputInfo.c_row_offsetTensor_,
        inputInfo.c_rowTensor_, 
        inputInfo.c_colTensor_,
        inputInfo.c_atomicTensor_ ,
        att[-inputInfo.nnz_c :],

        X_prime, 

        inputInfo.parts_t,
        inputInfo.parts_c,
        X_prime.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_ori)[0]
        
        return d_attention, d_input_prime, None

class MAGNNSpmm1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att, X_prime, inputInfo):
        #复原att
        # temp = inputInfo.values_templete.clone()
        # temp[temp!=0] = att
        # att = temp
        # SpMM: Neighbor AggreAGNNion.
        X_prime = MagicsphereGCN_cmake.forward_filter(inputInfo.row_pointers, inputInfo.column_index, att, inputInfo.values_templete,  X_prime, inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)
        #X_prime = MagicsphereGCN_cmake.forward_v2(inputInfo.row_pointers, inputInfo.column_index, att,  X_prime, inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)
        return X_prime

    @staticmethod
    def backward(ctx, X_prime_grad):
        return None, None, None

    
class MAGNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_prime, attention_w, inputInfo):
        
        # SDDMM: edge feature computation. 
        edge_feature = LibraAGNN_new.forward_tf32(
        inputInfo.t_rowNew_offsetTensor_, 
        inputInfo.t_blockTensor_,
        inputInfo.t_columnTensor_, 
        inputInfo.t_window_rowTensor_,
        inputInfo.t_binaryTensor_,

        inputInfo.c_row_offsetTensor_,
        inputInfo.c_rowTensor_, 
        inputInfo.c_colTensor_,

        X_prime,
        X_prime,

        inputInfo.nnz, 
        inputInfo.maxPart, 
        inputInfo.parts_t,

        X_prime.size(1), 
        inputInfo.num_nodes_ori, 
        inputInfo.num_nodes_ori,
        inputInfo.parts_c)[0]
                                                  
                                                  
        edge_feature = edge_feature * attention_w
       
        return edge_feature
    @staticmethod
    def backward(ctx, d_attention):
        d_attention_w = torch.sum(d_attention).view(1)
        return None, d_attention_w, None
    
class AGNNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AGNNConv, self).__init__()
        # gain1 = nn.init.calculate_gain("relu")
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))

        self.attention_w = torch.nn.Parameter(torch.randn(1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

        
    def forward(self, X, inputInfo):
        # 1. 特征降维
        X_prime = torch.mm(X, self.weights)
        
        # 2. 求att
        att = MAGNNFunction.apply(X_prime, self.attention_w, inputInfo)
        
        # # 3. exp
        # max_value= torch.max(att)
        # min_value= torch.min(att)
        # att = (att - min_value) / (max_value - min_value)
        #temp = att
        att = torch.exp(att)
        rows_sum = MAGNNSpmm.apply(att, inputInfo.ones, inputInfo)

        # 4. 特征更新
        h_prime = MAGNNSpmm.apply(att, X_prime, inputInfo)
        
        # # 5. softmax
        h_prime = h_prime.div(rows_sum)

        return h_prime