#绘制Libra，cusaprse, sputnik, Rode的图
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns
from scipy import stats
def mean(input_list):
    return round((sum(input_list) / len(input_list)),1)

flash= []
tcgnn= []
sputnik = []
cusparse = []
rode = []
dtc = []
total = 0
total_tcgnn = 0
libra_road = 0
#统计128的数据

# df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/tf32/filter_h100_spmm_tf32_result_new2_0215_128.csv')
# #df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/tf32/filter_h100_spmm_tf32_result_new2_0217_128.csv')
# df1['libra_tf32'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
# df1_tf32 = df1[['libra_tf32','dataSet','spmm_tcu']]

# df2 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/kernel_spmm/h100_128/rode_spmm_f32_n128.csv')
# df_res = pd.merge(df1_tf32, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
# #tcgnn
# df3 = pd.read_csv('/home/shijinliang/module/Libra/tcgnn/spmm_tcgnn_libra_h100_128.csv')
# df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）
# # 方图
# df4 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/kernel_spmm/h100_128/dtc_spmm_f32_n128.csv')
# df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）
# df5 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/res_4090/result_tir_spmm_h100_128.csv')
# df_res = pd.merge(df_res, df5, on='dataSet', how='inner')  # 使用内连接（inner join）

path = '/home/shijinliang/module/Libra-sc25/res/h100/spmm/tf32'
path1 = '/home/shijinliang/module/Libra-sc25/res/h100/spmm/fp16'

df1 = pd.read_csv(path + '/filter_h100_spmm_tf32_result_new2_0221_128.csv')
df1['libra_tf32'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df1_tf32 = df1[['libra_tf32','dataSet']]
#TCGNN
df3 = pd.read_csv(path + '/spmm_tcgnn_libra_h100_128.csv')
df_res = pd.merge(df1_tf32, df3, on='dataSet', how='inner')  # 使用内连接（inner join）
#DTC
df4 = pd.read_csv(path + '/dtc_spmm_f32_n128.csv')
df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）
df5 = pd.read_csv(path + '/h100_spmm_tf32_result_flash_128.csv')
df_res = pd.merge(df_res, df5, on='dataSet', how='inner')  # 使用内连接（inner join）

for index, row in df_res.iterrows():
    if row['tcgnn'] != 1000000000:
        tcgnn.append(round((row['tcgnn']/row['libra_tf32']),4))
        total_tcgnn +=1
    # sputnik.append(round((row['Sputnik_time']/row['libra_tf32']),4))
    # cusparse.append(round((row['cuSPARSE_time']/row['libra_tf32']),4))
    # rode.append(round((row['rode']/row['libra_tf32']),4))
    dtc.append(round((row['dtc']/row['libra_tf32']),4))
    flash.append(round((row['flash']/row['libra_tf32']),4))
    total+=1

#统计256的数据
# df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/fp16/h100_spmm_fp16_result_256.csv')
# df1['libra'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
# df2 = pd.read_csv('/home/shijinliang/module/RoDe-main/res_spmm_h100/result__spmm_f32_n256.csv')
# df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
# #tcgnn
# df3 = pd.read_csv('/home/shijinliang/module/Libra/tcgnn/tcgnn_libra_h100_256.csv')
# df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）
# # 方图
# df4 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/data_square.csv')
# df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）

# for index, row in df_res.iterrows():
#     tcgnn.append(round((row['tcgnn']/row['libra_tf32']),4))
#     sputnik.append(round((row['Sputnik_time']/row['libra_tf32']),4))
#     cusparse.append(round((row['cuSPARSE_time']/row['libra_tf32']),4))
#     rode.append(round((row['ours_time']/row['libra_tf32']),4))
#     total+=1
print()
# 统计分布
# 1. TC-GNN
a = 0
b = 0
c = 0
d = 0
e = 0
for item in tcgnn:
    if item < 1:
        a+=1
    elif item < 1.2:
        b+=1
    elif item < 1.5:
        c+=1
    elif item < 2:
        d+=1
    else:
        e+=1
print("TC-GNN distribution:")  
a_percen = round((a/total_tcgnn*100), 2)
b_percen = round((b/total_tcgnn*100), 2)
c_percen = round((c/total_tcgnn*100), 2)
d_percen = round((d/total_tcgnn*100), 2)
print("<1 ",a_percen, '%')
print("<1.2 ", b_percen,  '%')
print("<1.5 ", c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')
print("geo : ",stats.gmean(tcgnn) )
print("max : ",max(tcgnn) )
print()
# # 2. Sputnik
# a = 0
# b = 0
# c = 0
# d = 0
# e = 0
# for item in sputnik:
#     if item < 1:
#         a+=1
#     elif item < 1.2:
#         b+=1
#     elif item < 1.5:
#         c+=1
#     elif item < 2:
#         d+=1
#     else:
#         e+=1
# print("Sputnik distribution:")  
# a_percen = round((a/total*100), 2)
# b_percen = round((b/total*100), 2)
# c_percen = round((c/total*100), 2)
# d_percen = round((d/total*100), 2)
# print("<1 ",a_percen, '%')
# print("<1.2 ", b_percen,  '%')
# print("<1.5 ", c_percen,  '%')
# print("<2 ", d_percen,  '%')
# print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')
# print("geo : ",stats.gmean(sputnik) )
# print("max : ",max(sputnik) )
# print()
# # 3. cusparse
# a = 0
# b = 0
# c = 0
# d = 0
# e = 0
# for item in cusparse:
#     if item < 1:
#         a+=1
#     elif item < 1.2:
#         b+=1
#     elif item < 1.5:
#         c+=1
#     elif item < 2:
#         d+=1
#     else:
#         e+=1
# print("cuSPARSE distribution:")  
# a_percen = round((a/total*100), 2)
# b_percen = round((b/total*100), 2)
# c_percen = round((c/total*100), 2)
# d_percen = round((d/total*100), 2)
# print("<1 ",a_percen, '%')
# print("<1.2 ", b_percen,  '%')
# print("<1.5 ", c_percen,  '%')
# print("<2 ", d_percen,  '%')
# print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')
# print("geo : ",stats.gmean(cusparse) )
# print("max : ",max(cusparse) )
# print()
# # 4. Rode
# a = 0
# b = 0
# c = 0
# d = 0
# e = 0
# for item in rode:
#     if item < 1:
#         a+=1
#     elif item < 1.2:
#         b+=1
#     elif item < 1.5:
#         c+=1
#     elif item < 2:
#         d+=1
#     else:
#         e+=1
# print("Rode distribution: ", libra_road)  
# a_percen = round((a/total*100), 2)
# b_percen = round((b/total*100), 2)
# c_percen = round((c/total*100), 2)
# d_percen = round((d/total*100), 2)
# print("<1 ",a_percen, '%')
# print("<1.2 ", b_percen,  '%')
# print("<1.5 ", c_percen,  '%')
# print("<2 ", d_percen,  '%')
# print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')

# print("geo : ",stats.gmean(rode) )
# print("max : ",max(rode) )
# print()
# 5. DTC
a = 0
b = 0
c = 0
d = 0
e = 0
for item in dtc:
    if item < 1:
        a+=1
    elif item < 1.2:
        b+=1
    elif item < 1.5:
        c+=1
    elif item < 2:
        d+=1
    else:
        e+=1
print("DTC distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((d/total*100), 2)
print("<1 ",a_percen, '%')
print("<1.2 ", b_percen,  '%')
print("<1.5 ", c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')

print("dtc : ",stats.gmean(dtc) )
print("max : ",max(dtc) )
print()

# 6. Flash
a = 0
b = 0
c = 0
d = 0
e = 0
for item in flash:
    if item < 1:
        a+=1
    elif item < 1.2:
        b+=1
    elif item < 1.5:
        c+=1
    elif item < 2:
        d+=1
    else:
        e+=1
print("Flash distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((d/total*100), 2)
print("<1 ",a_percen, '%')
print("<1.2 ", b_percen,  '%')
print("<1.5 ", c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')

# filtered_list = [x for x in flash if x > 1]
print("flash : ",stats.gmean(flash) )
print("max : ",max(flash) )

print(total)