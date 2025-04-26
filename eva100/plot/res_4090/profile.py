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

tcgnn= []
sputnik = []
cusparse = []
rode = []
flash = []
total = 0
total_tcgnn = 0
libra_road = 0
#统计32的数据
df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/sddmm/tf32/new_filter_h100_sddmm_tf32_result_new0220_32.csv')
df1['libra'] = df1[['4', '8', '14', '18', '22', '30', '38', '50','sddmm_tcu', 'sddmm_cuda']].min(axis=1)
df1['flash'] = df1['sddmm_tcu']
df2 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/kernel_sddmm/h100_fp16_32/rode_sddmm_f32_n32_res.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
# #tcgnn
df3 = pd.read_csv('/home/shijinliang/module/Libra/tcgnn/sddmm_tcgnn_libra_h100_32.csv')
df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）
# 方图
# df4 = pd.read_csv('/home/shijinliang/module/DTC-sddmm_ASPLOS24-main/res/h100_32.csv')
# df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）

for index, row in df_res.iterrows():
    if row['tcgnn'] != 1000000000:
        tcgnn.append(round((row['tcgnn']/row['libra']),4))
        total_tcgnn +=1
    sputnik.append(round((row['Sputnik_time']/row['libra']),4))
    # cusparse.append(round((row['cuSPARSE_time']/row['libra']),4))
    rode.append(round((row['rode']/row['libra']),4))
    flash.append(round((row['flash']/row['libra']),4))
    total+=1
    if row['rode'] > row['libra']:
        libra_road+=1
#统计256的数据
# df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/sddmm/fp16/h100_sddmm_fp16_result_256.csv')
# df1['libra'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','sddmm_tcu', 'sddmm_cuda']].min(axis=1)
# df2 = pd.read_csv('/home/shijinliang/module/RoDe-main/res_spmm_h100/result__sddmm_f32_n256.csv')
# df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
# #tcgnn
# df3 = pd.read_csv('/home/shijinliang/module/Libra/tcgnn/tcgnn_libra_h100_256.csv')
# df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）
# # 方图
# df4 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/data_square.csv')
# df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）

# for index, row in df_res.iterrows():
#     tcgnn.append(round((row['tcgnn']/row['libra']),4))
#     sputnik.append(round((row['Sputnik_time']/row['libra']),4))
#     cusparse.append(round((row['cuSPARSE_time']/row['libra']),4))
#     rode.append(round((row['rode']/row['libra']),4))
#     total+=1

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

# 2. Sputnik
a = 0
b = 0
c = 0
d = 0
e = 0
for item in sputnik:
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
print("Sputnik distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((d/total*100), 2)
print("<1 ",a_percen, '%')
print("<1.2 ", b_percen,  '%')
print("<1.5 ", c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')
print("geo : ",stats.gmean(sputnik) )
print("max : ",max(sputnik) )
print()
# 3. cusparse
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
# print("tcgnn distribution:")  
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

# 4. Rode
a = 0
b = 0
c = 0
d = 0
e = 0
for item in rode:
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
print("Rode distribution: ", libra_road)  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((d/total*100), 2)
print("<1 ",a_percen, '%')
print("<1.2 ", b_percen,  '%')
print("<1.5 ", c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')

print("geo : ",stats.gmean(rode) )
print("max : ",max(rode) )
print()

# # 5. DTC
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
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((d/total*100), 2)
print("<1 ",a_percen, '%')
print("<1.2 ", b_percen,  '%')
print("<1.5 ", c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')

print("flash : ",stats.gmean(flash) )


print(total)