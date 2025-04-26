#绘制Libra，cusaprse, sputnik, Rode的图
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns
def mean(input_list):
    return round((sum(input_list) / len(input_list)),1)

path = '/home/shijinliang/module/Libra-sc25/eva100/plot/res_4090'
#CUDA-v2 + TCU-BCRS
df1 = pd.read_csv(path + '/filter_4090_spmm_fp16_result_128_0220.csv')
df1['libra_fp16'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df1 = df1[['libra_fp16','dataSet']]
#RoDe
df2 = pd.read_csv(path + '/rode_result_spmm_f32_n128_0215.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
##Tir
df5 = pd.read_csv(path + '/result_tir_spmm_h100_128.csv')
df_res = pd.merge(df_res, df5, on='dataSet', how='inner')  # 使用内连接（inner join）


num_edges = []
libra_G = []
libra_G_fp16 = []
sputnik_G = []
cusparse_G = []
rode_G = []
tir_G = []

total = 0
libra_effctive = 0
libra_speedup= []

for index, row in df_res.iterrows():
    compute = row['num_edges1']*128*2
    if (round((compute/row['libra_fp16'])*1e-6,4))>13000:
        continue
    libra_G_fp16.append(round((compute/row['libra_fp16'])*1e-6,4))
    sputnik_G.append(round((compute/row['Sputnik_time'])*1e-6,4))
    cusparse_G.append(round((compute/row['cuSPARSE_time'])*1e-6,4))
    rode_G.append(round((compute/row['rode'])*1e-6,4))
    tir_G.append(round((compute/row['time'])*1e-6,4))
    num_edges.append(int(row['num_edges1']))
    total+=1

print(total)

sorted_indices = sorted(range(len(num_edges)), key=lambda k: num_edges[k]) 
# sorted_indices = np.argsort(libra_G_fp16)

#按非零元进行排序
libra_G_fp16 = [libra_G_fp16[i] for i in sorted_indices]
sputnik_G = [sputnik_G[i] for i in sorted_indices]
cusparse_G = [cusparse_G[i] for i in sorted_indices]
rode_G = [rode_G[i] for i in sorted_indices]
tir_G = [tir_G[i] for i in sorted_indices]
#间隔取平均值
interval = 2
# 计算平均值的数量
num_intervals = len(libra_G_fp16) // interval
# 计算最后剩余的不足 interval 个数的数量
remainder = len(libra_G_fp16) % interval
num_edges = [num_edges[i] for i in sorted_indices]
num_edges = num_edges[::interval]
num_edges_str = []
for item in num_edges:
    num_edges_str.append(item)


# 使用列表推导式对每隔 interval 个值求平均值
libra_G_fp16_avg = [round(sum(libra_G_fp16[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

sputnik_G_avg = [round(sum(sputnik_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

cusparse_G_avg = [round(sum(cusparse_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

rode_G_avg = [round(sum(rode_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

tir_G_avg = [round(sum(tir_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]
# 如果有剩余的数，计算剩余的数的平均值并添加到平均值列表中
if remainder > 0:
    # num_edges_str = num_edges_str[:-1]

    last_avg = round(sum(libra_G_fp16[num_intervals * interval:]) / remainder, 4)
    libra_G_fp16_avg.append(last_avg)
    
    last_avg = round(sum(sputnik_G[num_intervals * interval:]) / remainder, 4)
    sputnik_G_avg.append(last_avg)
    
    last_avg = round(sum(cusparse_G[num_intervals * interval:]) / remainder, 4)
    cusparse_G_avg.append(last_avg)
    
    last_avg = round(sum(rode_G[num_intervals * interval:]) / remainder, 4)
    rode_G_avg.append(last_avg)

    last_avg = round(sum(tir_G[num_intervals * interval:]) / remainder, 4)
    tir_G_avg.append(last_avg)

# plt.figure(figsize=(5.5, 3.5))  # 设置宽度为 10，高度为 6
fig, ax = plt.subplots(figsize=(7, 2.5))
#vs 

num_edges_str = np.log10(num_edges_str)
ax.scatter(num_edges_str, tir_G_avg, color='green', s=12, label='TIR')
ax.scatter(num_edges_str, cusparse_G_avg, color='khaki', s=12, label='cuSPARSE')
ax.scatter(num_edges_str, sputnik_G_avg, color='pink', s=12, label='Sputnik')
ax.scatter(num_edges_str, rode_G_avg, color='tomato', s=12, label='RoDe')
ax.scatter(num_edges_str, libra_G_fp16_avg, color='blue', s=12, label='Libra-FP16')

# # 设置 x 轴为整数刻度，间隔为 1
# plt.xticks(np.arange(np.floor(min(num_edges_str)), np.ceil(max(num_edges_str)) + 1, 1))
# 设置边框宽度和网格线
ax.spines['top'].set_linewidth(0.5)  # 设置顶部边框宽度
ax.spines['right'].set_linewidth(0.5)  # 设置右边框宽度
ax.spines['left'].set_linewidth(0.5)  # 设置左边框宽度
ax.spines['bottom'].set_linewidth(0.5)  # 设置底部边框宽度

# 启用网格线
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# 设置标签和标题
plt.xlabel('Log10(#nonzeros of matrix)')
plt.ylabel('TCGNN Average')
plt.title('Scatter Plot with Integer X-axis')
# ax.legend()
# 显示图形
plt.savefig('/home/shijinliang/module/Libra-sc25/eva100/plot/kernel_spmm/4090_128/4090_spmm_fp16_128_cuda.png', dpi=800)
# 清空图形
plt.clf()
