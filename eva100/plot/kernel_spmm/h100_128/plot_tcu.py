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

path = '/home/shijinliang/module/Libra-sc25/res/h100/spmm/tf32'
path1 = '/home/shijinliang/module/Libra-sc25/res/h100/spmm/fp16'
# #CUDA-v2 + TCU-bitmap
# df1 = pd.read_csv(path1 + '/filter_h100_spmm_fp16_result_new_0217_128.csv')
# df1['libra_tf32'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
# df1['flash'] = df1['spmm_tcu']
# df1_tf32 = df1[['libra_tf32','dataSet','flash']]
# #TCGNN
# df3 = pd.read_csv(path + '/spmm_tcgnn_libra_h100_128.csv')
# df_res = pd.merge(df1_tf32, df3, on='dataSet', how='inner')  # 使用内连接（inner join）
# #DTC
# df4 = pd.read_csv(path + '/dtc_spmm_f32_n128.csv')
# df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）

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

num_edges = []
libra_G_tf32 = []
tcgnn = []
dtc_G = []
flash_G = []

total = 0
libra_effctive = 0
libra_speedup= []

for index, row in df_res.iterrows():
    compute = row['num_edges_x']*128*2
    # if (round((compute/row['libra_tf32'])*1e-6,4))>7000:
    #     continue
    libra_G_tf32.append(round((compute/row['libra_tf32'])*1e-6,4))
    tcgnn.append(round((compute/row['tcgnn'])*1e-6,4))
    dtc_G.append(round((compute/row['dtc'])*1e-6,4))
    flash_G.append(round((compute/row['flash'])*1e-6,4))
    num_edges.append(int(row['num_edges_x']))
    total+=1

print(total)

sorted_indices = sorted(range(len(num_edges)), key=lambda k: num_edges[k]) 
# sorted_indices = np.argsort(libra_G_tf32)

#按非零元进行排序
libra_G_tf32 = [libra_G_tf32[i] for i in sorted_indices]
tcgnn = [tcgnn[i] for i in sorted_indices]
dtc_G = [dtc_G[i] for i in sorted_indices]
flash_G = [flash_G[i] for i in sorted_indices]
#间隔取平均值
interval = 2
# 计算平均值的数量
num_intervals = len(libra_G_tf32) // interval
# 计算最后剩余的不足 interval 个数的数量
remainder = len(libra_G_tf32) % interval
num_edges = [num_edges[i] for i in sorted_indices]
num_edges = num_edges[::interval]
num_edges_str = []
for item in num_edges:
    num_edges_str.append(item)


# 使用列表推导式对每隔 interval 个值求平均值
libra_G_tf32_avg = [round(sum(libra_G_tf32[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

tcgnn_avg = [round(sum(tcgnn [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

dtc_G_avg = [round(sum(dtc_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

flash_G_avg = [round(sum(flash_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]
# 如果有剩余的数，计算剩余的数的平均值并添加到平均值列表中
if remainder > 0:
    # num_edges_str = num_edges_str[:-1]

    last_avg = round(sum(libra_G_tf32[num_intervals * interval:]) / remainder, 4)
    libra_G_tf32_avg.append(last_avg)
    
    last_avg = round(sum(tcgnn[num_intervals * interval:]) / remainder, 4)
    tcgnn_avg.append(last_avg)

    last_avg = round(sum(dtc_G[num_intervals * interval:]) / remainder, 4)
    dtc_G_avg.append(last_avg)

    last_avg = round(sum(flash_G[num_intervals * interval:]) / remainder, 4)
    flash_G_avg.append(last_avg)

fig, ax = plt.subplots(figsize=(7, 2.5))

num_edges_str = np.log10(num_edges_str)

# indices_to_remove = [i for i, x in enumerate(libra_G_tf32_avg) if x > 13000]
# libra_G_tf32_avg = [x for i, x in enumerate(libra_G_tf32_avg) if i not in indices_to_remove]
# tcgnn_avg = [x for i, x in enumerate(tcgnn_avg) if i not in indices_to_remove]
# dtc_G_avg = [x for i, x in enumerate(dtc_G_avg) if i not in indices_to_remove]
# flash_G_avg = [x for i, x in enumerate(flash_G_avg) if i not in indices_to_remove]
# num_edges_str = [x for i, x in enumerate(num_edges_str) if i not in indices_to_remove]

ax.scatter(num_edges_str, tcgnn_avg, color='lightgreen', s=12)
ax.scatter(num_edges_str, dtc_G_avg, color='plum', s=12)
ax.scatter(num_edges_str, flash_G_avg, color='lightblue', s=12)
ax.scatter(num_edges_str, libra_G_tf32_avg, color='blue', s=12)

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

# 显示图形
plt.savefig('/home/shijinliang/module/Libra-sc25/eva100/plot/kernel_spmm/h100_128/h100_spmm_tf32_128_tcu.png', dpi=800)
# 清空图形
plt.clf()
