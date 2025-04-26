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
path = '/home/shijinliang/module/Libra-sc25/res/h100/sddmm/fp16'
path1 = '/home/shijinliang/module/Libra-sc25/res/h100/sddmm/tf32'
#(CUDA-v2 + TCU-bitmap)
df1 = pd.read_csv(path + '/new_filter_h100_sddmm_fp16_result_new0221_32.csv')
# df1 = pd.read_csv(path + '/new_filter_h100_sddmm_fp16_result_128.csv')
df1['libra'] = df1[['8', '16', '24', '32', '40', '48', '56', '64', 'sddmm_tcu', 'sddmm_cuda']].min(axis=1)
df1['flash'] = df1[['sddmm_tcu']].min(axis=1)
#RoDe
df2 = pd.read_csv(path + '/rode_sddmm_f32_n32.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
#tcgnn
df3 = pd.read_csv(path + '/sddmm_tcgnn_libra_h100_32.csv')
df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）

num_edges = []
libra_G = []
libra_tf32_G = []
sputnik_G = []
cusparse_G = []
rode_G = []
tcgnn_G = []

total = 0
libra_effctive = 0
libra_speedup= []

for index, row in df_res.iterrows():
    compute = row.iloc[2]*32*2
    if (round((compute/row['libra'])*1e-6,4))>10000:
        continue
    libra_G.append(round((compute/row['libra'])*1e-6,4))
    libra_tf32_G.append(round((compute/row['flash'])*1e-6,4))
    if row['Sputnik_time']<0.001:
        row['Sputnik_time'] = 1000
    sputnik_G.append(round((compute/row['Sputnik_time'])*1e-6,4))
    rode_G.append(round((compute/row['rode'])*1e-6,4))
    tcgnn_G.append(round((compute/row['tcgnn'])*1e-6,4))
    num_edges.append(int(row.iloc[2]))
    if row['libra'] < row['rode']:
        libra_effctive+=1
        libra_speedup.append(round((row['rode']/row['libra']),4))
    total+=1

print("Libra加速 有效矩阵个数: " + str(libra_effctive) + " / " + str(total) + '; 平均加速比: ' + str(mean(libra_speedup)) + '; 最高加速比: ' + str(max(libra_speedup)) )

sorted_indices = sorted(range(len(num_edges)), key=lambda k: num_edges[k]) 

#按非零元进行排序
libra_G = [libra_G[i] for i in sorted_indices]
libra_tf32_G = [libra_tf32_G[i] for i in sorted_indices]
sputnik_G = [sputnik_G[i] for i in sorted_indices]
rode_G = [rode_G[i] for i in sorted_indices]
tcgnn_G = [tcgnn_G[i] for i in sorted_indices]

#间隔取平均值
interval = 2
# 计算平均值的数量
num_intervals = len(libra_G) // interval
# 计算最后剩余的不足 interval 个数的数量
remainder = len(libra_G) % interval
num_edges = [num_edges[i] for i in sorted_indices]
num_edges = num_edges[::interval]
num_edges_str = []
for item in num_edges:
    num_edges_str.append(item)

# 使用列表推导式对每隔 interval 个值求平均值
libra_G_avg = [round(sum(libra_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

libra_tf32_G_avg = [round(sum(libra_tf32_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

sputnik_G_avg = [round(sum(sputnik_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

rode_G_avg = [round(sum(rode_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

tcgnn_G_avg = [round(sum(tcgnn_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

# 如果有剩余的数，计算剩余的数的平均值并添加到平均值列表中
if remainder > 0:
    # num_edges_str = num_edges_str[:-1]
    last_avg = round(sum(libra_G[num_intervals * interval:]) / remainder, 4)
    libra_G_avg.append(last_avg)

    last_avg = round(sum(libra_tf32_G[num_intervals * interval:]) / remainder, 4)
    libra_tf32_G_avg.append(last_avg)
    
    last_avg = round(sum(sputnik_G[num_intervals * interval:]) / remainder, 4)
    sputnik_G_avg.append(last_avg)
    
    last_avg = round(sum(rode_G[num_intervals * interval:]) / remainder, 4)
    rode_G_avg.append(last_avg)
    
    last_avg = round(sum(tcgnn_G[num_intervals * interval:]) / remainder, 4)
    tcgnn_G_avg.append(last_avg)


fig, ax = plt.subplots(figsize=(5, 2.5))
num_edges_str = np.log10(num_edges_str)
num_edges_str = np.where(num_edges_str > 8.2, 8.0, num_edges_str)
print(len(num_edges_str))

ax.scatter(num_edges_str, sputnik_G_avg, color='pink', s=12, label='Sputnik')
ax.scatter(num_edges_str, rode_G_avg, color='tomato', s=12, label='RoDe')
# ax.scatter(num_edges_str, tcgnn_G_avg, color='lightgreen', s=12)
# ax.scatter(num_edges_str, libra_tf32_G_avg, color='lightblue', s=12)
ax.scatter(num_edges_str, libra_G_avg, color='blue', s=12)
print(len(libra_G_avg))



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
plt.savefig('/home/shijinliang/module/Libra-sc25/eva100/plot/kernel_sddmm/h100_32/h100_sddmm_fp16_32_cuda.png', dpi=800)
# 清空图形
plt.clf()
