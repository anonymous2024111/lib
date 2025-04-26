import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Sample data (replace with your own data)
res = []
mydataset = ['Zd_Jac3', 'rma10', 'ex19']
# dataset = ['Si5H12', 'copter1', 'lowThrust_7', 'mario001', 'onetone2']
dataset = ['GaAsH6', 'transient', 'onetone2', 'copter1', 'brack2']
df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/sddmm/tf32/2new_filter_h100_sddmm_tf32_result_neww_128.csv')
cur = 0
for index, row in df1.iterrows():
    # if cur < 10 and row['density'] ==3 :
    #     print(row['dataSet'])
    if row['dataSet'] in dataset:
        print(row['dataSet'])
        temp = []
        temp.append(row['sddmm_cuda'] / row['8'])
        temp.append(row['sddmm_cuda'] / row['16'])
        temp.append(row['sddmm_cuda'] / row['24'])
        temp.append(row['sddmm_cuda'] / row['32'])
        temp.append(row['sddmm_cuda'] / row['40'])
        temp.append(row['sddmm_cuda'] / row['48'])
        temp.append(row['sddmm_cuda'] / row['56'])
        temp.append(row['sddmm_cuda'] / row['64'])
        res.append(temp)
        cur +=1
print(res)
xticks_labels = ['8', '16', '24', '32', '40', '48', '56', '64']
# Normalize data by row (using MinMaxScaler)
scaler = MinMaxScaler(feature_range=(0.5, 1.5))  # 归一化到 [0, 2] 范围
data_normalized = np.array([scaler.fit_transform(np.array(row).reshape(-1, 1)).flatten() for row in res])
# Create a heatmap
plt.figure(figsize=(9, 3))
sns.heatmap(res, annot=True,  fmt=".2f", cmap="Blues", cbar_kws={'label': 'Speedup'}, square=True, xticklabels=xticks_labels)

# Add labels (example, adjust according to your axes)
plt.xlabel('ngs')
plt.ylabel('dw')
plt.title('Heatmap Example')

plt.savefig('/home/shijinliang/module/Libra/eva100/plot/ablation/preprocess_sddmm' +'.png', dpi=800)
