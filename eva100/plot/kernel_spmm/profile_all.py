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
dtc = []
flash= []
cusparse = []
sputnik = []
rode = []


tcgnn_= []
dtc_ = []
flash_ = []
cusparse_ = []
sputnik_ = []
rode_ = []

total = 0
total_tcgnn = 0
#统计128的数据

#统计TCU TF32
path = '/home/shijinliang/module/Libra-sc25/res/h100/spmm/tf32'
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
    dtc.append(round((row['dtc']/row['libra_tf32']),4))
    flash.append(round((row['flash']/row['libra_tf32']),4))
    total+=1

print()
# 统计分布
# 1. TC-GNN
a = 0
b = 0
c = 0
d = 0
for item in tcgnn:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("TC-GNN distribution:")  
a_percen = round((a/total_tcgnn*100), 2)
b_percen = round((b/total_tcgnn*100), 2)
c_percen = round((c/total_tcgnn*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
tcgnn_.append(a_percen)
tcgnn_.append(b_percen)
tcgnn_.append(c_percen)
tcgnn_.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("geo : ",stats.gmean(tcgnn) )
print("max : ",max(tcgnn) )
tcgnn_.append(round(stats.gmean(tcgnn),2) )
tcgnn_.append('>50x')
print()

# 2. DTC
a = 0
b = 0
c = 0
d = 0
for item in dtc:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("DTC-SpMM distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
dtc_.append(a_percen)
dtc_.append(b_percen)
dtc_.append(c_percen)
dtc_.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("geo : ",stats.gmean(dtc) )
print("max : ",max(dtc) )
dtc_.append(round(stats.gmean(dtc),2) )
dtc_.append(round(max(dtc),2))
print()

# 3. Flash
a = 0
b = 0
c = 0
d = 0
for item in flash:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("FlashSparse distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
flash_.append(a_percen)
flash_.append(b_percen)
flash_.append(c_percen)
flash_.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
# filtered_list = [x for x in flash if x != 1]
print("geo : ",stats.gmean(flash) )
print("max : ",max(flash) )
flash_.append(round(stats.gmean(flash),2) )
flash_.append(round(max(flash),2))
print()


#CUDA
path = '/home/shijinliang/module/Libra-sc25/res/h100/spmm/fp16'
#CUDA-v2 + TCU-BCRS
df1 = pd.read_csv(path + '/filter_h100_spmm_fp16_result_new_0218_128.csv')
df1['libra_fp16'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df1 = df1[['libra_fp16','dataSet']]
#RoDe
df2 = pd.read_csv(path + '/rode_spmm_f32_n128_0215.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
#Tir
df5 = pd.read_csv(path + '/result_tir_spmm_h100_128.csv')
df_res = pd.merge(df_res, df5, on='dataSet', how='inner')  # 使用内连接（inner join）

for index, row in df_res.iterrows():
    sputnik.append(round((row['Sputnik_time']/row['libra_fp16']),4))
    cusparse.append(round((row['cuSPARSE_time']/row['libra_fp16']),4))
    rode.append(round((row['rode']/row['libra_fp16']),4))
    total+=1

# 1. cusparse
a = 0
b = 0
c = 0
d = 0
for item in cusparse:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("cuSPARSE distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
cusparse_.append(a_percen)
cusparse_.append(b_percen)
cusparse_.append(c_percen)
cusparse_.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("geo : ",stats.gmean(cusparse) )
print("max : ",max(cusparse) )
cusparse_.append(round(stats.gmean(cusparse),2) )
cusparse_.append(round(max(cusparse),2))
print()

# 2. Sputnik
a = 0
b = 0
c = 0
d = 0
for item in sputnik:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("Sputnik distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
sputnik_.append(a_percen)
sputnik_.append(b_percen)
sputnik_.append(c_percen)
sputnik_.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("geo : ",stats.gmean(sputnik) )
print("max : ",max(sputnik) )
sputnik_.append(round(stats.gmean(sputnik),2) )
sputnik_.append(round(max(sputnik),2))
print()

# 3. Rode
a = 0
b = 0
c = 0
d = 0
for item in rode:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("RoDe distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
rode_.append(a_percen)
rode_.append(b_percen)
rode_.append(c_percen)
rode_.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("geo : ",stats.gmean(rode) )
print("max : ",max(rode) )
# rode_.append(round(stats.gmean(rode),2) )
rode_.append(round(stats.gmean(rode),2) )
rode_.append(round(max(rode), 2))
print()

'''
#RTX4090:
'''
tcgnn= []
dtc = []
flash= []
cusparse = []
sputnik = []
rode = []

tcgnn_1= []
dtc_1 = []
flash_1 = []
cusparse_1 = []
sputnik_1 = []
rode_1 = []

total = 0
total_tcgnn = 0
#统计32的数据
path = '/home/shijinliang/module/Libra-sc25/eva100/plot/res_4090'
#CUDA-v2 + TCU-bitmap
df1 = pd.read_csv(path + '/filter_4090_spmm_tf32_result_128_0221.csv')
df1['libra_tf32'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df1_tf32 = df1[['libra_tf32','dataSet']]
#TCGNN
df3 = pd.read_csv(path + '/spmm_tcgnn_libra_h100_128.csv')
df_res = pd.merge(df1_tf32, df3, on='dataSet', how='inner')  # 使用内连接（inner join）
#DTC
df4 = pd.read_csv(path + '/dtc_4090_128.csv')
df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）
df5 = pd.read_csv(path + '/4090_spmm_tf32_result_flash_128.csv')
df_res = pd.merge(df_res, df5, on='dataSet', how='inner')  # 使用内连接（inner join）

for index, row in df_res.iterrows():
    if row['tcgnn'] != 1000000000:
        tcgnn.append(round((row['tcgnn']/row['libra_tf32']),4))
        total_tcgnn +=1
    dtc.append(round((row['dtc']/row['libra_tf32']),4))
    flash.append(round((row['flash']/row['libra_tf32']),4))
    total+=1

print()
# 统计分布
# 1. TC-GNN
a = 0
b = 0
c = 0
d = 0
for item in tcgnn:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("TC-GNN distribution:")  
a_percen = round((a/total_tcgnn*100), 2)
b_percen = round((b/total_tcgnn*100), 2)
c_percen = round((c/total_tcgnn*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
tcgnn_1.append(a_percen)
tcgnn_1.append(b_percen)
tcgnn_1.append(c_percen)
tcgnn_1.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("geo : ",stats.gmean(tcgnn) )
print("max : ",max(tcgnn) )
tcgnn_1.append(round(stats.gmean(tcgnn),2) )
tcgnn_1.append('>50x')
print()

# 2. DTC
a = 0
b = 0
c = 0
d = 0
for item in dtc:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("DTC-SpMM distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
dtc_1.append(a_percen)
dtc_1.append(b_percen)
dtc_1.append(c_percen)
dtc_1.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("geo : ",stats.gmean(dtc) )
print("max : ",max(dtc) )
dtc_1.append(round(stats.gmean(dtc),2) )
dtc_1.append(round(max(dtc),2))
print()

# 3. Flash
a = 0
b = 0
c = 0
d = 0
for item in flash:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("FlashSparse distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
flash_1.append(a_percen)
flash_1.append(b_percen)
flash_1.append(c_percen)
flash_1.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
# filtered_list = [x for x in flash if x != 1]
print("geo : ",stats.gmean(flash) )
print("max : ",max(flash) )
flash_1.append(round(stats.gmean(flash),2) )
flash_1.append(round(max(flash),2))
print()


#CUDA
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

for index, row in df_res.iterrows():
    sputnik.append(round((row['Sputnik_time']/row['libra_fp16']),4))
    cusparse.append(round((row['cuSPARSE_time']/row['libra_fp16']),4))
    rode.append(round((row['rode']/row['libra_fp16']),4))
    total+=1

# 1. cusparse
a = 0
b = 0
c = 0
d = 0
for item in cusparse:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("cuSPARSE distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
cusparse_1.append(a_percen)
cusparse_1.append(b_percen)
cusparse_1.append(c_percen)
cusparse_1.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("geo : ",stats.gmean(cusparse) )
print("max : ",max(cusparse) )
cusparse_1.append(round(stats.gmean(cusparse),2) )
cusparse_1.append(round(max(cusparse),2))
print()

# 2. Sputnik
a = 0
b = 0
c = 0
d = 0
for item in sputnik:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("Sputnik distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
sputnik_1.append(a_percen)
sputnik_1.append(b_percen)
sputnik_1.append(c_percen)
sputnik_1.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("geo : ",stats.gmean(sputnik) )
print("max : ",max(sputnik) )
sputnik_1.append(round(stats.gmean(sputnik),2) )
sputnik_1.append(round(max(sputnik),2))
print()

# 3. Rode
a = 0
b = 0
c = 0
d = 0
for item in rode:
    if item < 1:
        a+=1
    elif item < 1.5:
        b+=1
    elif item < 2:
        c+=1
    else:
        d+=1
print("RoDe distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((100-a_percen-b_percen-c_percen),2)
rode_1.append(a_percen)
rode_1.append(b_percen)
rode_1.append(c_percen)
rode_1.append(d_percen)

print("<1 ",a_percen, '%')
print("<1.5 ", b_percen,  '%')
print("<2 ", c_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("geo : ",stats.gmean(rode) )
print("max : ",max(rode) )
rode_1.append(round(stats.gmean(rode),2) )
rode_1.append(round(max(rode), 2))
print()

output = []
text = 'TC-GNN & '
for i in range(6):
    if i<4:
        text += str(tcgnn_[i]) + '\% & '
    else:
        text += str(tcgnn_[i]) + ' & '
for i in range(6):
    if i<4:
        text += str(tcgnn_1[i]) + '\% & '
    elif i==4:
        text += str(tcgnn_1[i]) + ' & '
    else:
        text += str(tcgnn_1[i]) + 'x \\\\'
text +='\n'
output.append(text)

text = 'DTC-SpMM & '
for i in range(6):
    if i<4:
        text += str(dtc_[i]) + '\% & '
    else:
        text += str(dtc_[i]) + ' & '
for i in range(6):
    if i<4:
        text += str(dtc_1[i]) + '\% & '
    elif i==4:
        text += str(dtc_1[i]) + ' & '
    else:
        text += str(dtc_1[i]) + 'x \\\\'
text +='\n'
output.append(text)

text = 'FlashSparse & '
for i in range(6):
    if i<4:
        text += str(flash_[i]) + '\% & '
    else:
        text += str(flash_[i]) + ' & '
for i in range(6):
    if i<4:
        text += str(flash_1[i]) + '\% & '
    elif i==4:
        text += str(flash_1[i]) + ' & '
    else:
        text += str(flash_1[i]) + 'x \\\\'
text +='\n'
text +='\hline' 
text +='\n'
output.append(text)

text = 'cuSPARSE & '
for i in range(6):
    if i<4:
        text += str(cusparse_[i]) + '\% & '
    else:
        text += str(cusparse_[i]) + ' & '
for i in range(6):
    if i<4:
        text += str(cusparse_1[i]) + '\% & '
    elif i==4:
        text += str(cusparse_1[i]) + ' & '
    else:
        text += str(cusparse_1[i]) + 'x \\\\'
text +='\n'
output.append(text)

text = 'Sputnik & '
for i in range(6):
    if i<4:
        text += str(sputnik_[i]) + '\% & '
    else:
        text += str(sputnik_[i]) + ' & '
for i in range(6):
    if i<4:
        text += str(sputnik_1[i]) + '\% & '
    elif i==4:
        text += str(sputnik_1[i]) + ' & '
    else:
        text += str(sputnik_1[i]) + 'x \\\\'
text +='\n'
output.append(text)

text = 'RoDe & '
for i in range(6):
    if i<4:
        text += str(rode_[i]) + '\% & '
    else:
        text += str(rode_[i]) + ' & '
for i in range(6):
    if i<4:
        text += str(rode_1[i]) + '\% & '
    elif i==4:
        text += str(rode_1[i]) + ' & '
    else:
        text += str(rode_1[i]) + 'x \\\\'
text +='\n'
output.append(text)

print()


with open("/home/shijinliang/module/Libra-sc25/eva100/plot/kernel_spmm/output.txt", "w", encoding="utf-8") as file:
    file.writelines(output)

print("多行内容已写入到 output.txt 文件中")