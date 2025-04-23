import torch
import numpy as np
from scipy.sparse import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('eva100/abalation/')
from gcn.mtest import *
from gcn.mdataset import *
from gat.mtest_gat import *
from gat.mdataset_gat import *
import csv

def norm(spmm):
    if spmm<10 :
        return '{:.2f}'.format(spmm)
    elif spmm < 100 :
        return '{:.1f}'.format(spmm)
    else:
        return '{:.0f}'.format(spmm)
    
# ablation-study
if __name__ == "__main__":
    # # 获取第一个可用的 GPU 设备
    # gpu_device = torch.cuda.current_device()
    
    # # 打印 GPU 设备的名称
    # gpu = torch.cuda.get_device_name(gpu_device)
    # print(gpu)
    

    hidden = [64, 128, 256, 512]

    epoches = 10
    
    dataset = ['reddit', 'ogb', 'AmazonProducts', 'IGB_medium', 'IGB_large']


    dataset = ['IGB_large']

    dimN = 64
    # #TF32
    # # CSV 文件路径
    file_name = './eva100/abalation/' + 'mma-motivation-' + '-H100.txt'
    # with open(file_name, 'w') as file:
    #     file.write('H100 : \n')
    for data in dataset:
        res = data
        
        # GAT - FP16
        # with-16x8
        inputInfo_16_gat = MGCN_dataset_m16_gat()
        inputInfo_16_gat.m_block_16_8_r(data, dimN)

        
        
        # with-8x8
        inputInfo_8_gcn = MGCN_dataset_m16()
        inputInfo_8_gcn.m_block_8_8_r(data, dimN)

        
        mma_16 = round((inputInfo_16_gat.degrees.size(0)/128)*2,0)
        mma_8 = round((inputInfo_8_gcn.degrees.size(0))/64,0)
        
        res = res + ' & ' + ' & ' + str(mma_16)  
        res = res + ' & ' + ' & ' + ' & ' + str(mma_8)
    
        reduction  =  (mma_16 - mma_8) /mma_16
        
        res = res + ' & ' + ' & ' + str(round(reduction*100,2)) + '\%'
        res = res + ' \\\\ ' 

        with open(file_name, 'a') as file:
            file.write(res + '\n')
        print(data + ' is successed!')
        print()
print('导出成功！')
print()


