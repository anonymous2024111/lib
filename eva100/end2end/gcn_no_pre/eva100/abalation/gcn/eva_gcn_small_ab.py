import torch
import numpy as np
from scipy.sparse import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('eva100/abalation/gcn/')
from mtest import *
from mdataset import *
import csv

# ablation-study
if __name__ == "__main__":
    # 获取第一个可用的 GPU 设备
    gpu_device = torch.cuda.current_device()
    
    # 打印 GPU 设备的名称
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)
   
    dataset = ['Reddit2', 'ovcar', 'amazon','amazon0505',
            'yelp', 'sw620', 'dd',
            'HR_NO', 'HU_NO', 'ell', 'GitHub',
            'artist', 'comamazon', 
            'yeast', 'blog', 'DGraphFin', 'reddit', 'ogb', 'AmazonProducts', 'IGB_medium', 'IGB_large']
    

    hidden = [64, 128, 256, 512, 1024]

    epoches = 1000
    
    dataset32 = ['blog', 'reddit', 'amazon', 'IGB_medium','AmazonProducts']
    hugedataset32 = [ 'IGB_medium','AmazonProducts']
    dataset = ['blog', 'reddit', 'amazon', 'IGB_medium','AmazonProducts']
    hugedataset = [ 'IGB_medium','AmazonProducts']
    # dataset = ['blog', 'dd']
    # hidden = [64, 128]
    head= ['Dataset', 'Dim', 'Initial-16x1', 'with-16x1-MR-v2', 'with-8x1-v2', 'with-8x1-MR-v2']
    
    # #TF32
    # # CSV 文件路径
    csv_file = './eva100/abalation/gcn/' + 'tf32' + '-H100.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    #TF32
    for data in dataset32:
        for dimN in hidden:
            if data in hugedataset32 and dimN==1024 :
                break
            res = []
            res.append(data)
            res.append(str(dimN))
            
            # Initial-16x1
            inputInfo_16 = MGCN_dataset_m32()
            inputInfo_16.m_block_16_4_r(data, dimN)
            #res
            mgcn32_16 = test_tf32_v2_16(data, epoches, dimN, inputInfo_16)
            del inputInfo_16
            res.append(str(mgcn32_16))
            
            # with-16x1 +MR
            inputInfo_16_mr = MGCN_dataset_m32()
            inputInfo_16_mr.m_block_16_4_mr(data, dimN)
            # 16x1 v2
            mgcn32_16_mr = test_tf32_v2_16(data, epoches, dimN, inputInfo_16_mr)
            # del inputInfo_8
            
            # with-8x1
            inputInfo_8 = MGCN_dataset_m32()
            inputInfo_8.m_block_8_4_r(data, dimN)
            # 8x1 v2
            mgcn32_8_v2 = test_tf32_v2(data, epoches, dimN, inputInfo_8)
            # del inputInfo_8
            
            # with-8x1 +MR
            inputInfo_8_mr = MGCN_dataset_m32()
            inputInfo_8_mr.m_block_8_4_mr(data, dimN)
            # 8x1+MR v2
            mgcn32_8_mr_v2 = test_tf32_v2(data, epoches, dimN, inputInfo_8_mr)
            # del inputInfo_8_mr

            res.append(str(mgcn32_16_mr))
            res.append(str(mgcn32_8_v2))
            res.append(str(mgcn32_8_mr_v2))
            
            # res.append(str(round((mgcn32_16/mgcn32_8_v2),4)))
            # res.append(str(round((mgcn32_16/mgcn32_8_v2),4)))
            # res.append(str(round((mgcn32_16/mgcn32_8_mr_v2),4)))
            # print("Speed up v2-8x1 :" + str(round((mgcn32_16/mgcn32_8_v2),4)))
            # print("Speed up v2-MR :" + str(round((mgcn32_16/mgcn32_8_mr_v2),4)))
            
            # 写入 CSV 文件
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
        print(data + ' is successed!')
        print()
    # print(f'{csv_file} 导出成功！')
    # print()
    
    # # FP16
    # CSV 文件路径
    csv_file = './eva100/abalation/gcn/' + 'fp16' + '-H100.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
        
    # FP16
    for data in dataset:
        for dimN in hidden:
            if data in hugedataset and dimN==1024 :
                break
            res = []
            res.append(data)
            res.append(str(dimN))
            
            # Initial-16x1
            inputInfo_16 = MGCN_dataset_m16()
            inputInfo_16.m_block_16_8_r(data, dimN)
            #res
            mgcn16_16 = test_fp16_v2_16(data, epoches, dimN, inputInfo_16)
            del inputInfo_16
            res.append(str(mgcn16_16))
            
             # with-16x1
            inputInfo_16_mr = MGCN_dataset_m16()
            inputInfo_16_mr.m_block_16_8_mr(data, dimN)
            # 16x1 v2
            mgcn16_16_mr = test_fp16_v2_16(data, epoches, dimN, inputInfo_16_mr)
            del inputInfo_16_mr
            
            # with-8x1
            inputInfo_8 = MGCN_dataset_m16()
            inputInfo_8.m_block_8_8_r(data, dimN)
            # 8x1 v2
            mgcn16_8_v2 = test_fp16_v2(data, epoches, dimN, inputInfo_8)
            del inputInfo_8
            
            
            # + MR
            inputInfo_8_mr = MGCN_dataset_m16()
            inputInfo_8_mr.m_block_8_8_mr(data, dimN)
            # 8x1+MR v2
            mgcn16_8_mr_v2 = test_fp16_v2(data, epoches, dimN, inputInfo_8_mr)
            del inputInfo_8_mr
            
            # v2
            res.append(str(mgcn16_16_mr))
            res.append(str(mgcn16_8_v2))
            res.append(str(mgcn16_8_mr_v2))
            # v2 speedup
            # res.append(str(round((mgcn16_16/mgcn16_8_v2),4)))
            # res.append(str(round((mgcn16_16/mgcn16_8_mr_v2),4)))
            # print("Speed up v2-8x1 :" + str(round((mgcn16_16/mgcn16_8_v2),4)))
            # print("Speed up v2-MR :" + str(round((mgcn16_16/mgcn16_8_mr_v2),4)))
            
            # 写入 CSV 文件
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
        print(data + ' is successed!')
        print()
    print(f'{csv_file} 导出成功！')
    print()

    
  