import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt

import sys
sys.path.append('./eva4090/accuracy/gat')
from mydgl import test_dgl
from mypyg import test_pyg
from mgat import test_mgat
from mgat32 import test_mgat32
    
#PYG
def pygGCN(data, baseline, epoches, heads, hidden):
    baseline['pyg'] = test_pyg.test(data, epoches, heads, hidden)
    
#DGL
def dglGCN(data, baseline, epoches, heads, hidden):
    baseline['dgl'] = test_dgl.test(data, epoches, heads, hidden)
    
#MGAT
def mGAT(data, baseline, epoches, heads, hidden):
    baseline['mgat'] = test_mgat.test(data, epoches, heads, hidden)
    
#MGAT32
def mGAT32(data, baseline, epoches, heads, hidden):
    baseline['mgat32'] = test_mgat32.test(data, epoches, heads, hidden)
    
def list_gen() :
    # 声明一个空的3维列表
    my_list = []
    # 定义列表的维度大小
    dim1 = 4  # 第一维度大小
    # 使用嵌套循环创建空的三维列表
    for i in range(dim1):
        # 创建第一维度的空列表
        sublist1 = []
        my_list.append(sublist1)
    return my_list



if __name__ == "__main__":
    
    #用于统计4种算法的block数
    sddmm = dict()
    base_spmm = dict()
    dataset = ['cite', 'cora', 'cornell', 'ell', 'min', 'pubmed', 'question', 'texas', 'wiki', 'wis']
    #dataset = ['cite', 'cora', 'ell', 'min', 'question']
    # dataset = ['cite', 'cora', 'ell', 'min', 'pubmed', 'question', 'tol', 'wiki']
    # smallData = ['cornell', 'texas', 'wis']
    spmm_eva = list_gen()
    load = False
    epoches = 100
    if load :
        data = np.load('./eva4090/accuracy/gat/data.npz')
        spmm_eva = data['eva_result']
    else: 
        #循环读入每个数据集
        for data in dataset:
            base_spmm.clear()
            path = './dgl_dataset/accuracy/' + data + '.npz'
            graph_obj = np.load(path)
            head = 6
            # if data in smallData :
            #     head = 1
            #PYG
            pygGCN(data, base_spmm, epoches, head, 16)
            #DGL
            dglGCN(data, base_spmm, epoches, head, 16)
            #MGAT
            mGAT(data, base_spmm, epoches, head, 16)
            #MGAT32
            mGAT32(data, base_spmm, epoches, head, 16)
            
            sddmm[data] = base_spmm.copy()
        # print("GAT:")
        # for data in sddmm:
        #     print(data+": ", end="")
        #     print(sddmm[data])
        print("Result processing")
        #循环读入每个数据集
        for data in dataset:
            #SpMM: tcgnn, rabbit, mgnn
            spmm_eva[0].append(sddmm[data]['pyg'])
            spmm_eva[1].append(sddmm[data]['dgl'])
            spmm_eva[2].append(sddmm[data]['mgat'])
            spmm_eva[3].append(sddmm[data]['mgat32'])
        #save for npz
        np.savez('./eva4090/accuracy/gat/data.npz', 
            eva_result=spmm_eva)
    # print(spmm_eva)
    
    #plot
    categories = ['cite', 'cora', 'corn', 'ell', 'min', 'pub', 'que', 'tex', 'wiki', 'wis']   # 分组标签
    group_names = ['PYG-FP32', 'DGL-FP32', 'MGAT-FP16', 'MGAT-TF32'] # 分组名称
    values = np.array([spmm_eva[0], spmm_eva[1], spmm_eva[2], spmm_eva[3]])  # 每个分组的数值


    # 设置柱状图的参数
    bar_width = 0.15  # 柱状宽度
    index = np.arange(len(categories))  # x轴刻度位置

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 3))
    bars1 = ax.bar(index, values[0], bar_width, label=group_names[0], color='lightsteelblue')
    bars2 = ax.bar(index + bar_width, values[1], bar_width, label=group_names[1],color='cornflowerblue')
    bars3 = ax.bar(index + bar_width*2, values[2], bar_width, label=group_names[2],color='royalblue')
    bars4 = ax.bar(index + bar_width*3, values[3], bar_width, label=group_names[3],color='blue')
    # 添加标签和标题
    ax.set_ylabel('Accuracy(%)', fontweight='bold')
    ax.set_xticklabels(categories, rotation='vertical') 
    # for label in ax.get_yticklabels():
    #     label.set_weight('bold') 
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories)
    # 创建一个字体属性对象，设置加粗
    # font = FontProperties(weight='bold')
    # 添加图例，并设置文本属性
    font = {'size': 'small'}
    # ax.legend(loc='upper right',prop=font)
    ax.legend(prop=font)
    plt.ylim(50, 100)  # 将 Y 轴下限设置为 40
    # 调整图表布局，使得竖直方向的标签能够显示完整
    plt.subplots_adjust(bottom=0.2)
    # 获取纵坐标的刻度值
    y_ticks = ax.get_yticks()
    # 在每个刻度值处绘制水平背景线
    for y_tick in y_ticks:
        ax.axhline(y=y_tick, color='lightgray', linestyle='-',linewidth=0.5, zorder=0)


    # 去除最大值横线
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.savefig('./eva4090/accuracy/gat/gat-acc.png', dpi=800)
    print("success")
        
    