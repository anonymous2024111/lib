import torch
import Rabbit
import MagicsphereMRabbit
import MagicsphereBlock
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import time

#Nvidia
def blocks_num_nvidia(graph_obj, m, n, baseline):
    
    src_li = graph_obj['src_li']
    dst_li = graph_obj['dst_li']
    num_nodes=graph_obj['num_nodes']+m-(graph_obj['num_nodes']%m)
    num_edges = len(src_li)
    edge_index = np.stack([src_li, dst_li])
    val = [1] * num_edges
    scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes))
    scipy_csr = scipy_coo.tocsr()
    column_index = torch.IntTensor(scipy_csr.indices)
    row_pointers = torch.IntTensor(scipy_csr.indptr)
    block_num = MagicsphereBlock.blockProcess_nbs(row_pointers, column_index, m, n)
    baseline['nvidia'] = block_num
    print('NVIDIA: ' + str(block_num))
    
#tcgnn_gen
def blocks_num_tcgnngen(graph_obj, m, n, baseline):
    
    src_li = graph_obj['src_li']
    dst_li = graph_obj['dst_li']
    num_nodes=graph_obj['num_nodes']+m-(graph_obj['num_nodes']%m)
    num_edges = len(src_li)
    edge_index = np.stack([src_li, dst_li])
    val = [1] * num_edges
    scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes))
    scipy_csr = scipy_coo.tocsr()
    column_index = torch.IntTensor(scipy_csr.indices)
    row_pointers = torch.IntTensor(scipy_csr.indptr)
    block_num = MagicsphereBlock.blockProcess_tc(row_pointers, column_index, m, n)
    baseline['tcgnngen'] = block_num
    print('TC-GNN: ' + str(block_num))
    
#Rabbit
def blocks_num_rabbit(graph_obj, m, n, baseline):
    
    src_li = graph_obj['src_li']
    dst_li = graph_obj['dst_li']
    num_nodes=graph_obj['num_nodes']+m-(graph_obj['num_nodes']%m)
    num_edges = len(src_li)
    edge_index = np.stack([src_li, dst_li])
    #Rabbit
    edge_index= Rabbit.reorder(torch.IntTensor(edge_index))
    val = [1] * num_edges
    scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes))
    scipy_csr = scipy_coo.tocsr()
    column_index = torch.IntTensor(scipy_csr.indices)
    row_pointers = torch.IntTensor(scipy_csr.indptr)
    row, _, _ = MagicsphereBlock.blockProcess8_4_1(row_pointers, column_index)
    block_num = row[-1].numpy().item() / n
    baseline['rabbit'] = block_num
    print('Rabbit: ' + str(block_num))

    
#M-GNN
def blocks_num_mgnn(graph_obj, m, n, k, baseline):
    
    src_li = graph_obj['src_li']
    dst_li = graph_obj['dst_li']
    num_nodes=graph_obj['num_nodes']+m-(graph_obj['num_nodes']%m)
    num_edges = len(src_li)
    edge_index = np.stack([src_li, dst_li])
    #Rabbit
    edge_index, _ = MagicsphereMRabbit.reorder(torch.IntTensor(edge_index),graph_obj['num_nodes'],k)
    val = [1] * num_edges
    scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes))
    scipy_csr = scipy_coo.tocsr()
    column_index = torch.IntTensor(scipy_csr.indices)
    row_pointers = torch.IntTensor(scipy_csr.indptr)
    row, _, _ = MagicsphereBlock.blockProcess8_4_1(row_pointers, column_index)
    block_num = row[-1].numpy().item() / n
    baseline['mgnn'] = block_num
    print('MRabbit: ' + str(block_num))
        
def list_gen() :
    # 声明一个空的3维列表
    my_list = []
    # 定义列表的维度大小
    dim1 = 3  # 第一维度大小
    # 使用嵌套循环创建空的三维列表
    for i in range(dim1):
        # 创建第一维度的空列表
        sublist1 = []
        my_list.append(sublist1)
    return my_list

def autolabel(bars, ori):
    i=0
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(ori[i]),
                ha='center', va='bottom')
        i+=1

if __name__ == "__main__":
    
    #用于统计4种算法的block数
    spmm = dict()
    base_spmm = dict()
    dataset = ['ogb', 'amazon', 'yelp', 'reddit', 'blog', 'artist']
    spmm_eva = list_gen()
    spmm_eva_ori = list_gen()
    load = False
    if load :
        data = np.load('./eva/data_block/data1.npz')
        spmm_eva = data['eva_result']
        spmm_eva_ori = data['eva_result_ori']
    else: 
        #循环读入每个数据集
        for data in dataset:
            print(data)
            base_spmm.clear()
            path = './dgl_dataset/mythroughput/' + data + '.npz'
            graph_obj = np.load(path)
            #SpMM
            #Nvidia
            blocks_num_nvidia(graph_obj, 8, 4, base_spmm)
            #TC-GNN-gen
            blocks_num_tcgnngen(graph_obj, 8, 4, base_spmm)
            #Rabbit
            blocks_num_rabbit(graph_obj, 8, 4, base_spmm)
            #M-GNN
            blocks_num_mgnn(graph_obj, 8, 4, 6, base_spmm)
            spmm[data] = base_spmm.copy()
        print("SpMM:")
        for data in spmm:
            print(data+": ", end="")
            print(spmm[data])
        print("Result processing")
        #循环读入每个数据集
        for data in dataset:
            #SpMM: tcgnn, rabbit, mgnn
            nvidia_num = spmm[data]['nvidia']
            spmm_eva[0].append(round(((nvidia_num-spmm[data]['tcgnngen']) / nvidia_num),4))
            spmm_eva[1].append(round(((nvidia_num-spmm[data]['rabbit']) / nvidia_num),4))
            spmm_eva[2].append(round(((nvidia_num-spmm[data]['mgnn']) / nvidia_num),4))
            spmm_eva_ori[0].append(nvidia_num-spmm[data]['tcgnngen'])
            spmm_eva_ori[1].append(nvidia_num-spmm[data]['rabbit']) 
            spmm_eva_ori[2].append(nvidia_num-spmm[data]['mgnn'])
        #save for npz
        np.savez('./eva/data_block/data1.npz', 
            eva_result=spmm_eva,
            eva_result_ori=spmm_eva_ori)
    print(spmm_eva)
    
    #plot
    categories = ['OGB', 'Amazon', 'Yelp', 'Reddit', 'Blog', 'Artist']   # 分组标签
    group_names = ['SGT', 'Rabbit', 'MRabbit'] # 分组名称
    values = np.array([spmm_eva[0], spmm_eva[1], spmm_eva[2]])  # 每个分组的数值
    values *= 100
    print(values)

    # 设置柱状图的参数
    bar_width = 0.18  # 柱状宽度
    index = np.arange(len(categories))  # x轴刻度位置

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(8, 3))
    bars1 = ax.bar(index, values[0], bar_width, label=group_names[0], color='lightsteelblue')
    bars2 = ax.bar(index + bar_width, values[1], bar_width, label=group_names[1],color='cornflowerblue')
    bars3 = ax.bar(index + bar_width*2, values[2], bar_width, label=group_names[2],color='blue')
    # autolabel(bars1,spmm_eva_ori[0])
    # autolabel(bars2,spmm_eva_ori[1])
    # autolabel(bars3,spmm_eva_ori[2])

    # 添加标签和标题
    ax.set_ylabel('Reduction(%)', fontweight='bold')
    ax.set_xticklabels(categories) 
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
    plt.ylim(50, 90)  # 将 Y 轴下限设置为 40
    # 调整图表布局，使得竖直方向的标签能够显示完整
    plt.subplots_adjust(bottom=0.3)
    # 获取纵坐标的刻度值
    y_ticks = ax.get_yticks()
    # 在每个刻度值处绘制水平背景线
    for y_tick in y_ticks:
        ax.axhline(y=y_tick, color='lightgray', linestyle='-',linewidth=0.5, zorder=0)


    # 去除最大值横线
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.savefig('./eva4090/data_block/data1.png', dpi=800)
    print("success")
        
    