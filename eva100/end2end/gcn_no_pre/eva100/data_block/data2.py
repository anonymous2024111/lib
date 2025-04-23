import torch
import Rabbit
import MagicsphereMRabbit
import MagicsphereBlock
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
    
#Rabbit
def blocks_num_rabbit(graph_obj, result, m, n, datatype, baseline):
    
    src_li = graph_obj['src_li']
    dst_li = graph_obj['dst_li']
    num_nodes=graph_obj['num_nodes']+m-(graph_obj['num_nodes']%m)
    num_edges = len(src_li)
    edge_index = np.stack([src_li, dst_li])
    #Rabbit
    edge_index = Rabbit.reorder(torch.IntTensor(edge_index))
    val = [1] * num_edges
    scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes))
    scipy_csr = scipy_coo.tocsr()
    column_index = torch.IntTensor(scipy_csr.indices)
    row_pointers = torch.IntTensor(scipy_csr.indptr)
    label = 'rabbit' + str(m) + '-' + str(n)
    if n==4 :
        row, _, _ = MagicsphereBlock.blockProcess8_4_1(row_pointers, column_index)
        block_num = row[-1].numpy().item() / n
        baseline[label] = block_num
        print(label + ': ' + str(block_num))
    else :
        row, _, _ = MagicsphereBlock.blockProcess8_16(row_pointers, column_index)
        block_num = row[-1].numpy().item() / n
        baseline[label] = block_num
        print(label + ': ' + str(block_num))
    
#M-GNN
def blocks_num_mgnn(graph_obj, result, m, n, datatype, k, baseline):
    
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
    label = 'mgnn' + str(m) + '-' + str(n)
    if n==4 :
        row, _, _ = MagicsphereBlock.blockProcess8_4_1(row_pointers, column_index)
        block_num = row[-1].numpy().item() / n
        baseline[label] = block_num
        print(label + ': ' + str(block_num))
    else :
        row, _, _ = MagicsphereBlock.blockProcess8_16(row_pointers, column_index)
        block_num = row[-1].numpy().item() / n
        baseline[label] = block_num
        print(label + ': ' + str(block_num))
        
def list_gen() :
    # 声明一个空的3维列表
    my_list = []
    # 定义列表的维度大小
    dim1 = 2  # 第一维度大小
    # 使用嵌套循环创建空的三维列表
    for i in range(dim1):
        # 创建第一维度的空列表
        sublist1 = []
        my_list.append(sublist1)
    return my_list



if __name__ == "__main__":
    
    #用于统计4种算法的block数
    spmm = dict()
    base_spmm = dict()
    dataset = ['ogb', 'amazon', 'yelp', 'reddit', 'blog', 'artist']
    #dataset = ['ogb']
    spmm_eva = list_gen()
    load = True
    if load :
        data = np.load('./eva/data_block/data2.npz')
        spmm_eva = data['eva_result']
    else: 
        #循环读入每个数据集
        for data in dataset:
            print(data)
            base_spmm.clear()
            path = './dgl_dataset/mythroughput/' + data + '.npz'
            graph_obj = np.load(path)
            #SpMM
            #Rabbit8*4
            blocks_num_rabbit(graph_obj, spmm, 8, 4, data, base_spmm)
            #M-GNN8*4
            blocks_num_mgnn(graph_obj, spmm, 8, 4, data, 6, base_spmm)
            #Rabbit8*16
            blocks_num_rabbit(graph_obj, spmm, 8, 16, data, base_spmm)
            #M-GNN8*16
            blocks_num_mgnn(graph_obj, spmm, 8, 16, data, 6, base_spmm)
            spmm[data] = base_spmm.copy()
        print("SpMM:")
        for data in spmm:
            print(data+": ", end="")
            print(spmm[data])
        print("Result processing")
        #循环读入每个数据集
        for data in dataset:
            #SpMM: tcgnn, rabbit, mgnn
            spmm_eva[0].append(spmm[data]['rabbit8-4']-spmm[data]['mgnn8-4'])
            spmm_eva[1].append(spmm[data]['rabbit8-16']-spmm[data]['mgnn8-16'])
        #save for npz
        np.savez('./eva/data_block/data2.npz', 
            eva_result=spmm_eva)
    print(spmm_eva)
    
    #plot
    categories = ['OGB', 'Amazon', 'Yelp', 'Reddit', 'Blog', 'Artist']    # 分组标签
    group_names = ['SpMM:8x4', 'SDDMM:8x16'] # 分组名称
    values = np.array([spmm_eva[0], spmm_eva[1]])  # 每个分组的数值


    # 设置柱状图的参数
    bar_width = 0.18  # 柱状宽度
    index = np.arange(len(categories))  # x轴刻度位置

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(8, 3))
    bars1 = ax.bar(index, values[0], bar_width, label=group_names[0],color='cornflowerblue')
    bars2 = ax.bar(index + bar_width, values[1], bar_width, label=group_names[1],color='blue')
    # autolabel(bars1,spmm_eva_ori[0])
    # autolabel(bars2,spmm_eva_ori[1])
    # autolabel(bars3,spmm_eva_ori[2])

    # 添加标签和标题
    ax.set_ylabel('Reduced Blocks', fontweight='bold')
    ax.set_xticklabels(categories) 
    # for label in ax.get_yticklabels():
    #     label.set_weight('bold') 
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories)
    # 创建一个字体属性对象，设置加粗
    # font = FontProperties(weight='bold')
    # 添加图例，并设置文本属性
    font = {'size': 'small'}
    ax.legend(loc='upper right',prop=font)
    plt.ylim(1000, 1000000)  # 将 Y 轴下限设置为 40
    # 调整图表布局，使得竖直方向的标签能够显示完整
    plt.subplots_adjust(bottom=0.2)
    plt.yscale('log')  # y轴按照对数刻度显示
    # 获取纵坐标的刻度值
    y_ticks = ax.get_yticks()
    # 在每个刻度值处绘制水平背景线
    for y_tick in y_ticks:
        ax.axhline(y=y_tick, color='lightgray', linestyle='-',linewidth=0.5, zorder=0)


    # 去除最大值横线
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.savefig('./eva100/data_block/data2.png', dpi=800)
    print("success")
        
    