import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('./eva100/accuracy/gcn')
import seaborn as sns
import pandas as pd
    



if __name__ == "__main__":
    
    #用于统计4种算法的block数
    spmm = dict()
    base_spmm = dict()
    # dataset = ['cite', 'cora', 'cornell', 'ell', 'min', 'pubmed', 'question', 'texas', 'wiki', 'wis']
    load = False
    epoches = 300
    num_layers = 5
    hidden = 100
    
    res = dict()
    res['data'] = []
    res['baseline'] = []
    res['acc'] = []   
    with open('./eva100/accuracy/gcn/result.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # 跳过第一行
        for cur, row in enumerate(reader, start=0):
            res['data'].append(row[0])
            res['baseline'].append(row[1])
            res['acc'].append(float(row[2]))
            
    df = pd.DataFrame(res)
    mycolor = {'advisor':'cornflowerblue', 'tcgnn':'lightgreen', 
           'mgcn32':'gold', 'mgcn16':'orange', 'pyg':'tomato'}
    plt.figure(figsize=(5, 2))
    # sns.set(rc={"lines.linewidth": 0.5})
    # 对数据按照 'dim' 列进行升序排序
    sns.set_style("darkgrid")
    # 设置背景色为灰白色

    g = sns.barplot(x='data', y='acc', hue='baseline', data=df, 
                    palette='Blues_d', linewidth=1, legend=False, gap=0.1, width=0.8)

    plt.ylim(60, 100)  # 将 Y 轴下限设置为 40
    sns.despine(left=True, right=True, top=True)
    g.set_ylabel('Accuracy %',fontsize=14, fontweight='bold')
    g.set_xlabel('')
    plt.savefig('./eva100/accuracy/gcn/gcn.png', dpi=800)
    print("success")
    
