import os.path as osp
import argparse
import time
import torch
import sys
sys.path.append('eva100/end2end/agnn_no_pre')
from magnn.mdataset import *
from magnn.magnn_conv import *
from magnn.agnn_mgnn import *


def test(data, epoches, layers, featuredim, hidden, classes,  density, partsize_t, partsize_c, window, wide):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    inputInfo = MAGNN_dataset(data, featuredim , classes, density,  window, wide)

    inputInfo.to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes, layers).to(device)

    train(model, inputInfo, 10)
    torch.cuda.synchronize()
    start_time = time.time()
    train(model, inputInfo, epoches)
    # 记录程序结束时间
    torch.cuda.synchronize()
    end_time = time.time()
    # 计算程序执行时间（按秒算）
    execution_time = end_time - start_time
    print(round(execution_time,4))
    return round(execution_time,4)

# if __name__ == "__main__":

#     test('cite', 100, 64, 3, 64, 10)
