import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import sys
import argparse
import os





if __name__ == "__main__":
    
    #GCN
    # command = "python /home/shijinliang/module/sc24/Magicsphere-cmake/eva100/end2end/gcn_no_pre/eva_gcn_small.py"
    # exit_code = os.system(command)
    # if exit_code != 0:
    #     print("GCN-small filed")
    # else:
    #     print("GCN-small succces")
  
    # command = "python /home/shijinliang/module/sc24/Magicsphere-cmake/eva100/end2end/gcn_no_pre/eva_gcn.py"
    # exit_code = os.system(command)
    # if exit_code != 0:
    #     print("GCN filed")
    # else:
    #     print("GCN succces")
        
    #GAT
    # command = "python /home/shijinliang/module/sc24/Magicsphere-cmake/eva100/end2end/gat_no_pre/eva_gat_small.py"
    # exit_code = os.system(command)
    # if exit_code != 0:
    #     print("GAT-small filed")
    # else:
    #     print("GAT-small succces")

    command = "python /home/shijinliang/module/sc24/Magicsphere-cmake/eva100/end2end/gat_no_pre/eva_gat.py"
    exit_code = os.system(command)
    if exit_code != 0:
        print("GAT filed")
    else:
        print("GAT succces")
        
        
    #AGNN
    command = "python /home/shijinliang/module/sc24/Magicsphere-cmake/eva100/end2end/agnn_no_pre/eva_agnn_small.py"
    exit_code = os.system(command)
    if exit_code != 0:
        print("AGNN-small filed")
    else:
        print("AGNN-small succces")

    command = "python /home/shijinliang/module/sc24/Magicsphere-cmake/eva100/end2end/agnn_no_pre/eva_agnn.py"
    exit_code = os.system(command)
    if exit_code != 0:
        print("AGNN filed")
    else:
        print("AGNN succces")