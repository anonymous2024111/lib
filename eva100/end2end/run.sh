#!/bin/bash

# 1. run gcn
python ./gcn_no_pre/eva_gcn_100.py &&
python ./gcn_no_pre/eva_gcn_plot.py &&

# 2. run agnn
python ./gcn_no_pre/eva_agnn_100.py &&
python ./gcn_no_pre/eva_agnn_plot.py &&

# 3. run gat
python ./gcn_no_pre/eva_gat_100.py &&
python ./gcn_no_pre/eva_gat_plot.py 