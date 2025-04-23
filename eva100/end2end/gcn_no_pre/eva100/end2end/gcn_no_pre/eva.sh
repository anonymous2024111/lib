#!/bin/bash

folder="./eva100/end2end/gcn/result"

# 如果当前npz不存在
if [ ! -e "$folder/data-adv.npz" ]; then
    # 调用第一个 Python 程序
    python ./eva100/end2end/gcn/eva_gcn_small_bash.py advisor
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "---GNNAdvisor success.---"
    else
        echo "---GNNAdvisor failed.---"
    fi
fi
# 如果当前npz不存在
if [ ! -e "$folder/data-tc.npz" ]; then
    # 调用第二个 Python 程序
    python ./eva100/end2end/gcn/eva_gcn_small_bash.py tc
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "---TC-GNN success.---"
    else
        echo "---TC-GNN failed.---"
    fi
fi

# 如果当前npz不存在
if [ ! -e "$folder/data-mgcn.npz" ]; then
    # 调用第三个 Python 程序
    python ./eva100/end2end/gcn/eva_gcn_small_bash.py mgcn
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "---MGCN success.---"
    else
        echo "---MGCN failed.---"
    fi
fi

# 如果当前npz不存在
if [ ! -e "$folder/data-mgcn32npz" ]; then
    # 调用第四个 Python 程序
    python ./eva100/end2end/gcn/eva_gcn_small_bash.py mgcn32
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "---MGCN32 success.---"
    else
        echo "---MGCN32 failed.---"
    fi
fi

# 如果当前npz不存在
if [ ! -e "$folder/data-dgl.npz" ]; then
    # 调用第五个 Python 程序
    python ./eva100/end2end/gcn/eva_gcn_small_bash.py dgl
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "---DGL success.---"
    else
        echo "---DGL failed.---"
    fi
fi

# 如果当前npz不存在
if [ ! -e "$folder/data-pyg.npz" ]; then
    # 调用第六个 Python 程序
    python ./eva100/end2end/gcn/eva_gcn_small_bash.py pyg
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "---PYG success.---"
    else
        echo "---PYG failed.---"
    fi
fi



# #!/bin/bash

# # 调用第一个 Python 程序
# python ./eva100/end2end/gcn/eva_gcn_small_bash.py advisor
# exit_code=$?

# # 检查第一个程序的退出状态码
# if [ $exit_code -eq 0 ]; then
#     # 第一个程序执行成功，调用第二个 Python 程序
#     echo "---GNNAdvisor success.---"
#     python ./eva100/end2end/gcn/eva_gcn_small_bash.py tc
#     exit_code=$?
# fi

# # 检查第2个程序的退出状态码
# if [ $exit_code -eq 0 ]; then
#     # 第二个程序执行成功，调用第三个 Python 程序
#     echo "---TC-GNN success.---"
#     python ./eva100/end2end/gcn/eva_gcn_small_bash.py mgcn
# fi

# # 检查第3个程序的退出状态码
# if [ $exit_code -eq 0 ]; then
#     # 第二个程序执行成功，调用第三个 Python 程序
#     echo "---MGCN success.---"
#     python ./eva100/end2end/gcn/eva_gcn_small_bash.py mgcn32
# fi

# # 检查第4个程序的退出状态码
# if [ $exit_code -eq 0 ]; then
#     # 第二个程序执行成功，调用第三个 Python 程序
#     echo "---MGCN32 success.---"
#     python ./eva100/end2end/gcn/eva_gcn_small_bash.py dgl
# fi

# # 检查第5个程序的退出状态码
# if [ $exit_code -eq 0 ]; then
#     # 第二个程序执行成功，调用第三个 Python 程序
#     echo "---DGL success.---"
#     python ./eva100/end2end/gcn/eva_gcn_small_bash.py pyg
# fi

# # 检查第6个程序的退出状态码
# if [ $exit_code -eq 0 ]; then
#     # 第二个程序执行成功，调用第三个 Python 程序
#     echo "---PYG success---."
# fi