import matplotlib.pyplot as plt

# 示例数据：假设这是训练过程中每个 epoch 的准确率，最多 300 个 epoch
epochs = list(range(1, 301))  # epoch 从 1 到 300
flash_fp16 = []
flash_tf32 = []
pyg = []
dgl = []


#读取Flash-fp16
path= '/home/shijinliang/module/git-flashsprase-ae2/eva/accuracy/agnn/'
file_path = path + 'flash-pubmed-fp16.txt' 
with open(file_path, 'r') as file:
    for line in file:
        flash_fp16.append(float(line.strip())) 

#读取Flash-tf32
path= '/home/shijinliang/module/git-flashsprase-ae2/eva/accuracy/agnn/'
file_path = path + 'flash-pubmed-tf32.txt' 
with open(file_path, 'r') as file:
    for line in file:
        flash_tf32.append(float(line.strip())) 
        
#读取DGL
path= '/home/shijinliang/module/git-flashsprase-ae2/eva/accuracy/agnn/'
file_path = path + 'dgl-pubmed.txt' 
with open(file_path, 'r') as file:
    for line in file:
        dgl.append(float(line.strip()))
        
#读取PyG
path= '/home/shijinliang/module/git-flashsprase-ae2/eva/accuracy/agnn/'
file_path = path + 'pyg-pubmed.txt' 
with open(file_path, 'r') as file:
    for line in file:
        pyg.append(float(line.strip()))

# 创建绘图对象
plt.figure(figsize=(5, 2))


plt.plot(epochs, dgl, label="DGL", color='red', marker='o', linewidth=0.5, markersize=5)
plt.plot(epochs, pyg, label="PyG", color='green', marker='x', linewidth=0.5, markersize=5)
plt.plot(epochs, flash_tf32, label="Libra_TF32", color='dodgerblue', marker='x', linewidth=0.5, markersize=5)
plt.plot(epochs, flash_fp16, label="Libra-FP16", color='blue', marker='o', linewidth=0.5, markersize=5)

# 设置标题和标签
# plt.title("Accuracy Curve (300 Epochs)")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图形
plt.savefig('/home/shijinliang/module/git-flashsprase-ae2/eva/accuracy/agnn/agnn.png', dpi=800)
# 清空图形
plt.clf()