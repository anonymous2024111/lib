import torch

# 创建一个示例一维张量
tensor_example = torch.arange(1, 21)  # 创建一个从1到20的张量
print(tensor_example)
# 取前10个元素
first_10 = tensor_example[:10]

# 取后5个元素
last_5 = tensor_example[-10:]

print("前10个元素:", first_10)
print("后5个元素:", last_5)
