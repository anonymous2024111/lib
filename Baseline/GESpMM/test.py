import torch
import magiBlock
import mGCN

col = torch.tensor([1,3,11,1,7,1,2,5,5,7,0,0,3,8,9,11,1,6,2,7,3,6],dtype=torch.int32)
row = torch.tensor([0,3,5,6,8,8,8,10,11,14,16,18,20,20,22,22,22],dtype=torch.int32)
# col = torch.tensor([1,3,11,1,7,1,2,5,5,7,0],dtype=torch.int32)
# row = torch.tensor([0,3,5,6,8,8,8,10,11],dtype=torch.int32)
d=(row[1:] - row[:-1]).tolist()
degree = torch.tensor(d,dtype=torch.float16)
row1,col1,value1 = magiBlock.blockProcess8_8(row,col)
# print(torch.reshape(value1, (24, 8)))
a0 = torch.tensor([[ 0.2361,  0.6191, -0.4106,  0.7368,  0.3804,  0.5195,  0.2378]],dtype=torch.float16)
a0_matrix = a0.expand(14, -1)
rhs = torch.full((16, 1),  -0.4868,dtype=torch.float16)
# print(rhs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
row1=row1.to(device)
col1=col1.to(device)
value1=value1.to(device)
print(row1)
print(col1)
print(value1)

rhs=rhs.to(device)
a0_matrix=a0_matrix.to(device)
print(a0_matrix[:2])
output=mGCN.forward(row1, col1, value1, a0_matrix, 16, 7, 14)
# outputcpu=value1.to('cpu').numpy()
# for i in range(16):
#     for j in range(8):
#         print(outputcpu[ i*8 + j], end=" ")
#     print()
#     if i%8==7 :
#         print()
print(output)
# non_zero_tensor = output[output != 0]
# print(non_zero_tensor)

# result = example.add(3, 4)
# print(result)  # 输出结果为7
