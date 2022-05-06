import torch
import torch as t
from torch import nn
import torch.nn.functional as F

'''
torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

功能：一个批次的输入样本中，提取出滑动的局部区域块。

参数： kernel_size: 滑块的大小。

dilation：控制滑动过程中所跨越元素的个数。默认为1

padding：0填充。 默认为0

stride: 步长。默认为1。

输入： inputs (B, W, H, C)

输出： outputs (B, N, L) 
N:表示生成后每个局部块的大小。L：表示有多少个局部块。 

将每个 kernel size 的区域提取出来 拉平

'''

inputs = t.arange(25).reshape(1, 1, 5, 5).float()
print(inputs)
unfold = nn.Unfold(kernel_size=(3, 3))
# print(unfold(inputs))
print(unfold(inputs))

# kernel 展开 实现 二维卷积 不考虑 padding stride 默认 1
def convert_kernel_matrix(kernel, input_size):
    kernel_h, kernel_w = kernel.shape
    input_h, input_w = input_size

    num_output = (input_h - kernel_h + 1) * (input_w - kernel_w + 1)

    result = t.zeros((num_output, input_h*input_w)).float()
    count = 0
    for i in range(0, input_h - kernel_h + 1):
        for j in range(0, input_w - kernel_w + 1):
            padded_kernel = F.pad(kernel, (i, input_h - kernel_h - i, j, input_w - kernel_w - j))
            result[count] = padded_kernel.flatten()
            count += 1

    return result

kernel = t.randn(3,3)
input = t.randn(4,4)
print(input.reshape(-1,1))
kernel_matrix = convert_kernel_matrix(kernel, input.shape)
print(kernel_matrix) # [4, 16] 输出 特征图 2*2

out = t.matmul(kernel_matrix, input.reshape(-1,1))
print(out.reshape(2,2))

# 转置卷积 kenerl_matrix^T dot out.reshape(-1,1) => [16, 1]
out_transpose = t.matmul(kernel_matrix.transpose(1, 0), out.reshape(-1, 1))
print(out_transpose.reshape(4,4))

torch_output = F.conv_transpose2d(out.reshape(2,2).unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0))
print(torch_output)

# conv transpose 的另一种理解
x = t.tensor([[4, 3, 4], [2, 4, 3], [2, 3, 4]]).reshape(1, 1, 3, 3).float()
kernel = t.ones(size=(1, 1, 3, 3)).float()
output = F.conv_transpose2d(x, kernel)
print(output)

# 等价于 先插入 0
x = t.tensor([[4, 0, 3, 0, 4], [0, 0, 0, 0, 0], [2, 0, 4, 0, 3], [0, 0, 0, 0, 0], [2, 0, 3, 0,  4]]).reshape(1, 1, 5, 5).float()
output = F.conv2d(x, kernel)
print(output)
