import torch
from torch import nn
import torch.nn.functional as F

# common convlution
m = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2))

print(m.weight.shape)
print(m.bias.shape)

input = torch.randn(1, 3, 20, 20)
output = m(input)

print(output.shape)

# point-wise convlution 1*1 convlution
m = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(1, 1), stride=(2, 2))
input = torch.randn(1, 3, 20, 20)
output = m(input)

print(output.shape)

# depth-wise convlution 一个卷积核负责一个通道，一个通道只被一个卷积核卷积
m = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1), groups=3)
print(m.weight.shape)
print(m.bias.shape)

input = torch.randn(1, 3, 20, 20)
output = m(input)

print(output.shape)

# 三路残差分支的算子融合

in_channels = 2
out_channels = 2
h = 9
w = 9

x = torch.randn(1, in_channels, h, w)

###################### 原始组合： conv 3*3 + conv 1*1 + x ##############
conv3_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same')
conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

result0 = conv3_3(x) + conv1_1(x) + x

##################### 3*3 convert ###################
# point wise -> 3 * 3
pointwise_to_conv3_weight = F.pad(conv1_1.weight, [1,1,1,1,0,0,0,0]) # [2,2,1,1] -> [2,2,3,3]
conv1_3_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same')
conv1_3_3.weight = nn.Parameter(pointwise_to_conv3_weight)
conv1_3_3.bias = nn.Parameter(conv1_1.bias)

# x -> 3*3 通道间不进行运算，feature空间不进行计算
# 单一通道内的恒等运算
ones = torch.unsqueeze(F.pad(torch.ones(1,1), [1,1,1,1]), 0)
zeros = torch.unsqueeze(torch.zeros(3,3), 0)
ones_zeros = torch.unsqueeze(torch.cat([ones, zeros], 0), 0)
zeros_ones = torch.unsqueeze(torch.cat([zeros, ones], 0), 0)

identity_conv3_3_weight = torch.cat([ones_zeros, zeros_ones], 0)
print(identity_conv3_3_weight.shape)

identity_conv3_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same')
identity_conv3_3.weight = nn.Parameter(identity_conv3_3_weight)
identity_conv3_3.bias = nn.Parameter(torch.zeros(out_channels))

result1 = conv3_3(x) + conv1_3_3(x) + identity_conv3_3(x)

print(torch.all(torch.isclose(result0, result1)))

######### fuse 注意是 concat ###################
conv_fuse = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same')
conv_fuse.weight = nn.Parameter(conv3_3.weight.data + conv1_3_3.weight.data + identity_conv3_3.weight.data)
conv_fuse.bias = nn.Parameter(conv3_3.bias.data + conv1_3_3.bias.data + identity_conv3_3.bias.data)
result2 = conv_fuse(x)
print(torch.all(torch.isclose(result0, result2)))

