import torch
import torch.nn as nn
import math

'''
BatchNorm

    per channel across mini-batch
    
    cv
    
    训练时，考虑历史 batch mean var 的滑动平均
    测试时，则是最后一次 batch 的 mean var 的结果
    
    以 [bs, c, h, w] 为例，求每个channel 对 每个 batch 的均值
     
'''

x = torch.arange(0, 16).float().reshape(2, 4, 2)
print(x)
print(torch.mean(x, dim=(0, 2), keepdim=True))
print(torch.std(x, dim=(0, 2), keepdim=True, unbiased=False))

# compute manual
channel_mean = []
channel_std = []

batch = x.shape[0]
channel = x.shape[1]
feature_size = x.shape[2]

for c in range(channel):
    sum = 0
    for bs in range(batch):
        sum += x[bs, c, :].sum()
    channel_mean.append(sum / (batch * feature_size))

for c in range(channel):
    sum = 0
    for bs in range(batch):
        erro = (x[bs, c, :] - channel_mean[c]) * (x[bs, c, :] - channel_mean[c])
        sum += erro.sum()
    channel_std.append(math.sqrt(sum / (batch * feature_size)))

print(channel_mean, channel_std)

bn_res = (x - torch.mean(x, dim=(0, 2), keepdim=True)) / (torch.std(x, dim=(0, 2), keepdim=True, unbiased=False) + 1e-5)
print(bn_res.shape)

'''
layer norm
    
    per layer per sample 
    
    sequence
    
    compute over the last D dimension (normalize shape)
'''
normalize_shape = x.shape[-1]
mean = torch.mean(x, dim=-1, keepdim=True)
std = torch.std(x, dim=-1, unbiased=False, keepdim=True)
ln_res = (x - mean) / (std + 1e-5)
print(ln_res.shape)

'''
Instance Norm

    gan
    
    per sample , per channel
'''

mean = torch.mean(x, dim=1, keepdim=True)
std = torch.std(x, dim=1, unbiased=False, keepdim=True)
print(mean, std)
inorm = (x - mean) / (std + 1e-5)

'''
group norm

    per channel, per group (对 channel 进行 group 划分 ) 
'''

group_in = torch.split(x, split_size_or_sections=[1, 1])
group_out = []
for g_in in group_in:
    mean = torch.mean(g_in, dim=(0, 2), keepdim=True)
    std = torch.std(g_in, dim=(0, 2), keepdim=True, unbiased=False)
    g_res = (g_in - mean) / (std + 1e-5)
    group_out.append(g_res)
out = torch.cat(group_out, dim=1)

'''
weight nrom

    权重归一化 分离 权重为 模长 和 方向
'''

