import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 基于dilation 值的特征图取值方法
t = torch.randn(7, 7)
# dilation = 1
print(t[0:3, 0:3])

# dilation = 2
print(t[0:3:2, 0:3:2])

# dilation = 3
print(t[0:3:3, 0:3:3])

# groups
'''
in_channel, out_channel = 2, 4

groups = 2

sub_in_channel, sub_out_channel = 1, 2

groups > 1, 通道融合不需要完全充分，只需要在一个个group内进行融合，最后拼接

往往需要接 1*1 卷积（pointwise convolution）进行通道完全融合
'''

def matrix_mul_for_conv2d(x, kernel, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding, 0, 0, 0, 0))
    bs, in_channel, in_h, in_w = x.shape
    o_channel, _, kernel_h, kernel_w = kernel.shape

    assert o_channel % groups == 0 and in_channel % groups == 0

    x = x.reshape(bs, groups, in_channel // groups, in_h, in_w)
    kernel = kernel.reshape(groups, o_channel // groups, in_channel // groups, kernel_h, kernel_w)

    kernel_h = (kernel_h - 1) * (dilation - 1) + kernel_h
    kernel_w = (kernel_w - 1) * (dilation - 1) + kernel_w

    o_h = math.floor((in_h - kernel_h) / stride) + 1
    o_w = math.floor((in_w - kernel_w) / stride) + 1

    o_shape = (bs, groups, o_channel // groups, o_h, o_w)
    o = torch.zeros(o_shape)

    if bias is None:
        bias = torch.zeros(o_channel)

    for b in range(bs):
        for g in range(groups):
            for oc in range(o_channel // groups):
                for ic in range(in_channel // groups):
                    for i in range(0, in_h - kernel_h + 1, stride):
                        for j in range(0, in_w - kernel_w + 1, stride):
                            region = x[b, g, ic, i:i+kernel_h:dilation, j:j+kernel_w:dilation]
                            o[b, g, oc, i // stride, j // stride] += torch.sum(region * kernel[g, oc, ic])
                o[b, g, oc] += bias[g*(o_channel // groups) + oc]

    return o









