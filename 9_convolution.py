import torch as t
from torch import nn
import torch.nn.functional as F

# api
x = t.randn(1,3,4,4)
print(x)

conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3,3))
res = conv(x)

print(res.shape)    # torch.Size([1, 4, 2, 2])


out = F.conv2d(x, conv.weight)
print(out.shape)    # torch.Size([1, 4, 2, 2])

# conv

def convolution(x, bias, kernel, stride=1, padding=0):

    if padding > 0:
        F.pad(x, (padding, padding, padding, padding))

    in_h, in_w = x.shape
    k_h, k_w = kernel.shape

    out_h = (in_h - k_h) // stride + 1
    out_w = (in_w - k_w) // stride + 1
    output = t.zeros(out_h, out_w, dtype=t.float32)

    for i in range(0, in_h - k_h + 1, stride):
        for j in range(0, in_w - k_w + 1, stride):
            region = x[i:i+k_h, j:j+k_w]
            output[i//stride, j//stride] = t.sum(region * kernel) + bias

    return output

# 基于矩阵乘法实现卷积
def convolution_matrix(x, bias, kernel, stride=1, padding=0):
    if padding > 0:
        F.pad(x, (padding, padding, padding, padding))

    in_h, in_w = x.shape
    k_h, k_w = kernel.shape

    out_h = (in_h - k_h) // stride + 1
    out_w = (in_w - k_w) // stride + 1

    region_matrix = t.zeros(out_h*out_w, k_h*k_w).float()
    kernel_matrix = kernel.reshape(kernel.numel(), 1)
    row_index = 0
    for i in range(0, in_h - k_h + 1, stride):
        for j in range(0, in_w - k_w + 1, stride):
            region = x[i:i+k_h, j:j+k_w]
            region_matrix[row_index] = region.reshape(-1)
            row_index += 1
            
    output = t.matmul(region_matrix, kernel_matrix).reshape(out_h, out_w) + bias
    return output


x = t.randn(5,5)
k = t.randn(3,3)
b = t.randn(1)

out = convolution(x, b, k)
out = convolution_matrix(x, b, k)
print(out)


def convolution_batch(x, bias, kernel, stride=1, padding=0):
    if padding > 0:
        F.pad(x, (padding, padding, padding, padding, 0,0,0,0))

    batch, in_c, in_h, in_w = x.shape
    o_c, in_c, k_h, k_w = kernel.shape

    out_h = (in_h - k_h) // stride + 1
    out_w = (in_w - k_w) // stride + 1

    output = t.zeros(batch, o_c, out_h, out_w, dtype=t.float32)

    for bs in range(batch):
        for oc in range(o_c):
            for ic in range(in_c):
                for i in range(0, in_h - k_h + 1, stride):
                    for j in range(0, in_w - k_w + 1, stride):
                        region = x[bs, ic, i:i+k_h, j:j+k_w]
                        output[bs, oc, i//stride, j//stride] += t.sum(region * kernel[oc, ic])
            output[bs, oc] += bias[oc]

    return output

x = t.randn(2, 3, 5, 5)
k = t.randn(4, 3, 3, 3)
bias = t.randn(4)
y = convolution_batch(x, bias, k)
print(y.shape)