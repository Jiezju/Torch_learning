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
    in_h, in_w = x.shape
    k_h, k_w = kernel.shape

    if padding > 0:
        F.pad(x, (padding, padding, padding, padding))

    out_h = (in_h - k_h) // stride + 1
    out_w = (in_w - k_w) // stride + 1
    output = t.zeros(out_h, out_w, dtype=t.float32)

    for i in range(0, in_h - k_h + 1, stride):
        for j in range(0, in_w - k_w + 1, stride):
            region = x[i:i+k_h, j:j+k_w]
            output[i//stride, j//stride] = t.sum(region * kernel) + bias

    return output

x = t.randn(5,5)
k = t.randn(3,3)
b = t.randn(1)

out = convolution(x, b, k)
print(out)