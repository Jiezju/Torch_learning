import torch as t
from torch import nn
import torch.nn.functional as F


x = t.randn(1,3,4,4)
print(x)

conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3,3))
res = conv(x)

print(res.shape)    # torch.Size([1, 4, 2, 2])


out = F.conv2d(x, conv.weight)
print(out.shape)    # torch.Size([1, 4, 2, 2])