import torch
from torch import nn
import numpy as np

# pytorch
m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)

# mamual ratio 表示丢弃概率
def dropout(x, phase, ratio):
    if phase == 'TEST':
        y = x * (1 - ratio)
    else: # Train
        mask = np.random.binomial(1, 1 - ratio, x.shape)
        y = x * mask

    return y

output = dropout(input, 'TRAIN', 0.2)
