import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


model0 = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU())

print(model0._modules)

x = torch.randn(1,1,20,20)
y = model0(x)

sub_m = model0[1]


model1 = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))


class MyModulelist(nn.Module):
    def __init__(self):
        super(MyModulelist, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x

m_list = MyModulelist()
x = torch.randn(1,2,10)
y = m_list(x)


class MyModuledict(nn.Module):
    def __init__(self):
        super(MyModuledict, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x

x = torch.randn(1,1,10,10)

