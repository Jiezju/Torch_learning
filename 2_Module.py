import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == nn.Conv2d:
        m.weight.fill_(1.0)
        # print(m.weight)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 3, 3)
        self.register_buffer('running_mean', torch.zeros(3,5))

        self.register_module('bn1', nn.BatchNorm2d(3))

        sub_module = self.get_submodule('conv1')

        child_module = self.children()

        self.buffers_ = self._buffers
        self.parameters_ = self._parameters # 当前模块的参数为空，但是子模块不为空

        # 获取完整模块的参数，注意是 tensor
        params = []
        for param in self.parameters():
            if isinstance(param, torch.Tensor):
                print(True)
            params.append(param)
        self.modules = self._modules

        self.trace = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

if __name__ == '__main__':
    model = Model()
    model.apply(init_weights)
    print(model)

    print(model.conv1.weight.dtype)
    model.to(torch.double)
    print(model.conv1.weight.dtype)
    model.to(torch.float)

    state_dict = model.state_dict()

    x = torch.rand(1,1,10,10)

    out = model(x)
    print('Success !')