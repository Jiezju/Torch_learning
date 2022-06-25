import torch
import torch.nn as nn
import torch.nn.functional as F


def residual_branch(in_channel,
                    mid_channel,
                    out_channel,
                    kernel_size,
                    stride,
                    padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, mid_channel,
                  kernel_size, stride,
                  padding, bias=False),
        nn.BatchNorm2d(mid_channel),
        nn.ReLU(),
        nn.Conv2d(mid_channel, out_channel,
                  kernel_size, 1,
                  padding, bias=False),
        nn.BatchNorm2d(out_channel),
    )

class BasicBlockSearch(nn.Module):
    def __init__(self,
                 in_channel,
                 mid_channels,
                 out_channel,
                 stride=1):
        super(BasicBlockSearch, self).__init__()
        Cin, Cout = in_channel, out_channel
        C3, C5, C7 = mid_channels
        self.mid_channels = mid_channels
        self.module_list = nn.ModuleList([
            residual_branch(Cin, C3, Cout, 3, stride, 1),
            residual_branch(Cin, C5, Cout, 5, stride, 2),
            residual_branch(Cin, C7, Cout, 7, stride, 3),
        ])
        self.relu = nn.ReLU()
        # 设置架构可学习参数
        self.gates = nn.Parameter(torch.ones(3))
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(Cin, Cout,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(Cout)
            )

    def forward(self, x):
        residuals = []
        for m in self.module_list:
            residuals.append(m(x))

        probability = F.softmax(self.gates, dim=-1)
        merge_out = 0
        for r, p in zip(residuals, probability):
            merge_out += r * p

        out = self.shortcut(x) + merge_out
        out = self.relu(out)
        return out


class ResNetSearch(nn.Module):
    def __init__(self, depth=20, num_classes=10):
        super(ResNetSearch, self).__init__()
        n_blocks = (depth - 2) // 6

        self.in_channel = 16
        self.mid_channels = [16, 16, 16]
        self.conv = nn.Conv2d(
            3, self.in_channel,
            kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(n_blocks, 16, stride=1)
        self.layer2 = self._make_layer(n_blocks, 32, stride=2)
        self.layer3 = self._make_layer(n_blocks, 64, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_layer(self, n_blocks, out_channel, stride):
        multiplier = out_channel // self.in_channel
        self.mid_channels = [
            i * multiplier for i in self.mid_channels
        ]

        layers = []
        for i in range(n_blocks):
            layers.append(
                BasicBlockSearch(
                    self.in_channel,
                    self.mid_channels,
                    out_channel,
                    stride
                )
            )
            self.in_channel = out_channel
            stride = 1

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight.data, nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(-1, 64)
        x = self.linear(x)

        return x
