import torch
import torch.nn as nn
import torch.nn.functional as F


def residual_branch(in_channel, mid_channel, out_channel, kernel_size, stride,
                    padding):
    return nn.Sequential(
        nn.Conv2d(in_channel,
                  mid_channel,
                  kernel_size,
                  stride,
                  padding,
                  bias=False),
        nn.BatchNorm2d(mid_channel),
        nn.ReLU(),
        nn.Conv2d(mid_channel,
                  out_channel,
                  kernel_size,
                  1,
                  padding,
                  bias=False),
        nn.BatchNorm2d(out_channel),
    )


class BasicBlock(nn.Module):

    def __init__(self,
                 in_channel,
                 mid_channel,
                 out_channel,
                 kernel_size,
                 padding,
                 stride=1):
        super(BasicBlock, self).__init__()
        self.residual = residual_branch(in_channel, mid_channel, out_channel,
                                        kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel,
                          out_channel,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(out_channel))

    def forward(self, x):
        res = self.residual(x)
        out = self.shortcut(x) + res
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, cfg, depth=20, num_classes=10):
        super(ResNet, self).__init__()
        n_blocks = (depth - 2) // 6

        self.cfg = cfg
        self.in_channel = 16
        self.conv = nn.Conv2d(3,
                              self.in_channel,
                              kernel_size=3,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(cfg[:n_blocks], 16, stride=1)
        self.layer2 = self._make_layer(cfg[n_blocks:2 * n_blocks],
                                       32,
                                       stride=2)
        self.layer3 = self._make_layer(cfg[2 * n_blocks:], 64, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_layer(self, cfg, out_channel, stride):
        layers = []
        for c in cfg:
            kernel_size = c['kernel_size']
            padding = c['padding']
            mid_channel = c['mid_channel']
            layers.append(
                BasicBlock(self.in_channel, mid_channel, out_channel,
                           kernel_size, padding, stride))
            self.in_channel = out_channel
            stride = 1

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
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
