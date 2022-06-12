import torch
import torch.nn as nn
import math
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import datasets, transforms


class Binarize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input + 1e-20)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        # 限制梯度，对于 x 较大时，截止回传梯度
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output


class BinarizedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 binarize_input=True): # 控制输入是否二值化
        super(BinarizedLinear, self).__init__()
        self.binarize_input = binarize_input
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        nn.init.kaiming_uniform_(
            self.weight, a=math.sqrt(5)
        )

    def forward(self, x):
        if self.binarize_input:
            x = Binarize.apply(x)  # 二值化输入
        w = Binarize.apply(self.weight) # 二值化权重
        out = torch.matmul(x, w.t())
        return out


TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1000
LR = 0.01
EPOCH = 100
LOG_INTERVAL = 100

model = nn.Sequential(
    BinarizedLinear(784, 2048, False), # 第一层输入不进行二值化
    nn.BatchNorm1d(2048), # 二值化乘加运算结果可能导致输出结果异常变大，所以采用归一化结果进行归一化
    BinarizedLinear(2048, 2048),
    nn.BatchNorm1d(2048),
    BinarizedLinear(2048, 2048),
    nn.Dropout(0.5),
    nn.BatchNorm1d(2048),
    nn.Linear(2048, 10)
)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LR
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=40, gamma=0.1  # 每 40 个周期下降为原来的十分之一
)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '/Users/gaojie/knowledge/data/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=TRAIN_BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '/Users/gaojie/knowledge/data/mnist', train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=TEST_BATCH_SIZE, shuffle=False)

for epoch in range(EPOCH):
    for idx, (data, label) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        output = model(data.view(-1, 28 * 28))
        loss = F.cross_entropy(output, label)
        loss.backward()

        optimizer.step()
        # 控制权重范围
        for p in model.parameters():
            p.data.clamp_(-1, 1)

        if idx % LOG_INTERVAL == 0:
            print('Epoch %03d [%03d/%03d]\tLoss: %.4f' % (
                epoch, idx, len(train_loader), loss.item()
            ))

    correct_num = 0
    total_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            model.eval()
            output = model(data.view(-1, 28 * 28))
            pred = output.max(1)[1]
            correct_num += (pred == label).sum().item()
            total_num += len(data)

    acc = correct_num / total_num
    print('...Testing @ Epoch %03d\tAcc: %.4f' % (
        epoch, acc
    ))

    scheduler.step()
