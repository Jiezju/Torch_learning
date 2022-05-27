import torch.nn as nn
from collections import OrderedDict

class SimpleCNN(nn.Module):
    def __init__(self, in_channel=1):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 4, 5 ,2, 0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 8, 3, 2, 0)
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.reshape(-1, 200)

        out = self.linear(x)

        return out

# init weight
def init_weights(m):
    if m.__class__.__name__ == 'Conv2d':
        m.weight.data.normal_()
        # nn.init.kaiming_normal_(m.weight.data)

model = SimpleCNN()

model.apply(init_weights)

for k, v in model._modules.items():
    print(k, v)

# 返回各个可训练参数
for p in model.parameters():
    print(p.shape)

# list
model = nn.Sequential(nn.Conv2d(1, 4, 5, 2, 0), nn.ReLU())
print(model._modules['0'])

# dict
model = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(1, 4, 5, 2, 0)), ('relu1', nn.ReLU())]))
print(model.conv1)

'''
一般训练流程

model = init_model()
optimizer = optim.Optimizer(mdoel.parameters(), lr, mm)
scheduler = optim.lr_scheduler.Scheduler(optimizer)

for _ in range(epoch):
    for data, label in train_dataloader:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, label)
        loss.backward()
        optimizer.step()
        
    scheduler.step()
'''

