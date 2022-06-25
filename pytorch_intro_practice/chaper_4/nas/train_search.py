from model_search import ResNetSearch
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import argparse
import torch
import os
import misc

print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--depth', default=20, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--lr_min', default=0.001, type=float)
parser.add_argument('--mm', default=0.9, type=float)
parser.add_argument('--wd', default=3e-4, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--train_portion', default=0.5, type=float)
parser.add_argument('--arch_lr', default=1e-3, type=float)
parser.add_argument('--arch_wd', default=1e-3, type=float)

args = parser.parse_args()
args.num_classes = 10
args.device = 'cuda'
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.logdir = 'search-resnet%d' % args.depth

misc.prepare_logging(args)

print('==> Preparing data..')
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = datasets.CIFAR10(root='./data',
                              train=True,
                              download=True,
                              transform=transform)
num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size,
                                           sampler=SubsetRandomSampler(
                                               indices[:split]),
                                           pin_memory=True,
                                           num_workers=2)
val_loader = torch.utils.data.DataLoader(train_data,
                                         batch_size=args.batch_size,
                                         sampler=SubsetRandomSampler(
                                             indices[split:]),
                                         pin_memory=True,
                                         num_workers=2)

print('==> Initializing model...')

model = ResNetSearch(args.depth, args.num_classes)
model.to(args.device)
criterion = nn.CrossEntropyLoss()
criterion.to(args.device)

model_params = []
arch_params = []
# 单独使用模型参数和架构参数
for k, p in model.named_parameters():
    if k.endswith('gates'):
        arch_params.append(p)
    else:
        model_params.append(p)

# 两种参数分别使用优化器
arch_optim = optim.Adam(arch_params,
                        lr=args.arch_lr,
                        betas=(0.5, 0.999),
                        weight_decay=args.arch_wd)
model_optim = optim.SGD(model_params,
                        lr=args.lr,
                        momentum=args.mm,
                        weight_decay=args.wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optim,
                                                 args.epochs,
                                                 eta_min=args.lr_min)
writer = SummaryWriter(os.path.join(args.logdir, 'search-tb'))

train_counter = 0
valid_counter = 0


def train(epoch):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data = data.to(args.device)
        target = target.to(args.device)

        data_arch, target_arch = next(iter(val_loader))
        data_arch = data_arch.to(args.device)
        target_arch = target_arch.to(args.device)

        # 先更新架构参数， 然后更新模型权重
        arch_optim.zero_grad()
        output = model(data_arch)
        loss_arch = criterion(output, target_arch)
        loss_arch.backward()
        arch_optim.step()

        model_optim.zero_grad()
        output = model(data)
        loss_model = criterion(output, target)
        loss_model.backward()
        model_optim.step()

        if i % args.log_interval == 0:
            acc = (output.max(1)[1] == target).float().mean()
            print('Train Epoch: %d [%d/%d]  '
                  'Loss_A: %.4f, Loss_M: %.4f, Acc: %.4f' %
                  (epoch, i, len(train_loader), loss_arch.item(),
                   loss_model.item(), acc.item()))
            global train_counter
            writer.add_scalars('Loss/train_loss', {
                'loss_arch': loss_arch.item(),
                'loss_model': loss_model.item()
            }, train_counter)
            writer.add_scalar('Accuracy/train_acc', acc.item(), train_counter)
            for k, p in enumerate(arch_params):
                writer.add_scalars('Gates/gates_%d' % k, {
                    '3x3': p[0].item(),
                    '5x5': p[1].item(),
                    '7x7': p[2].item()
                }, train_counter)
            train_counter += 1


def evaluate(epoch):
    model.eval()
    loss_avg = 0
    acc_avg = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data = data.to(args.device)
            target = target.to(args.device)

            output = model(data)
            loss = criterion(output, target)
            acc = (output.max(1)[1] == target).float().mean()
            loss_avg += loss.item()
            acc_avg += acc.item()

    loss_avg /= len(val_loader)
    acc_avg /= len(val_loader)
    print('...Evaluate @ Epoch: %d  Loss: %.4f, Acc: %.4f' %
          (epoch, loss_avg, acc_avg))
    global valid_counter
    writer.add_scalar('Loss/valid_loss', loss_avg, valid_counter)
    writer.add_scalar('Accuracy/valid_acc', acc_avg, valid_counter)
    valid_counter += 1


for epoch in range(args.epochs):
    train(epoch)
    evaluate(epoch)
    scheduler.step()
    torch.save(model.state_dict(), os.path.join(args.logdir,
                                                'model_search.pth'))
