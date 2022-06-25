import torch
import argparse
import os
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import dataset
import misc
from torch.nn.parallel import DistributedDataParallel as DDP
import models


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='imagenet', type=str)
parser.add_argument('--arch', '-a', default='mobilenet_v2', type=str)
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--mm', default=0.9, type=float)
parser.add_argument('--wd', default=4e-5, type=float)
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--log_interval', default=50, type=int)
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--label_smooth', action='store_true')

args = parser.parse_args()
args.logdir = 'baseline-%s' % (args.arch)
if args.label_smooth:
    args.logdir += '-labelsmooth'

torch.backends.cudnn.benchmark = True

# 使用 nn.distributedDataParallel 进行数据的并行
args.gpu = args.local_rank % torch.cuda.device_count()
torch.cuda.set_device(args.gpu)
torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')
args.world_size = torch.distributed.get_world_size()

if args.local_rank == 0:
    misc.prepare_logging(args)


def print(msg):
    if args.local_rank == 0:
        misc.logger.info(msg)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


print("=> Using model {}".format(args.arch))
model = models.mobilenet_v2()
model = model.cuda()
# 建立分布式模型
model = DDP(model, device_ids=[args.gpu])

criterion = nn.CrossEntropyLoss().cuda()
if args.label_smooth:
    '''
    smooth 正则化损失函数

    '''
    class CrossEntropyLabelSmooth(nn.Module):
        def __init__(self, num_classes, epsilon):
            super(CrossEntropyLabelSmooth, self).__init__()
            self.num_classes = num_classes
            self.epsilon = epsilon
            self.logsoftmax = nn.LogSoftmax(dim=1)

        def forward(self, inputs, targets):
            log_probs = self.logsoftmax(inputs)
            # onehot
            targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            # smooth label y = (1- alpha) * y_one_hot + alpha / K
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (-targets * log_probs).mean(0).sum()
            return loss
    criterion = CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1).cuda()

print('==> Preparing data..')
train_loader, train_sampler = dataset.get_imagenet_loader(
    os.path.join(args.data, 'train'), args.batch_size, type='train'
)
test_loader = dataset.get_imagenet_loader(
    os.path.join(args.data, 'val'), 100, type='test'
)

# 提取模型权重，并设置不同的动量和权重 decay 信息
args.lr = args.lr * float(args.batch_size * args.world_size) / 256.
model_params = []
for params in model.parameters():
    ps = list(params.size())
    if len(ps) == 4 and ps[1] != 1:
        weight_decay = args.wd
    elif len(ps) == 2:
        weight_decay = args.wd
    else:
        weight_decay = 0
    item = {'params': params, 'weight_decay': weight_decay,
            'lr': args.lr, 'momentum': args.mm,
            'nesterov': True}
    model_params.append(item)

optimizer = torch.optim.SGD(model_params)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=0
)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    prefetcher = dataset.DataPrefetcher(train_loader)
    model.train()

    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)

            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))

        input, target = prefetcher.next()


def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    prefetcher = dataset.DataPrefetcher(val_loader)
    model.eval()

    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        reduced_loss = reduce_tensor(loss.data)
        prec1 = reduce_tensor(prec1)
        prec5 = reduce_tensor(prec5)

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        input, target = prefetcher.next()

    print(' * Test Epoch {0}, Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
          .format(epoch, top1=top1, top5=top5))

    return top1.avg


for epoch in range(args.epochs):
    train_sampler.set_epoch(epoch)
    train(train_loader, model, criterion, optimizer, epoch)
    prec1 = validate(test_loader, model, criterion, epoch)

    if args.local_rank == 0:
        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.logdir, 'checkpoint.pth'))

    lr_scheduler.step()
