from model import ResNet
from torchvision import transforms, datasets
from torch import optim
import torch.nn as nn
import argparse
import torch
import os
import misc

print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--depth', default=20, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--mm', default=0.9, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--epochs', default=160, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--ksize', default=3, type=int)
parser.add_argument('--cutout', action='store_true', default=False)
parser.add_argument('--cutout_length', type=int, default=16)

args = parser.parse_args()
args.num_classes = 10
args.device = 'cuda'
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.logdir = 'baseline-resnet%d-%dx%d' % (args.depth, args.ksize, args.ksize)
if args.cutout:
    args.logdir += '-cutout'

misc.prepare_logging(args)

print('==> Preparing data..')


class Cutout(object):

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        y = torch.randint(h, (1, ))
        x = torch.randint(w, (1, ))

        y1 = torch.clamp(y - self.length // 2, 0, h)
        y2 = torch.clamp(y + self.length // 2, 0, h)
        x1 = torch.clamp(x - self.length // 2, 0, w)
        x2 = torch.clamp(x + self.length // 2, 0, w)

        img[:, y1:y2, x1:x2] = 0.
        return img


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.cutout:
    transform_train.transforms.append(Cutout(args.cutout_length))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = datasets.CIFAR10(root='./data',
                              train=True,
                              download=True,
                              transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size,
                                           pin_memory=True,
                                           shuffle=True,
                                           num_workers=2)
test_data = datasets.CIFAR10(root='./data',
                             train=False,
                             download=True,
                             transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=args.batch_size,
                                          pin_memory=True,
                                          shuffle=False,
                                          num_workers=2)

print('==> Initializing model...')
model_cfg = []
n_blocks = (args.depth - 2) // 6
C_in = {'3': 16, '5': 16, '7': 16}
pad = {'3': 1, '5': 2, '7': 3}
c = C_in[str(args.ksize)]
p = pad[str(args.ksize)]

for i in range(3):
    for _ in range(n_blocks):
        model_cfg.append({
            'kernel_size': args.ksize,
            'padding': p,
            'mid_channel': c
        })
    c *= 2
print('Searched ResNet%d: ' % args.depth)
for c in model_cfg:
    print(str(c))

model = ResNet(model_cfg, args.depth, args.num_classes)
model.to(args.device)
criterion = nn.CrossEntropyLoss()
criterion.to(args.device)
optimizer = optim.SGD(model.parameters(),
                      lr=args.lr,
                      momentum=args.mm,
                      weight_decay=args.wd)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[80, 120],
                                           gamma=0.1)


def train(epoch):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data = data.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            acc = (output.max(1)[1] == target).float().mean()
            print('Train Epoch: %d [%d/%d]  '
                  'Loss: %.4f, Acc: %.4f' %
                  (epoch, i, len(train_loader), loss.item(), acc.item()))


def evaluate(epoch):
    model.eval()
    loss_avg = 0
    correct = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(args.device)
            target = target.to(args.device)

            output = model(data)
            loss = criterion(output, target)
            pred = output.max(1)[1]

            loss_avg += loss.item()
            correct += (pred == target).float().sum().item()

    loss_avg /= len(test_loader)
    acc = correct / len(test_loader.dataset)
    print('...Test @ Epoch: %d  Loss: %.4f, Acc: %.4f' %
          (epoch, loss_avg, acc))


for epoch in range(args.epochs):
    train(epoch)
    evaluate(epoch)
    scheduler.step()
    torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
