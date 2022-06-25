from dataset import ImageNet, custom_collate_fn
from gradcam import GradCAM
from torchvision import models, transforms
from torch.utils import data
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import misc
import helper
import torch
import argparse
import os

print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--arch', default='alexnet', type=str)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--max_alpha', default=5.2, type=float)
parser.add_argument('--alpha_step', default=0.2, type=float)

args = parser.parse_args()
args.device = 'cuda'
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.arch == 'alexnet':
    args.target_layer = ['features', '12']

elif args.arch == 'vgg16':
    args.target_layer = ['features', '30']

elif args.arch == 'resnet50':
    args.target_layer = ['layer4', '2']

else:
    raise NotImplementedError

args.logdir = 'wsol-%s' % args.arch

misc.prepare_logging(args)

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print('==> Preparing data..')

metadata = misc.load_pickle('imagenet_val_gt.pkl')
data_loader = data.DataLoader(ImageNet(metadata, tfm),
                              batch_size=args.batch_size,
                              collate_fn=custom_collate_fn,
                              num_workers=4,
                              pin_memory=True)

print('==> Initializing model...')

model = models.__dict__[args.arch](pretrained=True)
model.eval()
model.to(args.device)
explainer = GradCAM(model, args.target_layer)

num_alphas = len(np.arange(0, args.max_alpha, args.alpha_step))
correct_loc = torch.zeros(num_alphas)

for data, label, sizes, gt_bboxes in tqdm(data_loader):
    data, label = data.to(args.device), label.to(args.device)

    cam = explainer.generate_saliency_map(data, label)

    for i in range(len(data)):
        saliency_map = F.interpolate(cam[i].unsqueeze(0),
                                     sizes[i],
                                     mode='bilinear')
        saliency_map = saliency_map.squeeze().cpu().numpy()
        # 搜索最优的 alpha
        alphas = np.arange(0, args.max_alpha, args.alpha_step)
        for k, alpha in enumerate(alphas):
            pred_bbox = helper.getbb_from_heatmap(saliency_map, alpha)
            ious = helper.ious(pred_bbox, gt_bboxes[i])
            if max(ious) > 0.5:
                correct_loc[k] += 1

correct_loc /= len(data_loader.dataset)
alphas = np.arange(0, args.max_alpha, args.alpha_step)
for k, alpha in enumerate(alphas):
    print('loc acc = %.4f @ alpha = %.2f' % (correct_loc[k].item(), alpha))
misc.dump_pickle(correct_loc, os.path.join(args.logdir, 'loc_acc.pkl'))
