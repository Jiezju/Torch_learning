from model_search import ResNetSearch
import torch
import argparse
import os
import misc

parser = argparse.ArgumentParser()
parser.add_argument('--depth', default=20, type=int)

args = parser.parse_args()
args.num_classes = 10
args.model_weights_path = os.path.join(
    'logs/search-resnet%d' % args.depth,
    'model_search.pth'
)

model = ResNetSearch(args.depth, args.num_classes)
model.load_state_dict(torch.load(args.model_weights_path))
kernel_size = [3, 5, 7]
padding = [1, 2, 3]

config = []
for m in model.modules():
    if m.__class__.__name__ == 'BasicBlockSearch':
        select_idx = m.gates.argmax().item()
        mid_channel = m.mid_channels[select_idx]
        config.append({
            'kernel_size': kernel_size[select_idx],
            'padding': padding[select_idx],
            'mid_channel': mid_channel
        })
print('Searched ResNet%d: ' % args.depth)
for c in config:
    print(c)

misc.dump_pickle(config, os.path.join(
    'logs/search-resnet%d' % args.depth,
    'model_config.pkl'
))
