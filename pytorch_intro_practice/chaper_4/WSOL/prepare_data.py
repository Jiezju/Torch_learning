import misc
from collections import OrderedDict

metadata = OrderedDict()

with open('class_labels.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    filename, label = line.strip('\n').split(',')
    metadata[filename] = {'label': int(label)}

with open('image_sizes.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    filename, width, height = line.strip('\n').split(',')
    metadata[filename]['image_size'] = [int(height), int(width)]

with open('localization.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    filename, x1, y1, x2, y2 = line.strip('\n').split(',')
    if metadata[filename].get('gt_bboxes') is None:
        metadata[filename]['gt_bboxes'] = [
            [int(x1), int(y1), int(x2), int(y2)]
        ]
    else:
        metadata[filename]['gt_bboxes'].append(
            [int(x1), int(y1), int(x2), int(y2)]
        )

misc.dump_pickle(metadata, 'imagenet_val_gt.pkl')
