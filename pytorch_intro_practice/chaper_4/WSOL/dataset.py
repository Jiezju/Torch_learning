from torch.utils import data
import misc
import torch


class ImageNet(data.Dataset):
    def __init__(self, metadata, transform):
        self.metadata = metadata
        self.transform = transform

    def __getitem__(self, item):
        index = '000%05d' % (item + 1)
        name = 'val/ILSVRC2012_val_%s.JPEG' % index
        info = self.metadata[name]
        img = misc.pil_loader(name)
        img = self.transform(img)
        label = info['label']
        size = info['image_size']
        bbox = info['gt_bboxes']
        return img, label, size, bbox

    def __len__(self):
        return 50000


def custom_collate_fn(batch):
    imgs = []
    labels = []
    sizes = []
    bboxes = []
    for img, label, size, bbox in batch:
        imgs.append(img)
        labels.append(label)
        sizes.append(size)
        bboxes.append(bbox)

    imgs = torch.stack(imgs)
    labels = torch.Tensor(labels).long()

    return imgs, labels, sizes, bboxes