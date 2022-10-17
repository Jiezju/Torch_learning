import torch
import numpy as np
from PIL import Image
import torchvision.transforms as tfm
from torchvision.datasets import ImageFolder
import torch.utils.data as data

# 归一化参数
imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

# 去掉pytorch 默认的 数据提取（会有冗余的分类和检查），自定义更加高效


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def get_imagenet_loader(root, batch_size, type='train'):
    crop_scale = 0.25
    jitter_param = 0.4
    lighting_param = 0.1
    if type == 'train':
        transform = tfm.Compose([
            tfm.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            tfm.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            Lighting(lighting_param),
            tfm.RandomHorizontalFlip(),
        ])

    elif type == 'test':
        transform = tfm.Compose([
            tfm.Resize(256),
            tfm.CenterCrop(224),
        ])

    dataset = ImageFolder(root, transform)
    sampler = data.distributed.DistributedSampler(dataset)
    data_loader = data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=4,
        pin_memory=True, sampler=sampler,
        collate_fn=fast_collate
    )
    if type == 'train':
        return data_loader, sampler

    elif type == 'test':
        return data_loader


# 在模型计算的同时，实现数据加载，并转移到 gpu 上
class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [0.485 * 255, 0.456 * 255, 0.406 * 255]
        ).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229 * 255, 0.224 * 255, 0.225 * 255]
        ).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # 建立额外的 stream 进行数据加载和预处理
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
