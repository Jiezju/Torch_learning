# fake dataloader
import torch
from torch.utils import data

class FakeData(data.Dataset):
    def __init__(self, max_len):
        self.max_len = max_len

    def __getitem__(self, item):
        return torch.arange(item + 1)

    def __len__(self):
        return self.max_len

# 涉及到变长度样本情况，需要自定义 collate_fn
def custom_collate_fn(batch):
    lengths = [len(b) for b in batch]
    data = torch.zeros(len(batch), max(lengths))

    for i in range(len(batch)):
        end = lengths[i]
        data[i, :end] = batch[i]
        return data

loader = data.DataLoader(FakeData(10),
                         batch_size=5,
                         shuffle=True,
                         pin_memory=False,  # 使用 pin memory 加速 CPU  转移 GPU 的速度
                         collate_fn=custom_collate_fn)

for data in loader:
    print(data)