import os
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) + '.jpg'
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == '__main__':
    data = '/home/bright/Vscode_Project/pytorch/data/wheat_detection/'
    annotations_file = data + 'sample_submission.csv'
    img_dir = data + 'test/'
    # dataset 将数据封装，可以单个样本的获取
    data = CustomImageDataset(annotations_file, img_dir)
    # 将 data 封装迭代器，从而可以 batch 获取
    test_dataloader = DataLoader(data, batch_size=2, shuffle=True)
    test_features, test_labels = next(iter(test_dataloader))
    img = test_features[0].squeeze()
    plt.imshow(img)
    plt.show()
    print('Success!')
