import torch.utils.data as data
from PIL import Image
import os
import numpy as np

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


class ADVMNISTLoader(data.Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform

        self.imgs, self.labels = np.load(data_path, allow_pickle=True)
        self.n_data = len(self.imgs)

    def __getitem__(self, item):
        if self.transform is not None:

            # img = self.transform(self.imgs[item].reshape((28, 28, 1)))
            img = self.imgs[item]
            label = int(self.labels[item])

        return img, label

    def __len__(self):
        return self.n_data
