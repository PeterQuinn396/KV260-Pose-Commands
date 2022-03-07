import glob
import os

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from PIL import Image

from pathlib import Path

CATEGORIES = ['up', 'down', 'left', 'right', 'fist', 'palm']
num_classes = 6


class GestureDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        mid = 1920 // 2
        ext = 1080 // 2
        x = self.x[i] / 255.
        x = x[..., mid - ext:mid + ext]
        return x, self.y[i]


class GestureDatasetFromFile(Dataset):

    def __init__(self, folder):
        self.folder = Path(folder)

        dirs = [d for d in self.folder.glob("**/*") if d.is_dir()]

        self.x = []
        self.y = []
        for d in dirs:
            ims = list(d.glob("*.jpg"))
            ys = [CATEGORIES.index(str(d.stem))] * len(ims)
            self.x.extend(ims)
            self.y.extend(ys)

        assert len(self.x) == len(self.y), f"{len(self.x)} != {len(self.y)}"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        im = Image.open(self.x[i])
        arr = np.array(im)
        arr = np.moveaxis(arr, -1, 0)  # move channels to front for pytorch
        arr = arr.astype(float) / 255.
        x = torch.from_numpy(arr).float()
        y = torch.from_numpy(np.array(self.y[i])).long()
        return x, y


def get_dataloader(path='./custom_dataset/custom_dataset_test.pckl'):
    with open(path, 'rb') as f:
        pickled_dict = pickle.load(f)
    # train_x, train_y = np.array(pickled_dict["train_x"]), np.array(pickled_dict["train_y"])
    test_x, test_y = np.array(pickled_dict["test_x"]), np.array(pickled_dict["test_y"])

    # for i in range(2):

    # train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    test_x, test_y = torch.from_numpy(test_x), torch.from_numpy(test_y)

    # train_x, train_y = train_x.float(), train_y.long()
    test_x, test_y = test_x.float(), test_y.long()

    # train_dataset = GestureDataset(x=train_x, y=train_y)
    test_dataset = GestureDataset(x=test_x, y=test_y)

    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    return test_dataloader


def get_pickle_dataset(path='./custom_dataset/custom_dataset_test.pckl'):
    with open(path, 'rb') as f:
        pickled_dict = pickle.load(f)
    # train_x, train_y = np.array(pickled_dict["train_x"]), np.array(pickled_dict["train_y"])
    test_x, test_y = np.array(pickled_dict["test_x"]), np.array(pickled_dict["test_y"])
    # train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    test_x, test_y = torch.from_numpy(test_x), torch.from_numpy(test_y)
    # train_x, train_y = train_x.float(), train_y.long()
    test_x, test_y = test_x.float(), test_y.long()

    # train_dataset = GestureDataset(x=train_x, y=train_y)
    test_dataset = GestureDataset(x=test_x, y=test_y)

    print(f"Using {len(test_dataset)} test images")
    return test_dataset


def get_dataloader_files(folder):
    test_dataset = GestureDatasetFromFile(folder)

    print(f"Using {len(test_dataset)} test images")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    return test_dataloader


class MyResnet(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        num_classes = 6
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        self._resnet = model
        self.extra_ave_pool = nn.AvgPool2d((7, 7))

    def forward(self, x):
        x = self._resnet.conv1(x)
        x = self._resnet.bn1(x)
        x = self._resnet.relu(x)
        x = self._resnet.maxpool(x)

        x = self._resnet.layer1(x)
        x = self._resnet.layer2(x)
        x = self._resnet.layer3(x)
        x = self._resnet.layer4(x)

        x = self.extra_ave_pool(x)
        x = self._resnet.avgpool(x)
        x = torch.reshape(x, (-1, self._resnet.fc.in_features))
        x = self._resnet.fc(x)
        return x


def get_model():
    model = MyResnet()
    return model


def get_model_pose():
    return torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)

if __name__ == '__main__':
    model = get_model()
    print(model)
    x = torch.rand(1, 3, 1080, 1920)
    mid = 1920 // 2
    ext = 1080 // 2
    x = x[..., mid - ext:mid + ext]

    print(x.size())
    y = model(x)
    print(y)
    print(y.size())
