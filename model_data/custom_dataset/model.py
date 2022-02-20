
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

CATEGORIES = ['up', 'down', 'left', 'right', 'fist', 'palm']
num_classes = 6

class GestureDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def get_dataloader(path='./custom_dataset/custom_dataset.pckl'):
    with open('./custom_dataset.pckl', 'rb') as f:
        pickled_dict = pickle.load(f)
    train_x, train_y = np.array(pickled_dict["train_x"]), np.array(pickled_dict["train_y"])
    test_x, test_y =  np.array(pickled_dict["test_x"]), np.array(pickled_dict["test_y"])

    train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    test_x, test_y = torch.from_numpy(test_x), torch.from_numpy(test_y)

    train_x, train_y = train_x.float(), train_y.long()
    test_x, test_y = test_x.float(), test_y.long()

    train_dataset = GestureDataset(x=train_x, y=train_y)
    test_dataset = GestureDataset(x=test_x, y=test_y)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_dataloader  = DataLoader(test_dataset,  batch_size=4, shuffle=True, num_workers=2)

    return train_dataloader, test_dataloader


def get_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
