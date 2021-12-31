import torch

from .asl_dataset import CATEGORIES, ASLDataset, preprocess
from common import device


class GestureClassifyModel(torch.nn.Module):

    def __init__(self, path):
        super().__init__()

        # we will be loading our own state_dict, so we can leave pretrained false
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=False)
        # adjust final layer of the pretrained model to have the correct number of classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CATEGORIES))
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()

    def forward(self, input):
        return self.model.forward(input)


def loss_fn(v):
    return torch.nn.CrossEntropyLoss(v)


def get_dataloader():
    dataset = ASLDataset("/gdrive/My Drive/Projects/kv260/asl_dataset", transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    return dataloader


def test_vitis_compatible():
    pass


def train():
    pass


def test():
    pass
