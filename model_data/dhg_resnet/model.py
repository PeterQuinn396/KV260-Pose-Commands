
import torch


def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 14)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(2, 2), stride=(1, 1), bias=False)
    return model
