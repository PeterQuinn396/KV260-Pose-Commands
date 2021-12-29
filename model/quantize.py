
import torch
from .asl_dataset import ASLDataset, preprocess
from .model import loss_fn, GestureClassifyModel



def main():
    model = GestureClassifyModel("asl_resnext.pth")
