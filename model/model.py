import torch


from .asl_dataset import CATEGORIES, ASLDataset

class GestureClassifyModel(torch.nn.Module):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    # adjust final layer of the pretrained model to have the correct number of classes
    model.fc = torch.nn.Linear(model.fc.in_features, len(CATEGORIES))

    def __init__(self, path):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        # adjust final layer of the pretrained model to have the correct number of classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CATEGORIES))
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def forward(self, input):
        return self.model.forward(input)


def train():
    pass


def loss_fn(v):
    return torch.nn.CrossEntropyLoss(v)


def get_dataloader():
    dataset = ASLDataset("/gdrive/My Drive/Projects/kv260/asl_dataset", transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)


def test():
    pass
