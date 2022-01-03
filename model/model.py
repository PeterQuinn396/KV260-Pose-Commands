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
        # input is [BATCH_SIZE, 3, 224, 244]
        return self.model.forward(input)


def loss_fn(out, labels):
    return torch.nn.CrossEntropyLoss(out, labels)


def get_dataloader(data_set_path: str):
    dataset = ASLDataset(data_set_path, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    return dataloader


def test_vitis_compatible(model):
    """
    Run the torch jit check to make sure the model is quatizable
    by Vitis
    :param model:
    :return:
    """
    test_input = torch.rand((1,3,224,224)).to(device)
    try:
        tr_func = torch.jit.trace(model, test_input)
    except:
        return False
    return True


def train():
    pass


def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, max_samples=100):
    model.eval()

    running_corrects = 0
    running_loss = 0

    count = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == torch.argmax(labels, 1))

            # put a max on the number of samples tested
            count += len(inputs)
            if count > max_samples:
                break

    epoch_loss = running_loss / count
    epoch_acc = running_corrects.double() / count

    return epoch_acc, epoch_loss

