import torch

try:
    from .asl_dataset import CATEGORIES, ASLDataset, preprocess
    from .common import device
    from .resnet import resnext50_32x4d
except:
    from asl_dataset import CATEGORIES, ASLDataset, preprocess
    from common import device
    from resnet import resnext50_32x4d


class GestureClassifyModel(torch.nn.Module):

    def __init__(self, path):
        super().__init__()

        # we will be loading our own state_dict, so we can leave pretrained false
        self.model = resnext50_32x4d(pretrained=False)
        # adjust final layer of the pretrained model_data to have the correct number of classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CATEGORIES))
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()

    def forward(self, input):
        # input is [BATCH_SIZE, 3, 224, 244]
        return self.model.forward(input)


def get_loss_fn():
    return torch.nn.CrossEntropyLoss()


def get_dataloader(data_set_path: str, batch_size=4):
    dataset = ASLDataset(data_set_path, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def test_vitis_compatible(model):
    """
    Run the torch jit check to make sure the model_data is quatizable
    by Vitis
    :param model:
    :return:
    """
    test_input = torch.rand((1, 3, 224, 224)).to(device)
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

    loss_fn = get_loss_fn()
    count = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            lb_inds = torch.argmax(labels, 1)
            loss = loss_fn(outputs, lb_inds)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == lb_inds)

            # put a max on the number of samples tested
            count += len(inputs)
            if count > max_samples:
                break

    epoch_loss = running_loss / count
    epoch_acc = running_corrects.double() / count

    return epoch_acc, epoch_loss, count
