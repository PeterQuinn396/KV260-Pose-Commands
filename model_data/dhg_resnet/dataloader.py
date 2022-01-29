import numpy
import pickle

import torch
from torch.utils.data import DataLoader, Dataset

CATEGORIES = ['Grab', 'Tap', 'Expand', 'Pinch', 'Rotation Clockwise', 'Rotation Counter Clockwise',
              'Swipe Right', 'Swipe Left', 'Swipe Up','Swipe Down', 'Swipe X', 'Swipe +', 'Swipe V',
              'Shake']


def _load_data(filepath='./shrec_data.pckl'):
    """
    Returns hand gesture sequences (X) and their associated labels (Y).
    Each sequence has two different labels.
    The first label  Y describes the gesture class out of 14 possible gestures (e.g. swiping your hand to the right).
    The second label Y describes the gesture class out of 28 possible gestures (e.g. swiping your hand to the right
    with your index pointed, or not pointed).
    """
    file = open(filepath, 'rb')
    data = pickle.load(file, encoding='latin1')  # <<---- change to 'latin1' to 'utf8' if the data does not load
    file.close()
    return data['x_train'], data['x_test'], data['y_train_14'], data['y_train_28'], data['y_test_14'], data['y_test_28']


class GestureDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        x = x.reshape((3,100,22))
        return x, self.y[i]


def get_dataloader(path = 'dhg_data.pckl'):
    x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = _load_data(path)
    y_train_14, y_test_14 = numpy.array(y_train_14), numpy.array(y_test_14)
    y_train_28, y_test_28 = numpy.array(y_train_28), numpy.array(y_test_28)

    y_train = y_train_14
    y_test = y_test_14

    # Convert from numpy to torch format
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    # Ensure the label values are between 0 and n_classes-1
    if y_train.min() > 0:
        y_train = y_train - 1
    if y_test.min() > 0:
        y_test = y_test - 1

        # Ensure the data type is correct
    x_train, x_test = x_train.float(), x_test.float()
    y_train, y_test = y_train.long(), y_test.long()

    # Create the datasets
    train_dataset = GestureDataset(x=x_train, y=y_train)
    test_dataset = GestureDataset(x=x_test, y=y_test)

    # Pytorch dataloaders are used to group dataset items into batches
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

    return train_dataloader, test_dataloader
