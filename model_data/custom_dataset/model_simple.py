import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image

CATEGORIES = ['up', 'down', 'left', 'right', 'fist', 'palm']
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class GestureDatasetFromFile(Dataset):

    def __init__(self, folder):
        self.folder = Path(folder)

        dirs = [d for d in self.folder.glob("**/*") if d.is_dir()]

        self.x = []
        self.y = []
        self.tf = transforms.ToTensor()
        for d in dirs:
            ims = list(sorted(d.glob("*.jpg")))[0:50]
            ys = [CATEGORIES.index(str(d.stem))] * len(ims)
            self.x.extend(ims)
            self.y.extend(ys)

        assert len(self.x) == len(self.y), f"{len(self.x)} != {len(self.y)}"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        im = Image.open(self.x[i])
        im = np.array(im)
        y = torch.from_numpy(np.array(self.y[i]))
        return im, y


class PointDataset(Dataset):

    def __init__(self, image_dataset) -> None:
        super().__init__()
        self.hand_detector = mp_hands.Hands(min_detection_confidence=.5, min_tracking_confidence=.5, max_num_hands=1)
        self.image_dataset = image_dataset

    def get_3d_points(self, image):
        im = image[400:800, 500:1500, :]
        for i in range(4):
            results = self.hand_detector.process(im)  # trick the tracker
        arr = []
        if results.multi_hand_world_landmarks:
            for pt in results.multi_hand_world_landmarks[0].landmark:
                coords = [pt.x, pt.y, pt.z]
                arr.append(coords)
        else:
            arr = np.zeros(63)

        arr = np.array(arr).reshape(3, 1, 21).astype(np.float32).flatten()
        return torch.from_numpy(arr)

    def __getitem__(self, index):
        x, y = self.image_dataset[index]
        x = self.get_3d_points(x)
        return x, y

    def __len__(self):
        return len(self.image_dataset)


class SimpleMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(nn.Linear(63, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 6))

    def forward(self, x):
        return self._model(x)


def get_model():
    return SimpleMLP()

def get_dataset(path):
    d = GestureDatasetFromFile(path)
    return PointDataset(d)