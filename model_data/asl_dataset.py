import os
import torch
from torchvision import transforms
import glob
from PIL import Image

CATEGORIES= ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q',
             'r','s','t','u','v','w','x','y']

class ASLDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        path = os.path.join(root_dir)
        filenames = glob.glob(f"{path}/*.jpg")
        filenames = sorted(filenames)
        self.filenames = filenames
        self.files = []
        for fname in self.filenames:
            f = Image.open(fname)
            self.files.append(f)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        fname = self.filenames[i]
        jpg_name = fname.split('/')[-1]
        label = jpg_name[0]
        # input_image = Image.open(fname)
        input_image = self.files[i]
        if self.transform:
            input_image = self.transform(input_image)

        label_tensor = torch.zeros(len(CATEGORIES))
        ind = CATEGORIES.index(label)
        label_tensor[ind] = 1
        return input_image, label_tensor


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])