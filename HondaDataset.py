import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class Honda(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get image path
        img_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.root_dir, img_name)

        # Open image
        image = Image.open(img_path).convert("RGB")

        # Handle label if present
        if self.annotations.shape[1] > 1:
            y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        else:
            y_label = -1

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, y_label
