import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class Honda(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Name -> integer ID mapping
        self.class_to_idx = {
            'Accord': 0,
            'City': 1,
            'Civic': 2,
            'Pilot': 3,
            'Passport': 4,
            'Nsx': 5,
            'Odyssey': 6
        }

        # Invert dictionary for easy lookup later
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        class_name = os.path.splitext(img_name)[0].split('-')[0].strip().title()
        label = self.class_to_idx[class_name]

        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
