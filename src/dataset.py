import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from zipfile import ZipFile
from io import BytesIO
from PIL import Image
from typing import Optional, Callable


class ZipTrainDataset(Dataset):
    def __init__(self, zip_path, csv_name='dataset/train_solution.csv',
                 transform=None, ext='.jpg'):
        self.zip_path = zip_path
        self.csv_name = csv_name
        self.transform = transform
        self.ext = ext

        with ZipFile(self.zip_path, 'r') as z:
            with z.open(self.csv_name) as f:
                self.df = pd.read_csv(f, header=None)

        self.labels = torch.tensor(self.df[1].values, dtype=torch.long)
        self._zip = None

    def _get_zip(self):
        if self._zip is None:
            self._zip = ZipFile(self.zip_path, 'r')
        return self._zip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row[0]
        label = int(row[1])
        img_name = f"{img_id}{self.ext}"
        img_path_in_zip = f"dataset/train_images/{img_name}"
        z = self._get_zip()
        with z.open(img_path_in_zip) as f:
            image = Image.open(BytesIO(f.read())).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
