import os
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchaudio.transforms as TA
import torch.nn.functional as F

DIM = 128


# time shift-u de la augmentare
class TimeShift(torch.nn.Module):
    def __init__(self, max_shift=0.10):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x):
        if torch.rand(1).item() < 0.5:
            shift = int(torch.rand(1).item() * self.max_shift * x.shape[-1])
            x = torch.roll(x, shifts=shift, dims=-1)
        return x


# formula de normalizare la distributia standard
class Standardize(torch.nn.Module):
    def forward(self, x):
        return (x - x.mean()) / (x.std() + 1e-8)


# stretch imagine + transformarea in tensori + augmentare + normalizare
train_transform = T.Compose([
    T.Resize((DIM, DIM), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
    T.RandomHorizontalFlip(p=0.5),
    TimeShift(max_shift=0.10),
    TA.FrequencyMasking(freq_mask_param=9),
    TA.TimeMasking(time_mask_param=12),
    Standardize()
])

# acelsi lucru pt datele de validare (deci fara augment :))
val_transform = T.Compose([
    T.Resize((DIM, DIM), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
    Standardize()
])


# clasa pt setul de date
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(csv_file)
        self.images = []
        self.labels = []

        for _, row in df.iterrows():
            img_path = os.path.join(self.img_dir, str(row.iloc[0]))
            try:
                img = Image.open(img_path).convert('L')
                self.images.append(img)
                # reindexarea de la 0
                self.labels.append(torch.tensor(row.iloc[1] - 1, dtype=torch.long))
            except FileNotFoundError:
                continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y