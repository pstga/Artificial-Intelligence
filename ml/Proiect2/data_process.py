import os
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchaudio.transforms as TA
import torch.nn.functional as F


DIM = 128  # Secretul pentru salvarea axei scurte


class TimeShift(torch.nn.Module):
    def __init__(self, max_shift_pct=0.10):
        super().__init__()
        self.max_shift_pct = max_shift_pct

    def forward(self, x):
        if torch.rand(1).item() < 0.5:
            shift = int(torch.rand(1).item() * self.max_shift_pct * x.shape[-1])
            x = torch.roll(x, shifts=shift, dims=-1)
        return x


class Standardize(torch.nn.Module):
    def forward(self, x):
        return (x - torch.mean(x)) / (torch.std(x) + 1e-8)


train_transform = T.Compose([
    T.Resize((DIM, DIM), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
    T.RandomHorizontalFlip(p=0.5),
    TimeShift(max_shift_pct=0.10),
    TA.FrequencyMasking(freq_mask_param=9),
    TA.TimeMasking(time_mask_param=12),
])

val_transform = T.Compose([
    T.Resize((DIM, DIM), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = pd.read_csv(csv_file)
        self.images = []
        self.labels = []

        print(f"Preloading {len(self.annotations)} images...")
        for idx in range(len(self.annotations)):
            img_name = str(self.annotations.iloc[idx, 0])
            label = self.annotations.iloc[idx, 1]
            img_path = os.path.join(self.img_dir, img_name)
            try:
                image = Image.open(img_path).convert('L')
                self.images.append(image)
                # Conversie din 1-5 în 0-4 pentru PyTorch
                self.labels.append(torch.tensor(label - 1, dtype=torch.long))
            except FileNotFoundError:
                continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate_tta_classes(model, val_loader, weights_path=None, device='cuda'):
    """Evaluare cu TTA integrat direct în raport"""
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    model.eval()
    model.to(device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            p_orig = F.softmax(model(inputs), dim=1)
            p_left = F.softmax(model(torch.roll(inputs, shifts=-10, dims=-1)), dim=1)
            p_right = F.softmax(model(torch.roll(inputs, shifts=10, dims=-1)), dim=1)

            avg_probs = (p_orig + p_left + p_right) / 3.0
            preds = torch.argmax(avg_probs, dim=1)

            preds, targets = preds + 1, targets + 1  # Readucem la 1-5
            all_preds.extend(preds.cpu().int().numpy())
            all_labels.extend(targets.cpu().int().numpy())

    print(classification_report(all_labels, all_preds, zero_division=0))