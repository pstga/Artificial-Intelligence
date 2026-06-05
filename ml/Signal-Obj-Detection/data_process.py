import os
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.distributions.beta import Beta

def mixup_data(x, y, alpha=0.2):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample().item()
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)

    # Blend the images
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Keep both targets instead of interpolating them
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss against both sets of targets and weights it by lambda.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


import torch
import torchvision.transforms as T
import torchaudio.transforms as TA

# The maximum width/height of the masks.
# You may need to tune these depending on your spectrogram's actual dimensions!
FREQ_MASK_MAX = 20  # Max horizontal band thickness
TIME_MASK_MAX = 20  # Max vertical band thickness

train_transform = T.Compose([
    T.ToTensor(),  # Converts PIL Image to tensor [1, H, W]

    # 1. Frequency Masking: Hides specific frequency bands across the whole audio
    TA.FrequencyMasking(freq_mask_param=FREQ_MASK_MAX),

    # 2. Time Masking: Hides all frequencies for short bursts of time
    TA.TimeMasking(time_mask_param=TIME_MASK_MAX),

    # Optional: Apply a second time mask to make it harder, which is standard for SpecAugment
    TA.TimeMasking(time_mask_param=TIME_MASK_MAX),

    # Standardize as usual
    T.Lambda(lambda x: (x - torch.mean(x)) / (torch.std(x) + 1e-8))
])

# Validation transform remains clean
val_transform = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: (x - torch.mean(x)) / (torch.std(x) + 1e-8))
])

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = pd.read_csv(csv_file)

        self.images = []
        self.labels = []

        print(f"Preloading {len(self.annotations)} images and labels into RAM...")

        for idx in range(len(self.annotations)):
            img_name = str(self.annotations.iloc[idx, 0])
            label = self.annotations.iloc[idx, 1]

            img_path = os.path.join(self.img_dir, img_name)

            try:
                image = Image.open(img_path).convert('L')
                self.images.append(image)
                # CHANGED: Convert 1-5 to 0-4 and use dtype=long for CrossEntropy
                self.labels.append(torch.tensor(label - 1, dtype=torch.long))
            except FileNotFoundError:
                print(f"Warning: {img_name} not found. Skipping.")
                continue

        print("Preloading complete!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def evaluate_classes(model, val_loader, weights_path=None, device='cuda'):
    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)

            # CHANGED: Use argmax for classification
            preds = torch.argmax(logits, dim=1)

            # Revert 0-4 indexing back to 1-5 for readable reports
            preds = preds + 1
            targets = targets + 1

            all_preds.extend(preds.cpu().int().numpy())
            all_labels.extend(targets.cpu().int().numpy())

    print(classification_report(all_labels, all_preds, zero_division=0))