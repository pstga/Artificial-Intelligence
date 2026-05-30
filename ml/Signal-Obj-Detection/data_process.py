import os
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.distributions.beta import Beta


def mixup_data(x, y, alpha=0.1):
    if alpha > 0:
        lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample().item()
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)

    # Blend the audio
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Blend the object counts (e.g., 80% of a 5-object sound + 20% of a 1-object sound = 4.2 target)
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss against both sets of targets and weights it by lambda.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


train_transform = T.Compose([
    T.RandomAffine(degrees=0, translate=(0.04, 0.0)),
    T.ToTensor(),
    T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.1, 10), value=0),
    T.Lambda(lambda x: (x - torch.mean(x)) / (torch.std(x) + 1e-8))
])

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
                self.labels.append(torch.tensor(label, dtype=torch.float))
            except FileNotFoundError:
                print(f"Warning: {img_name} not found. Skipping.")
                continue

        print("Preloading complete!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Grab the raw PIL image from RAM
        image = self.images[idx]
        label = self.labels[idx]

        # 2. Apply the dynamic transformation here!
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
            # targets for validation are already raw floats in regression
            logits = model(inputs)

            # Convert continuous regression prediction to integer class
            preds = torch.round(logits).clamp(min=1, max=5)

            all_preds.extend(preds.cpu().int().numpy())
            all_labels.extend(targets.cpu().int().numpy())

    print(classification_report(all_labels, all_preds, zero_division=0))