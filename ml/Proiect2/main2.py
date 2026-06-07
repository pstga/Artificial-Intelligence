import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchaudio.transforms as TA
import copy

# ==========================================
# 1. CONFIGURĂRI & HIPERPARAMETRI
# ==========================================
DIR_TRAIN_IMG = "../../signal-object-detection/train"
DIR_TEST_IMG = "../../signal-object-detection/test"
CSV_TRAIN = "../../signal-object-detection/train.csv"
CSV_TEST = "../../signal-object-detection/sample_submission.csv"
FISIER_IESIRE = "submisie_aot_spectronet.csv"

DIM = 128  # ZOOM-ul salvator: de la 128x55 la 128x128
NR_FOLDURI = 5
EPOCI = 80
BATCH = 64
LR = 5e-4
WD = 1e-4
SEED = 197

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Folosim device: {DEVICE}")


# ==========================================
# 2. PREPROCESARE & DATASET
# ==========================================
class TimeShift(nn.Module):
    def __init__(self, max_shift_pct=0.10):
        super().__init__()
        self.max_shift_pct = max_shift_pct

    def forward(self, x):
        if torch.rand(1).item() < 0.5:
            shift = int(torch.rand(1).item() * self.max_shift_pct * x.shape[-1])
            x = torch.roll(x, shifts=shift, dims=-1)
        return x


# Pipeline Antrenament (Cu dilatare la 128x128 și normalizare)
train_transform = T.Compose([
    T.Resize((DIM, DIM), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    TimeShift(max_shift_pct=0.10),
    TA.FrequencyMasking(freq_mask_param=10),
    TA.TimeMasking(time_mask_param=10),
    T.Lambda(lambda x: (x - torch.mean(x)) / (torch.std(x) + 1e-8))
])

# Pipeline Validare/Test
val_transform = T.Compose([
    T.Resize((DIM, DIM), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Lambda(lambda x: (x - torch.mean(x)) / (torch.std(x) + 1e-8))
])


class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_name

        # Etichetele din CSV (1-5) devin (0-4) pentru PyTorch
        label = int(self.df.iloc[idx, 1]) - 1
        return image, label


# ==========================================
# 3. ARHITECTURA AOT-SPECTRONET
# ==========================================
class AddFreqCoords(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, _, h, w = x.size()
        y_coords = torch.linspace(-1, 1, steps=h, device=x.device)
        y_coords = y_coords.view(1, 1, h, 1).expand(b, 1, h, w)
        return torch.cat([x, y_coords], dim=1)


class AOTBlock(nn.Module):
    """Aggregated Contextual Transformations: Extrage multi-scale features simultan."""

    def __init__(self, channels, rates=(1, 2, 4, 8)):
        super().__init__()
        assert channels % len(rates) == 0
        group_dim = channels // len(rates)

        self.convs = nn.ModuleList([
            nn.Sequential(
                # padding = r și dilation = r păstrează dimensiunea spațială exactă
                nn.Conv2d(group_dim, group_dim, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(group_dim),
                nn.SiLU(inplace=True)
            ) for r in rates
        ])

        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # Tăiem tensorul pe canale în 4 bucăți
        splits = torch.chunk(x, len(self.convs), dim=1)
        # Aplicăm dilatările
        out = [conv(split) for split, conv in zip(splits, self.convs)]
        # Reasamblăm
        out = torch.cat(out, dim=1)
        out = self.project(out)
        return F.silu(x + out)


class Downsample(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))


class AOTSpectroNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.coords = AddFreqCoords()

        # Input: 1 canal imagine + 1 canal coordonate = 2
        # Câmp receptiv global direct din start cu stride=2
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        self.layer1 = AOTBlock(32)
        self.down1 = Downsample(32, 64)

        self.layer2 = AOTBlock(64)
        self.down2 = Downsample(64, 128)

        self.layer3 = AOTBlock(128)
        self.down3 = Downsample(128, 256)

        self.layer4 = AOTBlock(256)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.coords(x)
        x = self.stem(x)

        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.down3(x)
        x = self.layer4(x)

        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ==========================================
# 4. LOSS CONFIDENT & EMA
# ==========================================
class ConfidentCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, entropy_weight=0.15):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.entropy_weight = entropy_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        return ce_loss + (self.entropy_weight * entropy.mean())


class ModelEMA:
    def __init__(self, model, decay=0.99):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * model_v)


# ==========================================
# 5. TTA & BUCLE DE ANTRENAMENT
# ==========================================
def evalueaza_tta(model, val_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)

            # Original
            probs_orig = F.softmax(model(inputs), dim=1)
            # TTA Left Shift
            probs_left = F.softmax(model(torch.roll(inputs, shifts=-10, dims=-1)), dim=1)
            # TTA Right Shift
            probs_right = F.softmax(model(torch.roll(inputs, shifts=10, dims=-1)), dim=1)

            avg_probs = (probs_orig + probs_left + probs_right) / 3.0
            preds = torch.argmax(avg_probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    return np.array(all_preds), np.array(all_targets)


def main():
    print("=== START AOT-SPECTRONET PIPELINE ===")

    full_dataset = SpectrogramDataset(CSV_TRAIN, DIR_TRAIN_IMG)
    indices = np.arange(len(full_dataset))
    labels = full_dataset.df['label'].values - 1

    skf = StratifiedKFold(n_splits=NR_FOLDURI, shuffle=True, random_state=SEED)

    class_weights = torch.tensor([0.7, 1.0, 1.0, 1.3, 1.1], dtype=torch.float32).to(DEVICE)
    criterion = ConfidentCrossEntropyLoss(weight=class_weights, entropy_weight=0.15)

    best_fold_models = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n--- FOLD {fold + 1}/{NR_FOLDURI} ---")

        train_sub = torch.utils.data.Subset(SpectrogramDataset(CSV_TRAIN, DIR_TRAIN_IMG, train_transform), train_idx)
        val_sub = torch.utils.data.Subset(SpectrogramDataset(CSV_TRAIN, DIR_TRAIN_IMG, val_transform), val_idx)

        train_loader = DataLoader(train_sub, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

        model = AOTSpectroNet(num_classes=5).to(DEVICE)
        ema = ModelEMA(model, decay=0.99)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCI, eta_min=1e-6)

        best_acc = 0.0
        best_state = None

        for ep in range(EPOCI):
            model.train()
            train_loss = 0

            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema.update(model)

                train_loss += loss.item()

            scheduler.step()

            # Evaluare la fiecare 5 epoci sau in ultimele 15
            if (ep + 1) % 5 == 0 or ep > EPOCI - 15:
                preds, targets = evalueaza_tta(ema.ema, val_loader)
                acc = (preds == targets).mean()
                avg_loss = train_loss / len(train_loader)

                if acc > best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(ema.ema.state_dict())
                    print(f" Epoca {ep + 1:02d} | Loss: {avg_loss:.4f} | Val Acc TTA: {acc:.4f} *NOU RECORD*")
                else:
                    print(f" Epoca {ep + 1:02d} | Loss: {avg_loss:.4f} | Val Acc TTA: {acc:.4f}")

        print(f"--> Fold {fold + 1} incheiat cu Best Acc: {best_acc:.4f}")
        fold_scores.append(best_acc)
        best_fold_models.append(best_state)

        # Afisăm matricea și raportul pe cel mai bun model al foldului
        ema.ema.load_state_dict(best_state)
        preds, targets = evalueaza_tta(ema.ema, val_loader)
        print(
            classification_report(targets, preds, target_names=['Clasa 1', 'Clasa 2', 'Clasa 3', 'Clasa 4', 'Clasa 5']))

    print(f"\nMedia K-Fold AOT-SpectroNet: {np.mean(fold_scores):.4f}")

    # ==========================================
    # 6. PREDICȚIE FINALĂ ENSEMBLE TTA
    # ==========================================
    print("\nGenerare submisie...")
    test_ds = SpectrogramDataset(CSV_TEST, DIR_TEST_IMG, val_transform, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    final_probs = []
    image_names = []

    for x, names in test_loader:
        x = x.to(DEVICE)
        batch_probs = torch.zeros((x.size(0), 5), device=DEVICE)

        for state in best_fold_models:
            m = AOTSpectroNet(num_classes=5).to(DEVICE)
            m.load_state_dict(state)
            m.eval()

            with torch.no_grad():
                # TTA la inferență pe test
                p_orig = F.softmax(m(x), dim=1)
                p_left = F.softmax(m(torch.roll(x, shifts=-10, dims=-1)), dim=1)
                p_right = F.softmax(m(torch.roll(x, shifts=10, dims=-1)), dim=1)

                batch_probs += (p_orig + p_left + p_right) / 3.0

        batch_probs /= NR_FOLDURI
        final_probs.extend(torch.argmax(batch_probs, dim=1).cpu().numpy())
        image_names.extend(names)

    # Readucem la 1-5
    final_preds = np.array(final_probs) + 1
    pd.DataFrame({"id": image_names, "label": final_preds}).to_csv(FISIER_IESIRE, index=False)
    print(f"Salvat în: {FISIER_IESIRE}. Mult succes pe Kaggle!")


if __name__ == "__main__":
    main()