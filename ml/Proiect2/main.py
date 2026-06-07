import os
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from data_process import CustomImageDataset, train_transform, val_transform, evaluate_tta_classes
from model3 import *
from train import train_model


class FocalLoss(nn.Module):
    """
    Focal Loss - Taie atenția de pe exemplele ușoare și forțează
    rețeaua să învețe diferențele subtile dintre clasele grele.
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ─────────────────────────────────────────────
#  RESUME SETTINGS — edit these two lines only
RESUME_FOLD = 1  # fold to resume from (1-5); folds before this are skipped
RESUME_EPOCH = 140  # the epoch number that was saved (training will start at +1)


# Set RESUME_FOLD = 1 and RESUME_EPOCH = 0 to start fresh
# ─────────────────────────────────────────────


def make_optimizer_and_scheduler(model, num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=75, T_mult=2, eta_min=1e-6
    )
    return optimizer, scheduler


def load_checkpoint(path, model, optimizer, scheduler, device):
    """
    Loads a checkpoint. Handles two formats:
      - New format: dict with keys model_state / ema_state / optimizer_state /
                    scheduler_state / epoch / best_val_acc
      - Old format: bare state_dict (EMA weights only) saved by the original code
    Returns (start_epoch, best_val_acc, ema_state_or_None).
    """
    raw = torch.load(path, map_location=device)

    # ── New full checkpoint ──────────────────────────────────────────────────
    if isinstance(raw, dict) and 'optimizer_state' in raw:
        model.load_state_dict(raw['model_state'])
        optimizer.load_state_dict(raw['optimizer_state'])
        scheduler.load_state_dict(raw['scheduler_state'])
        start_epoch = raw['epoch'] + 1  # resume from next epoch
        best_val_acc = raw['best_val_acc']
        ema_state = raw['ema_state']
        print(f"  [checkpoint] full checkpoint — resuming from epoch {start_epoch}, "
              f"best val acc {best_val_acc:.2f}%")
        return start_epoch, best_val_acc, ema_state

    # ── Old format: bare EMA state dict ─────────────────────────────────────
    print(f"  [checkpoint] old-format weights detected — loading EMA state only.")
    print(f"  Optimizer/scheduler will be re-initialized; "
          f"resuming from epoch {RESUME_EPOCH + 1}.")

    # --- FIX 1: Actually load the weights into the model ---
    model.load_state_dict(raw)

    ema_state = raw  # the dict IS the state dict
    best_val_acc = 0.0  # unknown — will save on first improvement
    start_epoch = RESUME_EPOCH + 1

    # Fast-forward the scheduler to match the saved epoch so LR is correct
    for _ in range(start_epoch):
        scheduler.step()

    return start_epoch, best_val_acc, ema_state


def main():
    batch_size = 64
    num_epochs = 250
    num_classes = 5
    k_folds = 5
    save_dir = './kaggle_ensemble_models'

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_train_dataset = CustomImageDataset(
        csv_file='../../signal-object-detection/train.csv',
        img_dir='../../signal-object-detection/train/',
        transform=train_transform
    )
    full_val_dataset = CustomImageDataset(
        csv_file='../../signal-object-detection/train.csv',
        img_dir='../../signal-object-detection/train/',
        transform=val_transform
    )

    labels = [int(label) for label in full_train_dataset.labels]
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=197)
    dataset_indices = np.arange(len(full_train_dataset))

    for fold, (train_ids, val_ids) in enumerate(skf.split(dataset_indices, labels)):
        current_fold = fold + 1

        # ── Skip already-completed folds ────────────────────────────────────
        if current_fold < RESUME_FOLD:
            print(f"\n--- Skipping fold {current_fold}/{k_folds} (already complete) ---")
            continue

        print(f"\n--- INITIATING FOLD {current_fold}/{k_folds} ---")

        train_sub = Subset(full_train_dataset, train_ids)
        val_sub = Subset(full_val_dataset, val_ids)

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                                  num_workers=2, persistent_workers=True, pin_memory=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False,
                                num_workers=2, persistent_workers=True, pin_memory=True)

        model = SpectroNeXt(num_classes=num_classes)
        optimizer, scheduler = make_optimizer_and_scheduler(model, num_epochs)
        class_weights = torch.tensor([0.8, 1.0, 1.0, 1.1, 1.2],
                                     dtype=torch.float32).to(device)
        criterion = FocalLoss(weight=class_weights, gamma=2.0,
                              label_smoothing=0.04)

        # ── Resume if checkpoint exists for this fold ────────────────────────
        start_epoch = 0
        best_val_acc = 0.0
        ema_state = None

        checkpoint_path = os.path.join(save_dir, f'fold_{current_fold}_best_model.pth')
        if current_fold == RESUME_FOLD and RESUME_EPOCH > 0 and os.path.exists(checkpoint_path):
            print(f"  Loading checkpoint: {checkpoint_path}")
            start_epoch, best_val_acc, ema_state = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
        else:
            print(f"  Starting fold {current_fold} from scratch.")

        # ── Train ────────────────────────────────────────────────────────────
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            fold=current_fold,
            save_dir=save_dir,

            # --- FIX 2: Pass the resumed variables so train.py knows where to start ---
            start_epoch=start_epoch,
            best_val_acc=best_val_acc,
            ema_state=ema_state
        )

        print(f"\n[ Evaluating Best EMA Weights for Fold {current_fold} ]")
        evaluate_tta_classes(model, val_loader, weights_path=checkpoint_path)


if __name__ == "__main__":
    main()