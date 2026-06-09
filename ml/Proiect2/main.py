import os
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from data_process import CustomImageDataset, train_transform, val_transform
from model3 import *
from train import train_model

# calculeaza cross entropy loss pt a forta invatarea pe clasele pe care le greseste mai des
# parte din mecanismul de atentie
class FocalLoss(nn.Module):
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

# folosite pentru a relua antrenarea cand modificam anumite valori real time sa n-o ia de la capat
RESUME_FOLD = 1
RESUME_EPOCH = 0

# init optimizer si scheduler
def make_optimizer_and_scheduler(model, num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=75, T_mult=2, eta_min=1e-6
    )
    return optimizer, scheduler

# logica de revenire la punctul in care am ramas la resume
def load_checkpoint(path, model, optimizer, scheduler, device):
    raw = torch.load(path, map_location=device)

    if isinstance(raw, dict) and 'optimizer_state' in raw:
        model.load_state_dict(raw['model_state'])
        optimizer.load_state_dict(raw['optimizer_state'])
        scheduler.load_state_dict(raw['scheduler_state'])
        start_epoch = raw['epoch'] + 1  # resume
        best_val_acc = raw['best_val_acc']
        ema_state = raw['ema_state']
        return start_epoch, best_val_acc, ema_state

    model.load_state_dict(raw)
    ema_state = raw
    best_val_acc = 0.0  # ca sa salvam la primul improvement
    start_epoch = RESUME_EPOCH + 1

    for _ in range(start_epoch):
        scheduler.step()

    return start_epoch, best_val_acc, ema_state

def main():
    batch_size = 64
    num_epochs = 250
    num_classes = 5
    k_folds = 5
    save_dir = './kaggle_ensemble_models'

    # initializam si rulam pe gpu in loc de cpu !
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

    labels = [int(label.item()) for label in full_train_dataset.labels]
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=197)
    dataset_indices = np.arange(len(full_train_dataset))

    for fold, (train_ids, val_ids) in enumerate(skf.split(dataset_indices, labels)):
        current_fold = fold + 1

        # skip peste in cazul in care dam resume
        if current_fold < RESUME_FOLD:
            continue

        print(f"\n!!!!acum incep fold {current_fold}/{k_folds}")

        train_sub = Subset(full_train_dataset, train_ids)
        val_sub = Subset(full_val_dataset, val_ids)

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                                  num_workers=2, persistent_workers=True, pin_memory=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False,
                                num_workers=2, persistent_workers=True, pin_memory=True)

        model = SpectroModel(num_classes=num_classes)
        optimizer, scheduler = make_optimizer_and_scheduler(model, num_epochs)
        class_weights = torch.tensor([0.8, 1.0, 1.0, 1.1, 1.2],
                                     dtype=torch.float32).to(device)
        criterion = FocalLoss(weight=class_weights, gamma=2.0,
                              label_smoothing=0.04)

        # tot pt resume
        start_epoch = 0
        best_val_acc = 0.0
        ema_state = None

        checkpoint_path = os.path.join(save_dir, f'fold_{current_fold}_best_model.pth')
        if current_fold == RESUME_FOLD and RESUME_EPOCH > 0 and os.path.exists(checkpoint_path):
            start_epoch, best_val_acc, ema_state = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )

        # train
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
            start_epoch=start_epoch,
            best_val_acc=best_val_acc,
            ema_state=ema_state
        )

        print(f"\n am terminat fold {current_fold}")


if __name__ == "__main__":
    main()