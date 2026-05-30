import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

# Assuming these are properly defined in your other files
from data_process import CustomImageDataset, train_transform, val_transform, evaluate_classes
from model7 import CustomResNet_Large
from train import train_model


def main():
    batch_size = 256
    num_epochs = 75
    num_classes = 5
    learning_rate = 0.001
    k_folds = 1
    save_dir = './kaggle_ensemble_models'

    os.makedirs(save_dir, exist_ok=True)

    print("Loading Base Datasets...")

    # 1. Instantiate the dataset TWICE, using the two different transforms
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

    # 2. Setup K-Fold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=197)
    dataset_indices = np.arange(len(full_train_dataset))

    print(f"\n{'=' * 50}")
    print(f"  STARTING {k_folds}-FOLD CROSS VALIDATION")
    print(f"{'=' * 50}")

    # 3. Iterate through each fold
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_indices)):
        current_fold = fold + 1
        print(f"\n\n--- INITIATING FOLD {current_fold}/{k_folds} ---")

        # Subsets pointing to the CORRECT base dataset transforms
        train_sub = Subset(full_train_dataset, train_ids)
        val_sub = Subset(full_val_dataset, val_ids)

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        # MUST re-initialize the model, criterion, optimizer, and scheduler for every fold!
        model = CustomResNet_Large(num_classes=num_classes)
        criterion = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        # Train the model for this fold
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            mixup=True,
            fold=current_fold,
            save_dir=save_dir
        )

        print(f"✅ Fold {current_fold} Training Complete.")

        # Optionally evaluate the absolute best weights for this fold
        print(f"\n[ Evaluating Best Weights for Fold {current_fold} ]")
        best_weights_path = os.path.join(save_dir, f'fold_{current_fold}_best_model.pth')
        evaluate_classes(model, val_loader, weights_path=best_weights_path)

    print(f"\n{'=' * 50}")
    print(f"🎉 ALL {k_folds} FOLDS COMPLETE! Models saved in: {save_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()