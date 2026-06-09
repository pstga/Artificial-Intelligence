import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import *

# face un mixup intre 2 imagini pt antrenare - generalizare
def apply_mixup(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    return x_mix, y, y[perm], lam

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=80, fold=1,
                save_dir='./kaggle_ensemble_models', start_epoch=0, best_val_acc=0.0, ema_state=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ema = ModelEMA(model, decay=0.99)
    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"fold {fold} | epoca {epoch + 1}/{num_epochs}", leave=False):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)

            # mixup ul aplicat pe date
            if np.random.rand() < 0.4:
                inputs_mix, targets_a, targets_b, lam = apply_mixup(inputs, labels, alpha=0.2)
                outputs = model(inputs_mix)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            predicted = torch.argmax(outputs.data, dim=1)
            train_correct += (predicted == labels).sum().item()

            # gradientul sa nu creasca mai mult de 1 pt a nu face un jump prea mare
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # actualizam cu datele noi
            ema.update(model)

            train_loss += loss.item() * len(inputs)
            total_train += len(labels)

        scheduler.step()

        avg_train_loss = train_loss / total_train
        train_accuracy = 100 * train_correct / total_train

        # validare cu test-time-augmentation
        ema.ema.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # tta
                p_orig = F.softmax(ema.ema(inputs), dim=1)
                p_left = F.softmax(ema.ema(torch.roll(inputs, shifts=-10, dims=-1)), dim=1)
                p_right = F.softmax(ema.ema(torch.roll(inputs, shifts=10, dims=-1)), dim=1)
                avg_probs = (p_orig + p_left + p_right) / 3.0

                predicted = torch.argmax(avg_probs, dim=1)
                total_val += len(labels)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / total_val

        print(
            f"epoca [{epoch + 1}/{num_epochs}] train loss: {avg_train_loss:.4f} | train acc: {train_accuracy:.2f}% | val acc (cu tta): {val_accuracy:.2f}%")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            save_path = os.path.join(save_dir, f'fold_{fold}_best_model.pth')
            torch.save(ema.ema.state_dict(), save_path)
            print(f"NEW BEST !!!")
    return ema.ema