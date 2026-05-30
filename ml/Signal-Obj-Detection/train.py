import os
import torch
from tqdm import tqdm
from data_process import mixup_data

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=75, mixup=True, fold=1, save_dir='./kaggle_ensemble_models'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ==========================
        #      TRAINING PHASE
        # ==========================
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Fold {fold} | Epoch [{epoch + 1}/{num_epochs}] Train", leave=False):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)

            if not mixup:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                current_labels = labels
            else:
                mixed_inputs, mixed_labels = mixup_data(inputs, labels)
                outputs = model(mixed_inputs)
                loss = criterion(outputs, mixed_labels)
                current_labels = mixed_labels

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            # REGRESSION ACCURACY (TRAINING)
            predicted_counts = torch.round(outputs.data).clamp(min=1, max=5)
            true_counts = torch.round(current_labels.data).clamp(min=1, max=5)

            total_train += labels.size(0)
            train_correct += (predicted_counts == torch.round(true_counts)).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / total_train

        scheduler.step()

        # ==========================
        #     VALIDATION PHASE
        # ==========================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                # REGRESSION ACCURACY (VALIDATION)
                predicted_counts = torch.round(outputs.data).clamp(min=1, max=5)
                true_counts = torch.round(labels.data).clamp(min=1, max=5)

                total_val += labels.size(0)
                val_correct += (predicted_counts == true_counts).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * val_correct / total_val

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.2f}%")
        print("-" * 40)

        # Save the best epoch for this specific fold
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            save_path = os.path.join(save_dir, f'fold_{fold}_best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  ---> [Saved new best model for Fold {fold}!]")
        print("-" * 40)

    return model