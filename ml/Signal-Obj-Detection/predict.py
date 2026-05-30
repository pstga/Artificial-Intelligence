import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model6 import CustomResNet_Large  # Ensure this is the same class used in training
from data_process import val_transform  # Use the same transforms used during validation


class KaggleTestDataset(Dataset):
    def __init__(self, sample_sub_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.df = pd.read_csv(sample_sub_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, img_name


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = './kaggle_ensemble_models'

    # 1. Load Dataset
    test_ds = KaggleTestDataset(
        sample_sub_path='../../signal-object-detection/sample_submission.csv',
        img_dir='../../signal-object-detection/test/',
        transform=val_transform
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    # 2. Load the 5 models
    models = []
    print("Loading 5-fold ensemble...")
    for i in range(1, 6):
        model = CustomResNet_Large(num_classes=5)
        # Point to your best saved weights
        model.load_state_dict(torch.load(os.path.join(model_dir, f'fold_{i}_best_model.pth')))
        model.to(device)
        model.eval()
        models.append(model)
        print(f"Loaded Fold {i}")

    # 3. Inference
    all_preds = []
    image_names = []

    print("Running ensemble inference...")
    with torch.no_grad():
        for inputs, names in tqdm(test_loader):
            inputs = inputs.to(device)

            # Collect predictions from all 5 models
            fold_outputs = []
            for model in models:
                output = model(inputs)
                # Ensure output is flattened to [batch_size]
                fold_outputs.append(output.view(-1))

            # Stack them to shape [5, batch_size] and take the mean across fold dimension (0)
            ensemble_avg = torch.stack(fold_outputs).mean(dim=0)

            # Round and Clamp to 1-5 range
            final_preds = torch.round(ensemble_avg).clamp(min=1, max=5)

            all_preds.extend(final_preds.cpu().numpy().astype(int))
            image_names.extend(names)

    # 4. Save Submission
    submission = pd.DataFrame({'id': image_names, 'label': all_preds})
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv!")


if __name__ == "__main__":
    run_inference()