import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from model3 import SpectroModel
from data_process import val_transform


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

    # incarc setul de date
    test_ds = KaggleTestDataset(
        sample_sub_path='../../signal-object-detection/sample_submission.csv',
        img_dir='../../signal-object-detection/test/',
        transform=val_transform
    )

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # incarcam modelele pt 5-fold
    models = []
    for i in range(1, 6):
        model = SpectroModel(num_classes=5)
        weight_path = os.path.join(model_dir, f'fold_{i}_best_model.pth')

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"file not found la {weight_path}")

        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    all_preds = []
    image_names = []

    with torch.no_grad():
        for inputs, names in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)

            # soft-voting(cumulare de probabilitati)
            batch_probs = torch.zeros((len(inputs), 5), device=device)

            for model in models:
                # original
                p_orig = F.softmax(model(inputs), dim=1)

                # -10px
                inputs_left = torch.roll(inputs, shifts=-10, dims=-1)
                p_left = F.softmax(model(inputs_left), dim=1)

                # +10px
                inputs_right = torch.roll(inputs, shifts=10, dims=-1)
                p_right = F.softmax(model(inputs_right), dim=1)

                # media
                model_probs = (p_orig + p_left + p_right) / 3.0

                # adaugam peste tot
                batch_probs += model_probs

            ensemble_avg = batch_probs / len(models)

            final_preds = torch.argmax(ensemble_avg, dim=1) + 1
            all_preds.extend(final_preds.cpu().numpy().astype(int))
            image_names.extend(names)

    submission = pd.DataFrame({'id': image_names, 'label': all_preds})
    submission.to_csv('submission_spectronext.csv', index=False)

if __name__ == "__main__":
    run_inference()