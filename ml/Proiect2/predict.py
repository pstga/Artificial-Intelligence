import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from model3 import SpectroNeXt
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

        # Citim imaginea grayscale
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, img_name


def run_inference():
    print("=== START INFERENCE PIPELINE ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = './kaggle_ensemble_models'

    # 1. Încărcarea setului de test
    test_ds = KaggleTestDataset(
        sample_sub_path='../../signal-object-detection/sample_submission.csv',
        img_dir='../../signal-object-detection/test/',
        transform=val_transform
    )

    # Folosim num_workers=0 pentru a evita orice eroare de Windows la inferență
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # 2. Încărcarea celor 5 modele din K-Fold
    models = []
    print("Loading 5-fold SpectroNeXt ensemble (EMA weights)...")
    for i in range(1, 6):
        model = SpectroNeXt(num_classes=5)
        weight_path = os.path.join(model_dir, f'fold_{i}_best_model.pth')

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Nu găsesc greutățile pentru Fold {i} la: {weight_path}")

        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
        print(f" -> Loaded Fold {i}")

    all_preds = []
    image_names = []

    print("\nRunning ensemble inference with TTA (Test-Time Augmentation)...")
    with torch.no_grad():
        for inputs, names in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)

            # Matrice pentru a cumula probabilitățile (Soft Voting)
            batch_probs = torch.zeros((inputs.size(0), 5), device=device)

            for model in models:
                # TTA 1: Predicție pe imaginea originală
                p_orig = F.softmax(model(inputs), dim=1)

                # TTA 2: Time Shift Stânga (-10 pixeli)
                inputs_left = torch.roll(inputs, shifts=-10, dims=-1)
                p_left = F.softmax(model(inputs_left), dim=1)

                # TTA 3: Time Shift Dreapta (+10 pixeli)
                inputs_right = torch.roll(inputs, shifts=10, dims=-1)
                p_right = F.softmax(model(inputs_right), dim=1)

                # Mediem probabilitățile pentru acest model specific
                model_probs = (p_orig + p_left + p_right) / 3.0

                # Adăugăm la totalul ansamblului
                batch_probs += model_probs

            # Împărțim la numărul de modele (5) pentru a obține media finală absolută
            ensemble_avg = batch_probs / len(models)

            # Clasificare corectă: Argmax (0 la 4) + 1 pentru a readuce la indexul Kaggle (1 la 5)
            final_preds = torch.argmax(ensemble_avg, dim=1) + 1

            all_preds.extend(final_preds.cpu().numpy().astype(int))
            image_names.extend(names)

    # 3. Salvarea fișierului de submisie
    submission = pd.DataFrame({'id': image_names, 'label': all_preds})
    submission.to_csv('submission_spectronext.csv', index=False)
    print("\nInference complete! Submission saved to 'submission_spectronext.csv'!")


if __name__ == "__main__":
    run_inference()