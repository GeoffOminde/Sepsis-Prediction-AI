
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from models.sepsis_vae import SepsisVAE
from training.train_vae import prepare_mimic_data

print("Starting test...")
model = SepsisVAE(input_dim=17)
train, val, test = prepare_mimic_data()
print("Data prepared.")

features, labels = next(iter(test))
print("Got sample.")
model.eval()
with torch.no_grad():
    results = model.predict_sepsis(features, n_samples=10)
print("Prediction successful.")
print(results.keys())
