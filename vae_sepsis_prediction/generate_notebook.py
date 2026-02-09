
import nbformat as nbf
from pathlib import Path

# Create a new notebook
nb = nbf.v4.new_notebook()

# --- Section 1: Introduction ---
nb.cells.append(nbf.v4.new_markdown_cell("""
# 🏥 VAE-Based Sepsis Prediction System
## Generative AI for Early Sepsis Detection with Ethics & Responsibility

This notebook implements a complete **Variational Autoencoder (VAE)** based system for predicting sepsis risk from Electronic Health Records (EHR). 

### Key Features:
1. **Generative AI (VAE)**: Learns latent representations of health states and detects anomalies.
2. **Uncertainty Quantification**: Uses Monte Carlo sampling to provide confidence intervals.
3. **Synthetic Patient Generation**: Generates realistic synthetic patients for data augmentation.
4. **AI Ethics Framework**: Analyzes fairness and bias across demographics (Age, Gender).
"""))

# --- Section 2: Imports ---
nb.cells.append(nbf.v4.new_code_cell("""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, List, Optional
import time

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
"""))

# --- Section 3: Model Architecture ---
nb.cells.append(nbf.v4.new_markdown_cell("## 1. VAE Model Architecture"))
nb.cells.append(nbf.v4.new_code_cell("""
class SepsisVAE(nn.Module):
    \"\"\"
    Variational Autoencoder for Sepsis Risk Prediction
    \"\"\"
    def __init__(self, input_dim: int = 17, latent_dim: int = 8, hidden_dims: list = [64, 32, 16], dropout_rate: float = 0.2):
        super(SepsisVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent distribution
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Sepsis Classifier (from latent space)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        risk = self.classifier(z)
        return x_recon, mu, logvar, risk

    def predict_sepsis(self, x, n_samples=10):
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            risks = []
            for _ in range(n_samples):
                z = self.reparameterize(mu, logvar)
                risks.append(self.classifier(z))
            risks = torch.stack(risks)
            return {
                'risk_mean': risks.mean(0),
                'risk_std': risks.std(0),
                'risk_lower': torch.quantile(risks, 0.025, 0),
                'risk_upper': torch.quantile(risks, 0.975, 0)
            }

    def generate_synthetic_patients(self, n, sepsis=None):
        self.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim).to(device)
            if sepsis is not None:
                z = z + (1.0 if sepsis else -1.0)
            return self.decode(z)

def vae_loss_function(x_recon, x, mu, logvar, pred, true, beta=1.0, class_weight=2.0):
    recon_loss = F.mse_loss(x_recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    class_loss = F.binary_cross_entropy(pred, true.unsqueeze(1))
    return recon_loss + beta * kl_loss + class_weight * class_loss, recon_loss, kl_loss, class_loss
"""))

# --- Section 4: Data Preparation ---
nb.cells.append(nbf.v4.new_markdown_cell("## 2. Data Preparation"))
nb.cells.append(nbf.v4.new_code_cell("""
class EHRDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def get_simulated_data(n=5000):
    np.random.seed(42)
    # Vital signs and labs
    data = pd.DataFrame({
        'age': np.random.normal(65, 15, n).clip(18, 100),
        'gender': np.random.choice([0, 1], n),
        'hr': np.random.normal(85, 15, n).clip(40, 180),
        'rr': np.random.normal(16, 4, n).clip(8, 40),
        'temp': np.random.normal(37.0, 0.8, n).clip(35, 41),
        'sbp': np.random.normal(120, 20, n).clip(70, 200),
        'dbp': np.random.normal(75, 15, n).clip(40, 130),
        'spo2': np.random.normal(96, 3, n).clip(70, 100),
        'wbc': np.random.normal(9, 3, n).clip(2, 30),
        'lac': np.random.normal(1.5, 1.0, n).clip(0.5, 10),
        'crea': np.random.normal(1.0, 0.5, n).clip(0.5, 5),
        'plt': np.random.normal(250, 80, n).clip(50, 500),
        'bili': np.random.normal(0.8, 0.5, n).clip(0.2, 5)
    })
    # Derived features
    data['map'] = (data['sbp'] + 2 * data['dbp']) / 3
    data['pp'] = data['sbp'] - data['dbp']
    data['si'] = data['hr'] / data['sbp']
    data['age_norm'] = data['age'] / 100
    
    # sepsis label
    score = (data['hr']>100)*0.2 + (data['lac']>2)*0.3 + (data['sbp']<100)*0.2 + np.random.normal(0, 0.1, n)
    data['sepsis'] = (score > np.percentile(score, 85)).astype(int)
    
    return data

df = get_simulated_data()
X = df.drop('sepsis', axis=1).values
y = df['sepsis'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(X, y, df, test_size=0.2, random_state=42)

train_loader = DataLoader(EHRDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(EHRDataset(X_test, y_test), batch_size=64, shuffle=False)

print(f"Data ready. Train size: {len(X_train)}, Test size: {len(X_test)}")
"""))

# --- Section 5: Training ---
nb.cells.append(nbf.v4.new_markdown_cell("## 3. Model Training"))
nb.cells.append(nbf.v4.new_code_cell("""
model = SepsisVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50
history = {'loss': [], 'recon': [], 'kl': [], 'class': []}

print("Starting training...")
for epoch in range(epochs):
    model.train()
    epoch_losses = [0]*4
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar, risk = model(x)
        loss, recon, kl, cl = vae_loss_function(x_recon, x, mu, logvar, risk, y)
        loss.backward()
        optimizer.step()
        epoch_losses[0] += loss.item()
        epoch_losses[1] += recon.item()
        epoch_losses[2] += kl.item()
        epoch_losses[3] += cl.item()
        
    for i, k in enumerate(history.keys()):
        history[k].append(epoch_losses[i] / len(train_loader))
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {history['loss'][-1]:.4f}")

# Plot History
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Total Loss')
plt.title('Training Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['class'], label='Classification Loss')
plt.title('Clinical Prediction Loss')
plt.legend()
plt.show()
"""))

# --- Section 4: Ethics Monitor ---
nb.cells.append(nbf.v4.new_markdown_cell("## 4. AI Ethics and Responsibility Monitor"))
nb.cells.append(nbf.v4.new_code_cell("""
def analyze_fairness(model, X_test, y_test, df_test):
    model.eval()
    with torch.no_grad():
        results = model.predict_sepsis(torch.FloatTensor(X_test).to(device))
        preds = (results['risk_mean'].cpu().numpy() > 0.5).astype(int)
        
    df_results = df_test.copy()
    df_results['pred'] = preds
    df_results['label'] = y_test
    
    print("--- Fairness Analysis ---")
    # Gender Fairness
    for g in [0, 1]:
        mask = df_results['gender'] == g
        acc = (df_results[mask]['pred'] == df_results[mask]['label']).mean()
        gender_name = "Male" if g == 1 else "Female"
        print(f"Accuracy for {gender_name}: {acc:.2%}")
    
    # Age Fairness
    df_results['age_group'] = pd.cut(df_results['age'], bins=[0, 40, 65, 100], labels=['Young', 'Middle', 'Senior'])
    for group in df_results['age_group'].unique():
        mask = df_results['age_group'] == group
        acc = (df_results[mask]['pred'] == df_results[mask]['label']).mean()
        print(f"Accuracy for {group}: {acc:.2%}")

analyze_fairness(model, X_test, y_test, df_test)
"""))

# --- Section 5: Generative Capability ---
nb.cells.append(nbf.v4.new_markdown_cell("## 5. Generative AI Capabilities"))
nb.cells.append(nbf.v4.new_code_cell("""
# Generate 5 synthetic "High Risk" patient clinical profiles
synthetic_patients = model.generate_synthetic_patients(5, sepsis=True).cpu().numpy()
synthetic_df = pd.DataFrame(scaler.inverse_transform(synthetic_patients), columns=df.columns[:-1])

print("Generated Synthetic Sepsis Patient Profiles:")
display(synthetic_df.head())
"""))

# --- Section 6: Uncertainty Quantification ---
nb.cells.append(nbf.v4.new_markdown_cell("## 6. Uncertainty Quantification"))
nb.cells.append(nbf.v4.new_code_cell("""
# Select a random patient and show prediction with confidence interval
idx = np.random.randint(len(X_test))
patient = torch.FloatTensor(X_test[idx:idx+1]).to(device)
pred = model.predict_sepsis(patient, n_samples=50)

print(f"Patient ID: {idx}")
print(f"Predicted Sepsis Risk: {pred['risk_mean'].item():.1%}")
print(f"Confidence Interval: [{pred['risk_lower'].item():.1%}, {pred['risk_upper'].item():.1%}]")
print(f"Model Uncertainty (Std): {pred['risk_std'].item():.4f}")
"""))

# Save the notebook
with open('sepsis_vae_prediction_system.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook created: sepsis_vae_prediction_system.ipynb")
