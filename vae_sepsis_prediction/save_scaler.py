"""
Quick script to generate and save the feature scaler for deployment
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from training.train_vae import prepare_mimic_data
import joblib

# Prepare data to get the scaler
train_loader, val_loader, test_loader, scaler = prepare_mimic_data()

# Save scaler
output_path = Path("vae_output/deployment/feature_scaler.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, output_path)

print(f"✓ Scaler saved to: {output_path}")
