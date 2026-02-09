"""
Variational Autoencoder (VAE) for Sepsis Prediction
Generative AI model for analyzing EHR data and predicting sepsis risk

This VAE-based system:
1. Learns latent representations of patient health states
2. Detects anomalous patterns indicating sepsis risk
3. Generates synthetic patients for data augmentation
4. Provides uncertainty quantification for predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SepsisVAE(nn.Module):
    """
    Variational Autoencoder for Sepsis Risk Prediction
    
    Architecture:
    - Encoder: Maps patient features to latent distribution (μ, σ)
    - Latent Space: Low-dimensional representation of health state
    - Decoder: Reconstructs patient features from latent space
    - Classifier: Predicts sepsis risk from latent representation
    
    The VAE learns to:
    1. Compress patient data into meaningful latent features
    2. Identify sepsis-related patterns in latent space
    3. Generate realistic synthetic patients
    4. Provide uncertainty estimates
    """
    
    def __init__(self, 
                 input_dim: int = 17,
                 latent_dim: int = 8,
                 hidden_dims: list = [64, 32, 16],
                 dropout_rate: float = 0.2):
        """
        Initialize Sepsis VAE
        
        Args:
            input_dim: Number of input features (vital signs, labs, demographics)
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions for encoder/decoder
            dropout_rate: Dropout rate for regularization
        """
        super(SepsisVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder: Patient features → Latent distribution
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)  # Mean
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)  # Log variance
        
        # Decoder: Latent space → Reconstructed features
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Sepsis risk classifier from latent space
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Initialized SepsisVAE with {self.count_parameters():,} parameters")
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode patient features to latent distribution
        
        Args:
            x: Patient features [batch_size, input_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
        
        This allows backpropagation through the sampling process
        
        Args:
            mu: Mean [batch_size, latent_dim]
            logvar: Log variance [batch_size, latent_dim]
            
        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed features
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            x_reconstructed: Reconstructed features [batch_size, input_dim]
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE
        
        Args:
            x: Patient features [batch_size, input_dim]
            
        Returns:
            x_reconstructed: Reconstructed features
            mu: Latent mean
            logvar: Latent log variance
            sepsis_risk: Sepsis risk probability [batch_size, 1]
        """
        # Encode to latent distribution
        mu, logvar = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode to reconstruct features
        x_reconstructed = self.decode(z)
        
        # Predict sepsis risk from latent representation
        sepsis_risk = self.classifier(z)
        
        return x_reconstructed, mu, logvar, sepsis_risk
    
    def predict_sepsis(self, x: torch.Tensor, n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Predict sepsis risk with uncertainty quantification
        
        Uses Monte Carlo sampling from latent distribution to estimate uncertainty
        
        Args:
            x: Patient features [batch_size, input_dim]
            n_samples: Number of Monte Carlo samples for uncertainty
            
        Returns:
            Dictionary with:
                - risk_mean: Mean sepsis risk probability
                - risk_std: Standard deviation (uncertainty)
                - risk_lower: Lower confidence bound (2.5th percentile)
                - risk_upper: Upper confidence bound (97.5th percentile)
        """
        self.eval()
        with torch.no_grad():
            # Encode to latent distribution
            mu, logvar = self.encode(x)
            
            # Monte Carlo sampling
            risks = []
            for _ in range(n_samples):
                z = self.reparameterize(mu, logvar)
                risk = self.classifier(z)
                risks.append(risk)
            
            risks = torch.stack(risks, dim=0)  # [n_samples, batch_size, 1]
            
            # Calculate statistics
            risk_mean = risks.mean(dim=0)
            risk_std = risks.std(dim=0)
            risk_lower = torch.quantile(risks, 0.025, dim=0)
            risk_upper = torch.quantile(risks, 0.975, dim=0)
            
            return {
                'risk_mean': risk_mean,
                'risk_std': risk_std,
                'risk_lower': risk_lower,
                'risk_upper': risk_upper,
                'uncertainty': risk_std  # Epistemic uncertainty
            }
    
    def generate_synthetic_patients(self, n_patients: int, sepsis_condition: bool = None) -> torch.Tensor:
        """
        Generate synthetic patients using the decoder
        
        Args:
            n_patients: Number of synthetic patients to generate
            sepsis_condition: If True, generate sepsis patients; if False, healthy; if None, random
            
        Returns:
            synthetic_features: Generated patient features [n_patients, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(n_patients, self.latent_dim)
            
            # If conditioning on sepsis status, adjust latent space
            if sepsis_condition is not None:
                # This would be learned during training
                # For now, simple adjustment
                if sepsis_condition:
                    z = z + 1.0  # Shift towards sepsis region
                else:
                    z = z - 1.0  # Shift towards healthy region
            
            # Decode to generate features
            synthetic_features = self.decode(z)
            
            return synthetic_features
    
    def detect_anomalies(self, x: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
        """
        Detect anomalous patients (potential sepsis) using reconstruction error
        
        High reconstruction error indicates the patient is unusual/anomalous,
        which may indicate sepsis or other critical conditions
        
        Args:
            x: Patient features [batch_size, input_dim]
            threshold: Anomaly threshold (in standard deviations)
            
        Returns:
            is_anomalous: Boolean tensor indicating anomalies [batch_size]
        """
        self.eval()
        with torch.no_grad():
            x_reconstructed, _, _, _ = self.forward(x)
            
            # Calculate reconstruction error
            reconstruction_error = F.mse_loss(x_reconstructed, x, reduction='none').mean(dim=1)
            
            # Normalize by mean and std
            error_mean = reconstruction_error.mean()
            error_std = reconstruction_error.std()
            normalized_error = (reconstruction_error - error_mean) / (error_std + 1e-8)
            
            # Detect anomalies
            is_anomalous = normalized_error > threshold
            
            return is_anomalous
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation of patients
        
        Useful for visualization and analysis
        
        Args:
            x: Patient features [batch_size, input_dim]
            
        Returns:
            z: Latent representation [batch_size, latent_dim]
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return z

def vae_loss_function(x_reconstructed: torch.Tensor,
                     x_original: torch.Tensor,
                     mu: torch.Tensor,
                     logvar: torch.Tensor,
                     sepsis_pred: torch.Tensor,
                     sepsis_true: torch.Tensor,
                     beta: float = 1.0,
                     classification_weight: float = 1.0,
                     pos_weight: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """
    VAE loss function with sepsis classification
    
    Total Loss = Reconstruction Loss + β * KL Divergence + Classification Loss
    
    Args:
        x_reconstructed: Reconstructed features
        x_original: Original features
        mu: Latent mean
        logvar: Latent log variance
        sepsis_pred: Predicted sepsis risk
        sepsis_true: True sepsis labels
        beta: Weight for KL divergence (β-VAE)
        classification_weight: Weight for classification loss
        pos_weight: Weight for positive samples to handle imbalance
        
    Returns:
        Dictionary with individual loss components and total loss
    """
    # Reconstruction loss (MSE)
    reconstruction_loss = F.mse_loss(x_reconstructed, x_original, reduction='mean')
    
    # KL divergence loss
    # KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    
    # Classification loss (Binary Cross Entropy)
    if pos_weight is not None:
        classification_loss = F.binary_cross_entropy(
            sepsis_pred, 
            sepsis_true.unsqueeze(1),
            weight=pos_weight[sepsis_true.long()].unsqueeze(1) if pos_weight.dim() == 1 else pos_weight
        )
    else:
        classification_loss = F.binary_cross_entropy(sepsis_pred, sepsis_true.unsqueeze(1))
    
    # Total loss
    total_loss = reconstruction_loss + beta * kl_loss + classification_weight * classification_loss
    
    return {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'kl_loss': kl_loss,
        'classification_loss': classification_loss
    }

class EHRDataset(Dataset):
    """
    Dataset for Electronic Health Records
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize EHR dataset
        
        Args:
            features: Patient features [n_patients, n_features]
            labels: Sepsis labels [n_patients]
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Sepsis VAE - Generative AI Model")
    print("="*60)
    
    # Initialize model
    model = SepsisVAE(
        input_dim=17,  # 17 clinical features
        latent_dim=8,   # 8-dimensional latent space
        hidden_dims=[64, 32, 16]
    )
    
    print(f"\nModel Architecture:")
    print(f"  Input dimension: {model.input_dim}")
    print(f"  Latent dimension: {model.latent_dim}")
    print(f"  Total parameters: {model.count_parameters():,}")
    
    # Example forward pass
    batch_size = 32
    x = torch.randn(batch_size, 17)  # Random patient features
    
    x_recon, mu, logvar, risk = model(x)
    
    print(f"\nForward Pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstructed shape: {x_recon.shape}")
    print(f"  Latent mean shape: {mu.shape}")
    print(f"  Risk prediction shape: {risk.shape}")
    
    # Example prediction with uncertainty
    predictions = model.predict_sepsis(x, n_samples=10)
    print(f"\nPrediction with Uncertainty:")
    print(f"  Mean risk: {predictions['risk_mean'].mean().item():.3f}")
    print(f"  Uncertainty (std): {predictions['risk_std'].mean().item():.3f}")
    
    # Generate synthetic patients
    synthetic = model.generate_synthetic_patients(n_patients=10, sepsis_condition=True)
    print(f"\nSynthetic Patient Generation:")
    print(f"  Generated {synthetic.shape[0]} synthetic sepsis patients")
    
    print("\n✓ Sepsis VAE initialized successfully!")
