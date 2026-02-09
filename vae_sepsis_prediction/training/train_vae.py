"""
VAE Training Pipeline for Sepsis Prediction
Complete training system with real MIMIC-III data integration
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.sepsis_vae import SepsisVAE, vae_loss_function, EHRDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VAETrainer:
    """
    Complete training pipeline for Sepsis VAE
    """
    
    def __init__(self,
                 model: SepsisVAE,
                 learning_rate: float = 1e-3,
                 beta: float = 1.0,
                 classification_weight: float = 2.0,
                 pos_weight: float = None,
                 decision_threshold: float = 0.5,
                 device: str = None):
        """
        Initialize VAE trainer
        
        Args:
            model: SepsisVAE model
            learning_rate: Learning rate for optimizer
            beta: Weight for KL divergence (β-VAE)
            classification_weight: Weight for classification loss
            pos_weight: Weight for sepsis class (e.g. 5.0)
            decision_threshold: Threshold for binary classification
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.beta = beta
        self.classification_weight = classification_weight
        self.decision_threshold = decision_threshold
        
        # Setup pos_weight tensor
        if pos_weight is not None:
            self.pos_weight = torch.tensor([1.0, pos_weight])
        else:
            self.pos_weight = None
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'train_kl_loss': [],
            'val_kl_loss': [],
            'train_class_loss': [],
            'val_class_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        logger.info(f"VAE Trainer initialized on device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with average losses
        """
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_class_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            x_recon, mu, logvar, sepsis_pred = self.model(features)
            
            # Calculate loss
            losses = vae_loss_function(
                x_recon, features, mu, logvar,
                sepsis_pred, labels,
                beta=self.beta,
                classification_weight=self.classification_weight,
                pos_weight=self.pos_weight.to(self.device) if self.pos_weight is not None else None
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += losses['total_loss'].item()
            total_recon_loss += losses['reconstruction_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            total_class_loss += losses['classification_loss'].item()
            
            # Calculate accuracy (using configured decision threshold)
            predicted = (sepsis_pred > self.decision_threshold).float()
            correct += (predicted.squeeze() == labels).sum().item()
            total += labels.size(0)
        
        n_batches = len(train_loader)
        
        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'kl_loss': total_kl_loss / n_batches,
            'class_loss': total_class_loss / n_batches,
            'accuracy': correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with average losses
        """
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_class_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                x_recon, mu, logvar, sepsis_pred = self.model(features)
                
                # Calculate loss
                losses = vae_loss_function(
                    x_recon, features, mu, logvar,
                    sepsis_pred, labels,
                    beta=self.beta,
                    classification_weight=self.classification_weight,
                    pos_weight=self.pos_weight.to(self.device) if self.pos_weight is not None else None
                )
                
                # Accumulate metrics
                total_loss += losses['total_loss'].item()
                total_recon_loss += losses['reconstruction_loss'].item()
                total_kl_loss += losses['kl_loss'].item()
                total_class_loss += losses['classification_loss'].item()
                
                # Calculate accuracy
                predicted = (sepsis_pred > self.decision_threshold).float()
                correct += (predicted.squeeze() == labels).sum().item()
                total += labels.size(0)
        
        n_batches = len(val_loader)
        
        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'kl_loss': total_kl_loss / n_batches,
            'class_loss': total_class_loss / n_batches,
            'accuracy': correct / total
        }
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epochs: int = 100,
              early_stopping_patience: int = 15,
              save_dir: str = './checkpoints') -> Dict:
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("="*60)
        logger.info("Starting VAE Training")
        logger.info("="*60)
        logger.info(f"Epochs: {n_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Beta (KL weight): {self.beta}")
        logger.info(f"Classification weight: {self.classification_weight}")
        logger.info(f"Pos weight (Sepsis): {self.pos_weight[1].item() if self.pos_weight is not None else 1.0}")
        logger.info(f"Decision threshold: {self.decision_threshold}")
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['val_recon_loss'].append(val_metrics['recon_loss'])
            self.history['train_kl_loss'].append(train_metrics['kl_loss'])
            self.history['val_kl_loss'].append(val_metrics['kl_loss'])
            self.history['train_class_loss'].append(train_metrics['class_loss'])
            self.history['val_class_loss'].append(val_metrics['class_loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Log progress
            if (epoch + 1) % 5 == 0:
                logger.info(f"\nEpoch [{epoch+1}/{n_epochs}]")
                logger.info(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
                logger.info(f"  Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
                logger.info(f"  Recon: {val_metrics['recon_loss']:.4f} | KL: {val_metrics['kl_loss']:.4f} | Class: {val_metrics['class_loss']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'history': self.history
                }
                
                torch.save(checkpoint, save_dir / 'best_model.pth')
                logger.info(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        logger.info("\n" + "="*60)
        logger.info("Training Complete!")
        logger.info("="*60)
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Final validation accuracy: {self.history['val_accuracy'][-1]:.4f}")
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, save_dir / 'final_model.pth')
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_accuracy'], label='Train')
        axes[0, 1].plot(self.history['val_accuracy'], label='Validation')
        axes[0, 1].set_title('Classification Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Loss components
        axes[1, 0].plot(self.history['val_recon_loss'], label='Reconstruction')
        axes[1, 0].plot(self.history['val_kl_loss'], label='KL Divergence')
        axes[1, 0].plot(self.history['val_class_loss'], label='Classification')
        axes[1, 0].set_title('Validation Loss Components')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning curve
        axes[1, 1].plot(self.history['train_loss'], label='Train', alpha=0.7)
        axes[1, 1].plot(self.history['val_loss'], label='Validation', alpha=0.7)
        axes[1, 1].fill_between(range(len(self.history['train_loss'])),
                                self.history['train_loss'],
                                self.history['val_loss'],
                                alpha=0.2)
        axes[1, 1].set_title('Learning Curve')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training plot saved to: {save_path}")
        
        plt.close()

def augment_dataset_with_vae(features: np.ndarray, 
                             labels: np.ndarray, 
                             model: SepsisVAE, 
                             target_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment dataset using VAE's generative capabilities
    
    Args:
        features: Original patient features
        labels: Original sepsis labels
        model: Trained SepsisVAE model
        target_ratio: Desired ratio of sepsis patients in the final dataset
        
    Returns:
        augmented_features, augmented_labels
    """
    logger.info(f"Augmenting dataset using VAE (target sepsis ratio: {target_ratio:.1%})")
    
    n_total = len(labels)
    n_sepsis = np.sum(labels)
    n_healthy = n_total - n_sepsis
    
    # Calculate how many synthetic sepsis patients we need
    # n_sepsis_total / (n_healthy + n_sepsis_total) = target_ratio
    # n_sepsis_total = target_ratio * n_healthy / (1 - target_ratio)
    n_sepsis_target = int(target_ratio * n_healthy / (1 - target_ratio))
    n_to_generate = n_sepsis_target - n_sepsis
    
    if n_to_generate <= 0:
        logger.info("No augmentation needed. Dataset is already balanced.")
        return features, labels
    
    logger.info(f"Generating {n_to_generate} synthetic sepsis patients...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Generate in batches to avoid memory issues
    batch_size = 1000
    n_batches = (n_to_generate + batch_size - 1) // batch_size
    
    synthetic_features_list = []
    
    with torch.no_grad():
        for _ in range(n_batches):
            current_batch_size = min(batch_size, n_to_generate - len(synthetic_features_list) * batch_size)
            if current_batch_size <= 0: break
            
            # Generate synthetic sepsis patients
            synthetic_batch = model.generate_synthetic_patients(
                n_patients=current_batch_size, 
                sepsis_condition=True
            )
            synthetic_features_list.append(synthetic_batch.cpu().numpy())
    
    synthetic_features = np.vstack(synthetic_features_list)
    synthetic_labels = np.ones(len(synthetic_features))
    
    # Combine original and synthetic data
    augmented_features = np.vstack([features, synthetic_features])
    augmented_labels = np.concatenate([labels, synthetic_labels])
    
    logger.info(f"Augmentation complete. Final size: {len(augmented_labels)} (Sepsis: {np.sum(augmented_labels)/len(augmented_labels):.1%})")
    
    return augmented_features, augmented_labels

def prepare_mimic_data(data_path: str = None, 
                       augment_model: SepsisVAE = None,
                       target_ratio: float = 0.3) -> Tuple[DataLoader, DataLoader, DataLoader, any]:
    """
    Prepare MIMIC-III data for VAE training
    
    Args:
        data_path: Path to prepared MIMIC-III data
        augment_model: If provided, use this model to augment the training set
        target_ratio: Desired sepsis ratio if augmenting
        
    Returns:
        train_loader, val_loader, test_loader
    """
    logger.info("Preparing MIMIC-III data...")
    
    if data_path and Path(data_path).exists():
        # Load real MIMIC-III data
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data)} patients from MIMIC-III")
    else:
        # Generate realistic synthetic data for demonstration
        logger.info("Generating base EHR data for demonstration...")
        n_patients = 5000
        
        # Generate realistic patient features
        np.random.seed(42)
        
        data = pd.DataFrame({
            'age': np.random.normal(65, 15, n_patients).clip(18, 100),
            'gender': np.random.choice([0, 1], n_patients),
            'heart_rate': np.random.normal(85, 15, n_patients).clip(40, 180),
            'respiratory_rate': np.random.normal(16, 4, n_patients).clip(8, 40),
            'temperature': np.random.normal(37.0, 0.8, n_patients).clip(35, 41),
            'systolic_bp': np.random.normal(120, 20, n_patients).clip(70, 200),
            'diastolic_bp': np.random.normal(75, 15, n_patients).clip(40, 130),
            'oxygen_saturation': np.random.normal(96, 3, n_patients).clip(70, 100),
            'wbc_count': np.random.normal(9, 3, n_patients).clip(2, 30),
            'lactate': np.random.normal(1.5, 1.0, n_patients).clip(0.5, 10),
            'creatinine': np.random.normal(1.0, 0.5, n_patients).clip(0.5, 5),
            'platelet_count': np.random.normal(250, 80, n_patients).clip(50, 500),
            'bilirubin': np.random.normal(0.8, 0.5, n_patients).clip(0.2, 5),
        })
        
        # Derived features
        data['mean_arterial_pressure'] = (data['systolic_bp'] + 2 * data['diastolic_bp']) / 3
        data['pulse_pressure'] = data['systolic_bp'] - data['diastolic_bp']
        data['shock_index'] = data['heart_rate'] / data['systolic_bp']
        data['age_normalized'] = data['age'] / 100
        
        # Sepsis labels (based on clinical criteria)
        sepsis_score = (
            (data['age'] > 70) * 0.15 +
            (data['heart_rate'] > 100) * 0.2 +
            (data['respiratory_rate'] > 22) * 0.15 +
            (data['temperature'] > 38.3) * 0.15 +
            (data['temperature'] < 36) * 0.15 +
            (data['systolic_bp'] < 100) * 0.2 +
            (data['wbc_count'] > 12) * 0.15 +
            (data['wbc_count'] < 4) * 0.15 +
            (data['lactate'] > 2) * 0.25 +
            (data['creatinine'] > 1.5) * 0.1 +
            (data['platelet_count'] < 100) * 0.15 +
            np.random.normal(0, 0.1, n_patients)
        )
        
        # Original imbalanced threshold (15% sepsis)
        threshold = np.percentile(sepsis_score, 85)
        data['sepsis'] = (sepsis_score > threshold).astype(int)
    
    # Prepare features and labels
    feature_cols = [col for col in data.columns if col != 'sepsis']
    X = data[feature_cols].values
    y = data['sepsis'].values
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Optional Generative Augmentation
    if augment_model is not None:
        X_train, y_train = augment_dataset_with_vae(X_train, y_train, augment_model, target_ratio)
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Sepsis prevalence - Train: {y_train.mean():.1%}, Val: {y_val.mean():.1%}, Test: {y_test.mean():.1%}")
    
    # Create datasets
    train_dataset = EHRDataset(X_train, y_train)
    val_dataset = EHRDataset(X_val, y_val)
    test_dataset = EHRDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler

if __name__ == "__main__":
    # Prepare data
    train_loader, val_loader, test_loader = prepare_mimic_data()
    
    # Initialize model
    model = SepsisVAE(
        input_dim=17,
        latent_dim=8,
        hidden_dims=[64, 32, 16],
        dropout_rate=0.2
    )
    
    # Initialize trainer
    trainer = VAETrainer(
        model=model,
        learning_rate=1e-3,
        beta=1.0,
        classification_weight=2.0
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        early_stopping_patience=15,
        save_dir='./checkpoints'
    )
    
    # Plot training history
    trainer.plot_training_history(save_path='./checkpoints/training_history.png')
    
    print("\n✓ VAE training complete!")
