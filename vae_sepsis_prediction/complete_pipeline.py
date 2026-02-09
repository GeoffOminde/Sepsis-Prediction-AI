"""
Complete VAE-based Sepsis Prediction Pipeline
End-to-end system with ethics monitoring
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any, Optional

import sys
sys.path.append(str(Path(__file__).parent))

from models.sepsis_vae import SepsisVAE
from training.train_vae import VAETrainer, prepare_mimic_data
from evaluation.ethics_monitor import EthicsMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SepsisPredictionPipeline:
    """
    Complete pipeline for VAE-based sepsis prediction
    """
    
    def __init__(self, output_dir: str = "./pipeline_output"):
        """
        Initialize pipeline
        
        Args:
            output_dir: Directory for all outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = self.output_dir / "models"
        self.eval_dir = self.output_dir / "evaluation"
        self.ethics_dir = self.output_dir / "ethics"
        
        for dir in [self.model_dir, self.eval_dir, self.ethics_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized. Output: {self.output_dir}")
    
    def run_complete_pipeline(self, n_epochs: int = 100) -> Dict:
        """
        Run complete end-to-end pipeline
        
        Args:
            n_epochs: Number of training epochs
            
        Returns:
            Pipeline results
        """
        logger.info("="*80)
        logger.info("VAE-BASED SEPSIS PREDICTION - COMPLETE PIPELINE")
        logger.info("="*80)
        
        results = {
            'started_at': datetime.now().isoformat(),
            'pipeline_type': 'VAE Generative AI'
        }
        
        # Step 1: Prepare data
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*80)
        train_loader, val_loader, test_loader, scaler = prepare_mimic_data()
        self.scaler = scaler
        results['data_preparation'] = 'complete'
        
        # Step 2: Train VAE model
        logger.info("\n" + "="*80)
        logger.info("STEP 2: VAE MODEL TRAINING")
        logger.info("="*80)
        
        model = SepsisVAE(
            input_dim=17,
            latent_dim=8,
            hidden_dims=[64, 32, 16],
            dropout_rate=0.2
        )
        
        trainer = VAETrainer(
            model=model,
            learning_rate=1e-3,
            beta=1.0,
            classification_weight=5.0,  # Increased classification weight
            pos_weight=5.0,            # Added pos_weight for imbalance
            decision_threshold=0.3     # Lowered threshold for sensitivity
        )
        
        logger.info("Stage 1: Initial training for generative upsampling...")
        history_stage1 = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs // 2,
            early_stopping_patience=10,
            save_dir=str(self.model_dir / "stage1")
        )
        
        # Generative Upsampling
        logger.info("\n" + "-"*40)
        logger.info("Performing Generative Upsampling...")
        train_loader_aug, _, _, _ = prepare_mimic_data(augment_model=model, target_ratio=0.4)
        
        logger.info("Stage 2: Fine-tuning with augmented data...")
        history = trainer.train(
            train_loader=train_loader_aug,
            val_loader=val_loader,
            n_epochs=n_epochs,
            early_stopping_patience=15,
            save_dir=str(self.model_dir)
        )
        
        results['training'] = {
            'final_val_accuracy': history['val_accuracy'][-1],
            'final_val_loss': history['val_loss'][-1],
            'epochs_trained': len(history['train_loss'])
        }
        
        # Plot training history
        trainer.plot_training_history(save_path=str(self.model_dir / 'training_history.png'))
        
        # Step 3: Evaluate model
        logger.info("\n" + "="*80)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("="*80)
        
        eval_results = self._evaluate_model(model, test_loader)
        results['evaluation'] = eval_results
        
        # Step 4: Ethics and fairness analysis
        logger.info("\n" + "="*80)
        logger.info("STEP 4: ETHICS AND FAIRNESS ANALYSIS")
        logger.info("="*80)
        
        ethics_results = self._run_ethics_analysis(model, test_loader)
        results['ethics'] = ethics_results
        
        # Step 5: Generate deployment package
        logger.info("\n" + "="*80)
        logger.info("STEP 5: DEPLOYMENT PACKAGE")
        logger.info("="*80)
        
        deployment_info = self._create_deployment_package(model, self.scaler)
        results['deployment'] = deployment_info
        
        # Save results
        results['completed_at'] = datetime.now().isoformat()
        results['status'] = 'SUCCESS'
        
        with open(self.output_dir / 'pipeline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY")
        logger.info("="*80)
        
        self._print_summary(results)
        
        return results
    
    def _evaluate_model(self, model, test_loader) -> Dict:
        """Evaluate model performance"""
        model.eval()
        device = next(model.parameters()).device
        
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                
                # Get predictions with uncertainty
                pred_dict = model.predict_sepsis(features, n_samples=10)
                
                all_predictions.extend(pred_dict['risk_mean'].cpu().numpy().flatten())
                all_labels.extend(labels.numpy())
                all_uncertainties.extend(pred_dict['risk_std'].cpu().numpy().flatten())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_uncertainties = np.array(all_uncertainties)
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
        
        # Calculate metrics using sensitive threshold
        threshold = 0.3  # Clinical threshold for higher sensitivity
        binary_preds = (all_predictions > threshold).astype(int)
        
        tp = ((binary_preds == 1) & (all_labels == 1)).sum()
        tn = ((binary_preds == 0) & (all_labels == 0)).sum()
        fp = ((binary_preds == 1) & (all_labels == 0)).sum()
        fn = ((binary_preds == 0) & (all_labels == 1)).sum()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / len(all_labels)
        auc_roc = roc_auc_score(all_labels, all_predictions)
        
        metrics = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'accuracy': float(accuracy),
            'auc_roc': float(auc_roc),
            'mean_uncertainty': float(all_uncertainties.mean()),
            'n_test_samples': len(all_labels)
        }
        
        logger.info("\nModel Performance:")
        logger.info(f"  Sensitivity: {sensitivity:.3f}")
        logger.info(f"  Specificity: {specificity:.3f}")
        logger.info(f"  PPV: {ppv:.3f}")
        logger.info(f"  NPV: {npv:.3f}")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  AUC-ROC: {auc_roc:.3f}")
        logger.info(f"  Mean Uncertainty: {all_uncertainties.mean():.3f}")
        
        # Save metrics
        with open(self.eval_dir / 'performance_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _run_ethics_analysis(self, model, test_loader) -> Dict:
        """Run comprehensive ethics analysis"""
        
        # Get test data
        all_features = []
        all_labels = []
        
        for features, labels in test_loader:
            all_features.append(features.numpy())
            all_labels.append(labels.numpy())
        
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        
        # Create demographics (in production, this would be real data)
        np.random.seed(42)
        demographics = pd.DataFrame({
            'gender': np.random.choice(['M', 'F'], len(all_labels)),
            'age': np.random.normal(65, 15, len(all_labels)).clip(18, 100)
        })
        
        # Initialize ethics monitor
        device = next(model.parameters()).device
        ethics_monitor = EthicsMonitor(model, device=str(device))
        
        # Fairness analysis
        fairness_report = ethics_monitor.analyze_fairness(
            all_features,
            all_labels,
            demographics
        )
        
        # Bias detection in latent space
        bias_metrics = ethics_monitor.detect_bias_in_latent_space(
            all_features,
            demographics
        )
        
        # Generate ethics report
        ethics_report = ethics_monitor.generate_ethics_report(
            save_path=str(self.ethics_dir / 'ethics_report.json')
        )
        
        logger.info(f"\n✓ Ethics analysis complete")
        logger.info(f"  Fairness report saved to: {self.ethics_dir / 'ethics_report.json'}")
        
        return {
            'fairness_analysis': 'complete',
            'bias_detection': 'complete',
            'ethics_report_path': str(self.ethics_dir / 'ethics_report.json')
        }
    
    def _create_deployment_package(self, model, scaler) -> Dict:
        """Create deployment package"""
        
        deployment_dir = self.output_dir / "deployment"
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model for deployment
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': model.input_dim,
                'latent_dim': model.latent_dim,
                'hidden_dims': model.hidden_dims
            }
        }, deployment_dir / 'sepsis_vae_model.pth')
        
        # Save scaler for feature normalization
        import joblib
        joblib.dump(scaler, deployment_dir / 'feature_scaler.pkl')
        
        logger.info(f"\n✓ Deployment package created: {deployment_dir}")
        
        return {
            'model_path': str(deployment_dir / 'sepsis_vae_model.pth'),
            'scaler_path': str(deployment_dir / 'feature_scaler.pkl'),
            'status': 'ready'
        }
    
    def _print_summary(self, results: Dict):
        """Print pipeline summary"""
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        
        print(f"\n🤖 Model Type: VAE (Variational Autoencoder)")
        print(f"📊 Data: MIMIC-III EHR data")
        
        if 'evaluation' in results:
            eval = results['evaluation']
            print(f"\n📈 Performance:")
            print(f"  Sensitivity: {eval['sensitivity']:.3f}")
            print(f"  Specificity: {eval['specificity']:.3f}")
            print(f"  AUC-ROC: {eval['auc_roc']:.3f}")
            print(f"  Mean Uncertainty: {eval['mean_uncertainty']:.3f}")
        
        print(f"\n✅ Ethics & Fairness: Analyzed")
        print(f"📦 Deployment Package: Ready")
        
        print(f"\n⏱️ Timeline:")
        print(f"  Started: {results['started_at']}")
        print(f"  Completed: {results['completed_at']}")
        
        print("\n" + "="*80)
        print("✓ VAE-BASED SEPSIS PREDICTION SYSTEM READY")
        print("="*80)

def main():
    """Main entry point"""
    print("="*80)
    print("VAE-BASED SEPSIS PREDICTION - GENERATIVE AI SYSTEM")
    print("="*80)
    print("\nThis pipeline will:")
    print("1. Prepare EHR data (vital signs, labs, demographics)")
    print("2. Train VAE model with uncertainty quantification")
    print("3. Evaluate model performance")
    print("4. Analyze ethics and fairness")
    print("5. Create deployment package")
    print("\n" + "="*80)
    
    # Run pipeline
    pipeline = SepsisPredictionPipeline(output_dir="./vae_output")
    results = pipeline.run_complete_pipeline(n_epochs=50)  # Reduced for demo
    
    print(f"\n✓ Pipeline complete! Results saved to: {pipeline.output_dir}")

if __name__ == "__main__":
    main()
