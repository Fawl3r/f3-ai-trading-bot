#!/usr/bin/env python3
"""
Meta-Learner Ensemble System
Combines LightGBM, TimesNet, PPO, and baseline predictions using logistic regression
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import logging
import os
import json
import hashlib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """Utility class to load different model types"""
    
    @staticmethod
    def load_lightgbm(model_path: str):
        """Load LightGBM model"""
        try:
            model = joblib.load(model_path)
            # Load metadata
            metadata_path = model_path.replace('.pkl', '.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return model, metadata
        except Exception as e:
            logger.error(f"Failed to load LightGBM: {e}")
            return None, None
    
    @staticmethod
    def load_timesnet(model_path: str):
        """Load TimesNet model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Recreate model architecture
            from models.train_timesnet import TimesNet
            config = checkpoint['config']
            model = TimesNet(**config)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            
            return model, checkpoint
        except Exception as e:
            logger.error(f"Failed to load TimesNet: {e}")
            return None, None
    
    @staticmethod
    def load_ppo(model_path: str):
        """Load PPO model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Simple PPO wrapper for predictions
            class PPOPredictor(nn.Module):
                def __init__(self, state_dict):
                    super().__init__()
                    self.policy_net = nn.Sequential(
                        nn.Linear(20, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 4)
                    )
                    self.load_state_dict(state_dict, strict=False)
                    self.eval()
                
                def forward(self, x):
                    return torch.softmax(self.policy_net(x), dim=-1)
            
            model = PPOPredictor(checkpoint)
            return model, checkpoint
        except Exception as e:
            logger.error(f"Failed to load PPO: {e}")
            return None, None
    
    @staticmethod
    def load_baseline(model_path: str = None):
        """Load baseline model (TabNet or simple classifier)"""
        # For now, return a simple baseline
        class BaselinePredictor:
            def predict_proba(self, X):
                # Simple momentum-based baseline
                n_samples = X.shape[0]
                probs = np.zeros((n_samples, 3))
                
                for i in range(n_samples):
                    # Simple momentum signal
                    momentum = X[i, -1] if X.shape[1] > 0 else 0
                    
                    if momentum > 0.001:
                        probs[i] = [0.2, 0.3, 0.5]  # Bullish
                    elif momentum < -0.001:
                        probs[i] = [0.5, 0.3, 0.2]  # Bearish
                    else:
                        probs[i] = [0.33, 0.34, 0.33]  # Neutral
                
                return probs
        
        return BaselinePredictor(), {"type": "baseline"}

class MetaLearner:
    """Meta-learner that combines multiple model predictions"""
    
    def __init__(self, models_config: Dict[str, str]):
        """
        Initialize meta-learner
        
        Args:
            models_config: Dict mapping model names to file paths
                          e.g., {'lgbm': 'models/lgbm_SOL.pkl', 
                                'timesnet': 'models/timesnet_SOL.pt'}
        """
        self.models_config = models_config
        self.models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Load all models
        self._load_models()
    
    def _load_models(self):
        """Load all configured models"""
        logger.info("🔄 Loading models for meta-learning...")
        
        for model_name, model_path in self.models_config.items():
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                continue
            
            if model_name.startswith('lgbm'):
                model, metadata = ModelLoader.load_lightgbm(model_path)
            elif model_name.startswith('timesnet'):
                model, metadata = ModelLoader.load_timesnet(model_path)
            elif model_name.startswith('ppo'):
                model, metadata = ModelLoader.load_ppo(model_path)
            elif model_name.startswith('baseline'):
                model, metadata = ModelLoader.load_baseline(model_path)
            else:
                logger.warning(f"Unknown model type: {model_name}")
                continue
            
            if model is not None:
                self.models[model_name] = {'model': model, 'metadata': metadata}
                logger.info(f"✅ Loaded {model_name}")
            else:
                logger.error(f"❌ Failed to load {model_name}")
    
    def get_model_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all models"""
        predictions = []
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            
            try:
                if model_name.startswith('lgbm'):
                    # LightGBM prediction
                    pred = model.predict(X)
                    if pred.ndim == 1:
                        # Convert to probabilities
                        pred_probs = np.zeros((len(pred), 3))
                        for i, p in enumerate(pred):
                            pred_probs[i, int(p)] = 1.0
                        pred = pred_probs
                    predictions.append(pred)
                
                elif model_name.startswith('timesnet'):
                    # TimesNet prediction
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X)
                        if X_tensor.dim() == 2:
                            # Reshape for sequence input
                            X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension
                        
                        logits = model(X_tensor)
                        pred = torch.softmax(logits, dim=-1).numpy()
                    predictions.append(pred)
                
                elif model_name.startswith('ppo'):
                    # PPO prediction
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X)
                        pred = model(X_tensor).numpy()
                    predictions.append(pred)
                
                elif model_name.startswith('baseline'):
                    # Baseline prediction
                    pred = model.predict_proba(X)
                    predictions.append(pred)
                
                else:
                    logger.warning(f"Unknown prediction method for {model_name}")
            
            except Exception as e:
                logger.error(f"Prediction error for {model_name}: {e}")
                # Add dummy prediction to maintain shape
                dummy_pred = np.full((X.shape[0], 3), 1/3)
                predictions.append(dummy_pred)
        
        if not predictions:
            logger.error("No valid predictions obtained!")
            return np.full((X.shape[0], 3), 1/3)
        
        # Stack predictions
        stacked_predictions = np.stack(predictions, axis=-1)  # [n_samples, n_classes, n_models]
        
        # Reshape for meta-learning: [n_samples, n_classes * n_models]
        meta_features = stacked_predictions.reshape(X.shape[0], -1)
        
        return meta_features
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit meta-learner on training data"""
        logger.info("🧠 Training meta-learner...")
        
        # Get predictions from all models
        meta_features = self.get_model_predictions(X)
        
        # Scale features
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        
        # Train meta-model
        self.meta_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        self.meta_model.fit(meta_features_scaled, y)
        self.is_fitted = True
        
        # Evaluate training performance
        train_pred = self.meta_model.predict(meta_features_scaled)
        train_acc = accuracy_score(y, train_pred)
        
        logger.info(f"✅ Meta-learner trained with {train_acc:.4f} accuracy")
        logger.info(f"📊 Features shape: {meta_features_scaled.shape}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using meta-learner"""
        if not self.is_fitted:
            raise ValueError("Meta-learner not fitted yet!")
        
        # Get predictions from all models
        meta_features = self.get_model_predictions(X)
        
        # Scale features
        meta_features_scaled = self.scaler.transform(meta_features)
        
        # Meta-model prediction
        predictions = self.meta_model.predict(meta_features_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Meta-learner not fitted yet!")
        
        # Get predictions from all models
        meta_features = self.get_model_predictions(X)
        
        # Scale features
        meta_features_scaled = self.scaler.transform(meta_features)
        
        # Meta-model probabilities
        probabilities = self.meta_model.predict_proba(meta_features_scaled)
        
        return probabilities
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate meta-learner performance"""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        accuracy = accuracy_score(y, predictions)
        
        # Calculate individual model accuracies for comparison
        individual_accuracies = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            
            try:
                if model_name.startswith('lgbm'):
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                        if pred.ndim > 1:
                            pred = np.argmax(pred, axis=1)
                        individual_accuracies[model_name] = accuracy_score(y, pred)
                
                elif model_name.startswith('baseline'):
                    pred_probs = model.predict_proba(X)
                    pred = np.argmax(pred_probs, axis=1)
                    individual_accuracies[model_name] = accuracy_score(y, pred)
                
                # Add other model types as needed
                
            except Exception as e:
                logger.warning(f"Could not evaluate {model_name}: {e}")
        
        results = {
            'meta_accuracy': accuracy,
            'individual_accuracies': individual_accuracies,
            'improvement': accuracy - max(individual_accuracies.values()) if individual_accuracies else 0
        }
        
        return results
    
    def save(self, save_path: str):
        """Save meta-learner"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted meta-learner!")
        
        # Save meta-model and scaler
        meta_data = {
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'models_config': self.models_config,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(meta_data, save_path)
        logger.info(f"💾 Meta-learner saved: {save_path}")
    
    @classmethod
    def load(cls, load_path: str):
        """Load meta-learner"""
        meta_data = joblib.load(load_path)
        
        # Create instance
        instance = cls(meta_data['models_config'])
        instance.meta_model = meta_data['meta_model']
        instance.scaler = meta_data['scaler']
        instance.is_fitted = True
        
        logger.info(f"📁 Meta-learner loaded: {load_path}")
        return instance

def generate_synthetic_ensemble_data(n_samples=1000, n_features=20):
    """Generate synthetic data for meta-learner testing"""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate targets with some pattern
    linear_combo = X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2
    noise = np.random.normal(0, 0.1, n_samples)
    
    y_continuous = linear_combo + noise
    
    # Convert to classification
    y = np.where(y_continuous > 0.2, 2,  # Buy
                np.where(y_continuous < -0.2, 0, 1))  # Sell, Hold
    
    return X, y

def train_meta_learner(symbols=['SOL'], model_paths=None):
    """Train meta-learner with available models"""
    logger.info("🚀 Starting meta-learner training")
    
    # Default model paths if not provided
    if model_paths is None:
        model_paths = {}
        
        # Look for available models
        models_dir = 'models'
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.pkl') and 'lgbm' in filename:
                    model_paths['lgbm'] = os.path.join(models_dir, filename)
                elif filename.endswith('.pt') and 'timesnet' in filename:
                    model_paths['timesnet'] = os.path.join(models_dir, filename)
                elif filename.endswith('.pt') and 'ppo' in filename:
                    model_paths['ppo'] = os.path.join(models_dir, filename)
        
        # Add baseline
        model_paths['baseline'] = None
    
    logger.info(f"📊 Model paths: {model_paths}")
    
    # Generate synthetic data for testing
    X, y = generate_synthetic_ensemble_data(n_samples=2000, n_features=20)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train meta-learner
    meta_learner = MetaLearner(model_paths)
    meta_learner.fit(X_train, y_train)
    
    # Evaluate
    results = meta_learner.evaluate(X_test, y_test)
    
    logger.info("📊 Meta-learner evaluation results:")
    logger.info(f"🎯 Meta accuracy: {results['meta_accuracy']:.4f}")
    logger.info(f"📈 Individual accuracies: {results['individual_accuracies']}")
    logger.info(f"🚀 Improvement: {results['improvement']:.4f}")
    
    # Save meta-learner
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'models/meta_learner_{timestamp}.pkl'
    meta_learner.save(save_path)
    
    return meta_learner, results

def main():
    """Main function to train meta-learner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train meta-learner ensemble')
    parser.add_argument('--symbols', default='SOL,BTC,ETH', help='Comma-separated symbols')
    parser.add_argument('--lgbm_path', help='Path to LightGBM model')
    parser.add_argument('--timesnet_path', help='Path to TimesNet model')
    parser.add_argument('--ppo_path', help='Path to PPO model')
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(',')
    
    # Build model paths
    model_paths = {}
    if args.lgbm_path:
        model_paths['lgbm'] = args.lgbm_path
    if args.timesnet_path:
        model_paths['timesnet'] = args.timesnet_path
    if args.ppo_path:
        model_paths['ppo'] = args.ppo_path
    
    # Train meta-learner
    meta_learner, results = train_meta_learner(symbols, model_paths)
    
    logger.info("🎉 Meta-learner training complete!")
    logger.info(f"🎯 Final ensemble accuracy: {results['meta_accuracy']:.4f}")

if __name__ == "__main__":
    main() 