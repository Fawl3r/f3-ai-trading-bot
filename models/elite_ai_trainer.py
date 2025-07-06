#!/usr/bin/env python3
"""
Elite AI Training System - Double-Up Strategy
Multi-model ensemble with Optuna optimization for 70%+ win rate
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib
import optuna
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import lightgbm as lgb
import catboost as cb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Time Series Models
try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingDataset(Dataset):
    """PyTorch dataset for trading sequences"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BiLSTMModel(nn.Module):
    """Bi-directional LSTM for sequence modeling"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        return output

class TransformerModel(nn.Module):
    """Transformer model for time series"""
    
    def __init__(self, input_size: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.3):
        super().__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # Transformer
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        output = self.classifier(x)
        return output

class EliteAITrainer:
    """Elite AI training system for double-up strategy"""
    
    def __init__(self, data_dir: Path = Path("data/processed")):
        self.data_dir = data_dir
        self.models = {}
        self.best_params = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🚀 Using device: {self.device}")
    
    def load_data(self, coin: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load processed data for a coin"""
        coin_dir = self.data_dir / coin
        
        if not coin_dir.exists():
            raise FileNotFoundError(f"Data not found for {coin}")
        
        X = np.load(coin_dir / "X.npy")
        y = np.load(coin_dir / "y.npy")
        metadata = joblib.load(coin_dir / "metadata.pkl")
        
        logger.info(f"📊 Loaded {coin}: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
        return X, y, metadata
    
    def create_time_series_splits(self, X: np.ndarray, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time series splits for validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        for train_idx, val_idx in tscv.split(X):
            splits.append((train_idx, val_idx))
        
        return splits
    
    def optimize_lightgbm(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict:
        """Optimize LightGBM hyperparameters"""
        logger.info("🔧 Optimizing LightGBM...")
        
        # Flatten sequences for tree-based models
        X_flat = X.reshape(X.shape[0], -1)
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'verbosity': -1
            }
            
            # Time series cross-validation
            splits = self.create_time_series_splits(X_flat)
            scores = []
            
            for train_idx, val_idx in splits:
                X_train, X_val = X_flat[train_idx], X_flat[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_val)
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"✅ LightGBM optimization complete. Best AUC: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_catboost(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict:
        """Optimize CatBoost hyperparameters"""
        logger.info("🔧 Optimizing CatBoost...")
        
        # Flatten sequences for tree-based models
        X_flat = X.reshape(X.shape[0], -1)
        
        def objective(trial):
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'iterations': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                'random_strength': trial.suggest_float('random_strength', 0, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
                'od_type': 'Iter',
                'od_wait': 50,
                'random_seed': 42,
                'logging_level': 'Silent'
            }
            
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
            elif params['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = trial.suggest_float('subsample', 0.5, 1)
            
            # Time series cross-validation
            splits = self.create_time_series_splits(X_flat)
            scores = []
            
            for train_idx, val_idx in splits:
                X_train, X_val = X_flat[train_idx], X_flat[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = cb.CatBoostClassifier(**params)
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"✅ CatBoost optimization complete. Best AUC: {study.best_value:.4f}")
        return study.best_params
    
    def train_deep_model(self, X: np.ndarray, y: np.ndarray, model_type: str = 'bilstm') -> nn.Module:
        """Train deep learning model"""
        logger.info(f"🧠 Training {model_type.upper()} model...")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_dataset = TradingDataset(X_train, y_train)
        val_dataset = TradingDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Create model
        input_size = X.shape[2]
        
        if model_type == 'bilstm':
            model = BiLSTMModel(input_size=input_size).to(self.device)
        elif model_type == 'transformer':
            model = TransformerModel(input_size=input_size).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        # Training loop
        best_val_auc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    probs = F.softmax(outputs, dim=1)[:, 1]
                    
                    val_preds.extend(probs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_auc = roc_auc_score(val_targets, val_preds)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_{model_type}_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            scheduler.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{model_type}_model.pth'))
        logger.info(f"✅ {model_type.upper()} training complete. Best AUC: {best_val_auc:.4f}")
        
        return model
    
    def create_ensemble(self, models: Dict, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """Create ensemble meta-learner"""
        logger.info("🎯 Creating ensemble meta-learner...")
        
        # Get predictions from all models
        predictions = []
        
        # Tree-based models (use flattened data)
        X_flat = X.reshape(X.shape[0], -1)
        
        for name, model in models.items():
            if name in ['lightgbm', 'catboost']:
                if name == 'lightgbm':
                    preds = model.predict(X_flat)
                else:  # catboost
                    preds = model.predict_proba(X_flat)[:, 1]
            else:  # deep learning models
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(self.device)
                    outputs = model(X_tensor)
                    preds = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            predictions.append(preds)
        
        # Stack predictions
        stacked_preds = np.column_stack(predictions)
        
        # Train meta-learner
        meta_learner = LogisticRegression(random_state=42)
        meta_learner.fit(stacked_preds, y)
        
        logger.info("✅ Ensemble meta-learner created")
        return meta_learner
    
    def train_elite_models(self, coin: str) -> Dict:
        """Train all elite models for a coin"""
        logger.info(f"🚀 Training elite models for {coin}")
        
        # Load data
        X, y, metadata = self.load_data(coin)
        
        models = {}
        
        # 1. Optimize and train LightGBM
        lgb_params = self.optimize_lightgbm(X, y, n_trials=30)
        
        X_flat = X.reshape(X.shape[0], -1)
        lgb_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'num_boost_round': 1000
        })
        
        train_data = lgb.Dataset(X_flat, label=y)
        lgb_model = lgb.train(lgb_params, train_data)
        models['lightgbm'] = lgb_model
        
        # 2. Optimize and train CatBoost
        cb_params = self.optimize_catboost(X, y, n_trials=30)
        cb_params.update({
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'iterations': 1000,
            'random_seed': 42,
            'logging_level': 'Silent'
        })
        
        cb_model = cb.CatBoostClassifier(**cb_params)
        cb_model.fit(X_flat, y)
        models['catboost'] = cb_model
        
        # 3. Train Bi-LSTM
        bilstm_model = self.train_deep_model(X, y, 'bilstm')
        models['bilstm'] = bilstm_model
        
        # 4. Train Transformer (if enough data)
        if len(X) > 10000:
            transformer_model = self.train_deep_model(X, y, 'transformer')
            models['transformer'] = transformer_model
        
        # 5. Create ensemble
        ensemble = self.create_ensemble(models, X, y)
        models['ensemble'] = ensemble
        
        return models
    
    def evaluate_models(self, models: Dict, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate all models"""
        logger.info("📊 Evaluating models...")
        
        results = {}
        X_flat = X.reshape(X.shape[0], -1)
        
        for name, model in models.items():
            if name == 'ensemble':
                continue  # Skip ensemble for now
            
            if name in ['lightgbm', 'catboost']:
                if name == 'lightgbm':
                    preds = model.predict(X_flat)
                else:
                    preds = model.predict_proba(X_flat)[:, 1]
            else:  # deep learning models
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(self.device)
                    outputs = model(X_tensor)
                    preds = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            auc = roc_auc_score(y, preds)
            
            # Convert to binary predictions
            binary_preds = (preds > 0.5).astype(int)
            
            results[name] = {
                'auc': auc,
                'accuracy': np.mean(binary_preds == y),
                'precision': np.sum((binary_preds == 1) & (y == 1)) / np.sum(binary_preds == 1) if np.sum(binary_preds == 1) > 0 else 0,
                'recall': np.sum((binary_preds == 1) & (y == 1)) / np.sum(y == 1) if np.sum(y == 1) > 0 else 0
            }
            
            logger.info(f"{name}: AUC={auc:.4f}, Acc={results[name]['accuracy']:.4f}")
        
        return results
    
    def save_models(self, models: Dict, coin: str, output_dir: Path = Path("models/trained")):
        """Save trained models"""
        output_dir.mkdir(parents=True, exist_ok=True)
        coin_dir = output_dir / coin
        coin_dir.mkdir(exist_ok=True)
        
        for name, model in models.items():
            if name in ['lightgbm', 'catboost', 'ensemble']:
                joblib.dump(model, coin_dir / f"{name}_model.pkl")
            else:  # deep learning models
                torch.save(model.state_dict(), coin_dir / f"{name}_model.pth")
        
        logger.info(f"💾 Models saved to {coin_dir}")

def main():
    """Main training pipeline"""
    trainer = EliteAITrainer()
    
    # Elite coins for double-up strategy
    elite_coins = ['SOL', 'BTC', 'ETH']
    
    all_results = {}
    
    for coin in elite_coins:
        logger.info(f"\n{'='*50}")
        logger.info(f"TRAINING ELITE MODELS FOR {coin}")
        logger.info(f"{'='*50}")
        
        try:
            # Train models
            models = trainer.train_elite_models(coin)
            
            # Load data for evaluation
            X, y, _ = trainer.load_data(coin)
            
            # Evaluate models
            results = trainer.evaluate_models(models, X, y)
            all_results[coin] = results
            
            # Save models
            trainer.save_models(models, coin)
            
        except Exception as e:
            logger.error(f"❌ Error training {coin}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*50}")
    print("ELITE AI TRAINING COMPLETE")
    print(f"{'='*50}")
    
    for coin, results in all_results.items():
        print(f"\n{coin} Results:")
        for model, metrics in results.items():
            print(f"  {model}: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    print(f"\n🎯 Elite models ready for double-up strategy!")

if __name__ == "__main__":
    main() 