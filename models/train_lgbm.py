#!/usr/bin/env python3
"""
LightGBM + TSA-MAE Embeddings Trainer
Gradient-boosted trees with 64-dim TSA encoder embeddings for non-linear TA interactions
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import logging
import argparse
import os
import hashlib
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import joblib
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TSAMAEEncoderWrapper(nn.Module):
    """Wrapper for TSA-MAE encoder to extract embeddings"""
    
    def __init__(self, encoder_path: str):
        super().__init__()
        
        # Load encoder checkpoint
        checkpoint = torch.load(encoder_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # Extract encoder parameters
        embed_dim = config.get('embed_dim', 64)
        num_heads = config.get('nhead', 8)
        depth = config.get('depth', 4)
        
        # Build encoder architecture
        self.input_proj = nn.Linear(15, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 240, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        
        # Load weights
        try:
            if 'encoder_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
            else:
                # Extract encoder weights from full model
                encoder_weights = {}
                for k, v in checkpoint['state_dict'].items():
                    if 'input_proj' in k or 'pos_embed' in k or 'encoder' in k:
                        new_key = k.replace('input_projection', 'input_proj')
                        new_key = new_key.replace('pos_encoding', 'pos_embed')
                        encoder_weights[new_key] = v
                
                self.load_state_dict(encoder_weights, strict=False)
            
            logger.info(f"✅ Encoder loaded: {encoder_path}")
            logger.info(f"📊 Config: {embed_dim}d, {num_heads}h, {depth}L")
        except Exception as e:
            logger.error(f"❌ Failed to load encoder: {e}")
            raise
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """Extract embeddings from time series"""
        with torch.no_grad():
            x = self.input_proj(x)
            x = x + self.pos_embed
            x = self.encoder(x)
            # Global average pooling
            embeddings = x.mean(dim=1)
            return embeddings

def generate_synthetic_data(symbols=['SOL', 'BTC', 'ETH'], days=30):
    """Generate synthetic crypto data for training"""
    logger.info(f"📊 Generating synthetic data for {symbols} over {days} days")
    
    np.random.seed(42)
    data_frames = []
    
    for symbol in symbols:
        # Generate realistic price movements
        n_points = days * 24 * 60  # 1-minute bars
        
        # Base price
        base_price = {'SOL': 100, 'BTC': 45000, 'ETH': 2500}[symbol]
        
        # Generate returns with volatility clustering
        returns = np.random.normal(0, 0.002, n_points)
        volatility = np.random.exponential(0.01, n_points)
        returns = returns * (1 + volatility)
        
        # Create price series
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_points, freq='1min'),
            'symbol': symbol,
            'close': prices,
            'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_points))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_points))),
            'volume': np.random.lognormal(10, 1, n_points)
        })
        
        # Fix OHLC relationships
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Technical indicators
        df['rsi'] = 50 + 30 * np.sin(np.arange(n_points) / 100) + np.random.normal(0, 5, n_points)
        df['rsi'] = np.clip(df['rsi'], 0, 100)
        
        df['macd'] = np.random.normal(0, 0.1, n_points)
        df['bb_upper'] = df['close'] * 1.02
        df['bb_lower'] = df['close'] * 0.98
        df['atr'] = df['close'] * 0.02
        df['volume_sma'] = df['volume'].rolling(20).mean().fillna(df['volume'])
        df['price_change'] = df['close'].pct_change().fillna(0)
        df['volume_change'] = df['volume'].pct_change().fillna(0)
        
        # Add momentum indicators
        df['stoch_k'] = 50 + 40 * np.sin(np.arange(n_points) / 50) + np.random.normal(0, 10, n_points)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['williams_r'] = -50 + 40 * np.cos(np.arange(n_points) / 80) + np.random.normal(0, 15, n_points)
        
        # Generate targets (future returns)
        future_returns = df['close'].shift(-5) / df['close'] - 1
        df['target'] = np.where(future_returns > 0.002, 2,  # Buy
                               np.where(future_returns < -0.002, 0,  # Sell
                                       1))  # Hold
        
        df = df.dropna()
        data_frames.append(df)
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    logger.info(f"📈 Generated {len(combined_df)} samples with {combined_df['target'].value_counts().to_dict()} distribution")
    
    return combined_df

def prepare_features(df, encoder_model, window_size=240):
    """Prepare features with TSA-MAE embeddings"""
    logger.info("🔧 Preparing features with TSA-MAE embeddings")
    
    features = []
    targets = []
    
    # Technical analysis features
    ta_features = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr',
        'volume_sma', 'price_change', 'volume_change',
        'stoch_k', 'stoch_d'
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_model = encoder_model.to(device)
    
    # Process each symbol separately
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        
        for i in range(window_size, len(symbol_df)):
            # Get window of data
            window_data = symbol_df.iloc[i-window_size:i]
            
            # Standard TA features (current values)
            current_features = symbol_df.iloc[i][ta_features].values
            
            # TSA-MAE embeddings
            window_tensor = torch.FloatTensor(window_data[ta_features].values).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embeddings = encoder_model(window_tensor).cpu().numpy().flatten()
            
            # Combine features
            combined_features = np.concatenate([current_features, embeddings])
            features.append(combined_features)
            targets.append(symbol_df.iloc[i]['target'])
    
    features = np.array(features)
    targets = np.array(targets)
    
    logger.info(f"✅ Features prepared: {features.shape}, Targets: {len(targets)}")
    return features, targets

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for LightGBM hyperparameter tuning"""
    
    # Suggest hyperparameters
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'verbosity': -1,
        'random_state': 42
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_val, y_pred_classes)
    
    return accuracy

def train_lgbm_with_embeddings(encoder_path, symbols=['SOL', 'BTC', 'ETH'], 
                              optuna_trials=50, days=30):
    """Train LightGBM with TSA-MAE embeddings"""
    
    logger.info("🚀 Starting LightGBM + TSA-MAE training")
    logger.info(f"📊 Symbols: {symbols}, Trials: {optuna_trials}")
    
    # Load encoder
    encoder_model = TSAMAEEncoderWrapper(encoder_path)
    
    # Generate data
    df = generate_synthetic_data(symbols, days)
    
    # Prepare features
    X, y = prepare_features(df, encoder_model)
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    best_accuracy = 0
    best_model = None
    best_params = None
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logger.info(f"📊 Fold {fold + 1}/5")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Optuna optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=optuna_trials,
            show_progress_bar=True
        )
        
        # Train best model
        best_trial_params = study.best_params
        best_trial_params.update({
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42
        })
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            best_trial_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Evaluate
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val, y_pred_classes)
        
        logger.info(f"✅ Fold {fold + 1} Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = best_trial_params
    
    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_hash = hashlib.sha256(str(best_params).encode()).hexdigest()[:8]
    
    for symbol in symbols:
        model_path = f'models/lgbm_{symbol}_{timestamp}_{model_hash}.pkl'
        joblib.dump(best_model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': 'lightgbm',
            'symbols': symbols,
            'encoder_path': encoder_path,
            'best_params': best_params,
            'best_accuracy': best_accuracy,
            'timestamp': timestamp,
            'model_hash': model_hash,
            'feature_dim': X.shape[1]
        }
        
        metadata_path = f'models/lgbm_{symbol}_{timestamp}_{model_hash}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Model saved: {model_path}")
    
    logger.info(f"🏆 Best accuracy: {best_accuracy:.4f}")
    logger.info(f"📊 Best params: {best_params}")
    
    return best_model, best_params, best_accuracy

def main():
    parser = argparse.ArgumentParser(description='Train LightGBM with TSA-MAE embeddings')
    parser.add_argument('--encoder', required=True, help='Path to TSA-MAE encoder')
    parser.add_argument('--coins', default='SOL,BTC,ETH', help='Comma-separated coin symbols')
    parser.add_argument('--optuna_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--days', type=int, default=30, help='Days of synthetic data')
    
    args = parser.parse_args()
    
    symbols = args.coins.split(',')
    
    # Train model
    model, params, accuracy = train_lgbm_with_embeddings(
        args.encoder,
        symbols=symbols,
        optuna_trials=args.optuna_trials,
        days=args.days
    )
    
    logger.info("🎉 LightGBM + TSA-MAE training complete!")
    logger.info(f"🎯 Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 