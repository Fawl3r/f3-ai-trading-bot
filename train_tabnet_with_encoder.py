#!/usr/bin/env python3
"""
TabNet Fine-tuning with TSA-MAE Encoder
Uses frozen encoder embeddings for enhanced feature representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
import argparse
import os
from datetime import datetime
import hashlib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTabNet(nn.Module):
    """Simplified TabNet for crypto trading"""
    
    def __init__(self, input_dim, output_dim=3, encoder_dim=64):
        super().__init__()
        
        # Feature transformation
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.feature_transformer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism (simplified)
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        
        # Encoder integration
        self.encoder_proj = nn.Linear(encoder_dim, 64) if encoder_dim > 0 else None
        
        # Decision layers
        feature_dim = 64 + (64 if encoder_dim > 0 else 0)
        self.decision = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
        
        logger.info(f"TabNet: {input_dim} -> {feature_dim} -> {output_dim}")
    
    def forward(self, x, encoder_features=None):
        # Normalize input
        x = self.input_bn(x)
        
        # Feature transformation
        features = self.feature_transformer(x)
        
        # Attention weighting
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Integrate encoder features if available
        if encoder_features is not None and self.encoder_proj is not None:
            encoder_proj = self.encoder_proj(encoder_features)
            features = torch.cat([features, encoder_proj], dim=1)
        
        # Decision
        output = self.decision(features)
        
        return output, attention_weights

class TradingDataset(Dataset):
    """Trading dataset with features and labels"""
    
    def __init__(self, months=3, symbols=['SOL', 'BTC', 'ETH'], encoder_model=None):
        self.months = months
        self.symbols = symbols
        self.encoder_model = encoder_model
        
        logger.info(f"Creating trading dataset for {months} months")
        
        # Generate features and labels
        self.features, self.labels, self.sequences = self._create_trading_data()
        
        logger.info(f"Dataset: {len(self.features)} samples, {self.features.shape[1]} features")
    
    def _create_trading_data(self):
        """Create trading features and labels"""
        
        all_features = []
        all_labels = []
        all_sequences = []
        
        for symbol in self.symbols:
            logger.info(f"Processing {symbol}")
            
            # Generate price data
            n_days = self.months * 30
            n_points = n_days * 24 * 4  # 15-min bars for features
            
            # Price parameters
            base_price = {'SOL': 100, 'BTC': 45000, 'ETH': 3000}[symbol]
            vol = {'SOL': 0.025, 'BTC': 0.020, 'ETH': 0.022}[symbol]
            
            # Generate returns
            np.random.seed(42 + hash(symbol) % 1000)
            returns = np.random.normal(0, vol/np.sqrt(96), n_points)  # 15-min returns
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Calculate features (last 20 bars worth)
            lookback = 20
            
            for i in range(lookback, n_points - 5):  # Need 5 future bars for labels
                # Historical features
                price_window = prices[i-lookback:i]
                return_window = np.diff(np.log(price_window))
                
                # Technical features
                features = []
                
                # Price features
                features.extend([
                    price_window[-1] / price_window[0] - 1,  # Total return
                    np.mean(return_window),  # Mean return
                    np.std(return_window),   # Volatility
                    np.min(return_window),   # Min return
                    np.max(return_window),   # Max return
                ])
                
                # Momentum features
                features.extend([
                    (price_window[-1] - price_window[-5]) / price_window[-5],  # 5-bar momentum
                    (price_window[-1] - price_window[-10]) / price_window[-10], # 10-bar momentum
                    np.sum(return_window[-5:]),  # Recent momentum
                ])
                
                # Volatility features
                recent_vol = np.std(return_window[-5:])
                long_vol = np.std(return_window)
                features.extend([
                    recent_vol,
                    long_vol,
                    recent_vol / long_vol if long_vol > 0 else 1,  # Vol ratio
                ])
                
                # Trend features
                x = np.arange(len(price_window))
                slope = np.polyfit(x, price_window, 1)[0] / price_window[-1]
                features.append(slope)
                
                # RSI approximation
                gains = np.maximum(return_window, 0)
                losses = -np.minimum(return_window, 0)
                avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
                avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
                rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))
                features.append(rsi / 100)
                
                # Create label (future direction)
                future_prices = prices[i:i+5]
                future_return = (future_prices[-1] - prices[i]) / prices[i]
                
                # 3-class classification: down, hold, up
                if future_return < -0.01:  # Down > 1%
                    label = 0
                elif future_return > 0.01:  # Up > 1%
                    label = 2
                else:  # Hold
                    label = 1
                
                all_features.append(features)
                all_labels.append(label)
                
                # Create sequence for encoder (1-min bars equivalent)
                # Upsample 15-min to 1-min (simple interpolation)
                minute_prices = np.interp(
                    np.linspace(0, len(price_window)-1, 240),  # 240 1-min bars
                    np.arange(len(price_window)),
                    price_window
                )
                
                # Simple OHLCV approximation
                sequence = []
                for j in range(240):
                    p = minute_prices[j]
                    o = p * np.random.uniform(0.999, 1.001)
                    h = p * np.random.uniform(1.0, 1.002)
                    l = p * np.random.uniform(0.998, 1.0)
                    v = np.random.lognormal(10, 1)
                    
                    # Simple features
                    features_1min = [
                        o/p, h/p, l/p, p/minute_prices[0], v/1e6,
                        (p - minute_prices[max(0, j-1)]) / minute_prices[max(0, j-1)] if j > 0 else 0,
                        np.log(v/1e6), (h-l)/p, (p-o)/o, 1.0, 1.0, 0.5, 1.02, 0.98, 1.0
                    ]
                    sequence.append(features_1min)
                
                all_sequences.append(np.array(sequence, dtype=np.float32))
        
        # Convert to arrays
        features = np.array(all_features, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int64)
        sequences = np.array(all_sequences, dtype=np.float32)
        
        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return features, labels, sequences
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.from_numpy(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        sequence = torch.from_numpy(self.sequences[idx])
        
        return features, label, sequence

def load_encoder(encoder_path):
    """Load TSA-MAE encoder"""
    if not os.path.exists(encoder_path):
        logger.warning(f"Encoder not found: {encoder_path}")
        return None
    
    checkpoint = torch.load(encoder_path, map_location='cpu')
    logger.info(f"Loaded encoder: {encoder_path}")
    logger.info(f"Config: {checkpoint.get('config', {})}")
    
    # Create encoder model (simplified)
    class EncoderWrapper(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            self.input_proj = nn.Linear(15, embed_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, 240, embed_dim) * 0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*2,
                dropout=0.1, activation='gelu', batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, 4)
        
        def forward(self, x):
            x = self.input_proj(x)
            x = x + self.pos_embed
            x = self.encoder(x)
            return x.mean(dim=1)  # Global average pool
    
    encoder = EncoderWrapper()
    
    # Load weights (with error handling)
    try:
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        else:
            # Extract encoder weights from full model
            encoder_weights = {k.replace('encoder.', ''): v 
                             for k, v in checkpoint['state_dict'].items() 
                             if 'encoder' in k}
            encoder.load_state_dict(encoder_weights, strict=False)
        
        logger.info("✅ Encoder weights loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️  Could not load encoder weights: {e}")
        return None
    
    encoder.eval()
    return encoder

def train_tabnet_with_encoder(encoder_path=None, epochs=20, freeze_encoder=True):
    """Train TabNet with TSA-MAE encoder"""
    
    logger.info("🚀 TabNet Training with TSA-MAE Encoder")
    logger.info(f"📁 Encoder: {encoder_path}")
    logger.info(f"🧊 Freeze encoder: {freeze_encoder}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Device: {device}")
    
    # Load encoder
    encoder = load_encoder(encoder_path) if encoder_path else None
    if encoder:
        encoder.to(device)
        if freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
            logger.info("🧊 Encoder frozen")
    
    # Dataset
    dataset = TradingDataset(months=6, encoder_model=encoder)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Loaders
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2)
    
    logger.info(f"📊 Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Model
    encoder_dim = 64 if encoder else 0
    model = SimpleTabNet(input_dim=12, output_dim=3, encoder_dim=encoder_dim).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    best_acc = 0
    best_model_path = None
    
    for epoch in range(epochs):
        logger.info(f"\n📅 Epoch {epoch+1}/{epochs}")
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels, sequences in train_loader:
            features, labels, sequences = features.to(device), labels.to(device), sequences.to(device)
            
            # Get encoder features
            encoder_features = None
            if encoder:
                with torch.no_grad():
                    encoder_features = encoder(sequences)
            
            optimizer.zero_grad()
            
            # Forward
            outputs, attention = model(features, encoder_features)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_pred = []
        all_true = []
        
        with torch.no_grad():
            for features, labels, sequences in val_loader:
                features, labels, sequences = features.to(device), labels.to(device), sequences.to(device)
                
                encoder_features = None
                if encoder:
                    encoder_features = encoder(sequences)
                
                outputs, attention = model(features, encoder_features)
                loss = F.cross_entropy(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_pred.extend(predicted.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, average='weighted')
        
        logger.info(f"📊 Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"📊 Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        logger.info(f"📊 F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_hash = hashlib.sha256(f"tabnet_{timestamp}".encode()).hexdigest()[:8]
            best_model_path = f'models/tabnet_{timestamp}_{model_hash}.pt'
            
            torch.save({
                'state_dict': model.state_dict(),
                'config': {
                    'input_dim': 12,
                    'output_dim': 3,
                    'encoder_dim': encoder_dim,
                    'encoder_path': encoder_path
                },
                'best_val_acc': best_acc,
                'metrics': {
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                },
                'model_hash': model_hash,
                'timestamp': timestamp
            }, best_model_path)
            
            logger.info(f"💾 Best model saved: {best_model_path}")
    
    logger.info(f"\n🎉 TabNet Training Complete!")
    logger.info(f"🏆 Best Accuracy: {best_acc:.4f}")
    logger.info(f"📁 Best Model: {best_model_path}")
    
    return model, best_model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_encoder', type=str, help='Path to TSA-MAE encoder')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    
    args = parser.parse_args()
    
    # Train TabNet
    model, model_path = train_tabnet_with_encoder(
        encoder_path=args.use_encoder,
        epochs=args.epochs,
        freeze_encoder=args.freeze_encoder
    )
    
    print(f"\n🚀 TabNet Fine-tuning Complete!")
    print(f"📁 Model: {model_path}")
    print(f"🔄 Next: python train_ppo.py --encoder {args.use_encoder}")

if __name__ == "__main__":
    main() 