#!/usr/bin/env python3
"""
TimesNet - Long-range Transformer for Crypto Futures
Captures multi-day cyclical drift and long-term dependencies
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
import hashlib
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import json
from typing import Dict, List, Tuple, Optional
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimesBlock(nn.Module):
    """TimesNet block with period-based decomposition"""
    
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Inception-like conv blocks for multi-scale feature extraction
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])
        
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Multi-scale convolution
        x_conv = x.transpose(1, 2)  # [B, d_model, L]
        conv_outs = []
        
        for conv in self.conv_blocks:
            conv_out = F.gelu(conv(x_conv))
            conv_outs.append(conv_out)
        
        # Combine multi-scale features
        x_conv = torch.stack(conv_outs, dim=-1).mean(dim=-1)
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_model]
        
        x = self.conv_norm(x + self.conv_dropout(x_conv))
        
        # Feed forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class PeriodEmbedding(nn.Module):
    """Period-aware embedding for cyclical patterns"""
    
    def __init__(self, d_model, max_periods=10):
        super().__init__()
        
        self.d_model = d_model
        self.max_periods = max_periods
        
        # Learnable period embeddings
        self.period_embeddings = nn.Embedding(max_periods, d_model)
        
        # FFT-based period detection weights
        self.period_weights = nn.Parameter(torch.randn(max_periods))
        
    def detect_periods(self, x):
        """Detect dominant periods using FFT"""
        # x: [B, L, d_model]
        B, L, D = x.shape
        
        # Apply FFT along time dimension
        x_fft = torch.fft.rfft(x, dim=1)
        power_spectrum = torch.abs(x_fft) ** 2
        
        # Average across feature dimension
        power_spectrum = power_spectrum.mean(dim=-1)  # [B, L//2+1]
        
        # Find top periods
        freqs = torch.fft.rfftfreq(L, device=x.device)
        periods = 1.0 / (freqs + 1e-8)
        
        # Discretize periods to max_periods bins
        period_bins = torch.linspace(1, L//4, self.max_periods, device=x.device)
        
        # Assign each frequency to nearest period bin
        period_indices = torch.bucketize(periods, period_bins)
        period_indices = torch.clamp(period_indices, 0, self.max_periods - 1)
        
        # Aggregate power by period
        period_power = torch.zeros(B, self.max_periods, device=x.device)
        for i in range(self.max_periods):
            mask = (period_indices == i)
            if mask.any():
                period_power[:, i] = power_spectrum[:, mask].sum(dim=1)
        
        # Normalize
        period_power = F.softmax(period_power, dim=1)
        
        return period_power
    
    def forward(self, x):
        """Add period-aware embeddings"""
        # Detect periods
        period_weights = self.detect_periods(x)  # [B, max_periods]
        
        # Get period embeddings
        period_embs = self.period_embeddings.weight  # [max_periods, d_model]
        
        # Weighted combination
        weighted_embs = torch.matmul(period_weights, period_embs)  # [B, d_model]
        
        # Add to input
        x = x + weighted_embs.unsqueeze(1)
        
        return x

class TimesNet(nn.Module):
    """
    TimesNet: Long-range Transformer for Time Series
    Captures multi-day cyclical patterns and long-term dependencies
    """
    
    def __init__(self, 
                 input_dim=15,
                 d_model=128,
                 num_heads=8,
                 num_layers=6,
                 d_ff=512,
                 max_seq_len=1024,
                 num_classes=3,
                 dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )
        
        # Period embedding
        self.period_embedding = PeriodEmbedding(d_model)
        
        # TimesNet blocks
        self.blocks = nn.ModuleList([
            TimesBlock(d_model, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"TimesNet: {d_model}d, {num_heads}h, {num_layers}L, {total_params:,} params")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass"""
        B, L, D = x.shape
        
        # Truncate if too long
        if L > self.max_seq_len:
            x = x[:, -self.max_seq_len:, :]
            L = self.max_seq_len
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :L, :]
        
        # Add period embeddings
        x = self.period_embedding(x)
        
        # Apply TimesNet blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # [B, d_model]
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_embeddings(self, x):
        """Get embeddings for downstream tasks"""
        with torch.no_grad():
            B, L, D = x.shape
            
            if L > self.max_seq_len:
                x = x[:, -self.max_seq_len:, :]
                L = self.max_seq_len
            
            x = self.input_projection(x)
            x = x + self.pos_encoding[:, :L, :]
            x = self.period_embedding(x)
            
            for block in self.blocks:
                x = block(x)
            
            embeddings = x.mean(dim=1)
            
        return embeddings

class TimesNetDataset(Dataset):
    """Dataset for TimesNet training"""
    
    def __init__(self, data_df, sequence_length=1024, prediction_horizon=5):
        self.data = data_df
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Features
        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr',
            'volume_sma', 'price_change', 'volume_change',
            'stoch_k', 'stoch_d'
        ]
        
        # Normalize features
        self.scaler = StandardScaler()
        self.normalized_data = self.scaler.fit_transform(
            self.data[self.features].values
        )
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(sequence_length, len(self.normalized_data) - prediction_horizon):
            sequence = self.normalized_data[i-sequence_length:i]
            
            # Future return target
            future_price = self.data.iloc[i + prediction_horizon]['close']
            current_price = self.data.iloc[i]['close']
            future_return = (future_price - current_price) / current_price
            
            # Classify return
            if future_return > 0.002:
                target = 2  # Buy
            elif future_return < -0.002:
                target = 0  # Sell
            else:
                target = 1  # Hold
            
            self.sequences.append(sequence)
            self.targets.append(target)
        
        logger.info(f"Created {len(self.sequences)} sequences of length {sequence_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.LongTensor([self.targets[idx]])
        return sequence, target

def generate_long_term_data(symbols=['SOL', 'BTC', 'ETH'], days=90):
    """Generate synthetic long-term crypto data"""
    logger.info(f"📊 Generating long-term data for {symbols} over {days} days")
    
    np.random.seed(42)
    data_frames = []
    
    for symbol in symbols:
        # Generate longer sequences for cyclical patterns
        n_points = days * 24 * 60  # 1-minute bars
        
        # Base price
        base_price = {'SOL': 100, 'BTC': 45000, 'ETH': 2500}[symbol]
        
        # Generate returns with multiple cycles
        t = np.arange(n_points)
        
        # Daily cycle (1440 minutes)
        daily_cycle = 0.0005 * np.sin(2 * np.pi * t / 1440)
        
        # Weekly cycle (7 days)
        weekly_cycle = 0.001 * np.sin(2 * np.pi * t / (7 * 1440))
        
        # Monthly cycle (30 days)
        monthly_cycle = 0.002 * np.sin(2 * np.pi * t / (30 * 1440))
        
        # Random noise
        noise = np.random.normal(0, 0.001, n_points)
        
        # Combine cycles
        returns = daily_cycle + weekly_cycle + monthly_cycle + noise
        
        # Add volatility clustering
        volatility = np.random.exponential(0.005, n_points)
        returns = returns * (1 + volatility)
        
        # Create price series
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_points, freq='1min'),
            'symbol': symbol,
            'close': prices,
            'open': prices * (1 + np.random.normal(0, 0.0005, n_points)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_points))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_points))),
            'volume': np.random.lognormal(10, 0.5, n_points)
        })
        
        # Fix OHLC relationships
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Technical indicators with cyclical components
        df['rsi'] = 50 + 20 * np.sin(2 * np.pi * t / 1440) + np.random.normal(0, 5, n_points)
        df['rsi'] = np.clip(df['rsi'], 0, 100)
        
        df['macd'] = 0.1 * np.sin(2 * np.pi * t / (3 * 1440)) + np.random.normal(0, 0.05, n_points)
        df['bb_upper'] = df['close'] * 1.015
        df['bb_lower'] = df['close'] * 0.985
        df['atr'] = df['close'] * 0.015
        df['volume_sma'] = df['volume'].rolling(20).mean().fillna(df['volume'])
        df['price_change'] = df['close'].pct_change().fillna(0)
        df['volume_change'] = df['volume'].pct_change().fillna(0)
        
        # Stochastic oscillators with cycles
        df['stoch_k'] = 50 + 30 * np.sin(2 * np.pi * t / 720) + np.random.normal(0, 10, n_points)
        df['stoch_k'] = np.clip(df['stoch_k'], 0, 100)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean().fillna(df['stoch_k'])
        
        data_frames.append(df)
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    logger.info(f"📈 Generated {len(combined_df)} long-term samples")
    
    return combined_df

def train_timesnet(window=1024, d_model=128, heads=8, layers=6, 
                  symbols=['SOL', 'BTC', 'ETH'], days=90, epochs=100):
    """Train TimesNet model"""
    
    logger.info("🚀 Starting TimesNet training")
    logger.info(f"📊 Window: {window}, Model: {d_model}d, Heads: {heads}, Layers: {layers}")
    
    # Generate data
    df = generate_long_term_data(symbols, days)
    
    # Create dataset
    dataset = TimesNetDataset(df, sequence_length=window)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimesNet(
        input_dim=len(dataset.features),
        d_model=d_model,
        num_heads=heads,
        num_layers=layers,
        max_seq_len=window
    ).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).squeeze()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).squeeze()
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss/len(val_loader):.4f}, "
                       f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"timesnet_{window}_{d_model}_{heads}_{layers}"
    model_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    
    for symbol in symbols:
        model_path = f'models/timesnet_{symbol}_{timestamp}_{model_hash}.pt'
        
        checkpoint = {
            'state_dict': best_model_state,
            'config': {
                'input_dim': len(dataset.features),
                'd_model': d_model,
                'num_heads': heads,
                'num_layers': layers,
                'max_seq_len': window,
                'num_classes': 3
            },
            'model_hash': model_hash,
            'symbols': symbols,
            'best_val_acc': best_val_acc,
            'timestamp': timestamp,
            'scaler': dataset.scaler
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"💾 Model saved: {model_path}")
    
    logger.info(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    return model, best_val_acc

def main():
    parser = argparse.ArgumentParser(description='Train TimesNet model')
    parser.add_argument('--window', type=int, default=1024, help='Sequence window size')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--symbols', default='SOL,BTC,ETH', help='Comma-separated symbols')
    parser.add_argument('--days', type=int, default=90, help='Days of data')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(',')
    
    # Train model
    model, accuracy = train_timesnet(
        window=args.window,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        symbols=symbols,
        days=args.days,
        epochs=args.epochs
    )
    
    logger.info("🎉 TimesNet training complete!")
    logger.info(f"🎯 Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 