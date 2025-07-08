#!/usr/bin/env python3
"""
Time Series Masked AutoEncoder (TSA-MAE)
Pre-trains on 12 months of SOL/BTC/ETH data to learn market microstructure embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import logging
import json
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataset(Dataset):
    """Dataset for market time series data"""
    
    def __init__(self, data_df: pd.DataFrame, sequence_length: int = 60, 
                 features: List[str] = None):
        self.data = data_df
        self.sequence_length = sequence_length
        
        if features is None:
            self.features = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr',
                'volume_sma', 'price_change', 'volume_change'
            ]
        else:
            self.features = features
        
        # Normalize features
        self.scaler = StandardScaler()
        self.normalized_data = self.scaler.fit_transform(
            self.data[self.features].values
        )
        
        # Create sequences
        self.sequences = []
        for i in range(len(self.normalized_data) - sequence_length + 1):
            self.sequences.append(self.normalized_data[i:i + sequence_length])
        
        logger.info(f"Created {len(self.sequences)} sequences of length {sequence_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        return sequence

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesMAE(nn.Module):
    """
    Time Series Masked AutoEncoder
    
    Learns dense embeddings from market microstructure by reconstructing
    randomly masked portions of the time series.
    """
    
    def __init__(self, 
                 input_dim: int = 13,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 mask_ratio: float = 0.25):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"TSA-MAE initialized: {d_model}d, {nhead}h, {num_encoder_layers}+{num_decoder_layers}L")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def random_masking(self, x, mask_ratio=None):
        """
        Random masking of time series patches
        
        Args:
            x: [batch_size, seq_len, input_dim]
            mask_ratio: ratio of patches to mask
        
        Returns:
            x_masked: masked input
            mask: binary mask (1 = keep, 0 = remove)
            ids_restore: indices to restore original order
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        batch_size, seq_len, input_dim = x.shape
        len_keep = int(seq_len * (1 - mask_ratio))
        
        # Generate random noise for each sample
        noise = torch.rand(batch_size, seq_len, device=x.device)
        
        # Sort noise to get random indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, input_dim))
        
        # Generate binary mask: 1 = keep, 0 = remove
        mask = torch.ones([batch_size, seq_len], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio=None):
        """Forward pass through encoder"""
        # Embed patches
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Masking: length -> length * mask_ratio
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Apply transformer encoder
        encoded = self.encoder(x_masked)
        
        return encoded, mask, ids_restore
    
    def forward_decoder(self, encoded, ids_restore):
        """Forward pass through decoder"""
        batch_size, len_keep, d_model = encoded.shape
        seq_len = ids_restore.shape[1]
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(batch_size, seq_len - len_keep, 1)
        x_full = torch.cat([encoded, mask_tokens], dim=1)
        
        # Unshuffle to restore original order
        x_full = torch.gather(x_full, dim=1, 
                             index=ids_restore.unsqueeze(-1).repeat(1, 1, d_model))
        
        # Add positional encoding
        x_full = self.pos_encoding(x_full)
        
        # Apply transformer decoder (self-attention only)
        decoded = self.decoder(x_full, x_full)
        
        # Project to input dimension
        reconstructed = self.output_projection(decoded)
        
        return reconstructed
    
    def forward(self, x, mask_ratio=None):
        """Full forward pass"""
        encoded, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        reconstructed = self.forward_decoder(encoded, ids_restore)
        return reconstructed, mask
    
    def get_embeddings(self, x):
        """Get embeddings for downstream tasks"""
        with torch.no_grad():
            # Project input
            x = self.input_projection(x)
            x = self.pos_encoding(x)
            
            # Encode without masking
            encoded = self.encoder(x)
            
            # Global average pooling
            embeddings = encoded.mean(dim=1)  # [batch_size, d_model]
            
        return embeddings

class TSAMAETrainer:
    """Trainer for TSA-MAE"""
    
    def __init__(self, model: TimeSeriesMAE, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: DataLoader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            
            # Forward pass
            reconstructed, mask = self.model(batch)
            
            # Calculate loss only on masked portions
            loss = self.calculate_masked_loss(batch, reconstructed, mask, criterion)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self, dataloader: DataLoader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Forward pass
                reconstructed, mask = self.model(batch)
                
                # Calculate loss
                loss = self.calculate_masked_loss(batch, reconstructed, mask, criterion)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def calculate_masked_loss(self, target, pred, mask, criterion):
        """Calculate loss only on masked portions"""
        # mask: 1 = remove, 0 = keep
        # We want to calculate loss only on removed (masked) portions
        mask_expanded = mask.unsqueeze(-1).expand_as(target)
        
        # Apply mask
        target_masked = target * mask_expanded
        pred_masked = pred * mask_expanded
        
        # Calculate MSE loss
        loss = criterion(pred_masked, target_masked)
        
        # Normalize by number of masked elements
        loss = loss * mask_expanded.numel() / mask_expanded.sum()
        
        return loss
    
    def save_checkpoint(self, epoch: int, optimizer, loss: float, filepath: str):
        """Save training checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str, optimizer=None):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint['epoch'], checkpoint['loss']

def load_historical_data(symbols: List[str] = ['SOL', 'BTC', 'ETH'], 
                        months: int = 12) -> pd.DataFrame:
    """Load 12 months of historical data for specified symbols"""
    
    # For demo purposes, generate synthetic data
    # In production, replace with actual data loading from exchange APIs
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months)
    
    all_data = []
    
    for symbol in symbols:
        logger.info(f"Loading {months} months of data for {symbol}")
        
        # Generate synthetic OHLCV data (replace with real data)
        dates = pd.date_range(start_date, end_date, freq='1min')
        n_points = len(dates)
        
        # Base price with realistic movements
        base_price = {'SOL': 100, 'BTC': 45000, 'ETH': 3000}[symbol]
        
        # Generate realistic price movements
        np.random.seed(42 + hash(symbol) % 1000)  # Different seed per symbol
        returns = np.random.normal(0.0001, 0.015, n_points)  # Small returns with 1.5% volatility
        returns = np.clip(returns, -0.05, 0.05)  # Clip extreme values to ±5%
        
        # Calculate cumulative prices using log returns
        log_prices = np.log(base_price) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Ensure no infinite or NaN values
        prices = np.clip(prices, base_price * 0.1, base_price * 10)
        prices = np.nan_to_num(prices, nan=base_price)
        
        # Generate OHLCV with realistic spreads
        spread_factor = 0.002  # 0.2% spread
        
        data = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'open': prices * (1 + np.random.normal(0, spread_factor/2, n_points)),
            'high': prices * (1 + np.abs(np.random.normal(0, spread_factor, n_points))),
            'low': prices * (1 - np.abs(np.random.normal(0, spread_factor, n_points))),
            'close': prices,
            'volume': np.random.lognormal(8, 0.5, n_points),  # More realistic volume
        })
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        # Add technical indicators
        data['rsi'] = 50 + 20 * np.sin(np.arange(n_points) * 0.01) + np.random.normal(0, 5, n_points)
        data['rsi'] = np.clip(data['rsi'], 0, 100)
        
        data['macd'] = np.random.normal(0, 0.5, n_points)
        data['bb_upper'] = data['close'] * 1.02
        data['bb_lower'] = data['close'] * 0.98
        data['atr'] = data['close'] * 0.02
        data['volume_sma'] = data['volume'].rolling(20, min_periods=1).mean()
        data['price_change'] = data['close'].pct_change().fillna(0)
        data['volume_change'] = data['volume'].pct_change().fillna(0)
        
        # Clean up any remaining NaN or infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.bfill().ffill()  # Use bfill() and ffill() instead of deprecated method
        
        # Final validation
        for col in data.select_dtypes(include=[np.number]).columns:
            if col != 'timestamp':
                data[col] = np.clip(data[col], -1e10, 1e10)  # Clip extreme values
                data[col] = np.nan_to_num(data[col])  # Replace any remaining NaN/inf
        
        all_data.append(data)
        logger.info(f"Loaded {len(data)} data points for {symbol}")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Total combined data points: {len(combined_data)}")
    return combined_data

def pretrain_tsa_mae(symbols: List[str] = ['SOL', 'BTC', 'ETH'],
                     months: int = 12,
                     epochs: int = 100,
                     batch_size: int = 64,
                     learning_rate: float = 1e-4,
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Pre-train TSA-MAE on 12 months of SOL/BTC/ETH data
    """
    
    logger.info("🧠 Starting TSA-MAE Pre-training")
    logger.info(f"Symbols: {symbols}, Months: {months}, Device: {device}")
    
    # Load historical data
    logger.info("📊 Loading historical market data...")
    data = load_historical_data(symbols, months)
    
    # Create dataset
    dataset = MarketDataset(data, sequence_length=60)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    model = TimeSeriesMAE(
        input_dim=len(dataset.features),
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=3,
        mask_ratio=0.25
    )
    
    # Initialize trainer
    trainer = TSAMAETrainer(model, device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training loop
    logger.info("🚀 Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        
        # Validate
        val_loss = trainer.validate_epoch(val_loader, criterion)
        
        # Update learning rate
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                epoch, optimizer, val_loss,
                'models/tsa_mae_best.pt'
            )
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                epoch, optimizer, val_loss,
                f'models/tsa_mae_epoch_{epoch+1}.pt'
            )
    
    logger.info("✅ TSA-MAE Pre-training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': len(dataset.features),
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 3,
            'mask_ratio': 0.25
        },
        'scaler': dataset.scaler,
        'features': dataset.features,
        'symbols': symbols,
        'training_months': months
    }, 'models/tsa_mae_final.pt')
    
    return model, dataset.scaler, dataset.features

if __name__ == "__main__":
    # Pre-train TSA-MAE
    model, scaler, features = pretrain_tsa_mae(
        symbols=['SOL', 'BTC', 'ETH'],
        months=12,
        epochs=50,  # Reduced for demo
        batch_size=32,
        learning_rate=1e-4
    )
    
    print("🧠 TSA-MAE Pre-training Complete!")
    print("📁 Model saved to: models/tsa_mae_final.pt")
    print("🚀 Ready for PPO fine-tuning!") 