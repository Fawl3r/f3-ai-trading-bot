#!/usr/bin/env python3
"""
Optimized TSA-MAE for RTX 2080 Ti (11GB VRAM)
Tuned parameters for maximum performance on crypto 1-minute bars
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime, timedelta
import logging
import json
import os
import hashlib
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import duckdb
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized hyperparameters for RTX 2080 Ti (11GB)
WINDOW = 240        # 4h of 1-min bars
EMBED_DIM = 64      # Compact embedding
DEPTH = 4           # Encoder layers
NUM_HEADS = 8       # Attention heads
MLP_RATIO = 2.0     # MLP expansion ratio
MASK_RATIO = 0.65   # Masking ratio
BATCH_SIZE = 192    # Fits 11GB with fp16
EPOCHS = 50         # Training epochs
LR = 3e-4           # Learning rate

class OptimizedCryptoDataset(Dataset):
    """Optimized dataset for crypto 1-minute bars"""
    
    def __init__(self, symbols: List[str] = ['SOL', 'BTC', 'ETH'], 
                 months: int = 12, window_size: int = WINDOW):
        self.symbols = symbols
        self.window_size = window_size
        self.data = self._load_data(months)
        self.sequences = self._create_sequences()
        
        logger.info(f"Dataset created: {len(self.sequences)} sequences of {window_size} bars")
    
    def _load_data(self, months: int) -> pd.DataFrame:
        """Load and prepare crypto data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months)
        
        all_data = []
        
        for symbol in self.symbols:
            logger.info(f"Loading {months} months of {symbol} data...")
            
            # Generate realistic 1-minute crypto data
            dates = pd.date_range(start_date, end_date, freq='1min')
            n_points = len(dates)
            
            # Base price
            base_prices = {'SOL': 100, 'BTC': 45000, 'ETH': 3000}
            base_price = base_prices[symbol]
            
            # Generate realistic price movements
            np.random.seed(42 + hash(symbol) % 1000)
            
            # Base returns with realistic volatility
            base_vol = 0.015 / np.sqrt(1440)  # 1.5% daily vol -> 1-min
            returns = np.random.normal(0, base_vol, n_points)
            returns = np.clip(returns, -0.01, 0.01)  # Clip extreme moves
            
            # Add momentum and mean reversion
            momentum = np.convolve(returns, np.ones(10)/10, mode='same')
            mean_revert = -0.1 * np.cumsum(returns[-100:].mean() - returns)
            returns += 0.3 * momentum + 0.1 * mean_revert
            
            # Calculate prices
            log_prices = np.log(base_price) + np.cumsum(returns)
            close_prices = np.exp(log_prices)
            
            # Generate OHLC with realistic spreads
            spread = 0.001  # 0.1% spread
            high_prices = close_prices * (1 + np.abs(np.random.normal(0, spread/2, n_points)))
            low_prices = close_prices * (1 - np.abs(np.random.normal(0, spread/2, n_points)))
            open_prices = np.roll(close_prices, 1)
            open_prices[0] = close_prices[0]
            
            # Ensure OHLC consistency
            high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
            low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
            
            # Volume with realistic patterns
            base_volume = {'SOL': 1e5, 'BTC': 1e6, 'ETH': 5e5}[symbol]
            vol_returns = np.abs(returns) * 50 + np.random.exponential(0.5, n_points)
            volumes = base_volume * vol_returns
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'symbol': symbol,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes,
                'returns': returns,
                'log_volume': np.log(volumes + 1),
                'hl_ratio': (high_prices - low_prices) / close_prices,
                'oc_ratio': (close_prices - open_prices) / open_prices
            })
            
            # Add technical features
            df['sma_10'] = df['close'].rolling(10, min_periods=1).mean()
            df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['bb_upper'], df['bb_lower'] = self._calculate_bollinger(df['close'], 20)
            df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
            
            # Clean data
            df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill()
            
            all_data.append(df)
            logger.info(f"Generated {len(df)} 1-min bars for {symbol}")
        
        # Combine and sort
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        
        logger.info(f"Total dataset: {len(combined)} bars across {len(self.symbols)} symbols")
        return combined
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return upper, lower
    
    def _create_sequences(self) -> List[np.ndarray]:
        """Create sliding window sequences"""
        sequences = []
        
        # Features to use
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 'returns',
            'log_volume', 'hl_ratio', 'oc_ratio', 'sma_10', 'sma_50', 
            'rsi', 'bb_upper', 'bb_lower', 'volume_sma'
        ]
        
        # Group by symbol and create sequences
        for symbol in self.symbols:
            symbol_data = self.data[self.data['symbol'] == symbol][feature_cols].values
            
            # Normalize per symbol
            scaler = StandardScaler()
            symbol_data = scaler.fit_transform(symbol_data)
            
            # Create sliding windows
            for i in range(len(symbol_data) - self.window_size + 1):
                sequence = symbol_data[i:i + self.window_size]
                sequences.append(sequence.astype(np.float32))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.sequences[idx])

class OptimizedPositionalEncoding(nn.Module):
    """Memory-efficient positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = WINDOW):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]

class OptimizedTransformerBlock(nn.Module):
    """Memory-optimized transformer block"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class OptimizedTSAMAE(nn.Module):
    """
    Optimized Time Series Masked AutoEncoder for RTX 2080 Ti
    Memory-efficient design with gradient checkpointing
    """
    
    def __init__(self, 
                 input_dim: int = 15,
                 d_model: int = EMBED_DIM,
                 nhead: int = NUM_HEADS,
                 num_encoder_layers: int = DEPTH,
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = None,
                 dropout: float = 0.1,
                 mask_ratio: float = MASK_RATIO):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        
        if dim_feedforward is None:
            dim_feedforward = int(d_model * MLP_RATIO)
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = OptimizedPositionalEncoding(d_model)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            OptimizedTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            OptimizedTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Optimized TSA-MAE: {d_model}d, {nhead}h, {num_encoder_layers}E+{num_decoder_layers}D")
        logger.info(f"Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Efficient random masking"""
        B, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward through encoder with gradient checkpointing"""
        # Embed and add positional encoding
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # Masking
        x_masked, mask, ids_restore = self.random_masking(x)
        
        # Encoder layers with gradient checkpointing
        for layer in self.encoder_layers:
            if self.training:
                x_masked = torch.utils.checkpoint.checkpoint(layer, x_masked)
            else:
                x_masked = layer(x_masked)
        
        return x_masked, mask, ids_restore
    
    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Forward through decoder"""
        B, L_enc, D = x.shape
        L = ids_restore.shape[1]
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(B, L - L_enc, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle
        x_full = torch.gather(x_full, dim=1, 
                             index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # Add positional encoding
        x_full = self.pos_encoding(x_full)
        
        # Decoder layers
        for layer in self.decoder_layers:
            if self.training:
                x_full = torch.utils.checkpoint.checkpoint(layer, x_full)
            else:
                x_full = layer(x_full)
        
        # Output projection
        x_full = self.output_projection(x_full)
        
        return x_full
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass"""
        encoded, mask, ids_restore = self.forward_encoder(x)
        decoded = self.forward_decoder(encoded, ids_restore)
        return decoded, mask
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder embeddings for downstream tasks"""
        with torch.no_grad():
            x = self.input_projection(x)
            x = self.pos_encoding(x)
            
            for layer in self.encoder_layers:
                x = layer(x)
            
            # Global average pooling
            embeddings = x.mean(dim=1)
            
        return embeddings

class OptimizedTrainer:
    """Optimized trainer with mixed precision and monitoring"""
    
    def __init__(self, model: OptimizedTSAMAE, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.scaler = GradScaler()
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.gpu_memory_usage = []
        
        logger.info(f"Trainer initialized on {device}")
        logger.info(f"Model memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    def masked_mse_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculate MSE loss only on masked portions"""
        # mask: 1 = remove (calculate loss), 0 = keep (ignore)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Average over features
        
        # Apply mask and normalize
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Train one epoch with mixed precision"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                pred, mask = self.model(batch)
                loss = self.masked_mse_loss(pred, batch, mask)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 50 == 0:
                memory_gb = torch.cuda.max_memory_allocated() / 1e9
                self.gpu_memory_usage.append(memory_gb)
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, "
                           f"Loss: {loss.item():.6f}, "
                           f"GPU: {memory_gb:.2f}GB")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device, non_blocking=True)
                
                with autocast():
                    pred, mask = self.model(batch)
                    loss = self.masked_mse_loss(pred, batch, mask)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer, 
                       filepath: str, model_hash: str = None):
        """Save optimized checkpoint"""
        if model_hash is None:
            model_hash = hashlib.sha256(str(epoch).encode()).hexdigest()[:8]
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'input_dim': self.model.input_dim,
                'd_model': self.model.d_model,
                'mask_ratio': self.model.mask_ratio,
                'window_size': WINDOW
            },
            'model_hash': model_hash,
            'gpu_memory_peak': max(self.gpu_memory_usage) if self.gpu_memory_usage else 0
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath} (SHA: {model_hash})")

def train_optimized_tsa_mae():
    """Main training function optimized for RTX 2080 Ti"""
    
    logger.info("🚀 Starting Optimized TSA-MAE Training")
    logger.info(f"Target GPU: RTX 2080 Ti (11GB)")
    logger.info(f"Hyperparameters: W={WINDOW}, E={EMBED_DIM}, D={DEPTH}, B={BATCH_SIZE}")
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - GPU required for this training")
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    # Create dataset
    logger.info("📊 Loading crypto dataset...")
    dataset = OptimizedCryptoDataset(symbols=['SOL', 'BTC', 'ETH'], months=12)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize model
    model = OptimizedTSAMAE(input_dim=15)
    trainer = OptimizedTrainer(model, device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    
    # Training loop
    logger.info(f"🔥 Starting training for {EPOCHS} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        epoch_start = datetime.now()
        
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer)
        
        # Validate
        val_loss = trainer.validate_epoch(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Metrics
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        logger.info(f"GPU Memory: {max_memory:.2f}GB, Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_hash = hashlib.sha256(f"tsa_mae_{epoch}_{val_loss:.6f}".encode()).hexdigest()[:8]
            trainer.save_checkpoint(
                epoch, optimizer, 
                f'models/encoder_{datetime.now().strftime("%Y%m%d")}_{model_hash}.pt',
                model_hash
            )
        
        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                epoch, optimizer,
                f'models/tsa_mae_checkpoint_ep{epoch+1}.pt'
            )
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    logger.info("✅ Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Peak GPU memory: {max(trainer.gpu_memory_usage):.2f}GB")
    
    return model, trainer

if __name__ == "__main__":
    # Train optimized TSA-MAE
    model, trainer = train_optimized_tsa_mae()
    
    print("🧠 Optimized TSA-MAE Training Complete!")
    print("📊 Ready for downstream fine-tuning!")
    print("🎯 Encoder saved with SHA hash for tracking!") 