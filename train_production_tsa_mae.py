#!/usr/bin/env python3
"""
Production TSA-MAE Training for RTX 2080 Ti
Follows exact specifications: 240 window, 192 batch, 65% mask, 4h training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import logging
import hashlib
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RTX 2080 Ti Production Parameters (your exact specs)
WINDOW = 240        # 4h of 1-min bars
EMBED_DIM = 64      # Compact embedding dimension
DEPTH = 4           # Encoder layers
NUM_HEADS = 8       # Attention heads
MLP_RATIO = 2.0     # MLP expansion ratio
MASK_RATIO = 0.65   # 65% masking for better learning
BATCH_SIZE = 192    # Fits 11GB with fp16
EPOCHS = 50         # Full training epochs
LR = 3e-4           # Learning rate

class ProductionCryptoDataset(Dataset):
    """Production crypto dataset with realistic 1-minute bars"""
    
    def __init__(self, symbols=['SOL', 'BTC', 'ETH'], months=12):
        self.symbols = symbols
        self.months = months
        self.window_size = WINDOW
        
        logger.info(f"🏗️  Building {months}-month dataset for {symbols}")
        
        # Generate realistic crypto data
        self.data = self._generate_crypto_data()
        
        logger.info(f"📊 Dataset ready: {len(self.data):,} sequences of {WINDOW} bars")
    
    def _generate_crypto_data(self):
        """Generate realistic 1-minute crypto OHLCV data"""
        
        sequences = []
        
        for symbol in self.symbols:
            logger.info(f"📈 Generating {self.months}-month data for {symbol}")
            
            # Time series length
            days = self.months * 30
            n_points = days * 24 * 60  # 1-minute bars
            
            # Base parameters per symbol
            params = {
                'SOL': {'base_price': 100, 'vol': 0.025, 'volume_base': 2e5},
                'BTC': {'base_price': 45000, 'vol': 0.020, 'volume_base': 1e6}, 
                'ETH': {'base_price': 3000, 'vol': 0.022, 'volume_base': 5e5}
            }
            
            p = params[symbol]
            
            # Generate price walks with realistic microstructure
            np.random.seed(42 + hash(symbol) % 1000)
            
            # Returns with time-varying volatility (higher during US/EU hours)
            hourly_vol = np.array([
                0.7 + 0.5 * np.sin((h - 8) * np.pi / 12) if 6 <= h <= 22 else 0.5
                for h in range(24)
            ])
            
            vol_cycle = np.tile(hourly_vol, n_points // (24 * 60) + 1)[:n_points]
            base_vol = p['vol'] / np.sqrt(1440)  # Scale to 1-min
            returns = np.random.normal(0, base_vol * vol_cycle, n_points)
            
            # Add microstructure effects
            momentum = np.convolve(returns, np.ones(5)/5, mode='same')  # 5-min momentum
            mean_revert = -0.05 * np.cumsum(returns - np.mean(returns))  # Mean reversion
            
            returns += 0.1 * momentum + 0.02 * mean_revert
            returns = np.clip(returns, -0.02, 0.02)  # Clip extreme moves
            
            # Generate prices
            log_prices = np.log(p['base_price']) + np.cumsum(returns)
            prices = np.exp(log_prices)
            
            # Generate OHLC with realistic bid-ask spreads
            spread_bps = 5  # 5 bps spread
            spread = spread_bps / 10000
            
            highs = prices * (1 + np.abs(np.random.normal(0, spread, n_points)))
            lows = prices * (1 - np.abs(np.random.normal(0, spread, n_points)))
            opens = np.roll(prices, 1)
            opens[0] = prices[0]
            
            # Ensure OHLC consistency
            highs = np.maximum(highs, np.maximum(opens, prices))
            lows = np.minimum(lows, np.minimum(opens, prices))
            
            # Generate volume with correlation to price moves
            vol_base = p['volume_base']
            vol_multiplier = 1 + 5 * np.abs(returns) + np.random.exponential(0.2, n_points)
            volumes = vol_base * vol_multiplier
            
            # Create feature matrix
            n_features = 15  # OHLCV + technical indicators
            
            for i in range(n_points - WINDOW):
                # Extract window
                window_slice = slice(i, i + WINDOW)
                
                # Raw OHLCV
                o = opens[window_slice]
                h = highs[window_slice] 
                l = lows[window_slice]
                c = prices[window_slice]
                v = volumes[window_slice]
                
                # Technical features
                returns_win = np.diff(np.log(c), prepend=np.log(c[0]))
                log_vol = np.log(v + 1)
                hl_ratio = (h - l) / c
                oc_ratio = (c - o) / o
                
                # Moving averages
                sma_10 = pd.Series(c).rolling(10, min_periods=1).mean().values
                sma_50 = pd.Series(c).rolling(50, min_periods=1).mean().values
                
                # RSI
                delta = pd.Series(c).diff()
                gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).fillna(50).values
                
                # Bollinger bands
                bb_sma = pd.Series(c).rolling(20, min_periods=1).mean()
                bb_std = pd.Series(c).rolling(20, min_periods=1).std()
                bb_upper = (bb_sma + 2 * bb_std).values
                bb_lower = (bb_sma - 2 * bb_std).values
                
                # Volume SMA
                vol_sma = pd.Series(v).rolling(20, min_periods=1).mean().values
                
                # Combine features
                features = np.column_stack([
                    o/c[0], h/c[0], l/c[0], c/c[0], v/vol_base,  # Normalized OHLCV
                    returns_win, log_vol/10, hl_ratio, oc_ratio,  # Derived features
                    sma_10/c, sma_50/c, rsi/100,  # Technical indicators  
                    bb_upper/c, bb_lower/c, vol_sma/vol_base  # More indicators
                ])
                
                # Clean and normalize
                features = np.nan_to_num(features)
                sequences.append(features.astype(np.float32))
        
        logger.info(f"✅ Generated {len(sequences):,} sequences across {len(self.symbols)} symbols")
        return sequences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

class ProductionTSAMAE(nn.Module):
    """Production TSA-MAE optimized for RTX 2080 Ti"""
    
    def __init__(self, input_dim=15):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = EMBED_DIM
        self.mask_ratio = MASK_RATIO
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, EMBED_DIM)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(WINDOW, EMBED_DIM) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=int(EMBED_DIM * MLP_RATIO),
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, DEPTH)
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * 0.02)
        
        # Decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS, 
            dim_feedforward=int(EMBED_DIM * MLP_RATIO),
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, 2)
        
        # Output projection
        self.output_projection = nn.Linear(EMBED_DIM, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 TSA-MAE: {EMBED_DIM}d, {NUM_HEADS}h, {DEPTH}L, {total_params:,} params")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def random_masking(self, x):
        """Random masking for 65% of tokens"""
        B, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        # Generate random indices
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate mask (1 = masked, 0 = kept)
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x):
        # Embed and add positional encoding
        x = self.input_projection(x)
        x = x + self.pos_encoding
        
        # Random masking
        x_masked, mask, ids_restore = self.random_masking(x)
        
        # Encode visible tokens
        encoded = self.encoder(x_masked)
        
        # Reconstruct full sequence
        B, L_enc, D = encoded.shape
        L = ids_restore.shape[1]
        
        # Add mask tokens
        mask_tokens = self.mask_token.repeat(B, L - L_enc, 1)
        x_full = torch.cat([encoded, mask_tokens], dim=1)
        
        # Unshuffle
        x_full = torch.gather(x_full, dim=1, 
                             index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # Add positional encoding again
        x_full = x_full + self.pos_encoding
        
        # Decode
        decoded = self.decoder(x_full)
        
        # Project to input space
        reconstructed = self.output_projection(decoded)
        
        return reconstructed, mask
    
    def get_embeddings(self, x):
        """Extract encoder embeddings for downstream tasks"""
        with torch.no_grad():
            # Embed
            x = self.input_projection(x)
            x = x + self.pos_encoding
            
            # Encode (no masking)
            encoded = self.encoder(x)
            
            # Global average pooling
            embeddings = encoded.mean(dim=1)
            
        return embeddings

def train_production_tsa_mae(symbols=['SOL', 'BTC', 'ETH'], 
                           months=12,
                           epochs=EPOCHS,
                           batch_size=BATCH_SIZE):
    """Train production TSA-MAE with exact RTX 2080 Ti specifications"""
    
    logger.info("🚀 Production TSA-MAE Training")
    logger.info("=" * 60)
    logger.info(f"🎯 Target: RTX 2080 Ti (11GB) - 4 hour training")
    logger.info(f"📊 Config: {WINDOW}W, {EMBED_DIM}D, {DEPTH}L, {batch_size}B, {MASK_RATIO:.0%}M")
    
    # Verify GPU
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA required for production training")
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"🖥️  GPU: {gpu_name}")
    logger.info(f"💾 VRAM: {gpu_memory:.1f}GB")
    
    if gpu_memory < 10:
        logger.warning("⚠️  <10GB VRAM - consider reducing batch size")
    
    # Create dataset
    logger.info(f"📊 Loading {months}-month dataset...")
    dataset = ProductionCryptoDataset(symbols, months)
    
    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Optimized data loaders for 11GB GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    logger.info(f"📈 Train: {len(train_dataset):,} samples ({len(train_loader)} batches)")
    logger.info(f"📉 Val: {len(val_dataset):,} samples ({len(val_loader)} batches)")
    
    # Initialize model
    model = ProductionTSAMAE().to(device)
    
    # Mixed precision scaler
    scaler = GradScaler('cuda')
    
    # Optimizer (AdamW with weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.95)
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LR/10
    )
    
    # Training metrics
    train_losses = []
    val_losses = []
    gpu_memory_peak = 0
    
    # Training loop
    logger.info(f"🔥 Starting {epochs}-epoch training...")
    training_start = datetime.now()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = datetime.now()
        
        logger.info(f"\n📅 Epoch {epoch+1}/{epochs}")
        logger.info("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast('cuda'):
                pred, mask = model(batch)
                
                # Masked MSE loss (only on masked tokens)
                loss = F.mse_loss(pred, batch, reduction='none')
                loss = loss.mean(dim=-1)  # Average over features
                loss = (loss * mask).sum() / mask.sum()  # Only masked positions
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Memory tracking
            current_memory = torch.cuda.memory_allocated() / 1e9
            gpu_memory_peak = max(gpu_memory_peak, current_memory)
            
            # Progress logging
            if batch_idx % 50 == 0:
                logger.info(f"Batch {batch_idx:3d}/{len(train_loader)}, "
                           f"Loss: {loss.item():.6f}, "
                           f"GPU: {current_memory:.2f}GB")
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                
                with autocast('cuda'):
                    pred, mask = model(batch)
                    loss = F.mse_loss(pred, batch, reduction='none')
                    loss = loss.mean(dim=-1)
                    loss = (loss * mask).sum() / mask.sum()
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Epoch summary
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        logger.info(f"📊 Train Loss: {train_loss:.6f}")
        logger.info(f"📊 Val Loss: {val_loss:.6f}")
        logger.info(f"📊 Learning Rate: {current_lr:.2e}")
        logger.info(f"⏱️  Epoch Time: {epoch_time:.1f}s")
        logger.info(f"💾 Peak GPU: {gpu_memory_peak:.2f}GB")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Create model hash for tracking
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_str = f"tsa_mae_{WINDOW}_{EMBED_DIM}_{DEPTH}_{MASK_RATIO}"
            model_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
            
            # Save encoder for downstream use
            checkpoint = {
                'state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'config': {
                    'window': WINDOW,
                    'embed_dim': EMBED_DIM,
                    'mask_ratio': MASK_RATIO,
                    'input_dim': 15,
                    'd_model': EMBED_DIM,
                    'nhead': NUM_HEADS,
                    'depth': DEPTH,
                    'mlp_ratio': MLP_RATIO
                },
                'model_hash': model_hash,
                'symbols': symbols,
                'months': months,
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                'timestamp': timestamp,
                'gpu_peak_memory': gpu_memory_peak
            }
            
            model_path = f'models/encoder_{timestamp}_{model_hash}.pt'
            torch.save(checkpoint, model_path)
            
            logger.info(f"💾 Best encoder saved: {model_path}")
            logger.info(f"🏆 New best validation loss: {best_val_loss:.6f}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Training complete
    training_time = (datetime.now() - training_start).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Production TSA-MAE Training Complete!")
    logger.info("=" * 60)
    logger.info(f"⏱️  Total Time: {training_time/3600:.2f} hours")
    logger.info(f"🏆 Best Val Loss: {best_val_loss:.6f}")
    logger.info(f"💾 Peak GPU Memory: {gpu_memory_peak:.2f}GB / {gpu_memory:.1f}GB")
    logger.info(f"🔑 Model Hash: {model_hash}")
    logger.info(f"📁 Ready for TabNet/PPO fine-tuning")
    
    return model, model_hash, best_val_loss

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Production TSA-MAE for RTX 2080 Ti')
    
    parser.add_argument('--symbols', nargs='+', default=['SOL', 'BTC', 'ETH'],
                       help='Crypto symbols')
    parser.add_argument('--months', type=int, default=12,
                       help='Training data months')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size (192 for 11GB)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test (1 month, 10 epochs)')
    
    args = parser.parse_args()
    
    if args.quick:
        logger.info("🧪 Quick test mode")
        args.months = 1
        args.epochs = 10
        args.batch_size = 64
    
    # Train model
    model, model_hash, best_loss = train_production_tsa_mae(
        symbols=args.symbols,
        months=args.months,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    logger.info(f"\n🚀 Ready for downstream fine-tuning!")
    logger.info(f"🎯 Next commands:")
    logger.info(f"   python train_tabnet.py --use_encoder encoder_*_{model_hash}.pt")
    logger.info(f"   python train_ppo.py --encoder encoder_*_{model_hash}.pt")
    logger.info(f"   python register_policy.py --path models/policy_{model_hash}.pt")

if __name__ == "__main__":
    main() 