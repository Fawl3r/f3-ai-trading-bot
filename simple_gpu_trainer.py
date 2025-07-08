#!/usr/bin/env python3
"""
Simple GPU TSA-MAE Trainer for RTX 2080 Ti
Minimal implementation that works out of the box
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import logging
import hashlib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RTX 2080 Ti optimized parameters
BATCH_SIZE = 128    # Conservative for 11GB
EMBED_DIM = 64
DEPTH = 4
HEADS = 8
WINDOW = 240
MASK_RATIO = 0.65

class SimpleCryptoDataset(Dataset):
    """Simple crypto dataset that works"""
    
    def __init__(self, months=1):  # Start with 1 month for testing
        self.window_size = WINDOW
        self.data = self._generate_data(months)
        
    def _generate_data(self, months):
        """Generate simple synthetic crypto data"""
        days = months * 30
        n_points = days * 24 * 60  # 1-minute bars
        
        logger.info(f"Generating {n_points:,} data points for {months} months")
        
        # Simple price walk
        np.random.seed(42)
        returns = np.random.normal(0, 0.0005, n_points)  # 0.05% per minute volatility
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Simple OHLCV
        data = np.zeros((n_points - WINDOW, WINDOW, 5))  # [samples, time, features]
        
        for i in range(n_points - WINDOW):
            window_prices = prices[i:i + WINDOW]
            
            # Create OHLCV features
            ohlcv = np.column_stack([
                window_prices,  # close
                window_prices * 1.001,  # high
                window_prices * 0.999,  # low
                np.random.exponential(1000, WINDOW),  # volume
                np.random.normal(50, 10, WINDOW)  # rsi
            ])
            
            data[i] = ohlcv
        
        # Normalize
        scaler = StandardScaler()
        n_samples, n_time, n_features = data.shape
        data_reshaped = data.reshape(-1, n_features)
        data_normalized = scaler.fit_transform(data_reshaped)
        data = data_normalized.reshape(n_samples, n_time, n_features)
        
        logger.info(f"Dataset created: {data.shape}")
        return torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class SimpleTransformerEncoder(nn.Module):
    """Simple transformer encoder"""
    
    def __init__(self, input_dim=5, d_model=EMBED_DIM, nhead=HEADS, num_layers=DEPTH):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(WINDOW, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        logger.info(f"Encoder: {d_model}d, {nhead}h, {num_layers}L")
    
    def forward(self, x):
        # x: [batch, seq, features]
        x = self.input_projection(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        return x

class SimpleTSAMAE(nn.Module):
    """Simple TSA-MAE for quick testing"""
    
    def __init__(self, input_dim=5):
        super().__init__()
        
        self.encoder = SimpleTransformerEncoder(input_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * 0.02)
        self.decoder = nn.Linear(EMBED_DIM, input_dim)
        self.mask_ratio = MASK_RATIO
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model parameters: {total_params:,}")
    
    def random_mask(self, x):
        """Simple random masking"""
        B, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        # Random indices to keep
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Create mask (1 = masked, 0 = kept)
        mask = torch.ones([B, L], device=x.device)
        mask.scatter_(1, ids_keep, 0)
        
        return mask
    
    def forward(self, x):
        # Get mask
        mask = self.random_mask(x)
        
        # Encode
        encoded = self.encoder(x)
        
        # Simple reconstruction (decode all positions)
        reconstructed = self.decoder(encoded)
        
        return reconstructed, mask

def gpu_memory_info():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "No GPU"

def train_simple_mae():
    """Train simple MAE on GPU"""
    
    logger.info("🚀 Simple GPU TSA-MAE Training")
    logger.info("=" * 40)
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("❌ No GPU available")
        return None
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"🖥️  GPU: {gpu_name}")
    logger.info(f"💾 VRAM: {gpu_memory:.1f}GB")
    
    # Dataset
    logger.info("📊 Creating dataset...")
    dataset = SimpleCryptoDataset(months=1)  # Start small
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        pin_memory=True
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = SimpleTSAMAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = GradScaler()
    
    logger.info(gpu_memory_info())
    
    # Training loop
    epochs = 10
    best_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                pred, mask = model(batch)
                
                # MSE loss on masked tokens only
                loss = F.mse_loss(pred, batch, reduction='none')
                loss = loss.mean(dim=-1)  # Average over features
                loss = (loss * mask).sum() / mask.sum()  # Only masked positions
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
                logger.info(gpu_memory_info())
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                
                with autocast():
                    pred, mask = model(batch)
                    loss = F.mse_loss(pred, batch, reduction='none')
                    loss = loss.mean(dim=-1)
                    loss = (loss * mask).sum() / mask.sum()
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            
            # Create model hash
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_hash = hashlib.sha256(f"simple_mae_{timestamp}".encode()).hexdigest()[:8]
            
            # Save encoder for downstream use
            torch.save({
                'encoder_state_dict': model.encoder.state_dict(),
                'full_model_state_dict': model.state_dict(),
                'config': {
                    'input_dim': 5,
                    'd_model': EMBED_DIM,
                    'nhead': HEADS,
                    'num_layers': DEPTH,
                    'window': WINDOW,
                    'mask_ratio': MASK_RATIO
                },
                'model_hash': model_hash,
                'best_val_loss': best_loss,
                'timestamp': timestamp
            }, f'models/encoder_{model_hash}_simple.pt')
            
            logger.info(f"💾 Best model saved: encoder_{model_hash}_simple.pt")
        
        torch.cuda.empty_cache()
    
    logger.info(f"\n✅ Training complete! Best loss: {best_loss:.6f}")
    logger.info(f"📁 Model ready for downstream fine-tuning")
    
    return model, model_hash

if __name__ == "__main__":
    # Quick GPU test
    logger.info("🧪 Running Simple GPU TSA-MAE Test")
    
    model, model_hash = train_simple_mae()
    
    if model:
        print(f"\n🎉 Success! GPU training completed")
        print(f"🔑 Model hash: {model_hash}")
        print(f"📊 Ready for PPO fine-tuning and bandit registration")
        print(f"🚀 Your RTX 2080 Ti is working perfectly!")
    else:
        print("❌ Training failed") 