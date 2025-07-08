#!/usr/bin/env python3
"""
TSA-MAE Training Script for RTX 2080 Ti (11GB VRAM)
Optimized for 4-hour crypto 1-minute bar sequences
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
import hashlib
from datetime import datetime
import argparse
from pathlib import Path

# Add models to path
sys.path.append('models')
from optimized_tsa_mae import OptimizedTSAMAE, OptimizedCryptoDataset, OptimizedTrainer

# GPU-optimized hyperparameters for RTX 2080 Ti
WINDOW = 240        # 4h of 1-min bars
EMBED_DIM = 64      # Compact but effective
DEPTH = 4           # Encoder layers
NUM_HEADS = 8       # Attention heads
MLP_RATIO = 2.0     # MLP expansion
MASK_RATIO = 0.65   # Higher masking for better learning
BATCH_SIZE = 192    # Fits 11GB with fp16
EPOCHS = 50         # Training epochs
LR = 3e-4           # Learning rate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu_requirements():
    """Verify GPU setup for training"""
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available - RTX 2080 Ti required")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"🖥️  GPU: {gpu_name}")
    logger.info(f"💾 VRAM: {gpu_memory:.1f}GB")
    
    if gpu_memory < 10:
        logger.warning("⚠️  Less than 10GB VRAM detected - may need to reduce batch size")
    
    return True

def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"📊 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        return allocated, reserved
    return 0, 0

def create_model_hash() -> str:
    """Create unique hash for model versioning"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"tsa_mae_{WINDOW}_{EMBED_DIM}_{DEPTH}_{MASK_RATIO}"
    hash_obj = hashlib.sha256(config_str.encode())
    return f"{timestamp}_{hash_obj.hexdigest()[:8]}"

def train_tsa_mae_rtx2080ti(symbols=['SOL', 'BTC', 'ETH'], 
                           months=12,
                           epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           resume_from=None):
    """
    Train TSA-MAE optimized for RTX 2080 Ti
    
    Args:
        symbols: Crypto symbols to train on
        months: Historical data months  
        epochs: Training epochs
        batch_size: Batch size (192 fits 11GB)
        resume_from: Path to resume training
    """
    
    logger.info("🚀 RTX 2080 Ti TSA-MAE Training")
    logger.info("=" * 50)
    
    # Check GPU
    if not check_gpu_requirements():
        return None
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    # Create model hash for tracking
    model_hash = create_model_hash()
    logger.info(f"🔑 Model Hash: {model_hash}")
    
    # Load dataset
    logger.info(f"📊 Loading {months}-month dataset for {symbols}")
    dataset = OptimizedCryptoDataset(symbols=symbols, months=months, window_size=WINDOW)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )
    
    logger.info(f"📈 Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    logger.info(f"📉 Val: {len(val_dataset):,} samples, {len(val_loader)} batches")
    
    # Initialize model with exact specs
    model = OptimizedTSAMAE(
        input_dim=15,           # OHLCV + technical features  
        d_model=EMBED_DIM,      # 64-dim embeddings
        nhead=NUM_HEADS,        # 8 attention heads
        num_encoder_layers=DEPTH,  # 4 encoder layers
        num_decoder_layers=2,   # 2 decoder layers
        mask_ratio=MASK_RATIO   # 65% masking
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"🧠 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize trainer
    trainer = OptimizedTrainer(model, device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.95)  # Optimized for transformers
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LR/10
    )
    
    # Resume training if specified
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        logger.info(f"📂 Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"🔄 Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info(f"🔥 Starting training for {epochs} epochs")
    logger.info(f"⚙️  Batch size: {batch_size}, Window: {WINDOW}, Mask ratio: {MASK_RATIO}")
    
    best_val_loss = float('inf')
    training_start = datetime.now()
    
    for epoch in range(start_epoch, epochs):
        epoch_start = datetime.now()
        
        logger.info(f"\n📅 Epoch {epoch+1}/{epochs}")
        logger.info("-" * 30)
        
        # Monitor memory before training
        allocated, reserved = monitor_gpu_memory()
        
        # Training phase
        train_loss = trainer.train_epoch(train_loader, optimizer)
        
        # Validation phase  
        val_loss = trainer.validate_epoch(val_loader)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Calculate epoch time
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        # Log results
        logger.info(f"📊 Train Loss: {train_loss:.6f}")
        logger.info(f"📊 Val Loss: {val_loss:.6f}")
        logger.info(f"📊 Learning Rate: {current_lr:.2e}")
        logger.info(f"⏱️  Epoch Time: {epoch_time:.1f}s")
        
        # Monitor GPU utilization
        allocated, reserved = monitor_gpu_memory()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f'models/encoder_{model_hash}_best.pt'
            
            # Save encoder specifically for downstream use
            encoder_checkpoint = {
                'state_dict': model.state_dict(),
                'encoder_state_dict': {k: v for k, v in model.state_dict().items() 
                                     if 'encoder' in k or 'input_projection' in k or 'pos_encoding' in k},
                'config': {
                    'window': WINDOW,
                    'embed_dim': EMBED_DIM,
                    'mask_ratio': MASK_RATIO,
                    'input_dim': 15,
                    'd_model': EMBED_DIM,
                    'nhead': NUM_HEADS,
                    'depth': DEPTH
                },
                'model_hash': model_hash,
                'symbols': symbols,
                'months': months,
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(encoder_checkpoint, best_model_path)
            logger.info(f"💾 Best model saved: {best_model_path}")
            logger.info(f"🏆 New best validation loss: {best_val_loss:.6f}")
        
        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'models/tsa_mae_checkpoint_ep{epoch+1}_{model_hash}.pt'
            trainer.save_checkpoint(epoch, optimizer, checkpoint_path, model_hash)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Early stopping check (optional)
        if len(trainer.val_losses) > 10:
            recent_losses = trainer.val_losses[-10:]
            if all(loss >= best_val_loss for loss in recent_losses[-5:]):
                logger.info("📈 Validation loss plateaued - consider early stopping")
    
    # Training complete
    training_time = (datetime.now() - training_start).total_seconds()
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 TSA-MAE Training Complete!")
    logger.info("=" * 50)
    logger.info(f"⏱️  Total Training Time: {training_time/3600:.2f} hours")
    logger.info(f"🏆 Best Validation Loss: {best_val_loss:.6f}")
    logger.info(f"🔑 Model Hash: {model_hash}")
    logger.info(f"💾 Best Model: models/encoder_{model_hash}_best.pt")
    logger.info(f"📊 Peak GPU Memory: {max(trainer.gpu_memory_usage):.2f}GB")
    
    return model, model_hash, best_val_loss

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='TSA-MAE Training for RTX 2080 Ti')
    
    parser.add_argument('--symbols', nargs='+', default=['SOL', 'BTC', 'ETH'],
                       help='Crypto symbols to train on')
    parser.add_argument('--months', type=int, default=12,
                       help='Historical data months')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size (192 for 11GB GPU)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with reduced data')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        logger.info("🧪 Quick test mode - reduced parameters")
        args.months = 1
        args.epochs = 5
        args.batch_size = 32
    
    # Train model
    model, model_hash, best_loss = train_tsa_mae_rtx2080ti(
        symbols=args.symbols,
        months=args.months,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume_from=args.resume
    )
    
    if model:
        logger.info(f"🚀 Training successful!")
        logger.info(f"📁 Encoder ready for TabNet/PPO fine-tuning")
        logger.info(f"🎯 Next: python train_tabnet.py --use_encoder encoder_{model_hash}_best.pt")
    else:
        logger.error("❌ Training failed")

if __name__ == "__main__":
    main() 