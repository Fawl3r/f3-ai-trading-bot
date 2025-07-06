#!/usr/bin/env python3
"""
Parabolic Pattern Recognition Transformer
Specialized transformer for detecting burst and fade patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ParabolicTransformer(nn.Module):
    """Transformer model for parabolic pattern recognition"""
    
    def __init__(self, 
                 input_dim: int = 50,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 num_classes: int = 3):  # burst, fade, normal
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Pattern-specific attention heads
        self.burst_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.fade_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 for burst + fade attention
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Transpose for transformer (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Pattern-specific attention
        burst_attn, _ = self.burst_attention(encoded, encoded, encoded)
        fade_attn, _ = self.fade_attention(encoded, encoded, encoded)
        
        # Global average pooling
        burst_pooled = burst_attn.mean(dim=0)  # (batch_size, d_model)
        fade_pooled = fade_attn.mean(dim=0)    # (batch_size, d_model)
        
        # Combine features
        combined = torch.cat([burst_pooled, fade_pooled], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        confidence = self.confidence_head(combined)
        
        return logits, confidence

class ParabolicPatternTrainer:
    """Trainer for parabolic pattern recognition"""
    
    def __init__(self, 
                 model_params: Dict = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.model_params = model_params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.label_map = {'normal': 0, 'burst': 1, 'fade': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
    def prepare_sequences(self, df: pd.DataFrame, 
                         sequence_length: int = 50,
                         features: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        
        if features is None:
            # Select relevant features for parabolic detection
            features = [
                'roc_accel_smooth', 'vwap_gap', 'rsi_14', 'delta_volume',
                'obi_smooth', 'body_ratio', 'atr_14', 'volume',
                'ema_200_slope', 'price_vs_ema200'
            ]
        
        # Add basic price features
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(2)
        df['price_change_5'] = df['close'].pct_change(5)
        df['volume_change'] = df['volume'].pct_change()
        
        # Ensure all features exist
        available_features = [f for f in features if f in df.columns]
        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            logger.warning(f"Missing features: {missing}")
            features = available_features
        
        # Create sequences
        sequences = []
        labels = []
        
        for i in range(sequence_length, len(df)):
            # Get sequence
            seq_data = df.iloc[i-sequence_length:i][features].values
            sequences.append(seq_data)
            
            # Get label from signal_type
            signal_type = df.iloc[i]['signal_type']
            if 'burst' in signal_type:
                label = 'burst'
            elif 'fade' in signal_type:
                label = 'fade'
            else:
                label = 'normal'
            
            labels.append(self.label_map[label])
        
        return np.array(sequences), np.array(labels)
    
    def create_balanced_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create balanced dataset for training"""
        
        # Count samples per class
        unique, counts = np.unique(y, return_counts=True)
        min_count = min(counts)
        
        # Balance dataset
        balanced_X = []
        balanced_y = []
        
        for class_label in unique:
            class_indices = np.where(y == class_label)[0]
            selected_indices = np.random.choice(class_indices, min_count, replace=False)
            
            balanced_X.append(X[selected_indices])
            balanced_y.append(y[selected_indices])
        
        balanced_X = np.concatenate(balanced_X)
        balanced_y = np.concatenate(balanced_y)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(balanced_X))
        balanced_X = balanced_X[shuffle_idx]
        balanced_y = balanced_y[shuffle_idx]
        
        return balanced_X, balanced_y
    
    def train(self, df: pd.DataFrame, 
              sequence_length: int = 50,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              validation_split: float = 0.2) -> Dict:
        """Train the parabolic transformer"""
        
        logger.info("Preparing training data...")
        
        # Prepare sequences
        X, y = self.prepare_sequences(df, sequence_length)
        
        # Balance dataset
        X, y = self.create_balanced_dataset(X, y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Create model
        input_dim = X.shape[-1]
        self.model = ParabolicTransformer(
            input_dim=input_dim,
            **self.model_params
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = torch.FloatTensor(X_train[i:i+batch_size]).to(self.device)
                batch_y = torch.LongTensor(y_train[i:i+batch_size]).to(self.device)
                
                optimizer.zero_grad()
                logits, confidence = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_X = torch.FloatTensor(X_val[i:i+batch_size]).to(self.device)
                    batch_y = torch.LongTensor(y_val[i:i+batch_size]).to(self.device)
                    
                    logits, confidence = self.model(batch_X)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(logits.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            train_loss /= (len(X_train) // batch_size)
            val_loss /= (len(X_val) // batch_size)
            val_accuracy = correct / total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Training results
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_val_accuracy': val_accuracies[-1],
            'best_val_accuracy': max(val_accuracies)
        }
        
        logger.info(f"Training complete. Best validation accuracy: {results['best_val_accuracy']:.4f}")
        
        return results
    
    def predict(self, df: pd.DataFrame, sequence_length: int = 50) -> np.ndarray:
        """Make predictions on new data"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare sequences
        X, _ = self.prepare_sequences(df, sequence_length)
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Predict
        self.model.eval()
        predictions = []
        confidences = []
        
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(X_scaled), batch_size):
                batch_X = torch.FloatTensor(X_scaled[i:i+batch_size]).to(self.device)
                logits, confidence = self.model(batch_X)
                
                probs = F.softmax(logits, dim=1)
                predictions.extend(probs.cpu().numpy())
                confidences.extend(confidence.cpu().numpy())
        
        return np.array(predictions), np.array(confidences)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_params': self.model_params,
            'scaler': self.scaler,
            'label_map': self.label_map
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_params = checkpoint['model_params']
        self.scaler = checkpoint['scaler']
        self.label_map = checkpoint['label_map']
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Recreate model
        input_dim = self.scaler.n_features_in_
        self.model = ParabolicTransformer(
            input_dim=input_dim,
            **self.model_params
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from {filepath}")

def main():
    """Test the parabolic transformer"""
    
    # Import parabolic detector
    import sys
    sys.path.append('.')
    from features.parabola_detector import ParabolaDetector
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=2000, freq='H')
    
    # Simulate price data with parabolic patterns
    price = 100
    prices = [price]
    volumes = []
    
    for i in range(1999):
        # Add parabolic behavior
        if i % 300 == 0:  # Parabolic move
            change = np.random.normal(0.03, 0.01)
        elif i % 150 == 0:  # Exhaustion
            change = np.random.normal(-0.02, 0.01)
        else:
            change = np.random.normal(0.001, 0.01)
        
        price *= (1 + change)
        prices.append(price)
        volumes.append(np.random.exponential(1000))
    
    # Create OHLCV data
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': volumes
    })
    
    # Generate parabolic signals
    detector = ParabolaDetector()
    df = detector.process_complete_analysis(df)
    
    # Train transformer
    trainer = ParabolicPatternTrainer()
    results = trainer.train(df, epochs=50)
    
    print("Training Results:")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    
    # Test predictions
    predictions, confidences = trainer.predict(df)
    
    print("\nPrediction Distribution:")
    pred_labels = np.argmax(predictions, axis=1)
    unique, counts = np.unique(pred_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"{trainer.reverse_label_map[label]}: {count}")

if __name__ == "__main__":
    main() 