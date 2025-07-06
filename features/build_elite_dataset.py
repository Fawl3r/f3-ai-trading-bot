#!/usr/bin/env python3
"""
Elite Feature Engineering Pipeline - Double-Up Trading Bot
Advanced feature extraction for 70%+ win rate AI system
"""

import pandas as pd
import numpy as np
import talib
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EliteFeatureEngineer:
    """Elite-level feature engineering for double-up strategy"""
    
    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
        self.scalers = {}
        self.feature_names = []
        
    def load_candle_data(self, coin: str, data_dir: Path = Path("data/raw")) -> pd.DataFrame:
        """Load candle data from parquet files"""
        candle_file = data_dir / coin / "1m" / f"{coin}_1m_candles.parquet"
        
        if not candle_file.exists():
            raise FileNotFoundError(f"Candle data not found: {candle_file}")
        
        df = pd.read_parquet(candle_file)
        logger.info(f"📊 Loaded {len(df)} candles for {coin}")
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        logger.info("🔧 Calculating technical indicators...")
        
        # Price data
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Trend indicators
        df['sma_10'] = talib.SMA(close, timeperiod=10)
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
        
        # RSI family
        df['rsi_14'] = talib.RSI(close, timeperiod=14)
        df['rsi_7'] = talib.RSI(close, timeperiod=7)
        df['rsi_21'] = talib.RSI(close, timeperiod=21)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)
        df['stoch_rsi'] = talib.STOCHRSI(close)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR and volatility
        df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        df['atr_7'] = talib.ATR(high, low, close, timeperiod=7)
        df['natr'] = talib.NATR(high, low, close)
        
        # Volume indicators
        df['obv'] = talib.OBV(close, volume)
        df['ad'] = talib.AD(high, low, close, volume)
        df['adosc'] = talib.ADOSC(high, low, close, volume)
        
        # Momentum indicators
        df['mom_10'] = talib.MOM(close, timeperiod=10)
        df['roc_10'] = talib.ROC(close, timeperiod=10)
        df['cci'] = talib.CCI(high, low, close)
        df['williams_r'] = talib.WILLR(high, low, close)
        
        # Advanced indicators
        df['adx'] = talib.ADX(high, low, close)
        df['plus_di'] = talib.PLUS_DI(high, low, close)
        df['minus_di'] = talib.MINUS_DI(high, low, close)
        
        # Parabolic SAR
        df['sar'] = talib.SAR(high, low)
        
        # Commodity Channel Index
        df['cci_20'] = talib.CCI(high, low, close, timeperiod=20)
        
        return df
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced price-based features"""
        logger.info("💰 Calculating price features...")
        
        # Price changes
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_3'] = df['close'].pct_change(3)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        
        # High-low ratios
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        
        # Candle patterns
        df['doji'] = np.abs(df['open'] - df['close']) / (df['high'] - df['low'])
        df['hammer'] = ((df['close'] - df['low']) / (df['high'] - df['low'])) > 0.6
        df['shooting_star'] = ((df['high'] - df['close']) / (df['high'] - df['low'])) > 0.6
        
        # Gap analysis
        df['gap_up'] = (df['open'] > df['close'].shift(1)) & (df['open'] > df['high'].shift(1))
        df['gap_down'] = (df['open'] < df['close'].shift(1)) & (df['open'] < df['low'].shift(1))
        
        # Volume-price relationship
        df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        df['price_vs_vwap'] = df['close'] / df['vwap']
        
        # Volatility features
        df['volatility_5'] = df['price_change_1'].rolling(5).std()
        df['volatility_10'] = df['price_change_1'].rolling(10).std()
        df['volatility_20'] = df['price_change_1'].rolling(20).std()
        
        return df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        logger.info("📊 Calculating volume features...")
        
        # Volume moving averages
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volume trends
        df['volume_trend_5'] = df['volume'].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        df['volume_trend_10'] = df['volume'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Price-volume correlation
        df['pv_corr_5'] = df['close'].rolling(5).corr(df['volume'])
        df['pv_corr_10'] = df['close'].rolling(10).corr(df['volume'])
        
        # Volume spikes
        df['volume_spike'] = df['volume'] > (df['volume_sma_20'] * 2)
        
        return df
    
    def calculate_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features"""
        logger.info("🏗️ Calculating microstructure features...")
        
        # Trade intensity
        df['trade_intensity'] = df['volume'] / (df['high'] - df['low'])
        
        # Realized volatility
        df['realized_vol_5'] = np.sqrt(df['price_change_1'].rolling(5).var() * 1440)  # Annualized
        df['realized_vol_10'] = np.sqrt(df['price_change_1'].rolling(10).var() * 1440)
        
        # Amihud illiquidity
        df['amihud_illiq'] = np.abs(df['price_change_1']) / df['volume']
        
        # Roll's spread estimator
        df['roll_spread'] = 2 * np.sqrt(np.maximum(0, -df['price_change_1'].rolling(2).cov(df['price_change_1'].shift(1))))
        
        return df
    
    def calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime features"""
        logger.info("🎯 Calculating regime features...")
        
        # Trend strength
        df['trend_strength'] = np.abs(df['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        
        # Market state
        df['bull_market'] = (df['close'] > df['sma_50']) & (df['sma_10'] > df['sma_20'])
        df['bear_market'] = (df['close'] < df['sma_50']) & (df['sma_10'] < df['sma_20'])
        
        # Momentum regime
        df['momentum_regime'] = np.where(
            df['rsi_14'] > 70, 'overbought',
            np.where(df['rsi_14'] < 30, 'oversold', 'neutral')
        )
        
        # Volatility regime
        df['vol_regime'] = np.where(
            df['atr_14'] > df['atr_14'].rolling(50).quantile(0.8), 'high_vol',
            np.where(df['atr_14'] < df['atr_14'].rolling(50).quantile(0.2), 'low_vol', 'normal_vol')
        )
        
        return df
    
    def create_target_labels(self, df: pd.DataFrame, risk_reward_ratio: float = 4.0) -> pd.DataFrame:
        """Create target labels for 4:1 risk-reward strategy"""
        logger.info("🎯 Creating target labels...")
        
        labels = []
        
        for i in range(len(df)):
            if i >= len(df) - 100:  # Don't label last 100 bars
                labels.append(0)
                continue
            
            current_price = df.iloc[i]['close']
            atr = df.iloc[i]['atr_14']
            
            if pd.isna(atr) or atr <= 0:
                labels.append(0)
                continue
            
            # Define stop loss and take profit
            stop_loss = current_price - atr  # 1 ATR stop
            take_profit = current_price + (atr * risk_reward_ratio)  # 4 ATR target
            
            # Look forward to see if TP hit before SL
            future_data = df.iloc[i+1:i+101]  # Look 100 bars ahead
            
            hit_tp = False
            hit_sl = False
            
            for j, row in future_data.iterrows():
                if row['high'] >= take_profit:
                    hit_tp = True
                    break
                elif row['low'] <= stop_loss:
                    hit_sl = True
                    break
            
            # Label: 1 if TP hit first, 0 if SL hit first or no hit
            labels.append(1 if hit_tp and not hit_sl else 0)
        
        df['target'] = labels
        
        win_rate = np.mean(labels) * 100
        logger.info(f"✅ Target labels created. Win rate: {win_rate:.1f}%")
        
        return df
    
    def create_feature_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for ML models"""
        logger.info("🔄 Creating feature sequences...")
        
        # Select features (exclude non-numeric and target)
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'datetime', 'target', 'momentum_regime', 'vol_regime'
        ]]
        
        # Handle categorical features
        if 'momentum_regime' in df.columns:
            regime_dummies = pd.get_dummies(df['momentum_regime'], prefix='momentum')
            df = pd.concat([df, regime_dummies], axis=1)
            feature_cols.extend(regime_dummies.columns)
        
        if 'vol_regime' in df.columns:
            vol_dummies = pd.get_dummies(df['vol_regime'], prefix='vol')
            df = pd.concat([df, vol_dummies], axis=1)
            feature_cols.extend(vol_dummies.columns)
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        if len(df_clean) < self.lookback_window:
            raise ValueError(f"Not enough data after cleaning. Need {self.lookback_window}, got {len(df_clean)}")
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.lookback_window, len(df_clean)):
            # Get feature sequence
            sequence = df_clean.iloc[i-self.lookback_window:i][feature_cols].values
            target = df_clean.iloc[i]['target']
            
            # Skip if sequence contains NaN
            if not np.isnan(sequence).any():
                X_sequences.append(sequence)
                y_sequences.append(target)
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        self.feature_names = feature_cols
        
        logger.info(f"✅ Created {len(X)} sequences with {len(feature_cols)} features")
        logger.info(f"   Sequence shape: {X.shape}")
        logger.info(f"   Target distribution: {np.bincount(y)}")
        
        return X, y
    
    def scale_features(self, X: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """Scale features using RobustScaler"""
        logger.info("📏 Scaling features...")
        
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit_scaler:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)
        
        X_scaled = X_scaled.reshape(original_shape)
        
        logger.info(f"✅ Features scaled. Shape: {X_scaled.shape}")
        return X_scaled
    
    def build_complete_dataset(self, coins: List[str], data_dir: Path = Path("data/raw")) -> Dict:
        """Build complete dataset for all coins"""
        logger.info(f"🏗️ Building complete dataset for {coins}")
        
        datasets = {}
        
        for coin in coins:
            logger.info(f"\n📈 Processing {coin}...")
            
            try:
                # Load data
                df = self.load_candle_data(coin, data_dir)
                
                # Calculate all features
                df = self.calculate_technical_indicators(df)
                df = self.calculate_price_features(df)
                df = self.calculate_volume_features(df)
                df = self.calculate_market_microstructure(df)
                df = self.calculate_regime_features(df)
                df = self.create_target_labels(df)
                
                # Create sequences
                X, y = self.create_feature_sequences(df)
                
                # Scale features
                X_scaled = self.scale_features(X, fit_scaler=True)
                
                datasets[coin] = {
                    'X': X_scaled,
                    'y': y,
                    'raw_data': df,
                    'feature_names': self.feature_names,
                    'scaler': self.scaler,
                    'win_rate': np.mean(y) * 100
                }
                
                logger.info(f"✅ {coin}: {len(X)} samples, {np.mean(y)*100:.1f}% win rate")
                
            except Exception as e:
                logger.error(f"❌ Error processing {coin}: {e}")
                continue
        
        return datasets
    
    def save_datasets(self, datasets: Dict, output_dir: Path = Path("data/processed")):
        """Save processed datasets"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for coin, data in datasets.items():
            coin_dir = output_dir / coin
            coin_dir.mkdir(exist_ok=True)
            
            # Save arrays
            np.save(coin_dir / "X.npy", data['X'])
            np.save(coin_dir / "y.npy", data['y'])
            
            # Save metadata
            metadata = {
                'feature_names': data['feature_names'],
                'win_rate': data['win_rate'],
                'n_samples': len(data['X']),
                'n_features': len(data['feature_names']),
                'sequence_length': self.lookback_window
            }
            
            joblib.dump(metadata, coin_dir / "metadata.pkl")
            joblib.dump(data['scaler'], coin_dir / "scaler.pkl")
            
            logger.info(f"💾 Saved {coin} dataset to {coin_dir}")

def main():
    """Main feature engineering pipeline"""
    engineer = EliteFeatureEngineer(lookback_window=60)
    
    # Elite coins for double-up strategy
    elite_coins = ['SOL', 'BTC', 'ETH', 'AVAX', 'MATIC']
    
    # Build datasets
    datasets = engineer.build_complete_dataset(elite_coins)
    
    # Save datasets
    engineer.save_datasets(datasets)
    
    # Summary
    print(f"\n{'='*50}")
    print("ELITE FEATURE ENGINEERING COMPLETE")
    print(f"{'='*50}")
    
    for coin, data in datasets.items():
        print(f"{coin}: {len(data['X']):,} samples, {data['win_rate']:.1f}% win rate")
    
    print(f"\n🎯 Ready for elite model training!")

if __name__ == "__main__":
    main() 