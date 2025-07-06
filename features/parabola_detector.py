#!/usr/bin/env python3
"""
Parabolic Movement & Exhaustion Detection Module
Advanced features for catching parabolic breakouts and reversal points
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ParabolaDetector:
    """Advanced parabolic movement and exhaustion detection"""
    
    def __init__(self, lookback_period: int = 30):
        self.lookback_period = lookback_period
        
    def calculate_roc_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Rate of Change acceleration for burst detection"""
        
        # 3-period ROC
        df['roc_3'] = talib.ROC(df['close'].values, timeperiod=3)
        
        # 30-bar standard deviation of ROC
        df['roc_30_std'] = df['roc_3'].rolling(30).std()
        
        # ROC acceleration signal (burst when ROC > 2x standard deviation)
        df['roc_accel'] = df['roc_3'] / (df['roc_30_std'] * 2)
        df['roc_burst_signal'] = df['roc_accel'] > 1.0
        
        # Smooth acceleration for better signals
        df['roc_accel_smooth'] = df['roc_accel'].rolling(3).mean()
        
        return df
    
    def calculate_vwap_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP gap analysis for parabolic detection"""
        
        # VWAP calculation
        df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        
        # ATR for gap measurement
        df['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # VWAP gap in ATR terms
        df['vwap_gap'] = (df['close'] - df['vwap']) / df['atr_14']
        
        # Parabolic signals
        df['vwap_burst_long'] = df['vwap_gap'] > 3.0   # Price > 3 ATR above VWAP
        df['vwap_burst_short'] = df['vwap_gap'] < -3.0  # Price > 3 ATR below VWAP
        
        # VWAP mean reversion signals
        df['vwap_reversion_signal'] = np.abs(df['vwap_gap']) > 2.5
        
        return df
    
    def detect_rsi_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect RSI divergence for exhaustion signals"""
        
        # RSI calculation
        df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # Find swing highs and lows
        df['swing_high'] = df['high'].rolling(5, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(5, center=True).min() == df['low']
        
        # Initialize divergence signals
        df['bullish_divergence'] = False
        df['bearish_divergence'] = False
        
        # Look for divergence patterns
        for i in range(20, len(df)):
            # Bearish divergence: Higher highs in price, lower highs in RSI
            if df.iloc[i]['swing_high']:
                # Look back for previous swing high
                for j in range(i-20, i-5):
                    if j >= 0 and df.iloc[j]['swing_high']:
                        price_hh = df.iloc[i]['high'] > df.iloc[j]['high']
                        rsi_lh = df.iloc[i]['rsi_14'] < df.iloc[j]['rsi_14']
                        
                        if price_hh and rsi_lh:
                            df.iloc[i, df.columns.get_loc('bearish_divergence')] = True
                        break
            
            # Bullish divergence: Lower lows in price, higher lows in RSI
            if df.iloc[i]['swing_low']:
                # Look back for previous swing low
                for j in range(i-20, i-5):
                    if j >= 0 and df.iloc[j]['swing_low']:
                        price_ll = df.iloc[i]['low'] < df.iloc[j]['low']
                        rsi_hl = df.iloc[i]['rsi_14'] > df.iloc[j]['rsi_14']
                        
                        if price_ll and rsi_hl:
                            df.iloc[i, df.columns.get_loc('bullish_divergence')] = True
                        break
        
        return df
    
    def calculate_volume_climax(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect volume climax for exhaustion signals"""
        
        # 20-bar median volume
        df['volume_median_20'] = df['volume'].rolling(20).median()
        
        # Delta volume (current vs median)
        df['delta_volume'] = df['volume'] / df['volume_median_20']
        
        # Volume climax signal
        df['volume_climax'] = df['delta_volume'] > 4.0
        
        # Price stall detection (small body relative to ATR)
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['body_ratio'] = df['body_size'] / df['atr_14']
        df['price_stall'] = df['body_ratio'] < 0.3  # Small body
        
        # Combined climax signal
        df['climax_fade_signal'] = df['volume_climax'] & df['price_stall']
        
        return df
    
    def calculate_order_book_imbalance(self, df: pd.DataFrame, 
                                     bid_volume: Optional[np.ndarray] = None,
                                     ask_volume: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Calculate order book imbalance (simulated if no real data)"""
        
        if bid_volume is None or ask_volume is None:
            # Simulate order book imbalance based on price action
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            
            # Simulate bid/ask imbalance
            df['obi'] = np.where(
                df['price_change'] > 0,
                0.6 + 0.3 * np.random.random(len(df)),  # Bid favored on up moves
                0.4 - 0.3 * np.random.random(len(df))   # Ask favored on down moves
            )
        else:
            # Real order book imbalance
            total_volume = bid_volume + ask_volume
            df['obi'] = bid_volume / total_volume
        
        # Smooth the signal
        df['obi_smooth'] = df['obi'].rolling(3).mean()
        
        # Imbalance signals
        df['bid_imbalance'] = df['obi_smooth'] > 0.65
        df['ask_imbalance'] = df['obi_smooth'] < 0.35
        
        return df
    
    def generate_parabolic_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive parabolic and exhaustion signals"""
        
        # Calculate all components
        df = self.calculate_roc_acceleration(df)
        df = self.calculate_vwap_analysis(df)
        df = self.detect_rsi_divergence(df)
        df = self.calculate_volume_climax(df)
        df = self.calculate_order_book_imbalance(df)
        
        # Combine signals for burst detection
        df['burst_long_signal'] = (
            df['roc_burst_signal'] & 
            df['vwap_burst_long'] & 
            df['bid_imbalance']
        )
        
        df['burst_short_signal'] = (
            df['roc_burst_signal'] & 
            df['vwap_burst_short'] & 
            df['ask_imbalance']
        )
        
        # Combine signals for fade detection
        df['fade_long_signal'] = (
            df['bullish_divergence'] | 
            (df['climax_fade_signal'] & (df['vwap_gap'] < -2))
        )
        
        df['fade_short_signal'] = (
            df['bearish_divergence'] | 
            (df['climax_fade_signal'] & (df['vwap_gap'] > 2))
        )
        
        # Signal strength (0-1)
        df['burst_strength'] = (
            df['roc_accel_smooth'].clip(0, 3) / 3 * 0.4 +
            np.abs(df['vwap_gap']).clip(0, 5) / 5 * 0.4 +
            np.abs(df['obi_smooth'] - 0.5) * 4 * 0.2
        ).clip(0, 1)
        
        df['fade_strength'] = (
            (df['bearish_divergence'] | df['bullish_divergence']).astype(float) * 0.6 +
            df['climax_fade_signal'].astype(float) * 0.4
        ).clip(0, 1)
        
        return df
    
    def calculate_trend_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend context for signal filtering"""
        
        # 200-period EMA for major trend
        df['ema_200'] = talib.EMA(df['close'].values, timeperiod=200)
        df['ema_200_slope'] = df['ema_200'].diff(5) / df['ema_200']
        
        # Trend classification
        df['major_trend'] = np.where(
            df['ema_200_slope'] > 0.001, 'uptrend',
            np.where(df['ema_200_slope'] < -0.001, 'downtrend', 'sideways')
        )
        
        # Price relative to EMA
        df['price_vs_ema200'] = (df['close'] - df['ema_200']) / df['ema_200']
        
        return df
    
    def get_time_of_day_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply time-of-day filters for optimal signal timing"""
        
        # Extract hour from timestamp (assuming UTC)
        if 'datetime' in df.columns:
            df['hour_utc'] = pd.to_datetime(df['datetime']).dt.hour
        else:
            # Simulate time if not available
            df['hour_utc'] = np.random.randint(0, 24, len(df))
        
        # NY morning hours (13:00-18:00 UTC) for burst trades
        df['ny_morning'] = (df['hour_utc'] >= 13) & (df['hour_utc'] <= 18)
        
        # Filter burst signals by time
        df['burst_long_filtered'] = df['burst_long_signal'] & df['ny_morning']
        df['burst_short_filtered'] = df['burst_short_signal'] & df['ny_morning']
        
        # Fade signals allowed 24/7
        df['fade_long_filtered'] = df['fade_long_signal']
        df['fade_short_filtered'] = df['fade_short_signal']
        
        return df
    
    def process_complete_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete parabolic analysis pipeline"""
        
        logger.info("Running complete parabolic analysis...")
        
        # Generate all signals
        df = self.generate_parabolic_signals(df)
        df = self.calculate_trend_context(df)
        df = self.get_time_of_day_filter(df)
        
        # Final signal classification
        df['signal_type'] = 'normal'
        
        # Mark burst signals
        df.loc[df['burst_long_filtered'], 'signal_type'] = 'burst_long'
        df.loc[df['burst_short_filtered'], 'signal_type'] = 'burst_short'
        
        # Mark fade signals (only if not in same direction as major trend)
        fade_long_valid = df['fade_long_filtered'] & (df['major_trend'] != 'uptrend')
        fade_short_valid = df['fade_short_filtered'] & (df['major_trend'] != 'downtrend')
        
        df.loc[fade_long_valid, 'signal_type'] = 'fade_long'
        df.loc[fade_short_valid, 'signal_type'] = 'fade_short'
        
        # Signal confidence
        df['signal_confidence'] = np.where(
            df['signal_type'].str.contains('burst'), df['burst_strength'],
            np.where(df['signal_type'].str.contains('fade'), df['fade_strength'], 0.5)
        )
        
        logger.info("Parabolic analysis complete")
        return df

def main():
    """Test the parabolic detector"""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    
    # Simulate realistic price data with parabolic moves
    price = 100
    prices = [price]
    volumes = []
    
    for i in range(999):
        # Add some parabolic behavior
        if i % 200 == 0:  # Parabolic move every 200 periods
            change = np.random.normal(0.02, 0.01)  # Strong move
        else:
            change = np.random.normal(0.001, 0.01)  # Normal move
        
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
    
    # Run analysis
    detector = ParabolaDetector()
    df_analyzed = detector.process_complete_analysis(df)
    
    # Print results
    print("Parabolic Analysis Results:")
    print(f"Total signals: {len(df_analyzed[df_analyzed['signal_type'] != 'normal'])}")
    print(f"Burst long: {len(df_analyzed[df_analyzed['signal_type'] == 'burst_long'])}")
    print(f"Burst short: {len(df_analyzed[df_analyzed['signal_type'] == 'burst_short'])}")
    print(f"Fade long: {len(df_analyzed[df_analyzed['signal_type'] == 'fade_long'])}")
    print(f"Fade short: {len(df_analyzed[df_analyzed['signal_type'] == 'fade_short'])}")
    
    # Show sample signals
    signals = df_analyzed[df_analyzed['signal_type'] != 'normal'].head(10)
    print("\nSample signals:")
    print(signals[['datetime', 'close', 'signal_type', 'signal_confidence']].to_string())

if __name__ == "__main__":
    main() 