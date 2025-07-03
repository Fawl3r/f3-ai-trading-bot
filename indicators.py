import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Tuple, Optional
from config import (
    CMF_PERIOD, OBV_SMA_PERIOD, RSI_PERIOD, BB_PERIOD, BB_STD,
    ATR_PERIOD, EMA_FAST, EMA_SLOW, DIVERGENCE_LOOKBACK, PARABOLIC_THRESHOLD
)

class TechnicalIndicators:
    @staticmethod
    def chaikin_money_flow(df: pd.DataFrame, period: int = CMF_PERIOD) -> pd.Series:
        """
        Calculate Chaikin Money Flow (CMF)
        CMF = Sum(Money Flow Volume, n) / Sum(Volume, n)
        """
        # Money Flow Multiplier
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mf_multiplier = mf_multiplier.fillna(0)  # Handle division by zero
        
        # Money Flow Volume
        mf_volume = mf_multiplier * df['volume']
        
        # CMF
        cmf = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return cmf.fillna(0)
    
    @staticmethod
    def on_balance_volume(df: pd.DataFrame, sma_period: int = OBV_SMA_PERIOD) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate On Balance Volume (OBV) and its smoothed version
        """
        # Calculate price direction
        price_change = df['close'].diff()
        direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        
        # Calculate OBV
        obv = (direction * df['volume']).cumsum()
        obv_sma = obv.rolling(window=sma_period).mean()
        
        return obv, obv_sma
    
    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = BB_PERIOD, std_dev: float = BB_STD) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        bb = ta.bbands(df['close'], length=period, std=std_dev)
        return bb[f'BBL_{period}_{std_dev}'], bb[f'BBM_{period}_{std_dev}'], bb[f'BBU_{period}_{std_dev}']
    
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
        """Calculate RSI"""
        return ta.rsi(df['close'], length=period)
    
    @staticmethod
    def ema(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return ta.ema(df['close'], length=period)
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
        """Calculate Average True Range"""
        return ta.atr(df['high'], df['low'], df['close'], length=period)
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and add to dataframe"""
        df = df.copy()
        
        # Price-based indicators
        df['cmf'] = TechnicalIndicators.chaikin_money_flow(df)
        df['obv'], df['obv_sma'] = TechnicalIndicators.on_balance_volume(df)
        df['bb_lower'], df['bb_middle'], df['bb_upper'] = TechnicalIndicators.bollinger_bands(df)
        df['rsi'] = TechnicalIndicators.rsi(df)
        df['ema_fast'] = TechnicalIndicators.ema(df, EMA_FAST)
        df['ema_slow'] = TechnicalIndicators.ema(df, EMA_SLOW)
        df['atr'] = TechnicalIndicators.atr(df)
        
        # Additional indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        return df

class PatternDetector:
    @staticmethod
    def detect_divergence(df: pd.DataFrame, lookback: int = DIVERGENCE_LOOKBACK) -> str:
        """
        Detect bullish/bearish divergence between price and indicators
        Returns: 'bullish', 'bearish', or 'none'
        """
        if len(df) < lookback * 2:
            return 'none'
        
        # Get recent highs and lows
        recent_data = df.tail(lookback)
        
        # Price highs and lows
        price_high_idx = recent_data['high'].idxmax()
        price_low_idx = recent_data['low'].idxmin()
        
        current_price = df['close'].iloc[-1]
        current_cmf = df['cmf'].iloc[-1]
        current_obv = df['obv_sma'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        
        # Bearish divergence: higher price highs, lower indicator highs
        if (current_price > df['close'].iloc[price_high_idx] and
            current_cmf < df['cmf'].iloc[price_high_idx] and
            current_obv < df['obv_sma'].iloc[price_high_idx] and
            current_rsi < df['rsi'].iloc[price_high_idx]):
            return 'bearish'
        
        # Bullish divergence: lower price lows, higher indicator lows
        if (current_price < df['close'].iloc[price_low_idx] and
            current_cmf > df['cmf'].iloc[price_low_idx] and
            current_obv > df['obv_sma'].iloc[price_low_idx] and
            current_rsi > df['rsi'].iloc[price_low_idx]):
            return 'bullish'
        
        return 'none'
    
    @staticmethod
    def detect_parabolic_move(df: pd.DataFrame, threshold: float = PARABOLIC_THRESHOLD) -> bool:
        """
        Detect parabolic price movements
        Returns True if price moved more than threshold * ATR in recent bars
        """
        if len(df) < 10:
            return False
        
        recent_bars = 5
        price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-recent_bars])
        atr_value = df['atr'].iloc[-1]
        
        return price_change > (threshold * atr_value)
    
    @staticmethod
    def detect_range_break(df: pd.DataFrame, confirmation_bars: int = 3) -> str:
        """
        Detect range breakouts using Bollinger Bands
        Returns: 'breakout_up', 'breakout_down', or 'none'
        """
        if len(df) < confirmation_bars + 5:
            return 'none'
        
        current_price = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        
        # Check if price consistently above/below bands
        recent_closes = df['close'].tail(confirmation_bars)
        recent_uppers = df['bb_upper'].tail(confirmation_bars)
        recent_lowers = df['bb_lower'].tail(confirmation_bars)
        
        if all(recent_closes > recent_uppers):
            return 'breakout_up'
        elif all(recent_closes < recent_lowers):
            return 'breakout_down'
        
        return 'none'
    
    @staticmethod
    def detect_pullback(df: pd.DataFrame) -> str:
        """
        Detect pullbacks in trending markets using EMAs
        Returns: 'bullish_pullback', 'bearish_pullback', or 'none'
        """
        if len(df) < 20:
            return 'none'
        
        ema_fast = df['ema_fast'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        # Bullish trend (fast EMA > slow EMA) with price pullback to fast EMA
        if (ema_fast > ema_slow and 
            current_price <= ema_fast and 
            prev_price > ema_fast):
            return 'bullish_pullback'
        
        # Bearish trend (fast EMA < slow EMA) with price pullback to fast EMA
        if (ema_fast < ema_slow and 
            current_price >= ema_fast and 
            prev_price < ema_fast):
            return 'bearish_pullback'
        
        return 'none'
    
    @staticmethod
    def detect_reversal_signals(df: pd.DataFrame) -> str:
        """
        Detect potential reversal signals using multiple indicators
        Returns: 'bullish_reversal', 'bearish_reversal', or 'none'
        """
        if len(df) < 20:
            return 'none'
        
        rsi = df['rsi'].iloc[-1]
        cmf = df['cmf'].iloc[-1]
        obv_trend = df['obv_sma'].iloc[-1] - df['obv_sma'].iloc[-5]
        price_vs_bb = df['close'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        
        # Bullish reversal conditions
        if (rsi < 30 and  # Oversold
            cmf > 0 and   # Positive money flow
            obv_trend > 0 and  # OBV rising
            price_vs_bb <= bb_lower):  # Price at lower BB
            return 'bullish_reversal'
        
        # Bearish reversal conditions
        if (rsi > 70 and  # Overbought
            cmf < 0 and   # Negative money flow
            obv_trend < 0 and  # OBV falling
            price_vs_bb >= bb_upper):  # Price at upper BB
            return 'bearish_reversal'
        
        return 'none'

class VolumeAnalysis:
    @staticmethod
    def is_volume_confirming(df: pd.DataFrame, min_factor: float = 1.5) -> bool:
        """Check if current volume confirms the move"""
        if len(df) < 20:
            return False
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume_sma'].iloc[-1]
        
        return current_volume >= (avg_volume * min_factor)
    
    @staticmethod
    def get_volume_profile_signal(df: pd.DataFrame) -> str:
        """Analyze volume profile for trading signals"""
        if len(df) < 10:
            return 'none'
        
        volume_trend = df['volume'].rolling(5).mean().iloc[-1] - df['volume'].rolling(5).mean().iloc[-6]
        price_trend = df['close'].iloc[-1] - df['close'].iloc[-5]
        
        # Volume increasing with price = confirmation
        if volume_trend > 0 and price_trend > 0:
            return 'bullish_volume'
        elif volume_trend > 0 and price_trend < 0:
            return 'bearish_volume'
        
        return 'none' 