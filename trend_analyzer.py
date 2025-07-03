"""
Advanced Trend Direction Analysis Module
Provides comprehensive trend analysis and market direction detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import pandas_ta as ta
from indicators import TechnicalIndicators

class TrendDirection(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

class TrendStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

class TrendAnalyzer:
    """Advanced trend analysis with multiple timeframe confirmation"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        
    def analyze_trend_direction(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive trend direction analysis using multiple indicators
        """
        if len(df) < 50:
            return self._default_trend_analysis()
            
        # Calculate all trend indicators
        trend_signals = self._calculate_trend_signals(df)
        
        # Analyze market structure
        structure_analysis = self._analyze_market_structure(df)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(df, trend_signals)
        
        # Determine overall trend direction
        overall_direction = self._determine_overall_trend(trend_signals, structure_analysis)
        
        # Calculate trend confidence
        confidence = self._calculate_trend_confidence(trend_signals)
        
        return {
            'direction': overall_direction,
            'strength': trend_strength,
            'confidence': confidence,
            'signals': trend_signals,
            'structure': structure_analysis,
            'timeframe_alignment': self._check_timeframe_alignment(df),
            'momentum': self._analyze_momentum(df),
            'volume_confirmation': self._analyze_volume_trend(df)
        }
    
    def _calculate_trend_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate multiple trend indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Moving averages
        ema_8 = ta.ema(close, length=8)
        ema_21 = ta.ema(close, length=21)
        ema_50 = ta.ema(close, length=50)
        ema_200 = ta.ema(close, length=200)
        
        # MACD
        macd_data = ta.macd(close)
        macd = macd_data['MACD_12_26_9'] if 'MACD_12_26_9' in macd_data.columns else close * 0
        macd_signal = macd_data['MACDs_12_26_9'] if 'MACDs_12_26_9' in macd_data.columns else close * 0
        macd_hist = macd_data['MACDh_12_26_9'] if 'MACDh_12_26_9' in macd_data.columns else close * 0
        
        # ADX for trend strength
        adx = ta.adx(high, low, close, length=14)['ADX_14'] if ta.adx(high, low, close, length=14) is not None else close * 0 + 20
        
        # Parabolic SAR
        sar = ta.psar(high, low, close)['PSARl_0.02_0.2'] if ta.psar(high, low, close) is not None else close
        
        # Supertrend
        supertrend = self._calculate_supertrend(df)
        
        # Ichimoku components
        ichimoku = self._calculate_ichimoku(df)
        
        current_price = close.iloc[-1]
        
        return {
            'ma_alignment': {
                'ema8_vs_ema21': 1 if ema_8.iloc[-1] > ema_21.iloc[-1] else -1,
                'ema21_vs_ema50': 1 if ema_21.iloc[-1] > ema_50.iloc[-1] else -1,
                'ema50_vs_ema200': 1 if ema_50.iloc[-1] > ema_200.iloc[-1] else -1,
                'price_vs_ema8': 1 if current_price > ema_8.iloc[-1] else -1,
                'price_vs_ema21': 1 if current_price > ema_21.iloc[-1] else -1,
                'price_vs_ema200': 1 if current_price > ema_200.iloc[-1] else -1
            },
            'macd': {
                'line_vs_signal': 1 if macd.iloc[-1] > macd_signal.iloc[-1] else -1,
                'histogram': 1 if macd_hist.iloc[-1] > 0 else -1,
                'momentum': 1 if macd_hist.iloc[-1] > macd_hist.iloc[-2] else -1
            },
            'adx': {
                'strength': adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20,
                'trending': 1 if adx.iloc[-1] > 25 else 0
            },
            'sar': {
                'direction': 1 if current_price > sar.iloc[-1] else -1
            },
            'supertrend': supertrend,
            'ichimoku': ichimoku
        }
    
    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict:
        """Calculate Supertrend indicator"""
        hl2 = (df['high'] + df['low']) / 2
        atr = ta.atr(df['high'], df['low'], df['close'], length=period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] <= lower_band.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = -1
            elif df['close'].iloc[i] >= upper_band.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
        
        current_direction = direction.iloc[-1] if not pd.isna(direction.iloc[-1]) else 0
        
        return {
            'direction': int(current_direction),
            'value': supertrend.iloc[-1] if not pd.isna(supertrend.iloc[-1]) else df['close'].iloc[-1]
        }
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud components"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-26)
        
        current_price = close.iloc[-1]
        
        # Cloud analysis
        cloud_top = max(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]) if not pd.isna(senkou_span_a.iloc[-1]) else current_price
        cloud_bottom = min(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]) if not pd.isna(senkou_span_a.iloc[-1]) else current_price
        
        return {
            'price_vs_cloud': 1 if current_price > cloud_top else (-1 if current_price < cloud_bottom else 0),
            'tenkan_vs_kijun': 1 if tenkan_sen.iloc[-1] > kijun_sen.iloc[-1] else -1,
            'cloud_color': 1 if senkou_span_a.iloc[-1] > senkou_span_b.iloc[-1] else -1,
            'price_vs_kijun': 1 if current_price > kijun_sen.iloc[-1] else -1
        }
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure for trend confirmation"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Higher highs and higher lows analysis
        recent_data = df.tail(20)
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_data) - 2):
            if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and 
                highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]):
                swing_highs.append(highs.iloc[i])
            
            if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and 
                lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]):
                swing_lows.append(lows.iloc[i])
        
        # Analyze structure
        structure_score = 0
        
        if len(swing_highs) >= 2:
            if swing_highs[-1] > swing_highs[-2]:
                structure_score += 1
            else:
                structure_score -= 1
                
        if len(swing_lows) >= 2:
            if swing_lows[-1] > swing_lows[-2]:
                structure_score += 1
            else:
                structure_score -= 1
        
        return {
            'structure_score': structure_score,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'structure_trend': 'bullish' if structure_score > 0 else ('bearish' if structure_score < 0 else 'neutral')
        }
    
    def _calculate_trend_strength(self, df: pd.DataFrame, signals: Dict) -> TrendStrength:
        """Calculate overall trend strength"""
        strength_score = 0
        max_score = 0
        
        # MA alignment contribution
        ma_signals = signals['ma_alignment']
        ma_score = sum(ma_signals.values())
        strength_score += abs(ma_score)
        max_score += 6
        
        # ADX contribution
        adx_strength = signals['adx']['strength']
        if adx_strength > 40:
            strength_score += 3
        elif adx_strength > 25:
            strength_score += 2
        elif adx_strength > 15:
            strength_score += 1
        max_score += 3
        
        # MACD contribution
        macd_signals = signals['macd']
        macd_score = sum(macd_signals.values())
        strength_score += abs(macd_score)
        max_score += 3
        
        # Normalize strength score
        normalized_strength = (strength_score / max_score) * 5 if max_score > 0 else 2.5
        
        if normalized_strength >= 4.5:
            return TrendStrength.VERY_STRONG
        elif normalized_strength >= 3.5:
            return TrendStrength.STRONG
        elif normalized_strength >= 2.5:
            return TrendStrength.MODERATE
        elif normalized_strength >= 1.5:
            return TrendStrength.WEAK
        else:
            return TrendStrength.VERY_WEAK
    
    def _determine_overall_trend(self, signals: Dict, structure: Dict) -> TrendDirection:
        """Determine overall trend direction"""
        bullish_score = 0
        bearish_score = 0
        
        # MA alignment
        ma_signals = signals['ma_alignment']
        for signal in ma_signals.values():
            if signal > 0:
                bullish_score += 1
            else:
                bearish_score += 1
        
        # MACD
        macd_signals = signals['macd']
        for signal in macd_signals.values():
            if signal > 0:
                bullish_score += 1
            else:
                bearish_score += 1
        
        # Other indicators
        if signals['sar']['direction'] > 0:
            bullish_score += 1
        else:
            bearish_score += 1
            
        if signals['supertrend']['direction'] > 0:
            bullish_score += 1
        else:
            bearish_score += 1
        
        # Ichimoku
        ichimoku_signals = signals['ichimoku']
        for key, signal in ichimoku_signals.items():
            if signal > 0:
                bullish_score += 1
            elif signal < 0:
                bearish_score += 1
        
        # Structure analysis
        if structure['structure_score'] > 0:
            bullish_score += 2
        elif structure['structure_score'] < 0:
            bearish_score += 2
        
        # Determine direction
        total_signals = bullish_score + bearish_score
        if total_signals == 0:
            return TrendDirection.SIDEWAYS
            
        bullish_ratio = bullish_score / total_signals
        
        if bullish_ratio >= 0.8:
            return TrendDirection.STRONG_UPTREND
        elif bullish_ratio >= 0.65:
            return TrendDirection.UPTREND
        elif bullish_ratio <= 0.2:
            return TrendDirection.STRONG_DOWNTREND
        elif bullish_ratio <= 0.35:
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.SIDEWAYS
    
    def _calculate_trend_confidence(self, signals: Dict) -> float:
        """Calculate confidence in trend direction"""
        confirmations = 0
        total_indicators = 0
        
        # Count confirmations from each indicator group
        ma_confirmations = sum(1 for signal in signals['ma_alignment'].values() if signal != 0)
        macd_confirmations = sum(1 for signal in signals['macd'].values() if signal != 0)
        
        confirmations += ma_confirmations + macd_confirmations + 2  # SAR and Supertrend
        total_indicators += 6 + 3 + 2  # MA(6) + MACD(3) + Others(2)
        
        # Add Ichimoku confirmations
        ichimoku_confirmations = sum(1 for signal in signals['ichimoku'].values() if signal != 0)
        confirmations += ichimoku_confirmations
        total_indicators += 4
        
        return (confirmations / total_indicators) * 100 if total_indicators > 0 else 50.0
    
    def _check_timeframe_alignment(self, df: pd.DataFrame) -> Dict:
        """Check trend alignment across different lookback periods"""
        close = df['close']
        
        # Different timeframe EMAs
        ema_short = ta.ema(close, length=8)
        ema_medium = ta.ema(close, length=21)
        ema_long = ta.ema(close, length=50)
        
        current_price = close.iloc[-1]
        
        return {
            'short_term': 'bullish' if current_price > ema_short.iloc[-1] else 'bearish',
            'medium_term': 'bullish' if current_price > ema_medium.iloc[-1] else 'bearish',
            'long_term': 'bullish' if current_price > ema_long.iloc[-1] else 'bearish',
            'alignment_score': self._calculate_alignment_score(current_price, ema_short.iloc[-1], ema_medium.iloc[-1], ema_long.iloc[-1])
        }
    
    def _calculate_alignment_score(self, price: float, ema_short: float, ema_medium: float, ema_long: float) -> int:
        """Calculate alignment score for timeframes"""
        if price > ema_short > ema_medium > ema_long:
            return 3  # Perfect bullish alignment
        elif price < ema_short < ema_medium < ema_long:
            return -3  # Perfect bearish alignment
        elif price > ema_short > ema_medium:
            return 2
        elif price < ema_short < ema_medium:
            return -2
        elif price > ema_short:
            return 1
        elif price < ema_short:
            return -1
        else:
            return 0
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        rsi = ta.rsi(close, length=14)
        
        # Stochastic
        stoch_data = ta.stoch(high, low, close)
        stoch_k = stoch_data['STOCHk_14_3_3'] if stoch_data is not None and 'STOCHk_14_3_3' in stoch_data.columns else close * 0 + 50
        stoch_d = stoch_data['STOCHd_14_3_3'] if stoch_data is not None and 'STOCHd_14_3_3' in stoch_data.columns else close * 0 + 50
        
        # Rate of Change
        roc = ta.roc(close, length=10)
        
        return {
            'rsi': {
                'value': rsi.iloc[-1],
                'signal': 'overbought' if rsi.iloc[-1] > 70 else ('oversold' if rsi.iloc[-1] < 30 else 'neutral')
            },
            'stochastic': {
                'k': stoch_k.iloc[-1],
                'd': stoch_d.iloc[-1],
                'signal': 'overbought' if stoch_k.iloc[-1] > 80 else ('oversold' if stoch_k.iloc[-1] < 20 else 'neutral')
            },
            'roc': {
                'value': roc.iloc[-1],
                'signal': 'bullish' if roc.iloc[-1] > 0 else 'bearish'
            }
        }
    
    def _analyze_volume_trend(self, df: pd.DataFrame) -> Dict:
        """Analyze volume trend confirmation"""
        volume = df['volume']
        close = df['close']
        
        # Volume moving averages
        vol_ma_short = volume.rolling(window=10).mean()
        vol_ma_long = volume.rolling(window=30).mean()
        
        # Price and volume correlation
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        
        correlation = price_change.corr(volume_change)
        
        return {
            'volume_trend': 'increasing' if vol_ma_short.iloc[-1] > vol_ma_long.iloc[-1] else 'decreasing',
            'current_vs_average': volume.iloc[-1] / vol_ma_long.iloc[-1] if vol_ma_long.iloc[-1] > 0 else 1,
            'price_volume_correlation': correlation if not pd.isna(correlation) else 0,
            'volume_confirmation': 'strong' if volume.iloc[-1] > vol_ma_long.iloc[-1] * 1.5 else 'weak'
        }
    
    def _default_trend_analysis(self) -> Dict:
        """Return default analysis when insufficient data"""
        return {
            'direction': TrendDirection.SIDEWAYS,
            'strength': TrendStrength.WEAK,
            'confidence': 50.0,
            'signals': {},
            'structure': {},
            'timeframe_alignment': {},
            'momentum': {},
            'volume_confirmation': {}
        } 