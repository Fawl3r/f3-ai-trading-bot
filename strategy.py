import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from indicators import TechnicalIndicators, PatternDetector, VolumeAnalysis
from config import MIN_VOLUME_FACTOR, RANGE_BREAK_CONFIRMATION

class TradingSignal:
    def __init__(self, signal_type: str, confidence: float, entry_price: float, 
                 stop_loss: float = None, take_profit: float = None, reason: str = ""):
        self.signal_type = signal_type  # 'buy', 'sell', 'close', 'hold'
        self.confidence = confidence    # 0.0 to 1.0
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.reason = reason
        self.timestamp = datetime.now()

class AdvancedTradingStrategy:
    def __init__(self):
        self.last_signal = None
        self.last_signal_time = None
        self.trade_count_today = 0
        self.last_trade_date = None
        self.signal_cooldown = 300  # 5 minutes between signals
        
        # Strategy weights for signal confidence calculation
        self.weights = {
            'divergence': 0.25,
            'range_break': 0.20,
            'reversal': 0.20,
            'pullback': 0.15,
            'volume_confirmation': 0.10,
            'parabolic_exit': 0.10
        }
    
    def _reset_daily_counters(self):
        """Reset daily trade counters"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.trade_count_today = 0
            self.last_trade_date = today
    
    def _is_signal_valid(self, signal_type: str) -> bool:
        """Check if signal is valid based on cooldown and daily limits"""
        now = datetime.now()
        
        # Check cooldown period
        if (self.last_signal_time and 
            (now - self.last_signal_time).total_seconds() < self.signal_cooldown):
            return False
        
        # Check daily trade limit
        self._reset_daily_counters()
        if self.trade_count_today >= 10:  # Max 10 trades per day
            return False
        
        # Avoid signal flip-flopping
        if self.last_signal == signal_type:
            return False
        
        return True
    
    def _calculate_stop_loss_take_profit(self, df: pd.DataFrame, signal_type: str, 
                                       entry_price: float) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels"""
        atr = df['atr'].iloc[-1]
        
        if signal_type == 'buy':
            stop_loss = entry_price - (2 * atr)
            take_profit = entry_price + (3 * atr)
        else:  # sell
            stop_loss = entry_price + (2 * atr)
            take_profit = entry_price - (3 * atr)
        
        return stop_loss, take_profit
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze overall market structure and trend"""
        if len(df) < 50:
            return {'trend': 'sideways', 'strength': 0.5}
        
        # Trend analysis using EMAs
        ema_fast = df['ema_fast'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        
        if ema_fast > ema_slow * 1.01:  # 1% buffer
            trend = 'bullish'
        elif ema_fast < ema_slow * 0.99:  # 1% buffer
            trend = 'bearish'
        else:
            trend = 'sideways'
        
        # Trend strength based on EMA separation
        ema_separation = abs(ema_fast - ema_slow) / ema_slow
        strength = min(ema_separation * 100, 1.0)  # Cap at 1.0
        
        # Volume trend
        volume_trend = df['volume'].rolling(10).mean().iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        
        return {
            'trend': trend,
            'strength': strength,
            'volume_trend': volume_trend,
            'volatility': df['atr'].iloc[-1] / df['close'].iloc[-1]
        }
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trading signals based on comprehensive analysis
        Target: 80-90% accuracy through multi-factor confirmation
        """
        if len(df) < 50:
            return None
        
        # Calculate all indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        # Market structure analysis
        market_structure = self._analyze_market_structure(df)
        
        # Pattern detection
        divergence = PatternDetector.detect_divergence(df)
        range_break = PatternDetector.detect_range_break(df, RANGE_BREAK_CONFIRMATION)
        pullback = PatternDetector.detect_pullback(df)
        reversal = PatternDetector.detect_reversal_signals(df)
        parabolic = PatternDetector.detect_parabolic_move(df)
        
        # Volume analysis
        volume_confirming = VolumeAnalysis.is_volume_confirming(df, MIN_VOLUME_FACTOR)
        volume_signal = VolumeAnalysis.get_volume_profile_signal(df)
        
        # Current market data
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        cmf = df['cmf'].iloc[-1]
        bb_position = self._get_bb_position(df)
        
        # Signal generation logic with confidence scoring
        signal_type = 'hold'
        confidence = 0.0
        reasons = []
        
        # BULLISH SIGNALS
        bullish_score = 0.0
        
        # 1. Bullish divergence (high weight)
        if divergence == 'bullish':
            bullish_score += self.weights['divergence']
            reasons.append("Bullish divergence detected")
        
        # 2. Range breakout up
        if range_break == 'breakout_up' and volume_confirming:
            bullish_score += self.weights['range_break']
            reasons.append("Upward breakout with volume")
        
        # 3. Bullish reversal
        if reversal == 'bullish_reversal':
            bullish_score += self.weights['reversal']
            reasons.append("Bullish reversal pattern")
        
        # 4. Bullish pullback in uptrend
        if pullback == 'bullish_pullback' and market_structure['trend'] == 'bullish':
            bullish_score += self.weights['pullback']
            reasons.append("Pullback in bullish trend")
        
        # 5. Volume confirmation
        if volume_signal == 'bullish_volume':
            bullish_score += self.weights['volume_confirmation']
            reasons.append("Bullish volume profile")
        
        # Additional bullish factors
        if rsi < 35 and cmf > 0.1:  # Oversold with positive money flow
            bullish_score += 0.05
            reasons.append("Oversold with positive CMF")
        
        if bb_position == 'lower' and market_structure['trend'] != 'bearish':
            bullish_score += 0.05
            reasons.append("Price at lower Bollinger Band")
        
        # BEARISH SIGNALS
        bearish_score = 0.0
        
        # 1. Bearish divergence (high weight)
        if divergence == 'bearish':
            bearish_score += self.weights['divergence']
            reasons.append("Bearish divergence detected")
        
        # 2. Range breakout down
        if range_break == 'breakout_down' and volume_confirming:
            bearish_score += self.weights['range_break']
            reasons.append("Downward breakout with volume")
        
        # 3. Bearish reversal
        if reversal == 'bearish_reversal':
            bearish_score += self.weights['reversal']
            reasons.append("Bearish reversal pattern")
        
        # 4. Bearish pullback in downtrend
        if pullback == 'bearish_pullback' and market_structure['trend'] == 'bearish':
            bearish_score += self.weights['pullback']
            reasons.append("Pullback in bearish trend")
        
        # 5. Volume confirmation
        if volume_signal == 'bearish_volume':
            bearish_score += self.weights['volume_confirmation']
            reasons.append("Bearish volume profile")
        
        # Additional bearish factors
        if rsi > 65 and cmf < -0.1:  # Overbought with negative money flow
            bearish_score += 0.05
            reasons.append("Overbought with negative CMF")
        
        if bb_position == 'upper' and market_structure['trend'] != 'bullish':
            bearish_score += 0.05
            reasons.append("Price at upper Bollinger Band")
        
        # PARABOLIC EXIT SIGNAL
        if parabolic:
            return TradingSignal(
                signal_type='close',
                confidence=self.weights['parabolic_exit'],
                entry_price=current_price,
                reason="Parabolic move detected - exit signal"
            )
        
        # Determine final signal
        min_confidence = 0.6  # Minimum 60% confidence for trade
        
        if bullish_score > bearish_score and bullish_score >= min_confidence:
            signal_type = 'buy'
            confidence = bullish_score
        elif bearish_score > bullish_score and bearish_score >= min_confidence:
            signal_type = 'sell'
            confidence = bearish_score
        
        # Apply market structure filter
        if signal_type == 'buy' and market_structure['trend'] == 'bearish':
            confidence *= 0.7  # Reduce confidence for counter-trend trades
        elif signal_type == 'sell' and market_structure['trend'] == 'bullish':
            confidence *= 0.7
        
        # Final confidence check
        if confidence < min_confidence:
            signal_type = 'hold'
        
        # Validate signal
        if signal_type != 'hold' and not self._is_signal_valid(signal_type):
            return None
        
        # Create signal with stop loss and take profit
        if signal_type in ['buy', 'sell']:
            stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                df, signal_type, current_price
            )
            
            # Update tracking variables
            self.last_signal = signal_type
            self.last_signal_time = datetime.now()
            self.trade_count_today += 1
            
            return TradingSignal(
                signal_type=signal_type,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=" | ".join(reasons)
            )
        
        return None
    
    def _get_bb_position(self, df: pd.DataFrame) -> str:
        """Get current price position relative to Bollinger Bands"""
        current_price = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_middle = df['bb_middle'].iloc[-1]
        
        if current_price >= bb_upper:
            return 'upper'
        elif current_price <= bb_lower:
            return 'lower'
        elif current_price >= bb_middle:
            return 'upper_mid'
        else:
            return 'lower_mid'
    
    def should_exit_position(self, df: pd.DataFrame, position_side: str, 
                           entry_price: float, current_pnl_pct: float) -> bool:
        """
        Determine if current position should be exited
        """
        # Emergency exit on large drawdown
        if current_pnl_pct <= -5.0:  # 5% max loss
            return True
        
        # Parabolic move exit
        if PatternDetector.detect_parabolic_move(df):
            return True
        
        # Reversal signal exit
        reversal = PatternDetector.detect_reversal_signals(df)
        if ((position_side == 'long' and reversal == 'bearish_reversal') or
            (position_side == 'short' and reversal == 'bullish_reversal')):
            return True
        
        # Profit taking on strong signals
        if current_pnl_pct >= 4.0:  # 4% profit
            return True
        
        return False
    
    def get_strategy_stats(self) -> Dict[str, any]:
        """Get current strategy statistics"""
        return {
            'trades_today': self.trade_count_today,
            'last_signal': self.last_signal,
            'last_signal_time': self.last_signal_time,
            'signal_weights': self.weights
        } 