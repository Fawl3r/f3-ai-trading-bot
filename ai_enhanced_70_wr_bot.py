#!/usr/bin/env python3
"""
AI-Enhanced 70%+ Win Rate Trading Bot
Advanced ML Pattern Recognition & Ensemble Methods
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_risk_management import AdvancedRiskManager, RiskMetrics
from advanced_top_bottom_detector import AdvancedTopBottomDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AISignal:
    """AI-generated trading signal with confidence"""
    direction: str  # 'long', 'short', 'hold'
    confidence: float  # 0.0 to 1.0
    probability: float  # ML model probability
    features: Dict  # Feature values used
    ensemble_votes: Dict  # Individual model votes
    risk_score: float  # Risk assessment 0-1

@dataclass
class TradeResult:
    """Enhanced trade result with AI metrics"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    holding_time_minutes: float
    exit_reason: str
    ai_confidence: float
    ai_probability: float
    market_regime: str

class AIPatternRecognizer:
    """Advanced AI pattern recognition system"""
    
    def __init__(self):
        # Ensemble of ML models
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'nn': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
        
    def extract_features(self, candles: List[Dict], lookback: int = 50) -> np.ndarray:
        """Extract comprehensive features for ML models"""
        if len(candles) < lookback:
            return np.array([])
        
        recent_candles = candles[-lookback:]
        df = pd.DataFrame(recent_candles)
        
        features = []
        
        # Price-based features
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        
        # 1. Moving averages and trends
        sma_5 = np.mean(closes[-5:])
        sma_10 = np.mean(closes[-10:])
        sma_20 = np.mean(closes[-20:])
        ema_12 = self._calculate_ema(closes, 12)
        ema_26 = self._calculate_ema(closes, 26)
        
        features.extend([
            closes[-1] / sma_5 - 1,  # Price vs SMA5
            closes[-1] / sma_10 - 1,  # Price vs SMA10
            closes[-1] / sma_20 - 1,  # Price vs SMA20
            sma_5 / sma_10 - 1,  # SMA5 vs SMA10
            sma_10 / sma_20 - 1,  # SMA10 vs SMA20
            ema_12 / ema_26 - 1,  # MACD-like
        ])
        
        # 2. Momentum indicators
        rsi = self._calculate_rsi(closes, 14)
        roc_5 = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        roc_10 = (closes[-1] / closes[-11] - 1) if len(closes) >= 11 else 0
        
        features.extend([
            rsi / 100,  # Normalized RSI
            roc_5,  # 5-period rate of change
            roc_10,  # 10-period rate of change
        ])
        
        # 3. Volatility features
        returns = np.diff(closes) / closes[:-1]
        volatility_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
        volatility_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0
        atr = self._calculate_atr(highs, lows, closes, 14)
        
        features.extend([
            volatility_5,
            volatility_20,
            atr / closes[-1],  # Normalized ATR
            volatility_5 / volatility_20 if volatility_20 > 0 else 0,
        ])
        
        # 4. Volume features
        avg_volume_10 = np.mean(volumes[-10:])
        volume_ratio = volumes[-1] / avg_volume_10 if avg_volume_10 > 0 else 1
        volume_trend = np.polyfit(range(5), volumes[-5:], 1)[0] if len(volumes) >= 5 else 0
        
        features.extend([
            volume_ratio,
            volume_trend / avg_volume_10 if avg_volume_10 > 0 else 0,
        ])
        
        # 5. Price pattern features
        high_low_ratio = (highs[-1] - lows[-1]) / closes[-1] if closes[-1] > 0 else 0
        upper_shadow = (highs[-1] - max(df['open'].iloc[-1], closes[-1])) / closes[-1]
        lower_shadow = (min(df['open'].iloc[-1], closes[-1]) - lows[-1]) / closes[-1]
        
        features.extend([
            high_low_ratio,
            upper_shadow,
            lower_shadow,
        ])
        
        # 6. Support/Resistance features
        recent_highs = np.max(highs[-10:])
        recent_lows = np.min(lows[-10:])
        price_position = (closes[-1] - recent_lows) / (recent_highs - recent_lows) if recent_highs != recent_lows else 0.5
        
        features.extend([
            closes[-1] / recent_highs - 1,  # Distance from recent high
            closes[-1] / recent_lows - 1,   # Distance from recent low
            price_position,  # Position in recent range
        ])
        
        # 7. Trend strength features
        trend_strength = self._calculate_trend_strength(closes)
        trend_consistency = self._calculate_trend_consistency(closes)
        
        features.extend([
            trend_strength,
            trend_consistency,
        ])
        
        return np.array(features)
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1]
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return 0.0
        
        tr_values = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
        
        return np.mean(tr_values[-period:])
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return slope / prices[-1]  # Normalized slope
    
    def _calculate_trend_consistency(self, prices: np.ndarray) -> float:
        """Calculate how consistent the trend is"""
        if len(prices) < 10:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        positive_returns = np.sum(returns > 0)
        total_returns = len(returns)
        
        # Consistency score: how often returns match the overall trend
        overall_trend = 1 if prices[-1] > prices[0] else -1
        consistent_moves = np.sum((returns > 0) == (overall_trend > 0))
        
        return consistent_moves / total_returns
    
    def prepare_training_data(self, historical_candles: List[Dict], lookback: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with labels"""
        features_list = []
        labels_list = []
        
        for i in range(lookback, len(historical_candles) - 10):  # Leave 10 candles for future price
            # Extract features
            candle_slice = historical_candles[:i+1]
            features = self.extract_features(candle_slice, lookback)
            
            if len(features) == 0:
                continue
            
            # Create label based on future price movement
            current_price = historical_candles[i]['close']
            future_prices = [historical_candles[j]['close'] for j in range(i+1, min(i+11, len(historical_candles)))]
            
            if not future_prices:
                continue
            
            # Label logic: 1 for profitable long, 2 for profitable short, 0 for hold
            max_future = max(future_prices)
            min_future = min(future_prices)
            
            up_move = (max_future - current_price) / current_price
            down_move = (current_price - min_future) / current_price
            
            # Conservative labeling for high win rate
            if up_move > 0.015 and up_move > down_move * 1.5:  # 1.5% up move, clearly bullish
                label = 1  # Long
            elif down_move > 0.015 and down_move > up_move * 1.5:  # 1.5% down move, clearly bearish
                label = 2  # Short
            else:
                label = 0  # Hold
            
            features_list.append(features)
            labels_list.append(label)
        
        return np.array(features_list), np.array(labels_list)
    
    def train_models(self, historical_candles: List[Dict]) -> Dict:
        """Train the ensemble of ML models"""
        logger.info("🤖 Training AI models for pattern recognition...")
        
        # Prepare training data
        X, y = self.prepare_training_data(historical_candles)
        
        if len(X) == 0:
            logger.error("No training data available")
            return {}
        
        logger.info(f"Training on {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Label distribution: Hold={np.sum(y==0)}, Long={np.sum(y==1)}, Short={np.sum(y==2)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train each model
        results = {}
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            logger.info(f"{name} accuracy: {accuracy:.3f}")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        self.is_trained = True
        logger.info("🎯 AI model training completed!")
        
        return results
    
    def generate_signal(self, candles: List[Dict]) -> AISignal:
        """Generate AI-enhanced trading signal"""
        if not self.is_trained:
            return AISignal('hold', 0.0, 0.0, {}, {}, 1.0)
        
        # Extract features
        features = self.extract_features(candles)
        if len(features) == 0:
            return AISignal('hold', 0.0, 0.0, {}, {}, 1.0)
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions from ensemble
        ensemble_votes = {}
        probabilities = {}
        
        for name, model in self.models.items():
            prediction = model.predict(features_scaled)[0]
            ensemble_votes[name] = prediction
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                probabilities[name] = proba
        
        # Ensemble decision making
        votes = list(ensemble_votes.values())
        long_votes = votes.count(1)
        short_votes = votes.count(2)
        hold_votes = votes.count(0)
        
        total_models = len(self.models)
        
        # Conservative approach for high win rate
        min_consensus = 0.8  # 80% of models must agree
        
        if long_votes >= total_models * min_consensus:
            direction = 'long'
            confidence = long_votes / total_models
            avg_proba = np.mean([p[1] for p in probabilities.values() if len(p) > 1])
        elif short_votes >= total_models * min_consensus:
            direction = 'short'
            confidence = short_votes / total_models
            avg_proba = np.mean([p[2] for p in probabilities.values() if len(p) > 2])
        else:
            direction = 'hold'
            confidence = max(hold_votes, long_votes, short_votes) / total_models
            avg_proba = 0.5
        
        # Calculate risk score based on market conditions
        risk_score = self._calculate_risk_score(candles, features)
        
        # Create feature dictionary for analysis
        feature_dict = {
            'price_vs_sma5': features[0],
            'price_vs_sma10': features[1],
            'rsi_normalized': features[6],
            'volatility_ratio': features[10],
            'volume_ratio': features[11],
            'trend_strength': features[19],
            'trend_consistency': features[20]
        }
        
        return AISignal(
            direction=direction,
            confidence=confidence,
            probability=avg_proba,
            features=feature_dict,
            ensemble_votes=ensemble_votes,
            risk_score=risk_score
        )
    
    def _calculate_risk_score(self, candles: List[Dict], features: np.ndarray) -> float:
        """Calculate risk score for the current market conditions"""
        if len(candles) < 20:
            return 1.0
        
        # Factors that increase risk
        risk_factors = 0
        
        # High volatility
        if features[9] > 0.02:  # High volatility
            risk_factors += 0.3
        
        # Extreme RSI
        rsi = features[6] * 100
        if rsi > 80 or rsi < 20:
            risk_factors += 0.2
        
        # Low trend consistency
        if features[20] < 0.6:  # Low trend consistency
            risk_factors += 0.2
        
        # High volume spike
        if features[11] > 2.0:  # Volume ratio > 2
            risk_factors += 0.1
        
        # Gap in price
        recent_candles = candles[-5:]
        price_gaps = []
        for i in range(1, len(recent_candles)):
            gap = abs(recent_candles[i]['open'] - recent_candles[i-1]['close']) / recent_candles[i-1]['close']
            price_gaps.append(gap)
        
        if max(price_gaps) > 0.01:  # 1% gap
            risk_factors += 0.2
        
        return min(risk_factors, 1.0)

class AIEnhanced70WRBot:
    """AI-Enhanced Trading Bot targeting 70%+ Win Rate"""
    
    def __init__(self, initial_balance: float = 50.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades = []
        self.positions = {}
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
        # Initialize AI components
        self.ai_recognizer = AIPatternRecognizer()
        self.risk_manager = AdvancedRiskManager()
        self.detector = AdvancedTopBottomDetector()
        
        # Enhanced trading parameters for high win rate
        self.max_position_size = 0.01  # 1% of balance per trade (conservative)
        self.min_ai_confidence = 0.8   # 80% AI confidence required
        self.max_risk_score = 0.3      # Maximum risk score allowed
        self.stop_loss_pct = 0.01      # 1% stop loss (tight)
        self.take_profit_pct = 0.015   # 1.5% take profit (conservative R:R)
        self.min_trade_size = 0.001    # Minimum trade size
        
        # Market regime tracking
        self.current_regime = 'UNKNOWN'
        self.regime_confidence = 0.0
        
    def generate_training_data(self, symbol: str, days: int = 90) -> List[Dict]:
        """Generate comprehensive training data"""
        candles = []
        start_time = datetime.now() - timedelta(days=days)
        current_time = start_time
        
        # Base prices for different symbols
        base_prices = {
            'ETH': 3500.0,
            'BTC': 65000.0,
            'AVAX': 45.0,
            'DOGE': 0.35
        }
        
        base_price = base_prices.get(symbol, 3500.0)
        current_price = base_price
        
        # Generate realistic market data with various regimes
        regime_length = (days * 288) // 4  # 4 regimes over the period
        current_regime_candles = 0
        regime_type = 0  # 0: bull, 1: bear, 2: range, 3: volatile
        
        while current_time < datetime.now():
            # Switch regimes
            if current_regime_candles >= regime_length:
                regime_type = (regime_type + 1) % 4
                current_regime_candles = 0
            
            # Generate price movement based on regime
            if regime_type == 0:  # Bull trend
                base_move = np.random.normal(0.0002, 0.008)  # Slight upward bias
                volatility = 0.008
            elif regime_type == 1:  # Bear trend
                base_move = np.random.normal(-0.0002, 0.008)  # Slight downward bias
                volatility = 0.010
            elif regime_type == 2:  # Range bound
                base_move = np.random.normal(0, 0.005)  # No bias, low volatility
                volatility = 0.005
            else:  # High volatility
                base_move = np.random.normal(0, 0.015)  # No bias, high volatility
                volatility = 0.015
            
            # Add some noise and mean reversion
            price_change = base_move + np.random.normal(0, volatility)
            
            # Mean reversion factor
            deviation_from_base = (current_price - base_price) / base_price
            if abs(deviation_from_base) > 0.2:  # If price deviates too much
                price_change -= deviation_from_base * 0.1  # Pull back towards base
            
            # OHLC generation
            open_price = current_price
            close_price = current_price * (1 + price_change)
            
            # Realistic high/low with wicks
            wick_factor = np.random.uniform(0.3, 0.7)
            high_price = max(open_price, close_price) * (1 + abs(price_change) * wick_factor)
            low_price = min(open_price, close_price) * (1 - abs(price_change) * wick_factor)
            
            # Volume (correlated with volatility)
            base_volume = 1000 if symbol == 'ETH' else 10000
            volume = base_volume * (1 + abs(price_change) * 10) * np.random.lognormal(0, 0.5)
            
            candle = {
                'timestamp': int(current_time.timestamp() * 1000),
                'open': round(open_price, 8),
                'high': round(high_price, 8),
                'low': round(low_price, 8),
                'close': round(close_price, 8),
                'volume': round(volume, 4)
            }
            
            candles.append(candle)
            current_price = close_price
            current_time += timedelta(minutes=5)
            current_regime_candles += 1
        
        logger.info(f"Generated {len(candles)} training candles for {symbol}")
        return candles
    
    def train_ai_system(self, symbol: str = 'ETH') -> Dict:
        """Train the AI system on historical data"""
        logger.info(f"🧠 Training AI system for {symbol}...")
        
        # Generate comprehensive training data
        training_data = self.generate_training_data(symbol, days=180)  # 6 months of data
        
        # Train AI models
        training_results = self.ai_recognizer.train_models(training_data)
        
        return training_results
    
    def analyze_market_regime(self, candles: List[Dict]) -> Tuple[str, float]:
        """Analyze current market regime"""
        if len(candles) < 50:
            return 'UNKNOWN', 0.0
        
        recent_candles = candles[-50:]
        closes = [c['close'] for c in recent_candles]
        volumes = [c['volume'] for c in recent_candles]
        
        # Trend analysis
        trend_slope = np.polyfit(range(len(closes)), closes, 1)[0]
        trend_strength = abs(trend_slope) / closes[-1]
        
        # Volatility analysis
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns)
        
        # Volume analysis
        avg_volume = np.mean(volumes)
        recent_volume = np.mean(volumes[-10:])
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Regime classification
        if trend_strength > 0.001 and trend_slope > 0:
            regime = 'BULL_TREND'
            confidence = min(trend_strength * 1000, 1.0)
        elif trend_strength > 0.001 and trend_slope < 0:
            regime = 'BEAR_TREND'
            confidence = min(trend_strength * 1000, 1.0)
        elif volatility > 0.015:
            regime = 'HIGH_VOLATILITY'
            confidence = min(volatility * 50, 1.0)
        else:
            regime = 'RANGE_BOUND'
            confidence = 1.0 - trend_strength * 500
        
        return regime, confidence
    
    def should_trade(self, ai_signal: AISignal, market_regime: str) -> bool:
        """Determine if we should execute the trade based on strict criteria"""
        # Minimum AI confidence
        if ai_signal.confidence < self.min_ai_confidence:
            return False
        
        # Maximum risk score
        if ai_signal.risk_score > self.max_risk_score:
            return False
        
        # No trading in extreme market conditions
        if market_regime == 'HIGH_VOLATILITY' and ai_signal.risk_score > 0.2:
            return False
        
        # Require strong trend consistency for trend-following trades
        if market_regime in ['BULL_TREND', 'BEAR_TREND']:
            if ai_signal.features.get('trend_consistency', 0) < 0.7:
                return False
        
        # No trading if signal is 'hold'
        if ai_signal.direction == 'hold':
            return False
        
        return True
    
    def calculate_position_size(self, price: float, ai_signal: AISignal) -> float:
        """Calculate position size based on AI confidence and risk"""
        # Base position size
        base_risk = self.current_balance * self.max_position_size
        
        # Adjust based on AI confidence
        confidence_multiplier = ai_signal.confidence  # Higher confidence = larger size
        
        # Adjust based on risk score (lower risk = larger size)
        risk_multiplier = 1.0 - ai_signal.risk_score
        
        # Final position value
        position_value = base_risk * confidence_multiplier * risk_multiplier
        position_size = position_value / price
        
        # Ensure minimum trade size
        position_size = max(position_size, self.min_trade_size)
        
        # Ensure we don't exceed balance
        max_affordable = (self.current_balance * 0.95) / price
        position_size = min(position_size, max_affordable)
        
        return round(position_size, 8)
    
    def execute_trade(self, signal: AISignal, price: float, timestamp: datetime, 
                     position_size: float, market_regime: str) -> Optional[Dict]:
        """Execute a trade with AI-enhanced parameters"""
        if signal.direction not in ['long', 'short']:
            return None
        
        trade_value = position_size * price
        
        # Check if we can afford the trade
        if trade_value > self.current_balance * 0.95:
            return None
        
        # Calculate dynamic stop loss and take profit based on AI confidence
        base_stop = self.stop_loss_pct
        base_profit = self.take_profit_pct
        
        # Tighter stops for lower confidence, wider for higher confidence
        stop_multiplier = 2.0 - signal.confidence  # 1.2x to 2.0x
        profit_multiplier = 0.5 + signal.confidence  # 1.0x to 1.5x
        
        if signal.direction == 'long':
            stop_loss = price * (1 - base_stop * stop_multiplier)
            take_profit = price * (1 + base_profit * profit_multiplier)
        else:
            stop_loss = price * (1 + base_stop * stop_multiplier)
            take_profit = price * (1 - base_profit * profit_multiplier)
        
        trade = {
            'id': len(self.trades) + 1,
            'entry_time': timestamp,
            'symbol': 'ETH',  # Default symbol
            'side': signal.direction,
            'entry_price': price,
            'quantity': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'ai_confidence': signal.confidence,
            'ai_probability': signal.probability,
            'market_regime': market_regime,
            'status': 'open'
        }
        
        self.positions[trade['id']] = trade
        logger.info(f"🤖 AI Trade: {signal.direction} {position_size:.6f} @ ${price:.2f} "
                   f"(Conf: {signal.confidence:.2f}, Risk: {signal.risk_score:.2f})")
        
        return trade
    
    def check_exit_conditions(self, trade: Dict, current_price: float, 
                            timestamp: datetime) -> Optional[Tuple[str, float]]:
        """Check if trade should be exited"""
        if trade['side'] == 'long':
            if current_price <= trade['stop_loss']:
                return 'stop_loss', trade['stop_loss']
            elif current_price >= trade['take_profit']:
                return 'take_profit', trade['take_profit']
        else:  # short
            if current_price >= trade['stop_loss']:
                return 'stop_loss', trade['stop_loss']
            elif current_price <= trade['take_profit']:
                return 'take_profit', trade['take_profit']
        
        # Time-based exit (max 2 hours for high frequency)
        holding_time = (timestamp - trade['entry_time']).total_seconds() / 3600
        if holding_time > 2:
            return 'time_exit', current_price
        
        return None
    
    def close_trade(self, trade_id: int, exit_reason: str, exit_price: float, 
                   exit_time: datetime):
        """Close a trade and record results"""
        trade = self.positions[trade_id]
        
        # Calculate PnL
        if trade['side'] == 'long':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']
        
        pnl_pct = (pnl / (trade['entry_price'] * trade['quantity'])) * 100
        
        # Update balance
        self.current_balance += pnl
        
        # Track peak balance and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Record trade result
        holding_time = (exit_time - trade['entry_time']).total_seconds() / 60
        
        trade_result = TradeResult(
            entry_time=trade['entry_time'],
            exit_time=exit_time,
            symbol=trade['symbol'],
            side=trade['side'],
            entry_price=trade['entry_price'],
            exit_price=exit_price,
            quantity=trade['quantity'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_time_minutes=holding_time,
            exit_reason=exit_reason,
            ai_confidence=trade['ai_confidence'],
            ai_probability=trade['ai_probability'],
            market_regime=trade['market_regime']
        )
        
        self.trades.append(trade_result)
        del self.positions[trade_id]
        
        result_emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"{result_emoji} Closed {trade['side']}: PnL ${pnl:.2f} ({pnl_pct:.2f}%) - {exit_reason}")
    
    def run_backtest(self, symbol: str = 'ETH', days: int = 30) -> Dict:
        """Run AI-enhanced backtest"""
        logger.info(f"🚀 Starting AI-Enhanced 70%+ WR Backtest for {symbol}")
        
        # First train the AI system
        training_results = self.train_ai_system(symbol)
        logger.info(f"AI Training Results: {training_results}")
        
        # Generate test data
        test_candles = self.generate_training_data(symbol, days)
        
        # Process each candle
        for i in range(100, len(test_candles)):  # Start after enough data for AI
            current_candle = test_candles[i]
            current_time = datetime.fromtimestamp(current_candle['timestamp'] / 1000)
            current_price = current_candle['close']
            
            # Analyze market regime
            self.current_regime, self.regime_confidence = self.analyze_market_regime(test_candles[:i+1])
            
            # Check exit conditions for open positions
            positions_to_close = []
            for trade_id, trade in self.positions.items():
                exit_condition = self.check_exit_conditions(trade, current_price, current_time)
                if exit_condition:
                    exit_reason, exit_price = exit_condition
                    positions_to_close.append((trade_id, exit_reason, exit_price))
            
            # Close positions
            for trade_id, exit_reason, exit_price in positions_to_close:
                self.close_trade(trade_id, exit_reason, exit_price, current_time)
            
            # Generate new signals if no open positions
            if not self.positions:
                ai_signal = self.ai_recognizer.generate_signal(test_candles[:i+1])
                
                if self.should_trade(ai_signal, self.current_regime):
                    position_size = self.calculate_position_size(current_price, ai_signal)
                    
                    if position_size > 0:
                        self.execute_trade(ai_signal, current_price, current_time, 
                                         position_size, self.current_regime)
        
        # Close any remaining positions
        final_time = datetime.fromtimestamp(test_candles[-1]['timestamp'] / 1000)
        final_price = test_candles[-1]['close']
        
        for trade_id in list(self.positions.keys()):
            self.close_trade(trade_id, 'backtest_end', final_price, final_time)
        
        return self.calculate_results()
    
    def calculate_results(self) -> Dict:
        """Calculate comprehensive results"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return_pct': 0.0,
                'ai_avg_confidence': 0.0,
                'profitable_by_confidence': {}
            }
        
        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        win_rate = winning_trades / total_trades
        
        # AI-specific metrics
        ai_confidences = [t.ai_confidence for t in self.trades]
        avg_ai_confidence = np.mean(ai_confidences)
        
        # Performance by confidence level
        confidence_ranges = [(0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)]
        profitable_by_confidence = {}
        
        for low, high in confidence_ranges:
            range_trades = [t for t in self.trades if low <= t.ai_confidence < high]
            if range_trades:
                range_wins = len([t for t in range_trades if t.pnl > 0])
                range_wr = range_wins / len(range_trades)
                profitable_by_confidence[f"{low:.2f}-{high:.2f}"] = {
                    'trades': len(range_trades),
                    'win_rate': range_wr,
                    'avg_pnl': np.mean([t.pnl for t in range_trades])
                }
        
        # Performance by market regime
        regime_performance = {}
        for regime in ['BULL_TREND', 'BEAR_TREND', 'RANGE_BOUND', 'HIGH_VOLATILITY']:
            regime_trades = [t for t in self.trades if t.market_regime == regime]
            if regime_trades:
                regime_wins = len([t for t in regime_trades if t.pnl > 0])
                regime_wr = regime_wins / len(regime_trades)
                regime_performance[regime] = {
                    'trades': len(regime_trades),
                    'win_rate': regime_wr,
                    'avg_pnl': np.mean([t.pnl for t in regime_trades])
                }
        
        # Return metrics
        total_return_pct = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': self.max_drawdown * 100,
            'ai_avg_confidence': avg_ai_confidence,
            'profitable_by_confidence': profitable_by_confidence,
            'regime_performance': regime_performance,
            'avg_holding_time': np.mean([t.holding_time_minutes for t in self.trades])
        }

def main():
    """Run AI-Enhanced 70%+ Win Rate Bot"""
    print("🤖 AI-ENHANCED 70%+ WIN RATE TRADING BOT")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        {"symbol": "ETH", "days": 14, "name": "ETH 2-Week Test"},
        {"symbol": "ETH", "days": 30, "name": "ETH 1-Month Test"},
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n🧠 Running {config['name']}...")
        print("-" * 40)
        
        bot = AIEnhanced70WRBot(50.0)  # $50 starting balance
        results = bot.run_backtest(config['symbol'], config['days'])
        
        # Display results
        print(f"💰 RESULTS FOR {config['name']}:")
        print(f"   Initial Balance: ${results['initial_balance']:.2f}")
        print(f"   Final Balance: ${results['final_balance']:.2f}")
        print(f"   Total Return: {results['total_return_pct']:.2f}%")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   🎯 WIN RATE: {results['win_rate']:.1%}")
        print(f"   Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"   AI Avg Confidence: {results['ai_avg_confidence']:.2f}")
        print(f"   Avg Holding Time: {results['avg_holding_time']:.1f} minutes")
        
        # AI Performance Analysis
        print(f"\n🤖 AI PERFORMANCE BY CONFIDENCE LEVEL:")
        for conf_range, stats in results['profitable_by_confidence'].items():
            print(f"   {conf_range}: {stats['trades']} trades, {stats['win_rate']:.1%} WR, ${stats['avg_pnl']:.2f} avg PnL")
        
        # Market Regime Analysis
        print(f"\n📊 PERFORMANCE BY MARKET REGIME:")
        for regime, stats in results['regime_performance'].items():
            print(f"   {regime}: {stats['trades']} trades, {stats['win_rate']:.1%} WR, ${stats['avg_pnl']:.2f} avg PnL")
        
        all_results.append({
            'config': config['name'],
            'results': results
        })
    
    # Summary
    print(f"\n🎯 AI-ENHANCED SUMMARY")
    print("=" * 60)
    
    for result in all_results:
        r = result['results']
        status = "🎯 TARGET ACHIEVED!" if r['win_rate'] >= 0.7 else "🔄 Needs Optimization"
        print(f"{result['config']}: {r['win_rate']:.1%} WR, {r['total_return_pct']:.2f}% return - {status}")
    
    # Save detailed results
    with open('ai_enhanced_70wr_results.json', 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'target_win_rate': 0.70,
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Detailed AI results saved to 'ai_enhanced_70wr_results.json'")
    
    # Recommendations
    best_result = max(all_results, key=lambda x: x['results']['win_rate'])
    print(f"\n💡 BEST PERFORMANCE: {best_result['config']}")
    print(f"   Win Rate: {best_result['results']['win_rate']:.1%}")
    print(f"   Return: {best_result['results']['total_return_pct']:.2f}%")
    print(f"   AI Confidence: {best_result['results']['ai_avg_confidence']:.2f}")

if __name__ == "__main__":
    main() 