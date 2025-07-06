#!/usr/bin/env python3
"""
Improved Edge System - Fixing Negative Expectancy
Focus: Better R:R, stricter filters, ATR-based stops
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedEdgeSystem:
    """Trading system with improved edge and positive expectancy"""
    
    def __init__(self):
        # Ensemble with weighted models
        self.models = {
            'rf_main': {
                'model': RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=3,
                    min_samples_leaf=1, random_state=42
                ),
                'weight': 0.4  # Higher weight for better performer
            },
            'gb_secondary': {
                'model': GradientBoostingClassifier(
                    n_estimators=150, max_depth=8, learning_rate=0.1,
                    min_samples_split=5, random_state=42
                ),
                'weight': 0.2  # Lower weight for weaker model
            },
            'nn_lstm_proxy': {  # Simple NN as LSTM proxy for testing
                'model': MLPClassifier(
                    hidden_layer_sizes=(100, 50), activation='relu',
                    solver='adam', max_iter=500, random_state=42
                ),
                'weight': 0.4
            }
        }
        
        # Meta-learner for ensemble
        self.meta_learner = LogisticRegression(random_state=42)
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Improved thresholds
        self.min_prob_threshold = 0.60      # Raised from 0.55
        self.min_edge_score = 0.0025        # 0.25% minimum edge
        self.atr_multiplier_sl = 1.0        # 1 ATR stop loss
        self.atr_multiplier_tp = 4.0        # 4 ATR take profit (4:1 R:R)
        
        # Error buffer for online learning
        self.error_buffer = []
        self.max_buffer_size = 500
        
    def calculate_atr(self, candles: list, period: int = 14) -> float:
        """Calculate ATR for dynamic stops"""
        if len(candles) < period + 1:
            return 0.0
        
        high = np.array([c['high'] for c in candles[-period-1:]])
        low = np.array([c['low'] for c in candles[-period-1:]])
        close = np.array([c['close'] for c in candles[-period-1:]])
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        
        return np.mean(tr)
    
    def calculate_order_book_imbalance(self, bid_volume: float, ask_volume: float) -> float:
        """Calculate order book imbalance"""
        total = bid_volume + ask_volume
        if total == 0:
            return 0
        return (bid_volume - ask_volume) / total
    
    def is_high_volatility(self, candles: list, lookback: int = 100) -> bool:
        """Check if current volatility is above median"""
        if len(candles) < lookback:
            return True
        
        # Calculate rolling ATR values
        atr_values = []
        for i in range(lookback - 14, lookback):
            atr = self.calculate_atr(candles[:i+1])
            if atr > 0:
                atr_values.append(atr)
        
        if not atr_values:
            return True
        
        current_atr = self.calculate_atr(candles)
        median_atr = np.median(atr_values)
        
        return current_atr > median_atr
    
    def extract_quality_features(self, candles: list, lookback: int = 60) -> np.ndarray:
        """Extract high-quality features for better prediction"""
        if len(candles) < lookback:
            return np.array([])
        
        recent = candles[-lookback:]
        closes = np.array([c['close'] for c in recent])
        highs = np.array([c['high'] for c in recent])
        lows = np.array([c['low'] for c in recent])
        volumes = np.array([c['volume'] for c in recent])
        
        features = []
        
        # 1. Multi-timeframe momentum
        for period in [5, 10, 20, 40]:
            if len(closes) >= period:
                momentum = (closes[-1] - closes[-period]) / closes[-period]
                features.append(momentum)
        
        # 2. ATR-normalized moves
        atr = self.calculate_atr(candles)
        if atr > 0:
            atr_norm = atr / closes[-1]
            features.append(atr_norm)
            
            # Recent move in ATR terms
            if len(closes) >= 5:
                recent_move = (closes[-1] - closes[-5]) / atr
                features.append(recent_move)
        
        # 3. RSI with multiple periods
        for period in [7, 14, 21]:
            rsi = self._calculate_rsi(closes, period)
            features.append(rsi / 100)
        
        # 4. Volume profile
        vol_avg = np.mean(volumes[-20:])
        vol_std = np.std(volumes[-20:])
        if vol_avg > 0:
            vol_zscore = (volumes[-1] - vol_avg) / (vol_std if vol_std > 0 else 1)
            features.append(np.clip(vol_zscore, -3, 3))
        
        # 5. Market microstructure
        high_20 = np.max(highs[-20:])
        low_20 = np.min(lows[-20:])
        range_20 = high_20 - low_20
        
        if range_20 > 0:
            # Position in range
            position = (closes[-1] - low_20) / range_20
            features.append(position)
            
            # Proximity to extremes
            dist_high = (high_20 - closes[-1]) / closes[-1]
            dist_low = (closes[-1] - low_20) / closes[-1]
            features.extend([dist_high, dist_low])
        
        # 6. Trend strength with R-squared
        for period in [10, 20]:
            if len(closes) >= period:
                x = np.arange(period)
                slope, intercept = np.polyfit(x, closes[-period:], 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((closes[-period:] - y_pred) ** 2)
                ss_tot = np.sum((closes[-period:] - np.mean(closes[-period:])) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                features.append(r_squared * np.sign(slope))
        
        # 7. Bollinger Band position
        if len(closes) >= 20:
            bb_mean = np.mean(closes[-20:])
            bb_std = np.std(closes[-20:])
            if bb_std > 0:
                bb_zscore = (closes[-1] - bb_mean) / bb_std
                features.append(np.clip(bb_zscore, -3, 3))
        
        return np.array(features)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use EMA for smoothing
        alpha = 1.0 / period
        avg_gain = gains[0]
        avg_loss = losses[0]
        
        for i in range(1, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def prepare_training_data(self, candles: list) -> tuple:
        """Prepare training data with 4:1 R:R targets"""
        features_list = []
        labels_list = []
        meta_features_list = []
        
        for i in range(60, len(candles) - 30):
            candle_slice = candles[:i+1]
            features = self.extract_quality_features(candle_slice)
            
            if len(features) == 0:
                continue
            
            # Calculate dynamic stops
            atr = self.calculate_atr(candle_slice)
            current_price = candles[i]['close']
            
            stop_distance = atr * self.atr_multiplier_sl
            target_distance = atr * self.atr_multiplier_tp
            
            # Look ahead for outcomes
            future_prices = [candles[j]['close'] for j in range(i+1, min(i+31, len(candles)))]
            
            if not future_prices:
                continue
            
            max_future = max(future_prices)
            min_future = min(future_prices)
            
            # Check if 4:1 R:R achieved
            long_target_hit = max_future >= current_price + target_distance
            long_stop_hit = min_future <= current_price - stop_distance
            
            short_target_hit = min_future <= current_price - target_distance
            short_stop_hit = max_future >= current_price + stop_distance
            
            # Label based on which happens first
            if long_target_hit and not long_stop_hit:
                label = 1  # Long winner
            elif short_target_hit and not short_stop_hit:
                label = 2  # Short winner
            else:
                label = 0  # No clear edge
            
            features_list.append(features)
            labels_list.append(label)
            
            # Placeholder for meta features (will be filled after base models train)
            meta_features_list.append(np.zeros(len(self.models)))
        
        return np.array(features_list), np.array(labels_list), np.array(meta_features_list)
    
    def train_models(self, candles: list) -> dict:
        """Train ensemble with meta-learner"""
        logger.info("🧠 Training Improved Edge Models...")
        
        X, y, meta_X = self.prepare_training_data(candles)
        
        if len(X) == 0:
            return {}
        
        # Log class distribution
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Training samples: {len(X)}")
        for u, c in zip(unique, counts):
            label_name = ['Neutral', 'Long', 'Short'][int(u)]
            logger.info(f"  {label_name}: {c} ({c/len(X):.1%})")
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Train base models
        results = {}
        base_predictions = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            model.fit(X_train, y_train)
            
            # Get predictions for meta-learner
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                base_predictions.append(proba)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            
            # Edge-only accuracy
            edge_mask = y_test != 0
            if np.any(edge_mask):
                edge_accuracy = np.mean(y_pred[edge_mask] == y_test[edge_mask])
                results[name] = {
                    'overall_acc': accuracy,
                    'edge_acc': edge_accuracy
                }
                logger.info(f"{name}: {accuracy:.1%} overall, {edge_accuracy:.1%} on edge trades")
            else:
                results[name] = {'overall_acc': accuracy}
                logger.info(f"{name}: {accuracy:.1%} overall")
        
        # Train meta-learner if we have base predictions
        if base_predictions:
            # Stack base model predictions
            meta_features = np.hstack([p[:, 1:] for p in base_predictions])  # Exclude neutral class
            
            # Train meta-learner
            self.meta_learner.fit(meta_features, y_test)
            meta_accuracy = self.meta_learner.score(meta_features, y_test)
            logger.info(f"Meta-learner: {meta_accuracy:.1%} accuracy")
        
        self.is_trained = True
        return results
    
    def generate_signal(self, candles: list, order_book: dict = None) -> dict:
        """Generate high-quality signal with strict filters"""
        if not self.is_trained:
            return {'direction': 'hold', 'confidence': 0.0, 'edge': 0.0}
        
        features = self.extract_quality_features(candles)
        if len(features) == 0:
            return {'direction': 'hold', 'confidence': 0.0, 'edge': 0.0}
        
        # Scale features
        features = np.nan_to_num(features, nan=0.0)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get base model predictions
        base_probas = []
        weighted_votes = {'long': 0, 'short': 0, 'hold': 0}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            weight = model_info['weight']
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                base_probas.append(proba)
                
                # Weighted voting
                if len(proba) > 2:
                    weighted_votes['hold'] += proba[0] * weight
                    weighted_votes['long'] += proba[1] * weight
                    weighted_votes['short'] += proba[2] * weight
        
        # Determine direction from weighted ensemble
        max_vote = max(weighted_votes.values())
        if weighted_votes['long'] == max_vote and weighted_votes['long'] > self.min_prob_threshold:
            direction = 'long'
            confidence = weighted_votes['long']
        elif weighted_votes['short'] == max_vote and weighted_votes['short'] > self.min_prob_threshold:
            direction = 'short'
            confidence = weighted_votes['short']
        else:
            direction = 'hold'
            confidence = weighted_votes['hold']
        
        # Calculate edge
        atr = self.calculate_atr(candles)
        current_price = candles[-1]['close']
        atr_pct = atr / current_price if current_price > 0 else 0.01
        
        if direction != 'hold':
            # Expected value with 4:1 R:R
            win_prob = confidence
            reward = atr_pct * self.atr_multiplier_tp
            risk = atr_pct * self.atr_multiplier_sl
            edge = (win_prob * reward) - ((1 - win_prob) * risk)
        else:
            edge = 0.0
            win_prob = 0.0
        
        # Additional filters
        rsi = self._calculate_rsi(np.array([c['close'] for c in candles[-15:]]), 14)
        high_volatility = self.is_high_volatility(candles)
        
        # Order book filter
        obi = 0.0
        if order_book:
            bid_vol = order_book.get('bid_volume', 0)
            ask_vol = order_book.get('ask_volume', 0)
            obi = self.calculate_order_book_imbalance(bid_vol, ask_vol)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'edge': edge,
            'win_prob': win_prob,
            'atr': atr,
            'atr_pct': atr_pct,
            'rsi': rsi,
            'high_volatility': high_volatility,
            'obi': obi,
            'stop_distance': atr * self.atr_multiplier_sl,
            'target_distance': atr * self.atr_multiplier_tp,
            'risk_reward_ratio': self.atr_multiplier_tp / self.atr_multiplier_sl
        }
    
    def should_trade(self, signal: dict) -> bool:
        """Apply strict filters for positive edge"""
        if signal['direction'] == 'hold':
            return False
        
        # 1. Minimum edge requirement
        if signal['edge'] < self.min_edge_score:
            return False
        
        # 2. Probability gate
        if signal['confidence'] < self.min_prob_threshold:
            return False
        
        # 3. Volatility filter
        if not signal['high_volatility']:
            return False
        
        # 4. Additional quality filters (need 2 of 3)
        quality_checks = 0
        
        # RSI filter
        if (signal['direction'] == 'long' and signal['rsi'] < 30) or \
           (signal['direction'] == 'short' and signal['rsi'] > 70):
            quality_checks += 1
        
        # Order book filter
        if (signal['direction'] == 'long' and signal['obi'] > -0.05) or \
           (signal['direction'] == 'short' and signal['obi'] < 0.05):
            quality_checks += 1
        
        # Strong confidence
        if signal['confidence'] >= 0.65:
            quality_checks += 1
        
        return quality_checks >= 2
    
    def update_error_buffer(self, features: np.ndarray, predicted: int, actual: int):
        """Store prediction errors for online learning"""
        if predicted != actual:
            self.error_buffer.append({
                'features': features,
                'predicted': predicted,
                'actual': actual,
                'timestamp': datetime.now()
            })
            
            # Keep buffer size limited
            if len(self.error_buffer) > self.max_buffer_size:
                self.error_buffer.pop(0)
    
    def online_update(self):
        """Fine-tune models on error buffer"""
        if len(self.error_buffer) < 50:  # Need minimum samples
            return
        
        logger.info(f"🔄 Online update with {len(self.error_buffer)} error samples")
        
        # Prepare error data
        X_errors = np.array([e['features'] for e in self.error_buffer])
        y_errors = np.array([e['actual'] for e in self.error_buffer])
        
        # Scale features
        X_errors_scaled = self.scaler.transform(X_errors)
        
        # Fine-tune neural network (most adaptable)
        nn_model = self.models['nn_lstm_proxy']['model']
        nn_model.partial_fit(X_errors_scaled, y_errors)
        
        # Clear old errors
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.error_buffer = [e for e in self.error_buffer if e['timestamp'] > cutoff_time]

def run_improved_backtest():
    """Run backtest with improved system"""
    print("🚀 IMPROVED EDGE SYSTEM TEST")
    print("=" * 50)
    print("Focus: 4:1 R:R, Strict filters, Positive edge\n")
    
    # Initialize system
    system = ImprovedEdgeSystem()
    
    # Generate test data
    days = 60
    candles = []
    base_price = 100
    current_price = base_price
    
    start_time = datetime.now() - timedelta(days=days)
    
    for i in range(days * 288):  # 5-minute candles
        # Create varied market conditions
        regime = (i % 2000) / 2000
        
        if regime < 0.3:  # Trending
            trend = 0.0002
            volatility = 0.003
        elif regime < 0.7:  # Ranging
            trend = 0
            volatility = 0.002
        else:  # Volatile
            trend = -0.0001
            volatility = 0.005
        
        change = np.random.normal(trend, volatility)
        new_price = current_price * (1 + change)
        
        high = new_price * (1 + abs(np.random.normal(0, volatility/2)))
        low = new_price * (1 - abs(np.random.normal(0, volatility/2)))
        
        candle = {
            'timestamp': int((start_time + timedelta(minutes=i*5)).timestamp() * 1000),
            'open': current_price,
            'high': max(high, current_price, new_price),
            'low': min(low, current_price, new_price),
            'close': new_price,
            'volume': max(100, 1000 + np.random.normal(0, 200))
        }
        
        candles.append(candle)
        current_price = new_price
    
    # Train system
    training_split = int(len(candles) * 0.6)
    training_results = system.train_models(candles[:training_split])
    
    # Backtest
    balance = 10000
    trades = []
    
    for i in range(100, len(candles) - training_split - 40):
        current_idx = training_split + i
        current_price = candles[current_idx]['close']
        
        # Generate signal with simulated order book
        order_book = {
            'bid_volume': np.random.uniform(1000, 5000),
            'ask_volume': np.random.uniform(1000, 5000)
        }
        
        signal = system.generate_signal(candles[:current_idx + 1], order_book)
        
        if system.should_trade(signal):
            # Execute trade with 4:1 R:R
            entry_price = current_price
            position_size = balance * 0.01  # 1% risk per trade
            
            if signal['direction'] == 'long':
                stop_price = entry_price - signal['stop_distance']
                target_price = entry_price + signal['target_distance']
            else:
                stop_price = entry_price + signal['stop_distance']
                target_price = entry_price - signal['target_distance']
            
            # Simulate outcome
            exit_price = None
            exit_reason = None
            
            for j in range(1, min(41, len(candles) - current_idx)):
                future_candle = candles[current_idx + j]
                
                if signal['direction'] == 'long':
                    if future_candle['low'] <= stop_price:
                        exit_price = stop_price
                        exit_reason = 'stop'
                        break
                    elif future_candle['high'] >= target_price:
                        exit_price = target_price
                        exit_reason = 'target'
                        break
                else:
                    if future_candle['high'] >= stop_price:
                        exit_price = stop_price
                        exit_reason = 'stop'
                        break
                    elif future_candle['low'] <= target_price:
                        exit_price = target_price
                        exit_reason = 'target'
                        break
            
            if exit_price is None:
                exit_price = candles[min(current_idx + 40, len(candles) - 1)]['close']
                exit_reason = 'time'
            
            # Calculate P&L
            if signal['direction'] == 'long':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            
            pnl_amount = position_size * pnl_pct
            balance += pnl_amount
            
            trades.append({
                'direction': signal['direction'],
                'entry': entry_price,
                'exit': exit_price,
                'reason': exit_reason,
                'pnl_pct': pnl_pct,
                'edge': signal['edge'],
                'confidence': signal['confidence']
            })
            
            # Log trade
            if exit_reason == 'target':
                logger.info(f"✅ WIN: {signal['direction']} +{pnl_pct:.2%} (edge: {signal['edge']:.3%})")
            else:
                logger.info(f"❌ LOSS: {signal['direction']} {pnl_pct:.2%} (edge: {signal['edge']:.3%})")
    
    # Calculate final metrics
    if trades:
        df_trades = pd.DataFrame(trades)
        
        wins = df_trades[df_trades['pnl_pct'] > 0]
        losses = df_trades[df_trades['pnl_pct'] <= 0]
        
        win_rate = len(wins) / len(trades)
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        total_wins = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        print(f"\n📊 RESULTS:")
        print(f"   Total Trades: {len(trades)}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   💰 EXPECTANCY: {expectancy:.3%} per trade")
        print(f"   📈 Profit Factor: {profit_factor:.2f}")
        print(f"   Avg Win: {avg_win:.2%}")
        print(f"   Avg Loss: {avg_loss:.2%}")
        print(f"   Final Balance: ${balance:.2f}")
        print(f"   Total Return: {(balance - 10000) / 10000:.1%}")
        
        # Check if we meet targets
        print(f"\n✅ TARGET CHECKS:")
        print(f"   Expectancy > 0.25%: {'✅' if expectancy > 0.0025 else '❌'}")
        print(f"   Profit Factor > 1.3: {'✅' if profit_factor > 1.3 else '❌'}")
        print(f"   Positive Return: {'✅' if balance > 10000 else '❌'}")
        
        if expectancy > 0.0025 and profit_factor > 1.3:
            print(f"\n🎯 SUCCESS! System has positive edge!")
            print(f"🚀 Ready for deployment")
        else:
            print(f"\n🔄 Close but needs refinement")

if __name__ == "__main__":
    run_improved_backtest() 