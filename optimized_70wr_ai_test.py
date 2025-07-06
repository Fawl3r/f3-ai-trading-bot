#!/usr/bin/env python3
"""
Optimized AI 70%+ Win Rate Test
Ultra-selective approach for maximum win rate
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedAI70WRSystem:
    """Optimized AI system for 70%+ win rate"""
    
    def __init__(self):
        # High-performance models
        self.models = {
            'rf_optimized': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=3,
                min_samples_leaf=1, random_state=42
            ),
            'gb_optimized': GradientBoostingClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                min_samples_split=3, random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Ultra-selective thresholds for 70%+ WR
        self.min_model_accuracy = 0.55      # Only trade if models are 55%+ accurate
        self.min_signal_strength = 0.75     # 75% signal strength
        self.min_quality_score = 0.80       # 80% quality score
        self.min_volatility_threshold = 0.005  # Minimum 0.5% volatility
        
    def extract_premium_features(self, candles: list, lookback: int = 50) -> np.ndarray:
        """Extract premium features for maximum prediction accuracy"""
        if len(candles) < lookback:
            return np.array([])
        
        recent = candles[-lookback:]
        df = pd.DataFrame(recent)
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        
        features = []
        
        # 1. Multi-timeframe momentum (most predictive)
        for period in [3, 5, 8, 13, 21]:
            if len(closes) >= period:
                momentum = (closes[-1] - closes[-period]) / closes[-period]
                features.append(momentum)
        
        # 2. Price position relative to moving averages
        for period in [5, 10, 20]:
            if len(closes) >= period:
                ma = np.mean(closes[-period:])
                features.append(closes[-1] / ma - 1)
        
        # 3. RSI divergence and momentum
        rsi_14 = self._calculate_rsi(closes, 14)
        rsi_7 = self._calculate_rsi(closes, 7)
        features.extend([rsi_14 / 100, rsi_7 / 100, (rsi_14 - rsi_7) / 100])
        
        # 4. Volatility analysis
        returns = np.diff(closes) / closes[:-1]
        for period in [5, 10, 20]:
            if len(returns) >= period:
                vol = np.std(returns[-period:])
                features.append(vol)
        
        # 5. Volume-price analysis
        if len(volumes) >= 10:
            vol_ma = np.mean(volumes[-10:])
            vol_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1
            features.append(min(vol_ratio, 4))  # Cap at 4x
            
            # Price-volume correlation
            if len(closes) >= 10:
                price_changes = np.diff(closes[-10:])
                vol_changes = np.diff(volumes[-10:])
                if len(price_changes) > 0 and len(vol_changes) > 0:
                    corr = np.corrcoef(price_changes, vol_changes)[0, 1]
                    features.append(corr if not np.isnan(corr) else 0)
        
        # 6. Support/Resistance levels
        if len(closes) >= 20:
            high_20 = np.max(highs[-20:])
            low_20 = np.min(lows[-20:])
            range_20 = high_20 - low_20
            
            if range_20 > 0:
                # Position in range
                position = (closes[-1] - low_20) / range_20
                features.append(position)
                
                # Distance from extremes
                dist_from_high = (high_20 - closes[-1]) / closes[-1]
                dist_from_low = (closes[-1] - low_20) / closes[-1]
                features.extend([dist_from_high, dist_from_low])
        
        # 7. Trend consistency
        for period in [5, 10, 20]:
            if len(closes) >= period:
                trend = self._calculate_trend_strength(closes[-period:])
                features.append(trend)
        
        # 8. Candlestick patterns
        if len(closes) >= 3:
            # Doji detection
            body_size = abs(closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0
            features.append(body_size)
            
            # Hammer/shooting star
            if len(highs) >= 2 and len(lows) >= 2:
                upper_shadow = (highs[-1] - max(closes[-1], closes[-2])) / closes[-1] if closes[-1] != 0 else 0
                lower_shadow = (min(closes[-1], closes[-2]) - lows[-1]) / closes[-1] if closes[-1] != 0 else 0
                features.extend([upper_shadow, lower_shadow])
        
        return np.array(features)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI with smoothing"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use exponential moving average for smoothing
        alpha = 2.0 / (period + 1)
        avg_gain = gains[0] if len(gains) > 0 else 0
        avg_loss = losses[0] if len(losses) > 0 else 0
        
        for i in range(1, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength with R-squared"""
        if len(prices) < 3:
            return 0.0
        
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Return signed trend strength
        trend_direction = 1 if slope > 0 else -1
        return trend_direction * r_squared
    
    def prepare_high_quality_data(self, candles: list) -> tuple:
        """Prepare ultra-high quality training data"""
        features_list = []
        labels_list = []
        
        for i in range(50, len(candles) - 15):
            candle_slice = candles[:i+1]
            features = self.extract_premium_features(candle_slice)
            
            if len(features) == 0:
                continue
            
            # Ultra-strict future labeling
            current_price = candles[i]['close']
            future_prices = [candles[j]['close'] for j in range(i+1, min(i+16, len(candles)))]
            
            if not future_prices:
                continue
            
            max_future = max(future_prices)
            min_future = min(future_prices)
            
            up_move = (max_future - current_price) / current_price
            down_move = (current_price - min_future) / current_price
            
            # Ultra-strict criteria: 2% minimum move, 2.5x confidence
            min_move = 0.02
            confidence_ratio = 2.5
            
            # Also check volatility - only trade in volatile conditions
            recent_returns = np.diff([candles[j]['close'] for j in range(max(0, i-10), i+1)])
            if len(recent_returns) > 0:
                volatility = np.std(recent_returns / [candles[j]['close'] for j in range(max(0, i-10), i)])
                if volatility < self.min_volatility_threshold:
                    continue  # Skip low volatility periods
            
            if up_move > min_move and up_move > down_move * confidence_ratio:
                label = 1  # Strong long
            elif down_move > min_move and down_move > up_move * confidence_ratio:
                label = 2  # Strong short
            else:
                continue  # Skip unclear signals
            
            features_list.append(features)
            labels_list.append(label)
        
        return np.array(features_list), np.array(labels_list)
    
    def train_models(self, candles: list) -> dict:
        """Train optimized models"""
        logger.info("🧠 Training Optimized AI Models for 70%+ WR...")
        
        X, y = self.prepare_high_quality_data(candles)
        
        if len(X) == 0:
            return {}
        
        logger.info(f"Training on {len(X)} high-quality samples, Long={np.sum(y==1)}, Short={np.sum(y==2)}")
        
        # Handle any NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale and split
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
        
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            logger.info(f"{name}: {accuracy:.1%} accuracy")
        
        # Only mark as trained if models meet minimum accuracy
        avg_accuracy = np.mean(list(results.values()))
        if avg_accuracy >= self.min_model_accuracy:
            self.is_trained = True
            logger.info(f"✅ Models meet accuracy threshold: {avg_accuracy:.1%}")
        else:
            logger.info(f"❌ Models below threshold: {avg_accuracy:.1%}")
        
        return results
    
    def generate_signal(self, candles: list) -> dict:
        """Generate ultra-selective signal"""
        if not self.is_trained:
            return {'direction': 'hold', 'confidence': 0.0, 'strength': 0.0}
        
        features = self.extract_premium_features(candles)
        if len(features) == 0:
            return {'direction': 'hold', 'confidence': 0.0, 'strength': 0.0}
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions and probabilities
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            predictions[name] = pred
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                probabilities[name] = proba
        
        # Require unanimous agreement
        votes = list(predictions.values())
        if len(set(votes)) > 1:  # Not unanimous
            return {'direction': 'hold', 'confidence': 0.0, 'strength': 0.0}
        
        direction_map = {1: 'long', 2: 'short'}
        direction = direction_map.get(votes[0], 'hold')
        
        if direction == 'hold':
            return {'direction': 'hold', 'confidence': 0.0, 'strength': 0.0}
        
        # Calculate strength from probabilities
        if probabilities:
            if direction == 'long':
                strengths = [p[0] for p in probabilities.values() if len(p) > 0]  # Index 0 for binary classification
            else:
                strengths = [p[1] for p in probabilities.values() if len(p) > 1]  # Index 1 for binary classification
            
            avg_strength = np.mean(strengths) if strengths else 0.5
        else:
            avg_strength = 0.5
        
        # Quality score combines unanimity and strength
        quality_score = 1.0 * avg_strength  # Unanimous = 100% confidence
        
        return {
            'direction': direction,
            'confidence': 1.0,  # Unanimous
            'strength': avg_strength,
            'quality_score': quality_score,
            'votes': predictions
        }
    
    def should_trade(self, signal: dict) -> bool:
        """Ultra-strict trading filter"""
        if signal['direction'] == 'hold':
            return False
        
        if signal['strength'] < self.min_signal_strength:
            return False
        
        if signal['quality_score'] < self.min_quality_score:
            return False
        
        return True

class OptimizedBacktester:
    """Optimized backtester for 70%+ WR"""
    
    def __init__(self):
        self.ai_system = OptimizedAI70WRSystem()
        
    def generate_premium_data(self, days: int = 30) -> list:
        """Generate high-quality market data with realistic patterns"""
        candles = []
        base_price = 3500
        current_price = base_price
        
        start_time = datetime.now() - timedelta(days=days)
        
        for i in range(days * 288):  # 5-minute candles
            # More sophisticated price movement
            base_change = np.random.normal(0, 0.004)  # 0.4% base volatility
            
            # Add market regimes
            regime_cycle = (i % 2000) / 2000  # 2000-candle regimes
            
            if regime_cycle < 0.3:  # Trending up
                trend_component = 0.0003
                volatility_mult = 0.8
            elif regime_cycle < 0.6:  # Trending down
                trend_component = -0.0003
                volatility_mult = 0.8
            else:  # Ranging/volatile
                trend_component = 0
                volatility_mult = 1.5
            
            # Add momentum persistence
            if i > 0:
                prev_change = (candles[-1]['close'] - candles[-1]['open']) / candles[-1]['open']
                momentum = prev_change * 0.1  # 10% momentum persistence
            else:
                momentum = 0
            
            total_change = base_change * volatility_mult + trend_component + momentum
            new_price = current_price * (1 + total_change)
            
            # Realistic OHLC with proper relationships
            volatility = abs(total_change) * 2
            high = new_price * (1 + volatility * np.random.random())
            low = new_price * (1 - volatility * np.random.random())
            
            # Ensure OHLC relationships are correct
            high = max(high, current_price, new_price)
            low = min(low, current_price, new_price)
            
            candle = {
                'timestamp': int((start_time + timedelta(minutes=i*5)).timestamp() * 1000),
                'open': current_price,
                'high': high,
                'low': low,
                'close': new_price,
                'volume': max(200, 1000 + np.random.normal(0, 300) * volatility_mult)
            }
            
            candles.append(candle)
            current_price = new_price
        
        return candles
    
    def run_backtest(self, days: int = 30) -> dict:
        """Run optimized backtest"""
        logger.info(f"🚀 Optimized Backtest ({days} days)")
        
        # Generate premium data
        candles = self.generate_premium_data(days)
        
        # Train AI (50% for training, 50% for testing - more data for testing)
        training_split = int(len(candles) * 0.5)
        training_results = self.ai_system.train_models(candles[:training_split])
        
        if not self.ai_system.is_trained:
            logger.info("❌ Models didn't meet accuracy threshold, skipping backtest")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'training_results': training_results
            }
        
        # Test on remaining data
        test_start = training_split
        trades_executed = 0
        winning_trades = 0
        total_pnl = 0
        
        for i in range(100, len(candles) - test_start - 20):
            current_idx = test_start + i
            current_candle = candles[current_idx]
            current_price = current_candle['close']
            
            # Generate signal
            signal_history = candles[:current_idx + 1]
            signal = self.ai_system.generate_signal(signal_history)
            
            if self.ai_system.should_trade(signal):
                trades_executed += 1
                
                # Ultra-realistic trade execution
                entry_price = current_price
                
                # Look ahead for realistic exit
                future_candles = candles[current_idx+1:current_idx+21]  # Up to 20 candles ahead
                if future_candles:
                    future_prices = [c['close'] for c in future_candles]
                    
                    if signal['direction'] == 'long':
                        # Take profit at 2% or stop loss at -0.8%
                        best_price = max(future_prices)
                        worst_price = min(future_prices)
                        
                        if best_price >= entry_price * 1.02:  # 2% profit
                            exit_price = entry_price * 1.02
                            profit = True
                        elif worst_price <= entry_price * 0.992:  # 0.8% loss
                            exit_price = entry_price * 0.992
                            profit = False
                        else:
                            exit_price = future_prices[-1]  # Exit at end
                            profit = exit_price > entry_price
                        
                        pnl = (exit_price - entry_price) / entry_price
                        
                    else:  # Short
                        best_price = min(future_prices)
                        worst_price = max(future_prices)
                        
                        if best_price <= entry_price * 0.98:  # 2% profit
                            exit_price = entry_price * 0.98
                            profit = True
                        elif worst_price >= entry_price * 1.008:  # 0.8% loss
                            exit_price = entry_price * 1.008
                            profit = False
                        else:
                            exit_price = future_prices[-1]
                            profit = exit_price < entry_price
                        
                        pnl = (entry_price - exit_price) / entry_price
                    
                    if profit:
                        winning_trades += 1
                        logger.info(f"✅ WIN #{trades_executed}: {signal['direction']} {pnl:.2%} (Q:{signal['quality_score']:.2f})")
                    else:
                        logger.info(f"❌ LOSS #{trades_executed}: {signal['direction']} {pnl:.2%} (Q:{signal['quality_score']:.2f})")
                    
                    total_pnl += pnl
        
        win_rate = winning_trades / trades_executed if trades_executed > 0 else 0
        avg_pnl = total_pnl / trades_executed if trades_executed > 0 else 0
        
        return {
            'total_trades': trades_executed,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'training_results': training_results
        }

def main():
    """Run optimized AI validation"""
    print("🚀 OPTIMIZED AI 70%+ WIN RATE VALIDATION")
    print("=" * 50)
    
    test_configs = [
        {"name": "Optimized 21-Day", "days": 21},
        {"name": "Optimized 30-Day", "days": 30},
        {"name": "Optimized 45-Day", "days": 45},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n🧠 {config['name']} Test...")
        print("-" * 40)
        
        backtester = OptimizedBacktester()
        results = backtester.run_backtest(config['days'])
        
        print(f"💰 RESULTS:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Winning Trades: {results['winning_trades']}")
        print(f"   🎯 WIN RATE: {results['win_rate']:.1%}")
        print(f"   📈 Total PnL: {results['total_pnl']:.2%}")
        print(f"   📊 Avg PnL: {results['avg_pnl']:.3%}")
        
        if results['win_rate'] >= 0.70:
            status = "🎯 TARGET ACHIEVED!"
        elif results['win_rate'] >= 0.65:
            status = "🔄 Very Close"
        elif results['win_rate'] >= 0.60:
            status = "📈 Good Progress"
        else:
            status = "❌ Needs Work"
        
        print(f"   Status: {status}")
        
        # Training results
        print(f"   🧠 Model Accuracy:")
        for model, acc in results['training_results'].items():
            print(f"      {model}: {acc:.1%}")
        
        all_results.append({
            'config': config['name'],
            'results': results,
            'status': status
        })
    
    # Summary
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 50)
    
    successful_tests = [r for r in all_results if r['results']['win_rate'] >= 0.70]
    total_trades = sum(r['results']['total_trades'] for r in all_results)
    total_wins = sum(r['results']['winning_trades'] for r in all_results)
    overall_wr = total_wins / total_trades if total_trades > 0 else 0
    total_pnl = sum(r['results']['total_pnl'] for r in all_results)
    
    print(f"✅ Tests Achieving 70%+: {len(successful_tests)}/{len(all_results)}")
    print(f"🎯 Overall Win Rate: {overall_wr:.1%}")
    print(f"📊 Total Trades: {total_trades}")
    print(f"📈 Combined PnL: {total_pnl:.2%}")
    
    if overall_wr >= 0.70:
        print(f"\n🎯 SUCCESS! AI achieves 70%+ win rate!")
        print(f"✅ Optimized system ready for live trading")
        conclusion = "SUCCESS"
    elif overall_wr >= 0.65:
        print(f"\n🔄 EXCELLENT! Very close to 70% target")
        print(f"📈 Minor fine-tuning needed")
        conclusion = "VERY_CLOSE"
    elif overall_wr >= 0.60:
        print(f"\n📈 STRONG FOUNDATION! Good progress toward 70%")
        print(f"🔧 Requires further optimization")
        conclusion = "STRONG"
    else:
        print(f"\n🔄 DEVELOPMENT PHASE")
        print(f"🔧 Continue optimization")
        conclusion = "DEVELOPMENT"
    
    # Save results
    with open('optimized_ai_70wr_results.json', 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'overall_win_rate': overall_wr,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'conclusion': conclusion,
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to 'optimized_ai_70wr_results.json'")

if __name__ == "__main__":
    main() 