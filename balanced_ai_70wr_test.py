#!/usr/bin/env python3
"""
Balanced AI 70%+ Win Rate Test
Optimized thresholds for actual trading
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BalancedAI70WRSystem:
    """Balanced AI system targeting 70%+ win rate"""
    
    def __init__(self):
        # 3 complementary models for better ensemble
        self.models = {
            'rf_conservative': RandomForestClassifier(
                n_estimators=150, max_depth=8, min_samples_split=15,
                min_samples_leaf=8, random_state=42
            ),
            'gb_aggressive': GradientBoostingClassifier(
                n_estimators=120, max_depth=5, learning_rate=0.08,
                min_samples_split=12, random_state=42
            ),
            'lr_balanced': LogisticRegression(
                C=0.1, max_iter=1000, random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Balanced thresholds for 70%+ WR
        self.min_ensemble_agreement = 0.67   # 67% agreement (2/3 models)
        self.min_signal_strength = 0.65      # 65% signal strength
        self.min_confidence_score = 0.70     # 70% confidence
        
    def extract_enhanced_features(self, candles: list, lookback: int = 60) -> np.ndarray:
        """Extract enhanced features for better prediction"""
        if len(candles) < lookback:
            return np.array([])
        
        recent = candles[-lookback:]
        df = pd.DataFrame(recent)
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        
        features = []
        
        # 1. Multi-timeframe Moving Averages
        for period in [5, 10, 20, 50]:
            if len(closes) >= period:
                ma = np.mean(closes[-period:])
                features.append(closes[-1] / ma - 1)
        
        # 2. Moving Average Crossovers
        if len(closes) >= 20:
            sma_5 = np.mean(closes[-5:])
            sma_10 = np.mean(closes[-10:])
            sma_20 = np.mean(closes[-20:])
            
            features.extend([
                sma_5 / sma_10 - 1,
                sma_10 / sma_20 - 1,
                sma_5 / sma_20 - 1,
            ])
        
        # 3. RSI Multi-period
        for period in [7, 14, 21]:
            rsi = self._calculate_rsi(closes, period)
            features.append(rsi / 100)
        
        # 4. Volatility Analysis
        returns = np.diff(closes) / closes[:-1]
        for period in [5, 10, 20]:
            if len(returns) >= period:
                vol = np.std(returns[-period:])
                features.append(vol)
        
        # 5. Volume Analysis
        if len(volumes) >= 10:
            vol_sma = np.mean(volumes[-10:])
            vol_ratio = volumes[-1] / vol_sma if vol_sma > 0 else 1
            features.append(min(vol_ratio, 5))  # Cap at 5x
        
        # 6. Price Position and Range
        if len(closes) >= 20:
            high_20 = np.max(highs[-20:])
            low_20 = np.min(lows[-20:])
            range_20 = high_20 - low_20
            
            if range_20 > 0:
                price_position = (closes[-1] - low_20) / range_20
                features.append(price_position)
                
                # Distance from highs/lows
                features.append((high_20 - closes[-1]) / closes[-1])
                features.append((closes[-1] - low_20) / closes[-1])
        
        # 7. Momentum Indicators
        if len(closes) >= 10:
            momentum_5 = (closes[-1] - closes[-5]) / closes[-5]
            momentum_10 = (closes[-1] - closes[-10]) / closes[-10]
            features.extend([momentum_5, momentum_10])
        
        # 8. Trend Strength
        for period in [10, 20]:
            if len(closes) >= period:
                trend = self._calculate_trend(closes[-period:])
                features.append(trend)
        
        # 9. Bollinger Bands
        if len(closes) >= 20:
            bb_upper, bb_lower = self._calculate_bollinger_bands(closes, 20, 2)
            if bb_upper != bb_lower:
                bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower)
                features.append(bb_position)
        
        return np.array(features)
    
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
        return 100 - (100 / (1 + rs))
    
    def _calculate_trend(self, prices: np.ndarray) -> float:
        """Calculate trend strength"""
        if len(prices) < 3:
            return 0.0
        
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return slope / prices[-1] if prices[-1] != 0 else 0
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_dev: float) -> tuple:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return prices[-1], prices[-1]
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, lower
    
    def prepare_quality_training_data(self, candles: list) -> tuple:
        """Prepare high-quality training data with balanced labeling"""
        features_list = []
        labels_list = []
        
        for i in range(60, len(candles) - 12):
            candle_slice = candles[:i+1]
            features = self.extract_enhanced_features(candle_slice)
            
            if len(features) == 0:
                continue
            
            # Balanced future labeling for 70%+ WR
            current_price = candles[i]['close']
            future_prices = [candles[j]['close'] for j in range(i+1, min(i+13, len(candles)))]
            
            if not future_prices:
                continue
            
            max_future = max(future_prices)
            min_future = min(future_prices)
            
            up_move = (max_future - current_price) / current_price
            down_move = (current_price - min_future) / current_price
            
            # Balanced criteria: 1.5% minimum move, 2x confidence ratio
            min_move = 0.015
            confidence_ratio = 2.0
            
            if up_move > min_move and up_move > down_move * confidence_ratio:
                label = 1  # Long
            elif down_move > min_move and down_move > up_move * confidence_ratio:
                label = 2  # Short
            else:
                label = 0  # Hold
            
            features_list.append(features)
            labels_list.append(label)
        
        return np.array(features_list), np.array(labels_list)
    
    def train_models(self, candles: list) -> dict:
        """Train ensemble models"""
        logger.info("🧠 Training Balanced AI Models for 70%+ WR...")
        
        X, y = self.prepare_quality_training_data(candles)
        
        if len(X) == 0:
            return {}
        
        logger.info(f"Training on {len(X)} samples, Labels: Hold={np.sum(y==0)}, Long={np.sum(y==1)}, Short={np.sum(y==2)}")
        
        # Handle NaN values
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
        
        self.is_trained = True
        return results
    
    def generate_signal(self, candles: list) -> dict:
        """Generate balanced signal with quality scoring"""
        if not self.is_trained:
            return {'direction': 'hold', 'confidence': 0.0, 'strength': 0.0}
        
        features = self.extract_enhanced_features(candles)
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
        
        # Ensemble voting
        votes = list(predictions.values())
        long_votes = votes.count(1)
        short_votes = votes.count(2)
        hold_votes = votes.count(0)
        
        total_models = len(self.models)
        
        # Determine direction
        if long_votes >= total_models * self.min_ensemble_agreement:
            direction = 'long'
            confidence = long_votes / total_models
        elif short_votes >= total_models * self.min_ensemble_agreement:
            direction = 'short'
            confidence = short_votes / total_models
        else:
            direction = 'hold'
            confidence = max(hold_votes, long_votes, short_votes) / total_models
        
        # Calculate signal strength
        if probabilities and direction != 'hold':
            if direction == 'long':
                strengths = [p[1] for p in probabilities.values() if len(p) > 1]
            else:
                strengths = [p[2] for p in probabilities.values() if len(p) > 2]
            
            avg_strength = np.mean(strengths) if strengths else 0.5
        else:
            avg_strength = 0.5
        
        # Quality score
        quality_score = confidence * avg_strength
        
        return {
            'direction': direction,
            'confidence': confidence,
            'strength': avg_strength,
            'quality_score': quality_score,
            'votes': predictions
        }
    
    def should_trade(self, signal: dict) -> bool:
        """Balanced trading filter"""
        if signal['direction'] == 'hold':
            return False
        
        if signal['confidence'] < self.min_ensemble_agreement:
            return False
        
        if signal['strength'] < self.min_signal_strength:
            return False
        
        if signal['quality_score'] < self.min_confidence_score:
            return False
        
        return True

class BalancedBacktester:
    """Balanced backtester for 70%+ WR validation"""
    
    def __init__(self, balance: float = 100.0):
        self.initial_balance = balance
        self.current_balance = balance
        self.trades = []
        self.ai_system = BalancedAI70WRSystem()
        
    def generate_realistic_data(self, days: int = 21) -> list:
        """Generate more realistic market data"""
        candles = []
        base_price = 3500
        current_price = base_price
        
        start_time = datetime.now() - timedelta(days=days)
        
        for i in range(days * 288):  # 5-minute candles
            # More realistic price movement with trends
            base_change = np.random.normal(0, 0.006)  # 0.6% base volatility
            
            # Add market cycles
            cycle_position = (i % 1000) / 1000  # 1000-candle cycles
            trend_strength = np.sin(cycle_position * 2 * np.pi) * 0.0003
            
            # Add volatility clustering
            if i > 0 and abs(candles[-1]['close'] - candles[-1]['open']) / candles[-1]['open'] > 0.01:
                volatility_multiplier = 1.5  # Higher volatility after big moves
            else:
                volatility_multiplier = 1.0
            
            change = base_change * volatility_multiplier + trend_strength
            new_price = current_price * (1 + change)
            
            # Realistic OHLC
            high = new_price * (1 + abs(np.random.normal(0, 0.002)))
            low = new_price * (1 - abs(np.random.normal(0, 0.002)))
            
            candle = {
                'timestamp': int((start_time + timedelta(minutes=i*5)).timestamp() * 1000),
                'open': current_price,
                'high': max(high, current_price, new_price),
                'low': min(low, current_price, new_price),
                'close': new_price,
                'volume': max(500, 1200 + np.random.normal(0, 300))
            }
            
            candles.append(candle)
            current_price = new_price
        
        return candles
    
    def run_backtest(self, days: int = 21) -> dict:
        """Run balanced backtest"""
        logger.info(f"🚀 Balanced Backtest ({days} days)")
        
        # Generate realistic data
        candles = self.generate_realistic_data(days)
        
        # Train AI (70% for training)
        training_split = int(len(candles) * 0.7)
        training_results = self.ai_system.train_models(candles[:training_split])
        
        # Test on remaining 30%
        test_start = training_split
        trades_executed = 0
        winning_trades = 0
        total_pnl = 0
        
        for i in range(100, len(candles) - test_start - 15):
            current_idx = test_start + i
            current_candle = candles[current_idx]
            current_price = current_candle['close']
            
            # Generate signal
            signal_history = candles[:current_idx + 1]
            signal = self.ai_system.generate_signal(signal_history)
            
            if self.ai_system.should_trade(signal):
                trades_executed += 1
                
                # Simulate realistic trade execution
                entry_price = current_price
                
                # Look ahead for exit (realistic holding period)
                exit_prices = []
                for j in range(1, min(16, len(candles) - current_idx)):
                    if current_idx + j < len(candles):
                        exit_prices.append(candles[current_idx + j]['close'])
                
                if exit_prices:
                    if signal['direction'] == 'long':
                        # Take profit at 1.5% or stop loss at -1%
                        best_price = max(exit_prices)
                        worst_price = min(exit_prices)
                        
                        if best_price >= entry_price * 1.015:  # 1.5% profit
                            exit_price = entry_price * 1.015
                            profit = True
                        elif worst_price <= entry_price * 0.99:  # 1% loss
                            exit_price = entry_price * 0.99
                            profit = False
                        else:
                            exit_price = exit_prices[-1]  # Exit at end
                            profit = exit_price > entry_price
                        
                        pnl = (exit_price - entry_price) / entry_price
                        
                    else:  # Short
                        best_price = min(exit_prices)
                        worst_price = max(exit_prices)
                        
                        if best_price <= entry_price * 0.985:  # 1.5% profit
                            exit_price = entry_price * 0.985
                            profit = True
                        elif worst_price >= entry_price * 1.01:  # 1% loss
                            exit_price = entry_price * 1.01
                            profit = False
                        else:
                            exit_price = exit_prices[-1]
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
    """Run balanced AI validation"""
    print("🚀 BALANCED AI 70%+ WIN RATE VALIDATION")
    print("=" * 50)
    
    test_configs = [
        {"name": "Balanced 14-Day", "days": 14},
        {"name": "Balanced 21-Day", "days": 21},
        {"name": "Balanced 30-Day", "days": 30},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n🧠 {config['name']} Test...")
        print("-" * 40)
        
        backtester = BalancedBacktester()
        results = backtester.run_backtest(config['days'])
        
        print(f"💰 RESULTS:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Winning Trades: {results['winning_trades']}")
        print(f"   🎯 WIN RATE: {results['win_rate']:.1%}")
        print(f"   📈 Total PnL: {results['total_pnl']:.2%}")
        print(f"   📊 Avg PnL: {results['avg_pnl']:.2%}")
        
        if results['win_rate'] >= 0.70:
            status = "🎯 TARGET ACHIEVED!"
        elif results['win_rate'] >= 0.65:
            status = "🔄 Very Close to Target"
        elif results['win_rate'] >= 0.60:
            status = "📈 Promising Results"
        else:
            status = "❌ Below Target"
        
        print(f"   Status: {status}")
        
        # Training results
        print(f"   🧠 Training Accuracy:")
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
        print(f"✅ Balanced AI system is working perfectly")
        print(f"🚀 Ready for live paper trading")
        conclusion = "SUCCESS"
    elif overall_wr >= 0.65:
        print(f"\n🔄 EXCELLENT! Very close to 70% target")
        print(f"📈 Minor optimization needed")
        conclusion = "VERY_CLOSE"
    elif overall_wr >= 0.60:
        print(f"\n📈 PROMISING! Good foundation for 70%+")
        print(f"🔧 Requires refinement")
        conclusion = "PROMISING"
    else:
        print(f"\n⚠️  NEEDS WORK")
        print(f"🔧 Requires significant optimization")
        conclusion = "NEEDS_WORK"
    
    # Save results
    with open('balanced_ai_70wr_results.json', 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'overall_win_rate': overall_wr,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'conclusion': conclusion,
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to 'balanced_ai_70wr_results.json'")

if __name__ == "__main__":
    main() 