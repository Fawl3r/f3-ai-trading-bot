#!/usr/bin/env python3
"""
Quick AI 70%+ Win Rate Test
Fast validation of advanced AI capabilities
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

class QuickAI70WRSystem:
    """Quick AI system for 70%+ win rate validation"""
    
    def __init__(self):
        # Best performing models from advanced system
        self.models = {
            'rf_optimized': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=42
            ),
            'gb_optimized': GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                min_samples_split=10, random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Ultra-conservative thresholds for 70%+ WR
        self.min_ensemble_agreement = 0.9   # 90% agreement
        self.min_signal_strength = 0.8      # 80% signal strength
        
    def extract_key_features(self, candles: list, lookback: int = 50) -> np.ndarray:
        """Extract most predictive features quickly"""
        if len(candles) < lookback:
            return np.array([])
        
        recent = candles[-lookback:]
        df = pd.DataFrame(recent)
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        
        features = []
        
        # 1. Moving averages (most important)
        sma_5 = np.mean(closes[-5:])
        sma_10 = np.mean(closes[-10:])
        sma_20 = np.mean(closes[-20:])
        
        features.extend([
            closes[-1] / sma_5 - 1,
            closes[-1] / sma_10 - 1,
            closes[-1] / sma_20 - 1,
            sma_5 / sma_10 - 1,
            sma_10 / sma_20 - 1,
        ])
        
        # 2. RSI (momentum)
        rsi = self._calculate_rsi(closes, 14)
        features.append(rsi / 100)
        
        # 3. Volatility
        returns = np.diff(closes) / closes[:-1]
        vol_10 = np.std(returns[-10:]) if len(returns) >= 10 else 0
        vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0
        
        features.extend([
            vol_10,
            vol_20,
            vol_10 / vol_20 if vol_20 > 0 else 1,
        ])
        
        # 4. Volume
        vol_avg = np.mean(volumes[-10:])
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1
        features.append(vol_ratio)
        
        # 5. Price position in range
        high_20 = np.max(highs[-20:])
        low_20 = np.min(lows[-20:])
        price_position = (closes[-1] - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5
        features.append(price_position)
        
        # 6. Trend strength
        trend = self._calculate_trend(closes[-20:])
        features.append(trend)
        
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
    
    def prepare_training_data(self, candles: list) -> tuple:
        """Prepare high-quality training data with strict labeling"""
        features_list = []
        labels_list = []
        
        for i in range(50, len(candles) - 10):
            candle_slice = candles[:i+1]
            features = self.extract_key_features(candle_slice)
            
            if len(features) == 0:
                continue
            
            # Very strict future labeling for 70%+ WR
            current_price = candles[i]['close']
            future_prices = [candles[j]['close'] for j in range(i+1, min(i+11, len(candles)))]
            
            if not future_prices:
                continue
            
            max_future = max(future_prices)
            min_future = min(future_prices)
            
            up_move = (max_future - current_price) / current_price
            down_move = (current_price - min_future) / current_price
            
            # Ultra-strict criteria: 2.5% minimum move, 3x confidence ratio
            min_move = 0.025
            confidence_ratio = 3.0
            
            if up_move > min_move and up_move > down_move * confidence_ratio:
                label = 1  # Strong long
            elif down_move > min_move and down_move > up_move * confidence_ratio:
                label = 2  # Strong short
            else:
                label = 0  # Hold
            
            features_list.append(features)
            labels_list.append(label)
        
        return np.array(features_list), np.array(labels_list)
    
    def train_models(self, candles: list) -> dict:
        """Train models quickly"""
        logger.info("🧠 Quick Training AI Models for 70%+ WR...")
        
        X, y = self.prepare_training_data(candles)
        
        if len(X) == 0:
            return {}
        
        logger.info(f"Training on {len(X)} samples, Labels: Hold={np.sum(y==0)}, Long={np.sum(y==1)}, Short={np.sum(y==2)}")
        
        # Scale and split
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
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
        """Generate ultra-conservative signal"""
        if not self.is_trained:
            return {'direction': 'hold', 'confidence': 0.0, 'strength': 0.0}
        
        features = self.extract_key_features(candles)
        if len(features) == 0:
            return {'direction': 'hold', 'confidence': 0.0, 'strength': 0.0}
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            predictions[name] = pred
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                probabilities[name] = proba
        
        # Ensemble decision with ultra-strict requirements
        votes = list(predictions.values())
        long_votes = votes.count(1)
        short_votes = votes.count(2)
        hold_votes = votes.count(0)
        
        total_models = len(self.models)
        agreement_threshold = self.min_ensemble_agreement
        
        if long_votes >= total_models * agreement_threshold:
            direction = 'long'
            confidence = long_votes / total_models
        elif short_votes >= total_models * agreement_threshold:
            direction = 'short'
            confidence = short_votes / total_models
        else:
            direction = 'hold'
            confidence = max(hold_votes, long_votes, short_votes) / total_models
        
        # Calculate signal strength
        if probabilities and direction != 'hold':
            if direction == 'long':
                avg_strength = np.mean([p[1] for p in probabilities.values() if len(p) > 1])
            else:
                avg_strength = np.mean([p[2] for p in probabilities.values() if len(p) > 2])
        else:
            avg_strength = 0.5
        
        return {
            'direction': direction,
            'confidence': confidence,
            'strength': avg_strength,
            'votes': predictions
        }
    
    def should_trade(self, signal: dict) -> bool:
        """Ultra-strict trading filter"""
        if signal['direction'] == 'hold':
            return False
        
        if signal['confidence'] < self.min_ensemble_agreement:
            return False
        
        if signal['strength'] < self.min_signal_strength:
            return False
        
        return True

class QuickBacktester:
    """Quick backtester for validation"""
    
    def __init__(self, balance: float = 50.0):
        self.initial_balance = balance
        self.current_balance = balance
        self.trades = []
        self.ai_system = QuickAI70WRSystem()
        
    def generate_test_data(self, days: int = 14) -> list:
        """Generate realistic test data"""
        candles = []
        base_price = 3500
        current_price = base_price
        
        start_time = datetime.now() - timedelta(days=days)
        
        for i in range(days * 288):  # 5-minute candles
            # Create realistic price movement
            change = np.random.normal(0, 0.008)  # 0.8% volatility
            
            # Add some trend patterns
            if i % 100 < 30:  # Trending periods
                trend = 0.0005 if (i // 100) % 2 == 0 else -0.0005
                change += trend
            
            new_price = current_price * (1 + change)
            
            candle = {
                'timestamp': int((start_time + timedelta(minutes=i*5)).timestamp() * 1000),
                'open': current_price,
                'high': new_price * 1.003,
                'low': new_price * 0.997,
                'close': new_price,
                'volume': 1000 + np.random.normal(0, 200)
            }
            
            candles.append(candle)
            current_price = new_price
        
        return candles
    
    def run_backtest(self, days: int = 14) -> dict:
        """Run quick backtest"""
        logger.info(f"🚀 Quick Backtest ({days} days)")
        
        # Generate data
        candles = self.generate_test_data(days)
        
        # Train AI
        training_results = self.ai_system.train_models(candles[:int(len(candles)*0.7)])
        
        # Test
        test_start = int(len(candles) * 0.7)
        trades_executed = 0
        winning_trades = 0
        
        for i in range(100, len(candles) - test_start):
            current_candle = candles[test_start + i]
            current_price = current_candle['close']
            
            # Generate signal
            signal_history = candles[:test_start + i + 1]
            signal = self.ai_system.generate_signal(signal_history)
            
            if self.ai_system.should_trade(signal):
                # Simulate trade outcome
                trades_executed += 1
                
                # Look ahead to see if trade would be profitable
                future_prices = []
                for j in range(1, min(11, len(candles) - test_start - i)):
                    if test_start + i + j < len(candles):
                        future_prices.append(candles[test_start + i + j]['close'])
                
                if future_prices:
                    if signal['direction'] == 'long':
                        best_exit = max(future_prices)
                        profit = (best_exit - current_price) / current_price > 0.012  # 1.2% target
                    else:
                        best_exit = min(future_prices)
                        profit = (current_price - best_exit) / current_price > 0.012  # 1.2% target
                    
                    if profit:
                        winning_trades += 1
                        logger.info(f"✅ Winning {signal['direction']} trade #{trades_executed}")
                    else:
                        logger.info(f"❌ Losing {signal['direction']} trade #{trades_executed}")
        
        win_rate = winning_trades / trades_executed if trades_executed > 0 else 0
        
        return {
            'total_trades': trades_executed,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'training_results': training_results
        }

def main():
    """Run quick AI validation"""
    print("🚀 QUICK AI 70%+ WIN RATE VALIDATION")
    print("=" * 50)
    
    test_configs = [
        {"name": "Quick 7-Day", "days": 7},
        {"name": "Quick 14-Day", "days": 14},
        {"name": "Quick 21-Day", "days": 21},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n🧠 {config['name']} Test...")
        print("-" * 30)
        
        backtester = QuickBacktester()
        results = backtester.run_backtest(config['days'])
        
        print(f"💰 RESULTS:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Winning Trades: {results['winning_trades']}")
        print(f"   🎯 WIN RATE: {results['win_rate']:.1%}")
        
        if results['win_rate'] >= 0.70:
            status = "🎯 TARGET ACHIEVED!"
        elif results['win_rate'] >= 0.60:
            status = "🔄 Close to Target"
        else:
            status = "❌ Below Target"
        
        print(f"   Status: {status}")
        
        # Training results
        print(f"   Training Accuracy:")
        for model, acc in results['training_results'].items():
            print(f"      {model}: {acc:.1%}")
        
        all_results.append({
            'config': config['name'],
            'results': results,
            'status': status
        })
    
    # Summary
    print(f"\n🎯 SUMMARY")
    print("=" * 50)
    
    successful_tests = [r for r in all_results if r['results']['win_rate'] >= 0.70]
    total_trades = sum(r['results']['total_trades'] for r in all_results)
    total_wins = sum(r['results']['winning_trades'] for r in all_results)
    overall_wr = total_wins / total_trades if total_trades > 0 else 0
    
    print(f"Tests Achieving 70%+: {len(successful_tests)}/{len(all_results)}")
    print(f"Overall Win Rate: {overall_wr:.1%}")
    print(f"Total Trades: {total_trades}")
    
    if overall_wr >= 0.70:
        print(f"\n🎯 SUCCESS! AI can achieve 70%+ win rate!")
        print(f"✅ Advanced AI system is working as designed")
        print(f"🚀 Ready for live paper trading validation")
    elif overall_wr >= 0.60:
        print(f"\n🔄 PROMISING! AI shows strong potential")
        print(f"📈 Consider larger sample size and refinement")
    else:
        print(f"\n⚠️  NEEDS OPTIMIZATION")
        print(f"🔧 Requires further AI development")
    
    # Save results
    with open('quick_ai_70wr_results.json', 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'overall_win_rate': overall_wr,
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to 'quick_ai_70wr_results.json'")

if __name__ == "__main__":
    main() 