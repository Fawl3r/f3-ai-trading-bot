#!/usr/bin/env python3
"""
Enhanced Edge Optimizer
Focuses on expectancy and profit factor, not just win rate
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEdgeSystem:
    """Trading system focused on expectancy and edge, not vanity win rate"""
    
    def __init__(self):
        # Models remain the same
        self.models = {
            'rf_main': RandomForestClassifier(
                n_estimators=150, max_depth=12, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            ),
            'gb_main': GradientBoostingClassifier(
                n_estimators=150, max_depth=8, learning_rate=0.1,
                min_samples_split=5, random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Edge-focused thresholds
        self.min_model_agreement = 0.5     # 50% agreement is fine
        self.min_signal_strength = 0.55    # Lower threshold
        self.min_edge_score = 0.35         # Minimum expected value per trade
        
    def calculate_atr(self, candles: list, period: int = 14) -> float:
        """Calculate Average True Range"""
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
    
    def calculate_obv_divergence(self, candles: list, lookback: int = 20) -> float:
        """Calculate On-Balance Volume divergence"""
        if len(candles) < lookback:
            return 0.0
        
        recent = candles[-lookback:]
        closes = np.array([c['close'] for c in recent])
        volumes = np.array([c['volume'] for c in recent])
        
        # Calculate OBV
        obv = np.zeros(len(volumes))
        obv[0] = volumes[0]
        
        for i in range(1, len(volumes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        # Check for divergence
        price_trend = (closes[-1] - closes[0]) / closes[0]
        obv_trend = (obv[-1] - obv[0]) / abs(obv[0]) if obv[0] != 0 else 0
        
        # Divergence: price up but OBV down, or vice versa
        divergence = abs(price_trend - obv_trend) if np.sign(price_trend) != np.sign(obv_trend) else 0
        
        return divergence
    
    def extract_edge_features(self, candles: list, lookback: int = 50) -> np.ndarray:
        """Extract features focused on edge and expectancy"""
        if len(candles) < lookback:
            return np.array([])
        
        recent = candles[-lookback:]
        df = pd.DataFrame(recent)
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        
        features = []
        
        # 1. ATR for dynamic stops
        atr = self.calculate_atr(candles)
        atr_pct = atr / closes[-1] if closes[-1] != 0 else 0
        features.append(atr_pct)
        
        # 2. Price momentum (multi-timeframe)
        for period in [5, 10, 20]:
            if len(closes) >= period:
                momentum = (closes[-1] - closes[-period]) / closes[-period]
                features.append(momentum)
        
        # 3. RSI for oversold/overbought
        rsi = self._calculate_rsi(closes, 14)
        features.append(rsi / 100)
        
        # 4. OBV divergence
        obv_div = self.calculate_obv_divergence(candles)
        features.append(obv_div)
        
        # 5. Volume analysis
        vol_avg = np.mean(volumes[-10:])
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1
        features.append(min(vol_ratio, 3))
        
        # 6. Market structure
        high_20 = np.max(highs[-20:])
        low_20 = np.min(lows[-20:])
        if high_20 != low_20:
            position = (closes[-1] - low_20) / (high_20 - low_20)
            features.append(position)
            
            # Distance from extremes (for mean reversion)
            dist_high = (high_20 - closes[-1]) / closes[-1]
            dist_low = (closes[-1] - low_20) / closes[-1]
            features.extend([dist_high, dist_low])
        
        # 7. Volatility clustering
        returns = np.diff(closes) / closes[:-1]
        vol_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
        vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0
        vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1
        features.append(vol_ratio)
        
        # 8. Trend quality (R-squared)
        if len(closes) >= 10:
            x = np.arange(10)
            slope, intercept = np.polyfit(x, closes[-10:], 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((closes[-10:] - y_pred) ** 2)
            ss_tot = np.sum((closes[-10:] - np.mean(closes[-10:])) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            features.append(r_squared)
        
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
    
    def prepare_training_data(self, candles: list) -> tuple:
        """Prepare training data with focus on R:R opportunities"""
        features_list = []
        labels_list = []
        
        for i in range(50, len(candles) - 20):
            candle_slice = candles[:i+1]
            features = self.extract_edge_features(candle_slice)
            
            if len(features) == 0:
                continue
            
            # Calculate ATR at this point
            atr = self.calculate_atr(candle_slice)
            current_price = candles[i]['close']
            
            # Look for 3:1 R:R opportunities
            future_prices = [candles[j]['close'] for j in range(i+1, min(i+21, len(candles)))]
            
            if not future_prices:
                continue
            
            # Dynamic targets based on ATR
            stop_distance = atr * 1.0  # 1 ATR stop
            target_distance = atr * 3.0  # 3 ATR target (3:1 R:R)
            
            max_future = max(future_prices)
            min_future = min(future_prices)
            
            # Long opportunity
            if max_future >= current_price + target_distance:
                if min_future > current_price - stop_distance:
                    label = 1  # Strong long (hit target before stop)
                else:
                    label = 0  # Neutral (hit stop first)
            # Short opportunity
            elif min_future <= current_price - target_distance:
                if max_future < current_price + stop_distance:
                    label = 2  # Strong short (hit target before stop)
                else:
                    label = 0  # Neutral (hit stop first)
            else:
                label = 0  # No clear opportunity
            
            features_list.append(features)
            labels_list.append(label)
        
        return np.array(features_list), np.array(labels_list)
    
    def train_models(self, candles: list) -> dict:
        """Train models focused on edge"""
        logger.info("🧠 Training Edge-Optimized Models...")
        
        X, y = self.prepare_training_data(candles)
        
        if len(X) == 0:
            return {}
        
        logger.info(f"Training on {len(X)} samples, Neutral={np.sum(y==0)}, Long={np.sum(y==1)}, Short={np.sum(y==2)}")
        
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
            
            # Show classification report for edge trades only
            edge_mask = y_test != 0
            if np.any(edge_mask):
                logger.info(f"\n{name} Edge Trade Performance:")
                logger.info(classification_report(y_test[edge_mask], y_pred[edge_mask]))
        
        self.is_trained = True
        return results
    
    def generate_signal(self, candles: list) -> dict:
        """Generate signal with edge calculation"""
        if not self.is_trained:
            return {'direction': 'hold', 'confidence': 0.0, 'edge': 0.0}
        
        features = self.extract_edge_features(candles)
        if len(features) == 0:
            return {'direction': 'hold', 'confidence': 0.0, 'edge': 0.0}
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)
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
        
        # Calculate edge-based signal
        votes = list(predictions.values())
        long_votes = votes.count(1)
        short_votes = votes.count(2)
        
        # Calculate ATR for position sizing
        atr = self.calculate_atr(candles)
        current_price = candles[-1]['close']
        
        # Dynamic R:R based on market conditions
        atr_pct = atr / current_price if current_price != 0 else 0.01
        
        # Calculate expected value
        if long_votes > short_votes:
            direction = 'long'
            confidence = long_votes / len(self.models)
            win_prob = np.mean([p[1] for p in probabilities.values() if len(p) > 1])
            
            # Expected value = (win_prob * reward) - (loss_prob * risk)
            reward = atr_pct * 3.0  # 3 ATR target
            risk = atr_pct * 1.0    # 1 ATR stop
            edge = (win_prob * reward) - ((1 - win_prob) * risk)
            
        elif short_votes > long_votes:
            direction = 'short'
            confidence = short_votes / len(self.models)
            win_prob = np.mean([p[2] for p in probabilities.values() if len(p) > 2])
            
            reward = atr_pct * 3.0
            risk = atr_pct * 1.0
            edge = (win_prob * reward) - ((1 - win_prob) * risk)
            
        else:
            direction = 'hold'
            confidence = 0.0
            edge = 0.0
            win_prob = 0.0
        
        # Additional filters
        rsi = self._calculate_rsi(np.array([c['close'] for c in candles[-15:]]), 14)
        obv_div = self.calculate_obv_divergence(candles)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'edge': edge,
            'win_prob': win_prob,
            'atr': atr,
            'atr_pct': atr_pct,
            'rsi': rsi,
            'obv_divergence': obv_div,
            'stop_distance': atr * 1.0,
            'target_distance': atr * 3.0
        }
    
    def should_trade(self, signal: dict) -> bool:
        """Trade only positive edge opportunities"""
        if signal['direction'] == 'hold':
            return False
        
        # Require positive edge
        if signal['edge'] < self.min_edge_score / 100:  # Convert to decimal
            return False
        
        # Additional filters (at least 2 of 3)
        filters_passed = 0
        
        # RSI filter
        if (signal['direction'] == 'long' and signal['rsi'] < 30) or \
           (signal['direction'] == 'short' and signal['rsi'] > 70):
            filters_passed += 1
        
        # OBV divergence filter
        if signal['obv_divergence'] > 0.02:
            filters_passed += 1
        
        # Strong confidence filter
        if signal['confidence'] >= 0.75:
            filters_passed += 1
        
        return filters_passed >= 2

class EdgeOptimizedBacktester:
    """Backtester focused on expectancy and profit factor"""
    
    def __init__(self):
        self.ai_system = EnhancedEdgeSystem()
        self.initial_balance = 10000
        
    def calculate_position_size(self, balance: float, signal: dict, kelly_fraction: float = 0.25) -> float:
        """Kelly-inspired position sizing based on edge"""
        if signal['edge'] <= 0:
            return 0
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win prob, q = loss prob, b = win/loss ratio
        p = signal['win_prob']
        q = 1 - p
        b = 3.0  # 3:1 R:R
        
        kelly = (p * b - q) / b
        
        # Use fractional Kelly for safety
        position_pct = min(kelly * kelly_fraction, 0.03)  # Max 3% per trade
        
        return balance * position_pct
    
    def generate_market_data(self, days: int = 30) -> list:
        """Generate realistic market data"""
        candles = []
        base_price = 100
        current_price = base_price
        
        start_time = datetime.now() - timedelta(days=days)
        
        for i in range(days * 288):  # 5-minute candles
            # Market regimes
            regime_cycle = (i % 2000) / 2000
            
            if regime_cycle < 0.3:  # Trending
                trend = 0.0002
                volatility = 0.003
            elif regime_cycle < 0.6:  # Ranging
                trend = 0
                volatility = 0.002
            else:  # Volatile
                trend = -0.0001
                volatility = 0.005
            
            change = np.random.normal(trend, volatility)
            new_price = current_price * (1 + change)
            
            # OHLC
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
        
        return candles
    
    def run_backtest(self, days: int = 30) -> dict:
        """Run edge-optimized backtest"""
        logger.info(f"🚀 Edge-Optimized Backtest ({days} days)")
        
        # Generate data
        candles = self.generate_market_data(days)
        
        # Train
        training_split = int(len(candles) * 0.6)
        training_results = self.ai_system.train_models(candles[:training_split])
        
        # Test
        test_start = training_split
        balance = self.initial_balance
        trades = []
        equity_curve = [balance]
        
        for i in range(100, len(candles) - test_start - 30):
            current_idx = test_start + i
            current_candle = candles[current_idx]
            current_price = current_candle['close']
            
            # Generate signal
            signal_history = candles[:current_idx + 1]
            signal = self.ai_system.generate_signal(signal_history)
            
            if self.ai_system.should_trade(signal):
                # Calculate position size
                position_size = self.calculate_position_size(balance, signal)
                
                if position_size > 0:
                    # Simulate trade with ATR-based stops
                    entry_price = current_price
                    stop_price = entry_price - signal['stop_distance'] if signal['direction'] == 'long' else entry_price + signal['stop_distance']
                    target_price = entry_price + signal['target_distance'] if signal['direction'] == 'long' else entry_price - signal['target_distance']
                    
                    # Look ahead for exit
                    exit_price = None
                    exit_reason = None
                    
                    for j in range(1, min(31, len(candles) - current_idx)):
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
                        else:  # Short
                            if future_candle['high'] >= stop_price:
                                exit_price = stop_price
                                exit_reason = 'stop'
                                break
                            elif future_candle['low'] <= target_price:
                                exit_price = target_price
                                exit_reason = 'target'
                                break
                    
                    if exit_price is None:
                        exit_price = candles[min(current_idx + 30, len(candles) - 1)]['close']
                        exit_reason = 'time'
                    
                    # Calculate P&L
                    if signal['direction'] == 'long':
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price
                    
                    pnl_amount = position_size * pnl_pct
                    balance += pnl_amount
                    
                    trade = {
                        'entry_time': current_candle['timestamp'],
                        'direction': signal['direction'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'position_size': position_size,
                        'pnl_pct': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'balance': balance,
                        'edge': signal['edge'],
                        'win_prob': signal['win_prob']
                    }
                    
                    trades.append(trade)
                    equity_curve.append(balance)
                    
                    if exit_reason == 'target':
                        logger.info(f"✅ WIN #{len(trades)}: {signal['direction']} +{pnl_pct:.2%} (edge: {signal['edge']:.2%})")
                    else:
                        logger.info(f"❌ LOSS #{len(trades)}: {signal['direction']} {pnl_pct:.2%} (edge: {signal['edge']:.2%})")
        
        # Calculate metrics
        if trades:
            df_trades = pd.DataFrame(trades)
            
            wins = df_trades[df_trades['pnl_pct'] > 0]
            losses = df_trades[df_trades['pnl_pct'] <= 0]
            
            win_rate = len(wins) / len(trades)
            avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
            
            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            
            # Profit factor
            total_wins = wins['pnl_amount'].sum() if len(wins) > 0 else 0
            total_losses = abs(losses['pnl_amount'].sum()) if len(losses) > 0 else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Max drawdown
            equity_array = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (running_max - equity_array) / running_max
            max_drawdown = np.max(drawdown)
            
            # Sharpe ratio (simplified)
            returns = df_trades['pnl_pct'].values
            sharpe = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0
            
            results = {
                'total_trades': len(trades),
                'win_rate': win_rate,
                'expectancy': expectancy,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'final_balance': balance,
                'total_return': (balance - self.initial_balance) / self.initial_balance,
                'training_results': training_results
            }
        else:
            results = {
                'total_trades': 0,
                'win_rate': 0,
                'expectancy': 0,
                'profit_factor': 0,
                'training_results': training_results
            }
        
        return results

def main():
    """Run edge optimization tests"""
    print("🚀 EDGE-OPTIMIZED TRADING SYSTEM")
    print("=" * 50)
    print("Focus: Expectancy > Win Rate")
    print("Target: 3:1 R:R with positive edge\n")
    
    test_configs = [
        {"name": "30-Day Edge Test", "days": 30},
        {"name": "60-Day Edge Test", "days": 60},
        {"name": "90-Day Edge Test", "days": 90},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n🧠 {config['name']}...")
        print("-" * 40)
        
        backtester = EdgeOptimizedBacktester()
        results = backtester.run_backtest(config['days'])
        
        print(f"📊 RESULTS:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"   💰 EXPECTANCY: {results.get('expectancy', 0):.3%} per trade")
        print(f"   📈 Profit Factor: {results.get('profit_factor', 0):.2f}")
        
        if results['total_trades'] > 0:
            print(f"   Avg Win: {results.get('avg_win', 0):.2%}")
            print(f"   Avg Loss: {results.get('avg_loss', 0):.2%}")
            print(f"   Max Drawdown: {results.get('max_drawdown', 0):.1%}")
            print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"   Total Return: {results.get('total_return', 0):.1%}")
        
        # Model performance
        print(f"   🧠 Model Accuracy:")
        for model, acc in results.get('training_results', {}).items():
            print(f"      {model}: {acc:.1%}")
        
        all_results.append({
            'config': config['name'],
            'results': results
        })
    
    # Summary
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 50)
    
    total_trades = sum(r['results']['total_trades'] for r in all_results)
    
    if total_trades > 0:
        avg_expectancy = np.mean([r['results'].get('expectancy', 0) for r in all_results if r['results']['total_trades'] > 0])
        avg_profit_factor = np.mean([r['results'].get('profit_factor', 0) for r in all_results if r['results']['total_trades'] > 0])
        avg_win_rate = np.mean([r['results'].get('win_rate', 0) for r in all_results if r['results']['total_trades'] > 0])
        
        print(f"📊 Average Win Rate: {avg_win_rate:.1%}")
        print(f"💰 Average Expectancy: {avg_expectancy:.3%} per trade")
        print(f"📈 Average Profit Factor: {avg_profit_factor:.2f}")
        print(f"✅ Total Trades: {total_trades}")
        
        if avg_expectancy > 0.0035 and avg_profit_factor > 1.5:
            print(f"\n🎯 SUCCESS! Positive edge achieved!")
            print(f"✅ System is profitable with good risk-reward")
            print(f"🚀 Ready for deployment")
        else:
            print(f"\n🔄 Edge needs improvement")
            print(f"📈 Consider adjusting R:R or filters")
    
    # Save results
    with open('edge_optimization_results.json', 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'avg_expectancy': avg_expectancy if total_trades > 0 else 0,
            'avg_profit_factor': avg_profit_factor if total_trades > 0 else 0,
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to 'edge_optimization_results.json'")

if __name__ == "__main__":
    main() 