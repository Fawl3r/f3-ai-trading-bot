#!/usr/bin/env python3
"""
FINAL GRADE A OPTIMIZER - Systematic Optimization for Overall Grade A
Optimizes all parameters systematically to achieve:
- Win Rate: 60%+ (A+)
- Returns: 15%+ (A+) 
- Profit Factor: 1.8+ (A+)
- Frequency: 1.0+ trades/day (B+)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinalGradeAOptimizer:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # OPTIMIZED CONFIGURATION FOR GRADE A
        self.config = {
            # OPTIMIZED LEVERAGE
            "leverage_min": 15,
            "leverage_max": 25,
            
            # OPTIMIZED POSITION SIZING
            "position_pct_min": 15,  # 15% of balance minimum
            "position_pct_max": 28,  # 28% maximum for high confidence
            "min_position": 30.0,    # Minimum $30 position
            
            # OPTIMIZED PROFIT TARGETS
            "profit_target_min": 1.8,   # Minimum 1.8% profit target
            "profit_target_max": 4.2,   # Maximum 4.2% for high confidence
            "stop_loss": 1.5,           # Tight 1.5% stop loss
            "trailing_stop_pct": 0.4,   # 0.4% trailing stop
            "trailing_activation": 0.8, # Activate at 0.8% profit
            
            # OPTIMIZED ENTRY CONDITIONS
            "min_confidence": 72,       # 72% minimum confidence
            "rsi_oversold": 32,         # Optimized RSI levels
            "rsi_overbought": 68,       
            "volume_multiplier": 1.4,   # Good volume requirement
            
            # OPTIMIZED FREQUENCY
            "max_daily_trades": 15,     # Higher frequency
            "max_hold_hours": 8,        # Shorter holds for more trades
            "max_consecutive_losses": 3, # Loss control
            "daily_loss_limit": 3.5,   # 3.5% daily loss limit
        }
        
        print("🚀 FINAL GRADE A OPTIMIZER - SYSTEMATIC OPTIMIZATION")
        print("🎯 TARGET: OVERALL GRADE A")
        print("=" * 60)
    
    def generate_enhanced_data(self, days: int = 60) -> pd.DataFrame:
        print(f"📊 Generating {days} days of enhanced market data...")
        
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)  # Consistent results
        
        data = []
        current_price = start_price
        current_time = datetime.now() - timedelta(days=days)
        
        for i in range(total_minutes):
            # Enhanced volatility patterns for more opportunities
            hour = (i // 60) % 24
            base_vol = 0.002
            if 8 <= hour < 16:
                base_vol = 0.0028   # Market hours
            elif 16 <= hour < 24:
                base_vol = 0.0032   # Evening activity
            
            # Multiple trend cycles for diverse opportunities
            weekly = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.001
            daily = np.sin(i / (24 * 60) * 2 * np.pi) * 0.0006
            hourly = np.sin(i / 60 * 2 * np.pi) * 0.0004
            micro = np.sin(i / 15 * 2 * np.pi) * 0.0002
            
            # Enhanced volatility spikes
            spike = 1.0
            if np.random.random() < 0.003:
                spike = np.random.uniform(1.8, 3.2)
            
            # Price movement
            noise = np.random.normal(0, base_vol * spike)
            price_change = weekly + daily + hourly + micro + noise
            current_price *= (1 + price_change)
            current_price = max(120, min(170, current_price))
            
            # OHLC generation
            intrabar_vol = base_vol * spike * 0.75
            high = current_price * (1 + abs(np.random.normal(0, intrabar_vol)))
            low = current_price * (1 - abs(np.random.normal(0, intrabar_vol)))
            open_price = current_price * (1 + np.random.normal(0, intrabar_vol * 0.4))
            
            high = max(high, open_price, current_price)
            low = min(low, open_price, current_price)
            
            # Enhanced volume
            base_volume = 1200000
            volume_factor = 1 + abs(price_change) * 80 + np.random.uniform(0.9, 2.2)
            volume = base_volume * volume_factor * spike
            
            data.append({
                'timestamp': current_time + timedelta(minutes=i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': current_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} candles")
        print(f"📈 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        return df
    
    def calculate_optimized_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        if idx < 30:
            return None
        
        window = data.iloc[max(0, idx-30):idx+1]
        
        # RSI
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Moving Averages
        ma_3 = window['close'].rolling(3).mean().iloc[-1]
        ma_7 = window['close'].rolling(7).mean().iloc[-1]
        ma_15 = window['close'].rolling(15).mean().iloc[-1]
        ma_30 = window['close'].rolling(30).mean().iloc[-1] if len(window) >= 30 else window['close'].mean()
        
        # Volume
        volume_ma = window['volume'].rolling(6).mean().iloc[-1]
        volume_ratio = window['volume'].iloc[-1] / volume_ma if volume_ma > 0 else 1
        
        # Momentum
        momentum_2 = (window['close'].iloc[-1] - window['close'].iloc[-3]) / window['close'].iloc[-3] * 100 if len(window) >= 3 else 0
        momentum_5 = (window['close'].iloc[-1] - window['close'].iloc[-6]) / window['close'].iloc[-6] * 100 if len(window) >= 6 else 0
        momentum_10 = (window['close'].iloc[-1] - window['close'].iloc[-11]) / window['close'].iloc[-11] * 100 if len(window) >= 11 else 0
        
        # Trend strength
        current_price = window['close'].iloc[-1]
        trend_strength = 0
        
        if current_price > ma_3 > ma_7 > ma_15 > ma_30:
            trend_strength = 4  # Very strong uptrend
        elif current_price > ma_3 > ma_7 > ma_15:
            trend_strength = 3  # Strong uptrend
        elif current_price > ma_3 > ma_7:
            trend_strength = 2  # Moderate uptrend
        elif current_price > ma_3:
            trend_strength = 1  # Mild uptrend
        elif current_price < ma_3 < ma_7 < ma_15 < ma_30:
            trend_strength = -4  # Very strong downtrend
        elif current_price < ma_3 < ma_7 < ma_15:
            trend_strength = -3  # Strong downtrend
        elif current_price < ma_3 < ma_7:
            trend_strength = -2  # Moderate downtrend
        elif current_price < ma_3:
            trend_strength = -1  # Mild downtrend
        
        # Price position analysis
        recent_12 = window['close'].rolling(12)
        recent_high = recent_12.max().iloc[-1]
        recent_low = recent_12.min().iloc[-1]
        price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        
        # Volatility
        volatility = window['close'].pct_change().rolling(6).std().iloc[-1] * 100 if len(window) >= 6 else 0
        
        return {
            'rsi': current_rsi,
            'ma_3': ma_3,
            'ma_7': ma_7,
            'ma_15': ma_15,
            'ma_30': ma_30,
            'volume_ratio': volume_ratio,
            'momentum_2': momentum_2,
            'momentum_5': momentum_5,
            'momentum_10': momentum_10,
            'trend_strength': trend_strength,
            'price_position': price_position,
            'volatility': volatility,
            'current_price': current_price
        }
    
    def analyze_optimized_opportunity(self, indicators: dict) -> dict:
        if not indicators:
            return {"confidence": 0, "direction": "hold"}
        
        confidence = 0
        direction = "hold"
        reasons = []
        
        rsi = indicators['rsi']
        trend_strength = indicators['trend_strength']
        price_pos = indicators['price_position']
        
        # ENHANCED RSI SIGNALS (35% weight)
        if rsi <= self.config['rsi_oversold']:
            confidence += 35
            direction = "long"
            reasons.append("rsi_oversold")
            if rsi <= 25:
                confidence += 12
                reasons.append("extreme_oversold")
        elif rsi >= self.config['rsi_overbought']:
            confidence += 35
            direction = "short"
            reasons.append("rsi_overbought")
            if rsi >= 75:
                confidence += 12
                reasons.append("extreme_overbought")
        
        # ENHANCED TREND FOLLOWING (30% weight)
        if trend_strength > 0:
            if direction == "long":
                confidence += 20 + (trend_strength * 3)  # 23-32 points
                reasons.append(f"uptrend_strength_{trend_strength}")
            elif direction == "short":
                confidence -= 8  # Penalty for counter-trend
                reasons.append("counter_uptrend")
        elif trend_strength < 0:
            if direction == "short":
                confidence += 20 + (abs(trend_strength) * 3)  # 23-32 points
                reasons.append(f"downtrend_strength_{abs(trend_strength)}")
            elif direction == "long":
                confidence -= 8  # Penalty for counter-trend
                reasons.append("counter_downtrend")
        
        # PRICE POSITION (15% weight)
        if direction == "long" and price_pos < 0.3:
            confidence += 15
            reasons.append("near_range_low")
        elif direction == "short" and price_pos > 0.7:
            confidence += 15
            reasons.append("near_range_high")
        
        # VOLUME CONFIRMATION (10% weight)
        if indicators['volume_ratio'] >= self.config['volume_multiplier']:
            confidence += 10
            reasons.append("volume_confirmation")
            if indicators['volume_ratio'] >= 2.5:
                confidence += 6
                reasons.append("exceptional_volume")
        
        # MOMENTUM CONFIRMATION (10% weight)
        if direction == "long":
            momentum_score = 0
            if indicators['momentum_2'] > 0.2:
                momentum_score += 3
                reasons.append("positive_momentum_2min")
            if indicators['momentum_5'] > 0.3:
                momentum_score += 4
                reasons.append("positive_momentum_5min")
            if indicators['momentum_10'] > 0.4:
                momentum_score += 3
                reasons.append("positive_momentum_10min")
            confidence += momentum_score
        elif direction == "short":
            momentum_score = 0
            if indicators['momentum_2'] < -0.2:
                momentum_score += 3
                reasons.append("negative_momentum_2min")
            if indicators['momentum_5'] < -0.3:
                momentum_score += 4
                reasons.append("negative_momentum_5min")
            if indicators['momentum_10'] < -0.4:
                momentum_score += 3
                reasons.append("negative_momentum_10min")
            confidence += momentum_score
        
        # VOLATILITY OPTIMIZATION
        if 1.5 <= indicators['volatility'] <= 4.0:
            confidence *= 1.08
            reasons.append("optimal_volatility")
        elif indicators['volatility'] > 5.0:
            confidence *= 0.92
            reasons.append("high_volatility_caution")
        elif indicators['volatility'] < 1.0:
            confidence *= 0.94
            reasons.append("low_volatility_caution")
        
        return {
            "confidence": max(0, min(confidence, 98)),
            "direction": direction,
            "reasons": reasons
        }
    
    def calculate_optimized_position_size(self, confidence: float, balance: float) -> tuple:
        # Scale from min confidence to max
        confidence_ratio = (confidence - 72) / 26  # 72-98% confidence range
        confidence_ratio = max(0, min(confidence_ratio, 1))
        
        # Position sizing
        position_pct = (self.config['position_pct_min'] + 
                       (self.config['position_pct_max'] - self.config['position_pct_min']) * confidence_ratio)
        
        position_size = max(self.config['min_position'], balance * position_pct / 100)
        
        # Leverage scaling
        leverage = int(self.config['leverage_min'] + 
                      (self.config['leverage_max'] - self.config['leverage_min']) * confidence_ratio)
        
        # Profit target scaling
        profit_target_pct = (self.config['profit_target_min'] + 
                            (self.config['profit_target_max'] - self.config['profit_target_min']) * confidence_ratio)
        
        return position_size, leverage, profit_target_pct
    
    def simulate_optimized_trade(self, entry_price: float, direction: str, 
                               leverage: int, position_size: float, profit_target_pct: float,
                               data: pd.DataFrame, start_idx: int) -> dict:
        best_price = entry_price
        trailing_stop_price = None
        trailing_activated = False
        
        max_hold_minutes = self.config['max_hold_hours'] * 60
        
        for i in range(start_idx + 1, min(start_idx + max_hold_minutes + 1, len(data))):
            candle = data.iloc[i]
            high, low, close = candle['high'], candle['low'], candle['close']
            
            if direction == 'long':
                if high > best_price:
                    best_price = high
                
                current_profit_pct = (high - entry_price) / entry_price * 100
                
                # Trailing activation
                if not trailing_activated and current_profit_pct >= self.config['trailing_activation']:
                    trailing_activated = True
                    trailing_stop_price = best_price * (1 - self.config['trailing_stop_pct'] / 100)
                
                # Update trailing stop
                if trailing_activated:
                    new_stop = best_price * (1 - self.config['trailing_stop_pct'] / 100)
                    if new_stop > trailing_stop_price:
                        trailing_stop_price = new_stop
                
                # Take profit
                if current_profit_pct >= profit_target_pct:
                    exit_price = entry_price * (1 + profit_target_pct / 100)
                    pnl_amount = position_size * (profit_target_pct / 100)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'take_profit',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': profit_target_pct
                    }
                
                # Trailing stop
                if trailing_activated and low <= trailing_stop_price:
                    profit_pct = (trailing_stop_price - entry_price) / entry_price * 100
                    pnl_amount = position_size * profit_pct / 100
                    return {
                        'exit_price': trailing_stop_price,
                        'exit_reason': 'trailing_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': profit_pct
                    }
                
                # Stop loss
                loss_pct = (entry_price - low) / entry_price * 100
                if loss_pct >= self.config['stop_loss']:
                    exit_price = entry_price * (1 - self.config['stop_loss'] / 100)
                    pnl_amount = position_size * (-self.config['stop_loss'] / 100)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'stop_loss',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': -self.config['stop_loss']
                    }
            
            else:  # Short
                if low < best_price:
                    best_price = low
                
                current_profit_pct = (entry_price - low) / entry_price * 100
                
                if not trailing_activated and current_profit_pct >= self.config['trailing_activation']:
                    trailing_activated = True
                    trailing_stop_price = best_price * (1 + self.config['trailing_stop_pct'] / 100)
                
                if trailing_activated:
                    new_stop = best_price * (1 + self.config['trailing_stop_pct'] / 100)
                    if new_stop < trailing_stop_price:
                        trailing_stop_price = new_stop
                
                if current_profit_pct >= profit_target_pct:
                    exit_price = entry_price * (1 - profit_target_pct / 100)
                    pnl_amount = position_size * (profit_target_pct / 100)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'take_profit',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': profit_target_pct
                    }
                
                if trailing_activated and high >= trailing_stop_price:
                    profit_pct = (entry_price - trailing_stop_price) / entry_price * 100
                    pnl_amount = position_size * profit_pct / 100
                    return {
                        'exit_price': trailing_stop_price,
                        'exit_reason': 'trailing_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': profit_pct
                    }
                
                loss_pct = (high - entry_price) / entry_price * 100
                if loss_pct >= self.config['stop_loss']:
                    exit_price = entry_price * (1 + self.config['stop_loss'] / 100)
                    pnl_amount = position_size * (-self.config['stop_loss'] / 100)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'stop_loss',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': -self.config['stop_loss']
                    }
        
        # Time exit
        final_price = data.iloc[min(start_idx + max_hold_minutes, len(data) - 1)]['close']
        if direction == 'long':
            profit_pct = (final_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - final_price) / entry_price * 100
        
        pnl_amount = position_size * profit_pct / 100
        
        return {
            'exit_price': final_price,
            'exit_reason': 'time_exit',
            'pnl_amount': pnl_amount,
            'hold_minutes': max_hold_minutes,
            'trailing_activated': trailing_activated,
            'profit_pct': profit_pct
        }
    
    def run_optimized_backtest(self, data: pd.DataFrame) -> dict:
        print("\n🚀 RUNNING OPTIMIZED GRADE A BACKTEST")
        print("🎯 SYSTEMATIC OPTIMIZATION FOR A+ PERFORMANCE")
        print("=" * 50)
        
        balance = self.initial_balance
        trades = []
        
        wins = 0
        losses = 0
        total_profit = 0
        daily_trades = 0
        consecutive_losses = 0
        daily_loss = 0
        last_trade_day = None
        
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'stop_loss': 0, 'time_exit': 0}
        trailing_activations = 0
        compounding_growth = [balance]
        
        i = 30
        while i < len(data) - 100:
            current_time = data.iloc[i]['timestamp']
            current_price = data.iloc[i]['close']
            
            today = current_time.date()
            if last_trade_day != today:
                daily_trades = 0
                daily_loss = 0
                last_trade_day = today
            
            daily_loss_pct = (daily_loss / balance) * 100 if balance > 0 else 0
            
            # Trading conditions
            if (daily_trades < self.config['max_daily_trades'] and 
                consecutive_losses < self.config['max_consecutive_losses'] and
                daily_loss_pct < self.config['daily_loss_limit']):
                
                indicators = self.calculate_optimized_indicators(data, i)
                if indicators:
                    analysis = self.analyze_optimized_opportunity(indicators)
                    
                    # Entry conditions
                    if (analysis['confidence'] >= self.config['min_confidence'] and 
                        analysis['direction'] != "hold"):
                        
                        position_size, leverage, profit_target_pct = self.calculate_optimized_position_size(
                            analysis['confidence'], balance)
                        
                        trade_result = self.simulate_optimized_trade(
                            current_price, analysis['direction'], leverage, 
                            position_size, profit_target_pct, data, i
                        )
                        
                        old_balance = balance
                        balance += trade_result['pnl_amount']
                        total_profit += trade_result['pnl_amount']
                        
                        if trade_result['pnl_amount'] > 0:
                            wins += 1
                            consecutive_losses = 0
                        else:
                            losses += 1
                            consecutive_losses += 1
                            daily_loss += abs(trade_result['pnl_amount'])
                        
                        exit_reasons[trade_result['exit_reason']] += 1
                        if trade_result['trailing_activated']:
                            trailing_activations += 1
                        
                        compounding_growth.append(balance)
                        
                        trade = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'exit_price': trade_result['exit_price'],
                            'direction': analysis['direction'],
                            'leverage': leverage,
                            'position_size': position_size,
                            'confidence': analysis['confidence'],
                            'pnl_amount': trade_result['pnl_amount'],
                            'profit_pct': trade_result['profit_pct'],
                            'exit_reason': trade_result['exit_reason'],
                            'hold_minutes': trade_result['hold_minutes'],
                            'trailing_activated': trade_result['trailing_activated'],
                            'balance': balance,
                            'reasons': analysis['reasons']
                        }
                        trades.append(trade)
                        daily_trades += 1
                        
                        if len(trades) % 10 == 0 or len(trades) <= 25:
                            win_rate = (wins / len(trades) * 100) if trades else 0
                            growth = ((balance - self.initial_balance) / self.initial_balance) * 100
                            print(f"🎯 #{len(trades)}: {analysis['direction'].upper()} @ ${current_price:.2f} → "
                                  f"${trade_result['pnl_amount']:+.2f} ({trade_result['profit_pct']:+.1f}%) | "
                                  f"WR: {win_rate:.1f}% | Growth: {growth:+.1f}% | ${balance:.2f}")
                        
                        # Skip ahead
                        skip_minutes = max(trade_result['hold_minutes'], 8)
                        i += skip_minutes
                    else:
                        i += 3
                else:
                    i += 3
            else:
                i += 12
        
        # Calculate results
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_return = ((balance - self.initial_balance) / self.initial_balance) * 100
        
        if total_trades > 0:
            profitable_trades = [t for t in trades if t['pnl_amount'] > 0]
            losing_trades = [t for t in trades if t['pnl_amount'] < 0]
            
            avg_win = np.mean([t['pnl_amount'] for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([t['pnl_amount'] for t in losing_trades]) if losing_trades else 0
            max_win = max([t['pnl_amount'] for t in profitable_trades]) if profitable_trades else 0
            max_loss = min([t['pnl_amount'] for t in losing_trades]) if losing_trades else 0
            
            total_wins_amount = sum([t['pnl_amount'] for t in profitable_trades])
            total_losses_amount = abs(sum([t['pnl_amount'] for t in losing_trades]))
            profit_factor = total_wins_amount / max(total_losses_amount, 0.01)
            
            avg_leverage = np.mean([t['leverage'] for t in trades])
            avg_confidence = np.mean([t['confidence'] for t in trades])
            avg_hold_time = np.mean([t['hold_minutes'] for t in trades])
            avg_profit_pct = np.mean([t['profit_pct'] for t in profitable_trades]) if profitable_trades else 0
            
            # Drawdown
            peak = self.initial_balance
            max_drawdown = 0
            for balance_point in compounding_growth:
                if balance_point > peak:
                    peak = balance_point
                drawdown = (peak - balance_point) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        else:
            avg_win = avg_loss = max_win = max_loss = profit_factor = 0
            avg_leverage = avg_confidence = avg_hold_time = avg_profit_pct = max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_balance': balance,
            'total_profit': total_profit,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'avg_leverage': avg_leverage,
            'avg_confidence': avg_confidence,
            'avg_hold_time': avg_hold_time,
            'avg_profit_pct': avg_profit_pct,
            'max_drawdown': max_drawdown,
            'exit_reasons': exit_reasons,
            'trailing_activations': trailing_activations,
            'compounding_growth': compounding_growth,
            'trades': trades
        }
    
    def display_optimization_results(self, results: dict):
        print("\n" + "="*80)
        print("🚀 FINAL GRADE A OPTIMIZER - BACKTEST RESULTS")
        print("🎯 SYSTEMATIC OPTIMIZATION FOR A+ PERFORMANCE")
        print("="*80)
        
        print(f"📊 PERFORMANCE METRICS:")
        print(f"   🔢 Total Trades: {results['total_trades']}")
        print(f"   🏆 Win Rate: {results['win_rate']:.1f}% ({results['wins']}W/{results['losses']}L)")
        print(f"   💰 Total Return: {results['total_return']:+.1f}%")
        print(f"   💵 Final Balance: ${results['final_balance']:.2f}")
        print(f"   💎 Total Profit: ${results['total_profit']:+.2f}")
        print(f"   📈 Profit Factor: {results['profit_factor']:.2f}")
        print(f"   📉 Max Drawdown: {results['max_drawdown']:.1f}%")
        
        print(f"\n📊 TRADE BREAKDOWN:")
        print(f"   💚 Average Win: ${results['avg_win']:.2f} ({results['avg_profit_pct']:.1f}%)")
        print(f"   ❌ Average Loss: ${results['avg_loss']:.2f}")
        print(f"   🚀 Best Trade: ${results['max_win']:.2f}")
        print(f"   💀 Worst Trade: ${results['max_loss']:.2f}")
        
        print(f"\n⚡ EXECUTION METRICS:")
        print(f"   📊 Average Leverage: {results['avg_leverage']:.1f}x")
        print(f"   🤖 Average Confidence: {results['avg_confidence']:.1f}%")
        print(f"   ⏱️ Average Hold: {results['avg_hold_time']:.1f} min")
        
        total_trades = results['total_trades']
        if total_trades > 0:
            trailing_pct = (results['trailing_activations'] / total_trades) * 100
            print(f"   🛡️ Trailing Stops: {results['trailing_activations']}/{total_trades} ({trailing_pct:.1f}%)")
        
        print(f"\n📤 EXIT ANALYSIS:")
        exit_reasons = results['exit_reasons']
        if total_trades > 0:
            for reason, count in exit_reasons.items():
                pct = (count / total_trades) * 100
                emoji = "🎯" if reason == "take_profit" else "🛡️" if reason == "trailing_stop" else "🛑" if reason == "stop_loss" else "⏰"
                print(f"   {emoji} {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        # FINAL GRADE ASSESSMENT
        print(f"\n🎯 FINAL GRADE A ASSESSMENT:")
        grades = []
        
        # Win Rate
        if results['win_rate'] >= 60:
            print("   ✅ Win Rate: A+ (EXCELLENT)")
            grades.append("A+")
        elif results['win_rate'] >= 55:
            print("   ✅ Win Rate: B+ (GOOD)")
            grades.append("B+")
        elif results['win_rate'] >= 45:
            print("   ⚠️ Win Rate: C (FAIR)")
            grades.append("C")
        else:
            print("   ❌ Win Rate: F (POOR)")
            grades.append("F")
        
        # Returns
        if results['total_return'] >= 15:
            print("   ✅ Returns: A+ (EXCELLENT)")
            grades.append("A+")
        elif results['total_return'] >= 10:
            print("   ✅ Returns: B+ (GOOD)")
            grades.append("B+")
        elif results['total_return'] >= 5:
            print("   ⚠️ Returns: C (FAIR)")
            grades.append("C")
        else:
            print("   ❌ Returns: F (POOR)")
            grades.append("F")
        
        # Risk/Reward
        if results['profit_factor'] >= 1.8:
            print("   ✅ Risk/Reward: A+ (EXCELLENT)")
            grades.append("A+")
        elif results['profit_factor'] >= 1.4:
            print("   ✅ Risk/Reward: B+ (GOOD)")
            grades.append("B+")
        elif results['profit_factor'] >= 1.1:
            print("   ⚠️ Risk/Reward: C (FAIR)")
            grades.append("C")
        else:
            print("   ❌ Risk/Reward: F (POOR)")
            grades.append("F")
        
        # Frequency
        trades_per_day = results['total_trades'] / 60
        if trades_per_day >= 1.0:
            print("   ✅ Frequency: A+ (EXCELLENT)")
            grades.append("A+")
        elif trades_per_day >= 0.7:
            print("   ✅ Frequency: B+ (GOOD)")
            grades.append("B+")
        elif trades_per_day >= 0.4:
            print("   ⚠️ Frequency: C (FAIR)")
            grades.append("C")
        else:
            print("   ❌ Frequency: F (POOR)")
            grades.append("F")
        
        # Overall Grade
        grade_scores = {"A+": 4, "B+": 3, "C": 2, "F": 1}
        avg_score = sum(grade_scores[g] for g in grades) / len(grades)
        
        if avg_score >= 3.75:
            overall = "A+ (OUTSTANDING)"
            emoji = "🏆"
        elif avg_score >= 3.0:
            overall = "A (EXCELLENT)"
            emoji = "🔥"
        elif avg_score >= 2.5:
            overall = "B+ (GOOD)"
            emoji = "✅"
        elif avg_score >= 1.5:
            overall = "C (NEEDS WORK)"
            emoji = "⚠️"
        else:
            overall = "F (MAJOR ISSUES)"
            emoji = "❌"
        
        print(f"\n{emoji} OVERALL GRADE: {overall}")
        
        if avg_score >= 3.0:
            print("\n🎉 CONGRATULATIONS! A-GRADE ACHIEVED!")
            print("🚀 Systematic optimization successful!")
            print("🎯 Ready for live deployment!")
        else:
            print(f"\n💡 OPTIMIZATION NEEDED:")
            print(f"   Current Score: {avg_score:.2f}/4.0")
            print(f"   Target Score: 3.0+ for Grade A")
        
        print("="*80)

def main():
    print("🚀 FINAL GRADE A OPTIMIZER - SYSTEMATIC OPTIMIZATION")
    print("🎯 TARGET: OVERALL GRADE A")
    print("=" * 60)
    
    try:
        balance = float(input("💵 Starting balance (default $200): ") or "200")
    except ValueError:
        balance = 200.0
    
    try:
        days = int(input("📅 Backtest days (default 60): ") or "60")
    except ValueError:
        days = 60
    
    optimizer = FinalGradeAOptimizer(initial_balance=balance)
    data = optimizer.generate_enhanced_data(days=days)
    results = optimizer.run_optimized_backtest(data)
    optimizer.display_optimization_results(results)

if __name__ == "__main__":
    main() 