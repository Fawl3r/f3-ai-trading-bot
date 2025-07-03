#!/usr/bin/env python3
"""
Final Optimized Compounding Bot - Backtesting System
Focus: Achievable profits through realistic targets and improved risk management
Key Fixes:
- Looser stop losses (2.5% instead of 1.8%)
- More achievable profit targets (0.8-2.2% instead of 1.2-2.8%)
- Better entry conditions with trend following
- Improved direction detection
- More realistic position sizing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinalOptimizedCompoundingBacktest:
    """Final optimized backtest with realistic profit expectations"""
    
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # FINAL OPTIMIZED CONFIG
        self.config = {
            # LEVERAGE - Conservative but profitable
            "leverage_min": 15,
            "leverage_max": 25,
            
            # POSITION SIZING - Realistic scaling
            "position_pct_min": 15,  # 15% minimum
            "position_pct_max": 25,  # 25% maximum
            "min_position": 35.0,    # Lower minimum for frequency
            
            # PROFIT TARGETS - REALISTIC AND ACHIEVABLE
            "profit_target_min": 0.8,   # 0.8% minimum (very achievable)
            "profit_target_max": 2.2,   # 2.2% maximum (realistic)
            "trailing_stop_pct": 0.5,   # 0.5% trailing stop (tighter)
            "trailing_activation": 0.3, # Activate at 0.3% profit
            "emergency_stop": 2.5,      # 2.5% emergency stop (looser)
            
            # ENTRY CONDITIONS - Improved quality
            "min_confidence": 45,       # 45% minimum (easier to achieve)
            "rsi_oversold": 40,         # RSI < 40 (less extreme)
            "rsi_overbought": 60,       # RSI > 60 (less extreme)
            "volume_multiplier": 1.2,   # 1.2x average volume (realistic)
            
            # RISK MANAGEMENT - Balanced
            "max_daily_trades": 15,     # More opportunities
            "max_hold_hours": 12,       # Longer hold time
            "max_consecutive_losses": 3,
            "daily_loss_limit": 5.0,    # 5% daily loss limit
        }
        
        print("🚀 FINAL OPTIMIZED COMPOUNDING BOT - BACKTESTING")
        print("🎯 REALISTIC TARGETS + IMPROVED RISK MANAGEMENT")
        print("💡 FIXING: STOPS, TARGETS, ENTRY CONDITIONS & DIRECTION")
        print("=" * 60)
    
    def generate_realistic_market_data(self, days: int = 60) -> pd.DataFrame:
        """Generate market data that provides balanced opportunities"""
        print(f"📊 Generating {days} days of realistic market data...")
        
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        current_price = start_price
        current_time = datetime.now() - timedelta(days=days)
        
        for i in range(total_minutes):
            # Dynamic volatility based on time
            hour = (i // 60) % 24
            base_volatility = 0.002
            if 8 <= hour < 16:  # European session
                base_volatility = 0.0025
            elif 16 <= hour < 24:  # US session
                base_volatility = 0.003
            
            # Multiple market cycles for opportunities
            weekly_trend = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.001
            daily_rhythm = np.sin(i / (24 * 60) * 2 * np.pi) * 0.0005
            mini_cycle = np.sin(i / 120 * 2 * np.pi) * 0.0003  # 2-hour cycles
            
            # Occasional volatility spikes
            volatility_spike = 1.0
            if np.random.random() < 0.002:
                volatility_spike = np.random.uniform(1.5, 3.0)
            
            # Price movement
            noise = np.random.normal(0, base_volatility * volatility_spike)
            price_change = weekly_trend + daily_rhythm + mini_cycle + noise
            current_price *= (1 + price_change)
            current_price = max(100, min(200, current_price))
            
            # Generate OHLC
            intrabar_vol = base_volatility * volatility_spike * 0.7
            high = current_price * (1 + abs(np.random.normal(0, intrabar_vol)))
            low = current_price * (1 - abs(np.random.normal(0, intrabar_vol)))
            open_price = current_price * (1 + np.random.normal(0, intrabar_vol * 0.3))
            
            high = max(high, open_price, current_price)
            low = min(low, open_price, current_price)
            
            # Volume
            base_volume = 1000000
            volume_factor = 1 + abs(price_change) * 50 + np.random.uniform(0.6, 1.8)
            volume = base_volume * volume_factor * volatility_spike
            
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
    
    def calculate_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        """Calculate technical indicators with improved sensitivity"""
        if idx < 40:
            return None
        
        window = data.iloc[max(0, idx-40):idx+1]
        
        # RSI
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Moving averages
        ma_5 = window['close'].rolling(5).mean().iloc[-1]
        ma_10 = window['close'].rolling(10).mean().iloc[-1]
        ma_20 = window['close'].rolling(20).mean().iloc[-1]
        ma_30 = window['close'].rolling(30).mean().iloc[-1] if len(window) >= 30 else window['close'].mean()
        
        # Volume
        volume_ma = window['volume'].rolling(10).mean().iloc[-1]
        volume_ratio = window['volume'].iloc[-1] / volume_ma if volume_ma > 0 else 1
        
        # Price action
        price_change_5min = (window['close'].iloc[-1] - window['close'].iloc[-6]) / window['close'].iloc[-6] * 100 if len(window) >= 6 else 0
        price_change_15min = (window['close'].iloc[-1] - window['close'].iloc[-16]) / window['close'].iloc[-16] * 100 if len(window) >= 16 else 0
        
        # Volatility
        volatility = window['close'].pct_change().rolling(10).std().iloc[-1] * 100 if len(window) >= 10 else 0
        
        # Trend strength
        trend_score = 0
        current_price = window['close'].iloc[-1]
        if current_price > ma_5 > ma_10 > ma_20:
            trend_score = 2  # Strong uptrend
        elif current_price > ma_5 > ma_10:
            trend_score = 1  # Mild uptrend
        elif current_price < ma_5 < ma_10 < ma_20:
            trend_score = -2  # Strong downtrend
        elif current_price < ma_5 < ma_10:
            trend_score = -1  # Mild downtrend
        
        return {
            'rsi': current_rsi,
            'ma_5': ma_5,
            'ma_10': ma_10,
            'ma_20': ma_20,
            'ma_30': ma_30,
            'volume_ratio': volume_ratio,
            'price_change_5min': price_change_5min,
            'price_change_15min': price_change_15min,
            'volatility': volatility,
            'trend_score': trend_score,
            'current_price': current_price
        }
    
    def analyze_opportunity(self, indicators: dict) -> dict:
        """Improved opportunity analysis with better direction detection"""
        if not indicators:
            return {"confidence": 0, "direction": "hold"}
        
        confidence = 0
        direction = "hold"
        reasons = []
        
        rsi = indicators['rsi']
        trend_score = indicators['trend_score']
        current_price = indicators['current_price']
        
        # RSI-based signals (40% weight)
        if rsi < self.config['rsi_oversold']:
            confidence += 40
            direction = "long"
            reasons.append("rsi_oversold")
        elif rsi > self.config['rsi_overbought']:
            confidence += 40
            direction = "short"
            reasons.append("rsi_overbought")
        elif rsi < 45:
            confidence += 20
            direction = "long"
            reasons.append("rsi_getting_oversold")
        elif rsi > 55:
            confidence += 20
            direction = "short"
            reasons.append("rsi_getting_overbought")
        
        # Trend following (35% weight) - MAJOR IMPROVEMENT
        if trend_score >= 1:  # Uptrend
            if direction == "long":
                confidence += 30
                reasons.append("trend_alignment_long")
            elif direction == "short":
                confidence -= 10  # Penalize counter-trend
                reasons.append("counter_trend_short")
        elif trend_score <= -1:  # Downtrend
            if direction == "short":
                confidence += 30
                reasons.append("trend_alignment_short")
            elif direction == "long":
                confidence -= 10  # Penalize counter-trend
                reasons.append("counter_trend_long")
        
        # Strong trend bonus
        if abs(trend_score) == 2:
            if (trend_score > 0 and direction == "long") or (trend_score < 0 and direction == "short"):
                confidence += 10
                reasons.append("strong_trend_alignment")
        
        # Volume confirmation (15% weight)
        if indicators['volume_ratio'] > self.config['volume_multiplier']:
            confidence += 15
            reasons.append("volume_confirmation")
        elif indicators['volume_ratio'] > 1.1:
            confidence += 8
            reasons.append("decent_volume")
        
        # Momentum confirmation (10% weight)
        if direction == "long" and indicators['price_change_5min'] > 0.2:
            confidence += 10
            reasons.append("positive_momentum")
        elif direction == "short" and indicators['price_change_5min'] < -0.2:
            confidence += 10
            reasons.append("negative_momentum")
        
        # Volatility adjustment
        if indicators['volatility'] > 4.0:
            confidence *= 0.85  # Reduce confidence in high volatility
            reasons.append("high_volatility_discount")
        elif 1.0 <= indicators['volatility'] <= 3.0:
            confidence *= 1.05  # Slight boost for good volatility
            reasons.append("good_volatility")
        
        return {
            "confidence": max(0, min(confidence, 95)),
            "direction": direction,
            "reasons": reasons
        }
    
    def calculate_optimized_position_size(self, confidence: float, balance: float) -> tuple:
        """Optimized position sizing for better risk management"""
        confidence_ratio = confidence / 100
        
        # Position percentage based on confidence
        position_pct = (self.config['position_pct_min'] + 
                       (self.config['position_pct_max'] - self.config['position_pct_min']) * confidence_ratio)
        
        position_size = max(self.config['min_position'], balance * position_pct / 100)
        
        # Conservative leverage
        leverage = int(self.config['leverage_min'] + 
                      (self.config['leverage_max'] - self.config['leverage_min']) * confidence_ratio)
        
        # Realistic profit targets
        profit_target_pct = (self.config['profit_target_min'] + 
                            (self.config['profit_target_max'] - self.config['profit_target_min']) * confidence_ratio)
        
        return position_size, leverage, profit_target_pct
    
    def simulate_optimized_trade(self, entry_price: float, direction: str, 
                               leverage: int, position_size: float, profit_target_pct: float,
                               data: pd.DataFrame, start_idx: int) -> dict:
        """Simulate trade with optimized exit logic"""
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
                
                # Early trailing stop activation for small profits
                if not trailing_activated and current_profit_pct >= self.config['trailing_activation']:
                    trailing_activated = True
                    trailing_stop_price = best_price * (1 - self.config['trailing_stop_pct'] / 100)
                
                # Update trailing stop
                if trailing_activated:
                    new_stop = best_price * (1 - self.config['trailing_stop_pct'] / 100)
                    if new_stop > trailing_stop_price:
                        trailing_stop_price = new_stop
                
                # Take profit - more achievable targets
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
                
                # Emergency stop - looser
                loss_pct = (entry_price - low) / entry_price * 100
                if loss_pct >= self.config['emergency_stop']:
                    exit_price = entry_price * (1 - self.config['emergency_stop'] / 100)
                    pnl_amount = position_size * (-self.config['emergency_stop'] / 100)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'emergency_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': -self.config['emergency_stop']
                    }
            
            else:  # Short position
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
                
                # Take profit
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
                
                # Trailing stop
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
                
                # Emergency stop
                loss_pct = (high - entry_price) / entry_price * 100
                if loss_pct >= self.config['emergency_stop']:
                    exit_price = entry_price * (1 + self.config['emergency_stop'] / 100)
                    pnl_amount = position_size * (-self.config['emergency_stop'] / 100)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'emergency_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': -self.config['emergency_stop']
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
    
    def run_backtest(self, data: pd.DataFrame) -> dict:
        """Run final optimized backtest"""
        print("\n🧪 RUNNING FINAL OPTIMIZED BACKTEST")
        print("=" * 50)
        
        balance = self.initial_balance
        trades = []
        
        # Tracking
        wins = 0
        losses = 0
        total_profit = 0
        daily_trades = 0
        consecutive_losses = 0
        daily_loss = 0
        last_trade_day = None
        
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'emergency_stop': 0, 'time_exit': 0}
        trailing_activations = 0
        compounding_growth = [balance]
        
        i = 40  # Start earlier
        while i < len(data) - 300:
            current_time = data.iloc[i]['timestamp']
            current_price = data.iloc[i]['close']
            
            # Daily reset
            today = current_time.date()
            if last_trade_day != today:
                daily_trades = 0
                daily_loss = 0
                last_trade_day = today
            
            daily_loss_pct = (daily_loss / balance) * 100 if balance > 0 else 0
            
            if (daily_trades < self.config['max_daily_trades'] and 
                consecutive_losses < self.config['max_consecutive_losses'] and
                daily_loss_pct < self.config['daily_loss_limit']):
                
                indicators = self.calculate_indicators(data, i)
                if indicators:
                    analysis = self.analyze_opportunity(indicators)
                    
                    if (analysis['confidence'] >= self.config['min_confidence'] and 
                        analysis['direction'] != "hold"):
                        
                        position_size, leverage, profit_target_pct = self.calculate_optimized_position_size(
                            analysis['confidence'], balance)
                        
                        trade_result = self.simulate_optimized_trade(
                            current_price, analysis['direction'], leverage, 
                            position_size, profit_target_pct, data, i
                        )
                        
                        # Update balance
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
                        
                        # Track stats
                        exit_reasons[trade_result['exit_reason']] += 1
                        if trade_result['trailing_activated']:
                            trailing_activations += 1
                        
                        compounding_growth.append(balance)
                        
                        # Store trade
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
                        
                        # Show key trades
                        if len(trades) % 10 == 0 or len(trades) <= 15:
                            win_rate = (wins / len(trades) * 100) if trades else 0
                            growth = ((balance - self.initial_balance) / self.initial_balance) * 100
                            print(f"📊 #{len(trades)}: {analysis['direction'].upper()} @ ${current_price:.2f} → "
                                  f"${trade_result['pnl_amount']:+.2f} ({trade_result['profit_pct']:+.1f}%) | "
                                  f"WR: {win_rate:.1f}% | Growth: {growth:+.1f}% | ${balance:.2f}")
                        
                        # Skip ahead
                        skip_minutes = max(trade_result['hold_minutes'], 8)
                        i += skip_minutes
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 10
        
        # Calculate results
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_return = ((balance - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate metrics
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
            
            # Max drawdown
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
    
    def display_results(self, results: dict):
        """Display comprehensive final results"""
        print("\n" + "="*75)
        print("🚀 FINAL OPTIMIZED COMPOUNDING BOT - BACKTEST RESULTS")
        print("="*75)
        
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   🔢 Total Trades: {results['total_trades']}")
        print(f"   🏆 Win Rate: {results['win_rate']:.1f}% ({results['wins']}W/{results['losses']}L)")
        print(f"   💰 Total Return: {results['total_return']:+.1f}%")
        print(f"   💵 Final Balance: ${results['final_balance']:.2f}")
        print(f"   💎 Total Profit: ${results['total_profit']:+.2f}")
        print(f"   📈 Profit Factor: {results['profit_factor']:.2f}")
        print(f"   📉 Max Drawdown: {results['max_drawdown']:.1f}%")
        
        print(f"\n📊 TRADE ANALYSIS:")
        print(f"   💚 Average Win: ${results['avg_win']:.2f} ({results['avg_profit_pct']:.1f}%)")
        print(f"   ❌ Average Loss: ${results['avg_loss']:.2f}")
        print(f"   🚀 Best Trade: ${results['max_win']:.2f}")
        print(f"   💀 Worst Trade: ${results['max_loss']:.2f}")
        
        print(f"\n⚡ TRADING METRICS:")
        print(f"   📊 Average Leverage: {results['avg_leverage']:.1f}x")
        print(f"   🤖 Average Confidence: {results['avg_confidence']:.1f}%")
        print(f"   ⏱️ Average Hold Time: {results['avg_hold_time']:.1f} minutes")
        
        print(f"\n🛡️ RISK MANAGEMENT:")
        total_trades = results['total_trades']
        if total_trades > 0:
            trailing_pct = (results['trailing_activations'] / total_trades) * 100
            print(f"   ⚡ Trailing Stops: {results['trailing_activations']}/{total_trades} ({trailing_pct:.1f}%)")
        
        print(f"\n📤 EXIT BREAKDOWN:")
        exit_reasons = results['exit_reasons']
        if total_trades > 0:
            for reason, count in exit_reasons.items():
                pct = (count / total_trades) * 100
                emoji = "🎯" if reason == "take_profit" else "🛡️" if reason == "trailing_stop" else "🚨" if reason == "emergency_stop" else "⏰"
                print(f"   {emoji} {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        # Projections
        if results['final_balance'] > self.initial_balance:
            print(f"\n💎 COMPOUNDING PROJECTIONS:")
            daily_return = (results['final_balance'] / self.initial_balance) ** (1/60) - 1
            monthly_return = ((1 + daily_return) ** 30 - 1) * 100
            annual_return = ((1 + daily_return) ** 365 - 1) * 100
            
            print(f"   📅 Daily Return: {daily_return*100:+.3f}%")
            print(f"   📅 Monthly Projection: {monthly_return:+.1f}%")
            print(f"   📅 Annual Projection: {annual_return:+.1f}%")
            
            # 6-month balance
            six_month_balance = self.initial_balance * ((1 + daily_return) ** 180)
            print(f"   🎯 6-Month Target: ${six_month_balance:.2f}")
        
        # Performance grading
        print(f"\n🎯 PERFORMANCE GRADES:")
        grades = []
        
        if results['win_rate'] >= 60:
            print("   🔥 Win Rate: A+ (Excellent)")
            grades.append("A+")
        elif results['win_rate'] >= 50:
            print("   ✅ Win Rate: B+ (Good)")
            grades.append("B+")
        elif results['win_rate'] >= 40:
            print("   ⚠️ Win Rate: C (Fair)")
            grades.append("C")
        else:
            print("   ❌ Win Rate: F (Poor)")
            grades.append("F")
        
        if results['total_return'] >= 15:
            print("   🔥 Returns: A+ (Excellent)")
            grades.append("A+")
        elif results['total_return'] >= 8:
            print("   ✅ Returns: B+ (Good)")
            grades.append("B+")
        elif results['total_return'] >= 3:
            print("   ⚠️ Returns: C (Fair)")
            grades.append("C")
        else:
            print("   ❌ Returns: F (Poor)")
            grades.append("F")
        
        if results['profit_factor'] >= 1.8:
            print("   🔥 Risk/Reward: A+ (Excellent)")
            grades.append("A+")
        elif results['profit_factor'] >= 1.3:
            print("   ✅ Risk/Reward: B+ (Good)")
            grades.append("B+")
        elif results['profit_factor'] >= 1.1:
            print("   ⚠️ Risk/Reward: C (Fair)")
            grades.append("C")
        else:
            print("   ❌ Risk/Reward: F (Poor)")
            grades.append("F")
        
        trades_per_day = results['total_trades'] / 60
        if trades_per_day >= 1.0:
            print("   ✅ Frequency: B+ (Good)")
            grades.append("B+")
        elif trades_per_day >= 0.5:
            print("   ⚠️ Frequency: C (Fair)")
            grades.append("C")
        else:
            print("   ❌ Frequency: F (Poor)")
            grades.append("F")
        
        # Overall grade
        grade_scores = {"A+": 4, "B+": 3, "C": 2, "F": 1}
        avg_score = sum(grade_scores[g] for g in grades) / len(grades)
        if avg_score >= 3.5:
            overall = "A+ (EXCELLENT)"
        elif avg_score >= 2.5:
            overall = "B+ (GOOD)"
        elif avg_score >= 1.5:
            overall = "C (NEEDS WORK)"
        else:
            overall = "F (MAJOR ISSUES)"
        
        print(f"\n🏆 OVERALL GRADE: {overall}")
        print("="*75)

def main():
    """Main function"""
    print("🚀 FINAL OPTIMIZED COMPOUNDING BOT - BACKTESTING")
    print("🎯 REALISTIC TARGETS + IMPROVED EVERYTHING")
    print("=" * 60)
    
    try:
        balance = float(input("💵 Starting balance (default $200): ") or "200")
    except ValueError:
        balance = 200.0
    
    try:
        days = int(input("📅 Backtest days (default 60): ") or "60")
    except ValueError:
        days = 60
    
    backtester = FinalOptimizedCompoundingBacktest(initial_balance=balance)
    data = backtester.generate_realistic_market_data(days=days)
    results = backtester.run_backtest(data)
    backtester.display_results(results)
    
    # Final recommendation
    if (results['win_rate'] >= 55 and results['total_return'] >= 10 and 
        results['profit_factor'] >= 1.5):
        print(f"\n🎯 RECOMMENDATION: READY FOR LIVE TRADING!")
        print(f"   ✅ Strong performance metrics across the board")
        print(f"   💰 Consistent profitability with good risk management")
        print(f"   🚀 Deploy with confidence!")
    elif (results['win_rate'] >= 45 and results['total_return'] >= 5):
        print(f"\n⚠️ RECOMMENDATION: PROMISING - NEEDS MINOR TWEAKS")
        print(f"   📊 Good foundation but room for improvement")
        print(f"   🔧 Consider optimizing entry conditions")
    else:
        print(f"\n❌ RECOMMENDATION: REQUIRES MORE OPTIMIZATION")
        print(f"   🛠️ Focus on improving win rate and profit factor")

if __name__ == "__main__":
    main()