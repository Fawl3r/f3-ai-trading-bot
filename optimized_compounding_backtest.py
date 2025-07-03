#!/usr/bin/env python3
"""
Optimized Compounding Bot - Backtesting System
Focus: Consistent profitability and compounding, not specific profit targets
Features:
- Dynamic position sizing based on balance growth
- Tighter risk management for consistency
- Higher confidence thresholds for quality trades
- Compound growth optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OptimizedCompoundingBacktest:
    """Backtest the Optimized Compounding Bot"""
    
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # OPTIMIZED CONFIG FOR COMPOUNDING
        self.config = {
            # LEVERAGE - More conservative for consistency
            "leverage_min": 15,
            "leverage_max": 25,
            
            # POSITION SIZING - Dynamic based on balance
            "position_pct_min": 15,  # 15% minimum of balance
            "position_pct_max": 25,  # 25% maximum of balance
            "min_position": 50.0,    # Minimum position size
            
            # PROFIT TARGETS - Flexible, not fixed
            "profit_target_min": 1.5,   # 1.5% minimum profit target
            "profit_target_max": 3.0,   # 3.0% maximum profit target
            "trailing_stop_pct": 0.6,   # 0.6% trailing stop (tighter)
            "trailing_activation": 0.3, # Activate at 0.3% profit (earlier)
            "emergency_stop": 1.5,      # 1.5% emergency stop (tighter)
            
            # ENTRY CONDITIONS - Higher quality
            "min_confidence": 55,       # 55% minimum confidence (higher)
            "rsi_oversold": 30,         # RSI < 30 = strong oversold
            "rsi_overbought": 70,       # RSI > 70 = strong overbought
            "volume_multiplier": 1.5,   # 1.5x average volume (higher)
            
            # RISK MANAGEMENT - More conservative
            "max_daily_trades": 8,      # Fewer, higher quality trades
            "max_hold_hours": 6,        # Shorter hold time
            "max_consecutive_losses": 2, # Stop after 2 losses
            "daily_loss_limit": 3.0,    # Stop if daily loss > 3%
        }
        
        print("🚀 OPTIMIZED COMPOUNDING BOT - BACKTESTING")
        print("💎 FOCUS: CONSISTENT PROFITS & COMPOUNDING")
        print("📊 DYNAMIC POSITION SIZING & TIGHT RISK MANAGEMENT")
        print("=" * 60)
    
    def generate_realistic_market_data(self, days: int = 60) -> pd.DataFrame:
        """Generate 60 days of realistic SOL market data"""
        print(f"📊 Generating {days} days of realistic market data...")
        
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        current_price = start_price
        current_time = datetime.now() - timedelta(days=days)
        
        for i in range(total_minutes):
            # Time-based volatility
            hour = (i // 60) % 24
            if 0 <= hour < 8:
                volatility = 0.0015
            elif 8 <= hour < 16:
                volatility = 0.0022
            else:
                volatility = 0.0030
            
            # Market cycles
            weekly_trend = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.0006
            daily_cycle = np.sin(i / (24 * 60) * 2 * np.pi) * 0.0003
            
            # Random shocks (less frequent for more realistic trading)
            if np.random.random() < 0.0008:
                shock = np.random.choice([-1, 1]) * np.random.uniform(0.015, 0.04)
                volatility *= 2.5
            else:
                shock = 0
            
            # Price movement
            noise = np.random.normal(0, volatility)
            price_change = weekly_trend + daily_cycle + noise + shock
            current_price *= (1 + price_change)
            current_price = max(90, min(220, current_price))
            
            # Generate OHLC with better consistency
            intrabar_vol = volatility * 0.5
            high = current_price * (1 + abs(np.random.normal(0, intrabar_vol)))
            low = current_price * (1 - abs(np.random.normal(0, intrabar_vol)))
            open_price = current_price * (1 + np.random.normal(0, intrabar_vol * 0.3))
            
            high = max(high, open_price, current_price)
            low = min(low, open_price, current_price)
            
            # Volume with better correlation to price movement
            base_volume = 1200000
            volume_factor = 1 + abs(price_change) * 60 + np.random.uniform(0.4, 1.8)
            if abs(price_change) > 0.008:
                volume_factor *= 1.3
            volume = base_volume * volume_factor
            
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
        """Enhanced indicator calculations"""
        if idx < 60:
            return None
        
        window = data.iloc[max(0, idx-60):idx+1]
        
        # RSI with better smoothing
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Multiple moving averages for better trend detection
        ma_5 = window['close'].rolling(5).mean().iloc[-1]
        ma_10 = window['close'].rolling(10).mean().iloc[-1]
        ma_20 = window['close'].rolling(20).mean().iloc[-1]
        ma_50 = window['close'].rolling(50).mean().iloc[-1] if len(window) >= 50 else window['close'].mean()
        
        # Volume analysis with trend
        volume_ma = window['volume'].rolling(20).mean().iloc[-1]
        volume_trend = window['volume'].rolling(5).mean().iloc[-1] / volume_ma if volume_ma > 0 else 1
        
        # Price momentum and volatility
        if len(window) >= 10:
            momentum_short = (window['close'].iloc[-1] - window['close'].iloc[-5]) / window['close'].iloc[-5] * 100
            momentum_long = (window['close'].iloc[-1] - window['close'].iloc[-10]) / window['close'].iloc[-10] * 100
        else:
            momentum_short = momentum_long = 0
        
        # Volatility measure
        volatility = window['close'].pct_change().rolling(20).std().iloc[-1] * 100
        
        return {
            'rsi': current_rsi,
            'ma_5': ma_5,
            'ma_10': ma_10,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'volume_trend': volume_trend,
            'momentum_short': momentum_short,
            'momentum_long': momentum_long,
            'volatility': volatility,
            'current_price': window['close'].iloc[-1]
        }
    
    def analyze_opportunity(self, indicators: dict) -> dict:
        """Enhanced opportunity analysis for higher quality trades"""
        if not indicators:
            return {"confidence": 0, "direction": "hold"}
        
        confidence = 0
        direction = "hold"
        reasons = []
        
        current_price = indicators['current_price']
        rsi = indicators['rsi']
        
        # RSI signals with stronger thresholds (50% weight)
        if rsi < self.config['rsi_oversold']:
            confidence += 50
            direction = "long"
            reasons.append("strong_oversold")
        elif rsi > self.config['rsi_overbought']:
            confidence += 50
            direction = "short"
            reasons.append("strong_overbought")
        elif rsi < 40 and direction == "hold":
            confidence += 25
            direction = "long"
            reasons.append("oversold")
        elif rsi > 60 and direction == "hold":
            confidence += 25
            direction = "short"
            reasons.append("overbought")
        
        # Multi-timeframe trend analysis (30% weight)
        if (current_price > indicators['ma_5'] > indicators['ma_10'] > indicators['ma_20']):
            if direction == "long" or direction == "hold":
                confidence += 30
                direction = "long"
                reasons.append("strong_uptrend")
        elif (current_price < indicators['ma_5'] < indicators['ma_10'] < indicators['ma_20']):
            if direction == "short" or direction == "hold":
                confidence += 30
                direction = "short"
                reasons.append("strong_downtrend")
        elif current_price > indicators['ma_10']:
            if direction == "long":
                confidence += 15
                reasons.append("uptrend")
        elif current_price < indicators['ma_10']:
            if direction == "short":
                confidence += 15
                reasons.append("downtrend")
        
        # Volume confirmation (15% weight)
        if indicators['volume_trend'] > self.config['volume_multiplier']:
            confidence += 15
            reasons.append("high_volume")
        elif indicators['volume_trend'] > 1.2:
            confidence += 8
            reasons.append("good_volume")
        
        # Momentum alignment (5% weight)
        if direction == "long" and indicators['momentum_short'] > 0.5:
            confidence += 5
            reasons.append("positive_momentum")
        elif direction == "short" and indicators['momentum_short'] < -0.5:
            confidence += 5
            reasons.append("negative_momentum")
        
        # Volatility filter - reduce confidence in high volatility
        if indicators['volatility'] > 3.0:
            confidence *= 0.8
            reasons.append("high_volatility_discount")
        
        return {
            "confidence": min(confidence, 95),
            "direction": direction,
            "reasons": reasons
        }
    
    def calculate_dynamic_position_size(self, confidence: float, balance: float) -> tuple:
        """Calculate dynamic position size and leverage based on balance and confidence"""
        # Position size scales with confidence and balance
        confidence_ratio = confidence / 100
        
        # Base percentage of balance to risk
        position_pct = (self.config['position_pct_min'] + 
                       (self.config['position_pct_max'] - self.config['position_pct_min']) * confidence_ratio)
        
        position_size = max(self.config['min_position'], balance * position_pct / 100)
        
        # Dynamic leverage based on confidence
        leverage = int(self.config['leverage_min'] + 
                      (self.config['leverage_max'] - self.config['leverage_min']) * confidence_ratio)
        
        # Dynamic profit target based on confidence
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
                
                # Calculate current profit
                current_profit_pct = (high - entry_price) / entry_price * 100
                
                # Activate trailing stop earlier
                if not trailing_activated and current_profit_pct >= self.config['trailing_activation']:
                    trailing_activated = True
                    trailing_stop_price = best_price * (1 - self.config['trailing_stop_pct'] / 100)
                
                # Update trailing stop
                if trailing_activated:
                    new_stop = best_price * (1 - self.config['trailing_stop_pct'] / 100)
                    if new_stop > trailing_stop_price:
                        trailing_stop_price = new_stop
                
                # Take profit at dynamic target
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
                
                # Trailing stop hit
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
                
                # Emergency stop
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
            
            else:  # Short position - similar logic but inverted
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
        """Run optimized backtest focused on compounding"""
        print("\n🧪 RUNNING OPTIMIZED COMPOUNDING BACKTEST")
        print("=" * 55)
        
        balance = self.initial_balance
        trades = []
        
        # Performance tracking
        wins = 0
        losses = 0
        total_profit = 0
        daily_trades = 0
        consecutive_losses = 0
        daily_loss = 0
        last_trade_day = None
        
        # Enhanced tracking
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'emergency_stop': 0, 'time_exit': 0}
        trailing_activations = 0
        compounding_growth = [balance]
        
        i = 60
        while i < len(data) - 500:
            current_time = data.iloc[i]['timestamp']
            current_price = data.iloc[i]['close']
            
            # Reset daily counters
            today = current_time.date()
            if last_trade_day != today:
                daily_trades = 0
                daily_loss = 0
                last_trade_day = today
            
            # Enhanced trade conditions
            daily_loss_pct = (daily_loss / balance) * 100 if balance > 0 else 0
            
            if (daily_trades < self.config['max_daily_trades'] and 
                consecutive_losses < self.config['max_consecutive_losses'] and
                daily_loss_pct < self.config['daily_loss_limit']):
                
                indicators = self.calculate_indicators(data, i)
                if indicators:
                    analysis = self.analyze_opportunity(indicators)
                    
                    if (analysis['confidence'] >= self.config['min_confidence'] and 
                        analysis['direction'] != "hold"):
                        
                        # Calculate dynamic parameters
                        position_size, leverage, profit_target_pct = self.calculate_dynamic_position_size(
                            analysis['confidence'], balance)
                        
                        # Simulate the trade
                        trade_result = self.simulate_optimized_trade(
                            current_price, analysis['direction'], leverage, 
                            position_size, profit_target_pct, data, i
                        )
                        
                        # Update balance and compound
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
                        
                        # Track performance
                        exit_reasons[trade_result['exit_reason']] += 1
                        if trade_result['trailing_activated']:
                            trailing_activations += 1
                        
                        compounding_growth.append(balance)
                        
                        # Store trade with enhanced data
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
                            'balance_growth': (balance - old_balance) / old_balance * 100,
                            'reasons': analysis['reasons']
                        }
                        trades.append(trade)
                        daily_trades += 1
                        
                        # Progress updates
                        if len(trades) % 5 == 0:
                            win_rate = (wins / len(trades) * 100) if trades else 0
                            growth = ((balance - self.initial_balance) / self.initial_balance) * 100
                            print(f"📊 Trade #{len(trades)}: {analysis['direction'].upper()} @ ${current_price:.2f} → "
                                  f"${trade_result['pnl_amount']:+.2f} ({trade_result['profit_pct']:+.1f}%) | "
                                  f"WR: {win_rate:.1f}% | Growth: {growth:+.1f}% | Balance: ${balance:.2f}")
                        
                        # Skip ahead based on result
                        skip_minutes = max(trade_result['hold_minutes'], 15)
                        i += skip_minutes
                    else:
                        i += 3
                else:
                    i += 1
            else:
                i += 20
        
        # Calculate comprehensive performance metrics
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
            
            # Compounding metrics
            max_balance = max(compounding_growth)
            max_drawdown = ((max_balance - min(compounding_growth[compounding_growth.index(max_balance):])) / max_balance) * 100
            
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
        """Display comprehensive results focused on compounding performance"""
        print("\n" + "="*70)
        print("🚀 OPTIMIZED COMPOUNDING BOT - BACKTEST RESULTS")
        print("="*70)
        
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
        print(f"   🚀 Maximum Win: ${results['max_win']:.2f}")
        print(f"   💀 Maximum Loss: ${results['max_loss']:.2f}")
        
        print(f"\n⚡ TRADING METRICS:")
        print(f"   📊 Average Leverage: {results['avg_leverage']:.1f}x")
        print(f"   🤖 Average Confidence: {results['avg_confidence']:.1f}%")
        print(f"   ⏱️ Average Hold Time: {results['avg_hold_time']:.1f} minutes")
        
        print(f"\n🛡️ RISK MANAGEMENT:")
        total_trades = results['total_trades']
        if total_trades > 0:
            trailing_pct = (results['trailing_activations'] / total_trades) * 100
            print(f"   ⚡ Trailing Stop Activations: {results['trailing_activations']}/{total_trades} ({trailing_pct:.1f}%)")
        
        print(f"\n📤 EXIT BREAKDOWN:")
        exit_reasons = results['exit_reasons']
        if total_trades > 0:
            for reason, count in exit_reasons.items():
                pct = (count / total_trades) * 100
                reason_display = reason.replace('_', ' ').title()
                print(f"   {reason_display}: {count} ({pct:.1f}%)")
        
        # Compounding analysis
        growth = results['compounding_growth']
        if len(growth) > 1:
            print(f"\n💎 COMPOUNDING ANALYSIS:")
            max_balance = max(growth)
            print(f"   🎯 Peak Balance: ${max_balance:.2f}")
            
            # Monthly projection
            daily_return = (results['final_balance'] / self.initial_balance) ** (1/60) - 1
            monthly_return = ((1 + daily_return) ** 30 - 1) * 100
            annual_return = ((1 + daily_return) ** 365 - 1) * 100
            
            print(f"   📅 Projected Monthly Return: {monthly_return:+.1f}%")
            print(f"   📅 Projected Annual Return: {annual_return:+.1f}%")
        
        # Performance assessment
        print(f"\n💡 PERFORMANCE ASSESSMENT:")
        if results['win_rate'] >= 70:
            print("   🔥 EXCELLENT win rate! Strong consistency.")
        elif results['win_rate'] >= 60:
            print("   ✅ GOOD win rate. Solid performance.")
        elif results['win_rate'] >= 50:
            print("   ⚠️ MODERATE win rate. Room for improvement.")
        else:
            print("   ❌ LOW win rate. Needs optimization.")
        
        if results['total_return'] >= 30:
            print("   💰 EXCELLENT returns! Highly profitable.")
        elif results['total_return'] >= 15:
            print("   💚 GOOD returns. Profitable strategy.")
        elif results['total_return'] >= 5:
            print("   ⚠️ MODEST returns. Consider optimization.")
        else:
            print("   ❌ POOR returns. Needs improvement.")
        
        # Trading frequency
        trades_per_day = results['total_trades'] / 60
        print(f"\n📅 TRADING FREQUENCY:")
        print(f"   📊 Average: {trades_per_day:.1f} trades per day")
        
        print("="*70)

def main():
    """Main optimized backtesting function"""
    print("🚀 OPTIMIZED COMPOUNDING BOT - BACKTESTING SYSTEM")
    print("💎 FOCUS: CONSISTENT PROFITS & COMPOUND GROWTH")
    print("=" * 60)
    
    try:
        balance = float(input("💵 Enter starting balance (default $200): ") or "200")
    except ValueError:
        balance = 200.0
    
    try:
        days = int(input("📅 Backtest period in days (default 60): ") or "60")
    except ValueError:
        days = 60
    
    backtester = OptimizedCompoundingBacktest(initial_balance=balance)
    
    # Generate market data
    data = backtester.generate_realistic_market_data(days=days)
    
    # Run backtest
    results = backtester.run_backtest(data)
    
    # Display results
    backtester.display_results(results)
    
    # Investment recommendation
    if (results['win_rate'] >= 65 and results['total_return'] >= 20 and 
        results['profit_factor'] >= 1.8 and results['max_drawdown'] <= 15):
        print(f"\n🎯 RECOMMENDATION: EXCELLENT FOR LIVE DEPLOYMENT!")
        print(f"   Strong win rate, good returns, manageable drawdown.")
        print(f"   Expected compound growth: Highly promising.")
    elif (results['win_rate'] >= 55 and results['total_return'] >= 10 and 
          results['profit_factor'] >= 1.3):
        print(f"\n⚠️ RECOMMENDATION: GOOD POTENTIAL - CONSIDER LIVE TESTING")
        print(f"   Decent performance metrics. Start with smaller position.")
    else:
        print(f"\n❌ RECOMMENDATION: NEEDS OPTIMIZATION")
        print(f"   Performance below acceptable thresholds for live trading.")

if __name__ == "__main__":
    main()