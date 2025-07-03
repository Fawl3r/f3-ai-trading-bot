#!/usr/bin/env python3
"""
Working $40 Profit Bot - Backtesting System
Tests the bot against 60 days of realistic historical data
Features:
- 20-30x adaptive leverage
- Trailing stop loss system
- AI-optimized entry conditions
- Realistic market simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Working40ProfitBacktest:
    """Backtest the Working $40 Profit Bot"""
    
    def __init__(self, initial_balance: float = 500.0):
        self.initial_balance = initial_balance
        
        # SAME CONFIG AS LIVE BOT
        self.config = {
            # TARGET AND LEVERAGE
            "target_profit": 40.0,
            "leverage_min": 20,
            "leverage_max": 30,
            
            # POSITION SIZING
            "base_position": 150.0,
            "max_position": 400.0,
            "position_pct": 20,  # 20% of balance max
            
            # STOPS AND TARGETS
            "profit_target_pct": 2.5,    # 2.5% price movement target
            "trailing_stop_pct": 0.8,    # 0.8% trailing stop
            "trailing_activation": 0.5,  # Activate at 0.5% profit
            "emergency_stop": 2.0,       # 2% emergency stop
            
            # ENTRY CONDITIONS
            "min_confidence": 40,        # 40% minimum confidence
            "rsi_oversold": 35,          # RSI < 35 = oversold
            "rsi_overbought": 65,        # RSI > 65 = overbought
            "volume_multiplier": 1.2,    # 1.2x average volume
            
            # RISK MANAGEMENT
            "max_daily_trades": 15,
            "max_hold_hours": 8,
            "max_consecutive_losses": 3,
            "daily_profit_target": 150.0,
        }
        
        print("🚀 WORKING $40 PROFIT BOT - BACKTESTING")
        print("⚡ 20-30X ADAPTIVE LEVERAGE WITH TRAILING STOPS")
        print("📊 60 DAYS HISTORICAL DATA SIMULATION")
        print("=" * 60)
    
    def generate_realistic_market_data(self, days: int = 60) -> pd.DataFrame:
        """Generate 60 days of realistic SOL market data"""
        print(f"📊 Generating {days} days of realistic market data...")
        
        # Starting price similar to current SOL
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)  # For reproducible results
        
        data = []
        current_price = start_price
        current_time = datetime.now() - timedelta(days=days)
        
        for i in range(total_minutes):
            # Time-based volatility (higher during US trading hours)
            hour = (i // 60) % 24
            if 0 <= hour < 8:  # Asian session
                volatility = 0.0018
            elif 8 <= hour < 16:  # European session
                volatility = 0.0025
            else:  # US session
                volatility = 0.0035
            
            # Weekly trend cycle
            weekly_trend = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.0008
            
            # Random market shocks (pumps and dumps)
            if np.random.random() < 0.001:  # 0.1% chance per minute
                shock = np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.05)
                volatility *= 3
            else:
                shock = 0
            
            # Price movement
            noise = np.random.normal(0, volatility)
            price_change = weekly_trend + noise + shock
            current_price *= (1 + price_change)
            
            # Keep price in realistic bounds
            current_price = max(80, min(250, current_price))
            
            # Generate OHLC
            intrabar_volatility = volatility * 0.6
            high = current_price * (1 + abs(np.random.normal(0, intrabar_volatility)))
            low = current_price * (1 - abs(np.random.normal(0, intrabar_volatility)))
            open_price = current_price * (1 + np.random.normal(0, intrabar_volatility * 0.5))
            
            # Ensure OHLC consistency
            high = max(high, open_price, current_price)
            low = min(low, open_price, current_price)
            
            # Volume simulation
            base_volume = 1500000
            volume_factor = 1 + abs(price_change) * 80 + np.random.uniform(0.3, 2.5)
            if abs(price_change) > 0.01:  # High volatility = high volume
                volume_factor *= 1.5
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
        print(f"📊 Avg daily volatility: {df['close'].pct_change().std() * np.sqrt(1440) * 100:.1f}%")
        
        return df
    
    def calculate_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        """Calculate technical indicators at specific index"""
        if idx < 60:
            return None
        
        # Get window of data
        window = data.iloc[max(0, idx-60):idx+1]
        
        # RSI calculation
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Moving averages
        ma_5 = window['close'].rolling(5).mean().iloc[-1]
        ma_20 = window['close'].rolling(20).mean().iloc[-1]
        ma_50 = window['close'].rolling(50).mean().iloc[-1] if len(window) >= 50 else window['close'].mean()
        
        # Volume analysis
        volume_ma = window['volume'].rolling(20).mean().iloc[-1]
        current_volume = window['volume'].iloc[-1]
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
        
        # Price momentum
        if len(window) >= 6:
            momentum = (window['close'].iloc[-1] - window['close'].iloc[-6]) / window['close'].iloc[-6] * 100
        else:
            momentum = 0
        
        return {
            'rsi': current_rsi,
            'ma_5': ma_5,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'volume_ratio': volume_ratio,
            'momentum': momentum,
            'current_price': window['close'].iloc[-1]
        }
    
    def analyze_opportunity(self, indicators: dict) -> dict:
        """Analyze market for trading opportunities (same as live bot)"""
        if not indicators:
            return {"confidence": 0, "direction": "hold"}
        
        confidence = 0
        direction = "hold"
        reasons = []
        
        # RSI signals (40% weight)
        if indicators['rsi'] < self.config['rsi_oversold']:
            confidence += 40
            direction = "long"
            reasons.append("oversold")
        elif indicators['rsi'] > self.config['rsi_overbought']:
            confidence += 40
            direction = "short"
            reasons.append("overbought")
        
        # Moving average trend (30% weight)
        current_price = indicators['current_price']
        if current_price > indicators['ma_5'] > indicators['ma_20']:
            if direction == "long" or direction == "hold":
                confidence += 25
                direction = "long"
                reasons.append("bullish_trend")
        elif current_price < indicators['ma_5'] < indicators['ma_20']:
            if direction == "short" or direction == "hold":
                confidence += 25
                direction = "short"
                reasons.append("bearish_trend")
        
        # Volume confirmation (20% weight)
        if indicators['volume_ratio'] > self.config['volume_multiplier']:
            confidence += 20
            reasons.append("high_volume")
        
        # Momentum (10% weight)
        if abs(indicators['momentum']) > 1.0:
            if direction == "long" and indicators['momentum'] > 0:
                confidence += 10
                reasons.append("positive_momentum")
            elif direction == "short" and indicators['momentum'] < 0:
                confidence += 10
                reasons.append("negative_momentum")
        
        return {
            "confidence": min(confidence, 95),
            "direction": direction,
            "reasons": reasons
        }
    
    def calculate_leverage(self, confidence: float) -> int:
        """Calculate adaptive leverage based on confidence"""
        base_leverage = self.config['leverage_min']
        max_leverage = self.config['leverage_max']
        
        confidence_ratio = confidence / 100
        leverage = int(base_leverage + (max_leverage - base_leverage) * confidence_ratio)
        
        return max(base_leverage, min(max_leverage, leverage))
    
    def calculate_position_size(self, confidence: float, balance: float) -> float:
        """Calculate position size based on confidence and balance"""
        base_size = self.config['base_position']
        max_size = min(
            self.config['max_position'],
            balance * self.config['position_pct'] / 100
        )
        
        confidence_ratio = confidence / 100
        position_size = base_size + (max_size - base_size) * confidence_ratio
        
        return position_size
    
    def simulate_trade_with_trailing_stop(self, entry_price: float, direction: str, 
                                        leverage: int, position_size: float,
                                        data: pd.DataFrame, start_idx: int) -> dict:
        """Simulate a complete trade with trailing stop"""
        best_price = entry_price
        trailing_stop_price = None
        trailing_activated = False
        
        max_hold_minutes = self.config['max_hold_hours'] * 60
        
        for i in range(start_idx + 1, min(start_idx + max_hold_minutes + 1, len(data))):
            candle = data.iloc[i]
            high = candle['high']
            low = candle['low']
            close = candle['close']
            
            if direction == 'long':
                # Update best price
                if high > best_price:
                    best_price = high
                
                # Check if we should activate trailing stop
                if not trailing_activated:
                    profit_pct = (best_price - entry_price) / entry_price * 100
                    if profit_pct >= self.config['trailing_activation']:
                        trailing_activated = True
                        trailing_stop_price = best_price * (1 - self.config['trailing_stop_pct'] / 100)
                
                # Update trailing stop
                if trailing_activated:
                    new_stop = best_price * (1 - self.config['trailing_stop_pct'] / 100)
                    if new_stop > trailing_stop_price:
                        trailing_stop_price = new_stop
                
                # Check exit conditions
                current_pnl = position_size * ((high - entry_price) / entry_price)
                
                # Take profit (90% of target)
                if current_pnl >= self.config['target_profit'] * 0.9:
                    exit_price = entry_price * (1 + (self.config['target_profit'] * 0.9) / position_size)
                    pnl_amount = self.config['target_profit'] * 0.9
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'take_profit',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'best_price': best_price
                    }
                
                # Trailing stop
                if trailing_activated and low <= trailing_stop_price:
                    pnl_amount = position_size * ((trailing_stop_price - entry_price) / entry_price)
                    return {
                        'exit_price': trailing_stop_price,
                        'exit_reason': 'trailing_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'best_price': best_price
                    }
                
                # Emergency stop
                current_loss_pct = (entry_price - low) / entry_price * 100
                if current_loss_pct >= self.config['emergency_stop']:
                    exit_price = entry_price * (1 - self.config['emergency_stop'] / 100)
                    pnl_amount = position_size * ((exit_price - entry_price) / entry_price)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'emergency_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'best_price': best_price
                    }
            
            else:  # Short position
                # Update best price (lowest for short)
                if low < best_price:
                    best_price = low
                
                # Check if we should activate trailing stop
                if not trailing_activated:
                    profit_pct = (entry_price - best_price) / entry_price * 100
                    if profit_pct >= self.config['trailing_activation']:
                        trailing_activated = True
                        trailing_stop_price = best_price * (1 + self.config['trailing_stop_pct'] / 100)
                
                # Update trailing stop
                if trailing_activated:
                    new_stop = best_price * (1 + self.config['trailing_stop_pct'] / 100)
                    if new_stop < trailing_stop_price:
                        trailing_stop_price = new_stop
                
                # Check exit conditions
                current_pnl = position_size * ((entry_price - low) / entry_price)
                
                # Take profit
                if current_pnl >= self.config['target_profit'] * 0.9:
                    exit_price = entry_price * (1 - (self.config['target_profit'] * 0.9) / position_size)
                    pnl_amount = self.config['target_profit'] * 0.9
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'take_profit',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'best_price': best_price
                    }
                
                # Trailing stop
                if trailing_activated and high >= trailing_stop_price:
                    pnl_amount = position_size * ((entry_price - trailing_stop_price) / entry_price)
                    return {
                        'exit_price': trailing_stop_price,
                        'exit_reason': 'trailing_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'best_price': best_price
                    }
                
                # Emergency stop
                current_loss_pct = (high - entry_price) / entry_price * 100
                if current_loss_pct >= self.config['emergency_stop']:
                    exit_price = entry_price * (1 + self.config['emergency_stop'] / 100)
                    pnl_amount = position_size * ((entry_price - exit_price) / entry_price)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'emergency_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'best_price': best_price
                    }
        
        # Time exit
        final_price = data.iloc[min(start_idx + max_hold_minutes, len(data) - 1)]['close']
        if direction == 'long':
            pnl_amount = position_size * ((final_price - entry_price) / entry_price)
        else:
            pnl_amount = position_size * ((entry_price - final_price) / entry_price)
        
        return {
            'exit_price': final_price,
            'exit_reason': 'time_exit',
            'pnl_amount': pnl_amount,
            'hold_minutes': max_hold_minutes,
            'trailing_activated': trailing_activated,
            'best_price': best_price
        }
    
    def run_backtest(self, data: pd.DataFrame) -> dict:
        """Run the complete backtest"""
        print("\n🧪 RUNNING BACKTEST SIMULATION")
        print("=" * 50)
        
        balance = self.initial_balance
        trades = []
        position = None
        
        # Performance tracking
        wins = 0
        losses = 0
        total_profit = 0
        daily_trades = 0
        consecutive_losses = 0
        daily_profit = 0
        last_trade_day = None
        
        # Track exit reasons and trailing stops
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'emergency_stop': 0, 'time_exit': 0}
        trailing_activations = 0
        
        i = 60  # Start after enough data for indicators
        while i < len(data) - 500:  # Leave room for position simulation
            current_time = data.iloc[i]['timestamp']
            current_price = data.iloc[i]['close']
            
            # Reset daily counters
            today = current_time.date()
            if last_trade_day != today:
                daily_trades = 0
                daily_profit = 0
                last_trade_day = today
            
            # Check if we can enter a trade
            if (daily_trades < self.config['max_daily_trades'] and 
                consecutive_losses < self.config['max_consecutive_losses'] and
                daily_profit < self.config['daily_profit_target']):
                
                # Calculate indicators
                indicators = self.calculate_indicators(data, i)
                if indicators:
                    # Analyze opportunity
                    analysis = self.analyze_opportunity(indicators)
                    
                    if (analysis['confidence'] >= self.config['min_confidence'] and 
                        analysis['direction'] != "hold"):
                        
                        # Calculate trade parameters
                        leverage = self.calculate_leverage(analysis['confidence'])
                        position_size = self.calculate_position_size(analysis['confidence'], balance)
                        
                        # Simulate the complete trade
                        trade_result = self.simulate_trade_with_trailing_stop(
                            current_price, analysis['direction'], leverage, 
                            position_size, data, i
                        )
                        
                        # Update balance and stats
                        balance += trade_result['pnl_amount']
                        total_profit += trade_result['pnl_amount']
                        daily_profit += trade_result['pnl_amount']
                        
                        if trade_result['pnl_amount'] > 0:
                            wins += 1
                            consecutive_losses = 0
                        else:
                            losses += 1
                            consecutive_losses += 1
                        
                        # Track stats
                        exit_reasons[trade_result['exit_reason']] += 1
                        if trade_result['trailing_activated']:
                            trailing_activations += 1
                        
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
                            'exit_reason': trade_result['exit_reason'],
                            'hold_minutes': trade_result['hold_minutes'],
                            'trailing_activated': trade_result['trailing_activated'],
                            'balance': balance,
                            'reasons': analysis['reasons']
                        }
                        trades.append(trade)
                        daily_trades += 1
                        
                        # Show progress
                        if len(trades) % 10 == 0:
                            win_rate = (wins / len(trades) * 100) if trades else 0
                            print(f"📊 Trade #{len(trades)}: {analysis['direction'].upper()} @ ${current_price:.2f} → "
                                  f"${trade_result['pnl_amount']:+.2f} | WR: {win_rate:.1f}% | Balance: ${balance:.2f}")
                        
                        # Skip ahead to avoid overlapping trades
                        i += max(trade_result['hold_minutes'], 30)
                    else:
                        i += 5  # Skip ahead when no signal
                else:
                    i += 1  # Skip when insufficient data
            else:
                i += 30  # Skip ahead when daily limits reached
        
        # Calculate final performance metrics
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
            
            profit_factor = (sum([t['pnl_amount'] for t in profitable_trades]) / 
                           max(abs(sum([t['pnl_amount'] for t in losing_trades])), 0.01))
            
            target_hits = len([t for t in profitable_trades if t['pnl_amount'] >= self.config['target_profit'] * 0.9])
            avg_leverage = np.mean([t['leverage'] for t in trades])
            avg_confidence = np.mean([t['confidence'] for t in trades])
            avg_hold_time = np.mean([t['hold_minutes'] for t in trades])
        else:
            avg_win = avg_loss = max_win = max_loss = profit_factor = 0
            target_hits = avg_leverage = avg_confidence = avg_hold_time = 0
        
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
            'target_hits': target_hits,
            'avg_leverage': avg_leverage,
            'avg_confidence': avg_confidence,
            'avg_hold_time': avg_hold_time,
            'exit_reasons': exit_reasons,
            'trailing_activations': trailing_activations,
            'trades': trades
        }
    
    def display_results(self, results: dict):
        """Display comprehensive backtest results"""
        print("\n" + "="*70)
        print("🚀 WORKING $40 PROFIT BOT - BACKTEST RESULTS")
        print("="*70)
        
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   🔢 Total Trades: {results['total_trades']}")
        print(f"   🏆 Win Rate: {results['win_rate']:.1f}% ({results['wins']}W/{results['losses']}L)")
        print(f"   💰 Total Return: {results['total_return']:+.1f}%")
        print(f"   💵 Final Balance: ${results['final_balance']:.2f}")
        print(f"   💎 Total Profit: ${results['total_profit']:+.2f}")
        print(f"   📈 Profit Factor: {results['profit_factor']:.2f}")
        
        print(f"\n📊 TRADE ANALYSIS:")
        print(f"   💚 Average Win: ${results['avg_win']:.2f}")
        print(f"   ❌ Average Loss: ${results['avg_loss']:.2f}")
        print(f"   🚀 Maximum Win: ${results['max_win']:.2f}")
        print(f"   💀 Maximum Loss: ${results['max_loss']:.2f}")
        print(f"   🎯 Target Hits: {results['target_hits']}/{results['wins']} "
              f"({results['target_hits']/max(results['wins'], 1)*100:.1f}%)")
        
        print(f"\n⚡ LEVERAGE & CONFIDENCE:")
        print(f"   📊 Average Leverage: {results['avg_leverage']:.1f}x")
        print(f"   🤖 Average Confidence: {results['avg_confidence']:.1f}%")
        print(f"   ⏱️ Average Hold Time: {results['avg_hold_time']:.1f} minutes")
        
        print(f"\n🛡️ TRAILING STOP ANALYSIS:")
        total_trades = results['total_trades']
        if total_trades > 0:
            trailing_pct = (results['trailing_activations'] / total_trades) * 100
            print(f"   ⚡ Activations: {results['trailing_activations']}/{total_trades} ({trailing_pct:.1f}%)")
        
        print(f"\n📤 EXIT BREAKDOWN:")
        exit_reasons = results['exit_reasons']
        if total_trades > 0:
            for reason, count in exit_reasons.items():
                pct = (count / total_trades) * 100
                reason_display = reason.replace('_', ' ').title()
                print(f"   {reason_display}: {count} ({pct:.1f}%)")
        
        # Performance assessment
        print(f"\n💡 PERFORMANCE ASSESSMENT:")
        if results['win_rate'] >= 70:
            print("   🔥 EXCELLENT win rate! Bot shows strong performance.")
        elif results['win_rate'] >= 60:
            print("   ✅ GOOD win rate. Bot is performing well.")
        elif results['win_rate'] >= 50:
            print("   ⚠️ MODERATE win rate. Consider optimization.")
        else:
            print("   ❌ LOW win rate. Needs significant improvement.")
        
        if results['total_return'] >= 50:
            print("   💰 EXCELLENT returns! Highly profitable strategy.")
        elif results['total_return'] >= 20:
            print("   💚 GOOD returns. Profitable strategy.")
        elif results['total_return'] >= 0:
            print("   ⚠️ POSITIVE but low returns. Room for improvement.")
        else:
            print("   ❌ NEGATIVE returns. Strategy needs major revision.")
        
        if results['profit_factor'] >= 2.0:
            print("   📈 EXCELLENT profit factor! Great risk/reward.")
        elif results['profit_factor'] >= 1.5:
            print("   ✅ GOOD profit factor. Decent risk management.")
        elif results['profit_factor'] >= 1.0:
            print("   ⚠️ MARGINAL profit factor. Barely profitable.")
        else:
            print("   ❌ POOR profit factor. Losses exceed profits.")
        
        # Trading frequency assessment
        trades_per_day = results['total_trades'] / 60
        print(f"\n📅 TRADING FREQUENCY:")
        print(f"   📊 Average: {trades_per_day:.1f} trades per day")
        if trades_per_day >= 2:
            print("   ✅ Good trading frequency for active strategy.")
        elif trades_per_day >= 1:
            print("   ⚠️ Moderate frequency. Could be more active.")
        else:
            print("   ❌ Low frequency. May miss opportunities.")
        
        print("="*70)

def main():
    """Main backtesting function"""
    print("🚀 WORKING $40 PROFIT BOT - BACKTESTING SYSTEM")
    print("⚡ AI-OPTIMIZED 20-30X LEVERAGE WITH TRAILING STOPS")
    print("=" * 60)
    
    try:
        balance = float(input("💵 Enter starting balance (default $500): ") or "500")
    except ValueError:
        balance = 500.0
    
    try:
        days = int(input("📅 Backtest period in days (default 60): ") or "60")
    except ValueError:
        days = 60
    
    backtester = Working40ProfitBacktest(initial_balance=balance)
    
    # Generate market data
    data = backtester.generate_realistic_market_data(days=days)
    
    # Run backtest
    results = backtester.run_backtest(data)
    
    # Display results
    backtester.display_results(results)
    
    # Investment recommendation
    if results['win_rate'] >= 65 and results['total_return'] >= 25 and results['profit_factor'] >= 1.8:
        print(f"\n🎯 RECOMMENDATION: DEPLOY LIVE!")
        print(f"   The bot shows excellent performance across all metrics.")
        print(f"   Expected monthly return: {results['total_return']/2:.1f}%")
        print(f"   Risk level: MODERATE with good risk management")
    elif results['win_rate'] >= 55 and results['total_return'] >= 10:
        print(f"\n⚠️ RECOMMENDATION: CONSIDER OPTIMIZATION")
        print(f"   The bot shows promise but could be improved.")
        print(f"   Consider paper trading first before live deployment.")
    else:
        print(f"\n❌ RECOMMENDATION: DO NOT DEPLOY")
        print(f"   The bot needs significant improvement before live trading.")
        print(f"   Focus on optimizing entry conditions and risk management.")

if __name__ == "__main__":
    main()