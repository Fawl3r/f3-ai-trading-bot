#!/usr/bin/env python3
"""
MATHEMATICAL GRADE A BOT
Pure Mathematical Approach for Grade A Performance

Based on proven statistical patterns:
- Mean reversion trades with statistical backing
- Simple but effective entry/exit logic
- Mathematical position sizing
- Optimized for: 60%+ WR, 15%+ Returns, 1.8+ PF, 1.0+ Trades/Day
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MathematicalGradeABot:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # MATHEMATICAL CONFIGURATION
        self.config = {
            # Simple but effective parameters
            "position_base": 0.08,      # 8% base position
            "position_max": 0.15,       # 15% max position
            "leverage": 12,             # Conservative 12x leverage
            
            # Mathematical targets
            "profit_target": 0.018,     # 1.8% profit target
            "stop_loss": 0.012,         # 1.2% stop loss (1.5:1 R:R)
            "trail_distance": 0.004,    # 0.4% trailing distance
            "trail_start": 0.008,       # Start trail at 0.8%
            
            # Entry thresholds (mathematically optimized)
            "rsi_low": 25,              # Strong oversold
            "rsi_high": 75,             # Strong overbought
            "volume_min": 1.2,          # 1.2x average volume
            "price_dev": 0.015,         # 1.5% deviation from MA
            
            # Trade management
            "max_daily": 8,             # Max 8 trades per day
            "max_hold": 480,            # Max 8 hours hold
            "cooldown": 30,             # 30 min cooldown
        }
        
        print("🧮 MATHEMATICAL GRADE A BOT")
        print("📊 PURE STATISTICAL APPROACH")
        print("🎯 TARGET: OVERALL GRADE A")
        print("=" * 50)
    
    def generate_market_data(self, days: int = 60) -> pd.DataFrame:
        """Generate realistic market data with mathematical properties"""
        print(f"📊 Generating {days} days of market data...")
        
        # Start parameters
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        price = start_price
        time = datetime.now() - timedelta(days=days)
        
        for i in range(total_minutes):
            # Time-based volatility
            hour = (i // 60) % 24
            if 8 <= hour <= 16:
                vol = 0.0025  # Market hours
            elif 17 <= hour <= 23:
                vol = 0.002   # Evening
            else:
                vol = 0.0015  # Night
            
            # Price movement with trends
            trend = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.0008  # Weekly cycle
            noise = np.random.normal(0, vol)
            price_change = trend + noise
            
            # Apply change
            price *= (1 + price_change)
            price = max(110, min(180, price))  # Boundaries
            
            # OHLC
            spread = vol * 0.8
            high = price * (1 + abs(np.random.normal(0, spread)))
            low = price * (1 - abs(np.random.normal(0, spread)))
            open_p = price * (1 + np.random.normal(0, spread * 0.5))
            
            # Ensure OHLC validity
            high = max(high, price, open_p)
            low = min(low, price, open_p)
            
            # Volume
            base_vol = 800000
            vol_mult = 1 + abs(price_change) * 100 + np.random.uniform(0.5, 2.0)
            volume = base_vol * vol_mult
            
            data.append({
                'timestamp': time + timedelta(minutes=i),
                'open': open_p,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} candles")
        print(f"📈 Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        return df
    
    def calculate_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        """Calculate mathematical indicators"""
        if idx < 20:
            return None
        
        # Get window
        window = data.iloc[max(0, idx-20):idx+1]
        current = window.iloc[-1]
        
        # RSI calculation
        delta = window['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        # Moving averages
        ma_5 = window['close'].rolling(5).mean().iloc[-1]
        ma_20 = window['close'].rolling(20).mean().iloc[-1]
        
        # Volume
        vol_avg = window['volume'].rolling(10).mean().iloc[-1]
        vol_ratio = current['volume'] / vol_avg if vol_avg > 0 else 1
        
        # Price deviation from MA
        price_dev = (current['close'] - ma_20) / ma_20 if ma_20 > 0 else 0
        
        # Volatility
        volatility = window['close'].pct_change().rolling(10).std().iloc[-1]
        
        return {
            'rsi': current_rsi,
            'ma_5': ma_5,
            'ma_20': ma_20,
            'price': current['close'],
            'vol_ratio': vol_ratio,
            'price_dev': price_dev,
            'volatility': volatility or 0
        }
    
    def check_entry(self, indicators: dict) -> dict:
        """Mathematical entry logic"""
        if not indicators:
            return {'signal': False, 'direction': None, 'confidence': 0}
        
        rsi = indicators['rsi']
        price = indicators['price']
        ma_20 = indicators['ma_20']
        vol_ratio = indicators['vol_ratio']
        price_dev = indicators['price_dev']
        
        # Entry conditions
        direction = None
        confidence = 0
        
        # Long conditions
        if (rsi <= self.config['rsi_low'] and 
            price_dev <= -self.config['price_dev'] and
            vol_ratio >= self.config['volume_min']):
            direction = 'long'
            confidence = 70 + min(20, (self.config['rsi_low'] - rsi) * 2)
        
        # Short conditions
        elif (rsi >= self.config['rsi_high'] and 
              price_dev >= self.config['price_dev'] and
              vol_ratio >= self.config['volume_min']):
            direction = 'short'
            confidence = 70 + min(20, (rsi - self.config['rsi_high']) * 2)
        
        return {
            'signal': direction is not None,
            'direction': direction,
            'confidence': confidence
        }
    
    def calculate_position_size(self, balance: float, confidence: float) -> float:
        """Mathematical position sizing"""
        conf_factor = confidence / 100
        position_pct = self.config['position_base'] + \
                      (self.config['position_max'] - self.config['position_base']) * conf_factor
        return balance * position_pct
    
    def simulate_trade(self, entry_idx: int, entry_price: float, direction: str, 
                      position_size: float, data: pd.DataFrame) -> dict:
        """Simulate trade with mathematical exit logic"""
        
        # Trade parameters
        if direction == 'long':
            profit_target = entry_price * (1 + self.config['profit_target'])
            stop_loss = entry_price * (1 - self.config['stop_loss'])
        else:
            profit_target = entry_price * (1 - self.config['profit_target'])
            stop_loss = entry_price * (1 + self.config['stop_loss'])
        
        # Tracking
        best_price = entry_price
        trail_price = None
        trail_active = False
        
        # Simulate
        max_idx = min(entry_idx + self.config['max_hold'], len(data) - 1)
        
        for i in range(entry_idx + 1, max_idx + 1):
            candle = data.iloc[i]
            high, low, close = candle['high'], candle['low'], candle['close']
            
            if direction == 'long':
                # Update best price
                if high > best_price:
                    best_price = high
                
                # Check profit target
                if high >= profit_target:
                    pnl = position_size * self.config['profit_target']
                    return {
                        'exit_price': profit_target,
                        'exit_reason': 'profit_target',
                        'pnl': pnl,
                        'hold_minutes': i - entry_idx,
                        'success': True
                    }
                
                # Trailing stop logic
                unrealized_pct = (high - entry_price) / entry_price
                if not trail_active and unrealized_pct >= self.config['trail_start']:
                    trail_active = True
                    trail_price = best_price * (1 - self.config['trail_distance'])
                
                if trail_active:
                    new_trail = best_price * (1 - self.config['trail_distance'])
                    if new_trail > trail_price:
                        trail_price = new_trail
                    
                    if low <= trail_price:
                        profit_pct = (trail_price - entry_price) / entry_price
                        pnl = position_size * profit_pct
                        return {
                            'exit_price': trail_price,
                            'exit_reason': 'trailing_stop',
                            'pnl': pnl,
                            'hold_minutes': i - entry_idx,
                            'success': pnl > 0
                        }
                
                # Stop loss
                if low <= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss',
                        'pnl': pnl,
                        'hold_minutes': i - entry_idx,
                        'success': False
                    }
            
            else:  # short
                # Update best price
                if low < best_price:
                    best_price = low
                
                # Check profit target
                if low <= profit_target:
                    pnl = position_size * self.config['profit_target']
                    return {
                        'exit_price': profit_target,
                        'exit_reason': 'profit_target',
                        'pnl': pnl,
                        'hold_minutes': i - entry_idx,
                        'success': True
                    }
                
                # Trailing stop logic
                unrealized_pct = (entry_price - low) / entry_price
                if not trail_active and unrealized_pct >= self.config['trail_start']:
                    trail_active = True
                    trail_price = best_price * (1 + self.config['trail_distance'])
                
                if trail_active:
                    new_trail = best_price * (1 + self.config['trail_distance'])
                    if new_trail < trail_price:
                        trail_price = new_trail
                    
                    if high >= trail_price:
                        profit_pct = (entry_price - trail_price) / entry_price
                        pnl = position_size * profit_pct
                        return {
                            'exit_price': trail_price,
                            'exit_reason': 'trailing_stop',
                            'pnl': pnl,
                            'hold_minutes': i - entry_idx,
                            'success': pnl > 0
                        }
                
                # Stop loss
                if high >= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss',
                        'pnl': pnl,
                        'hold_minutes': i - entry_idx,
                        'success': False
                    }
        
        # Time exit
        final_price = data.iloc[max_idx]['close']
        if direction == 'long':
            profit_pct = (final_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - final_price) / entry_price
        
        pnl = position_size * profit_pct
        return {
            'exit_price': final_price,
            'exit_reason': 'time_exit',
            'pnl': pnl,
            'hold_minutes': self.config['max_hold'],
            'success': pnl > 0
        }
    
    def run_backtest(self, data: pd.DataFrame) -> dict:
        """Run mathematical backtest"""
        print("\n🧮 RUNNING MATHEMATICAL BACKTEST")
        print("=" * 40)
        
        # Initialize
        balance = self.initial_balance
        trades = []
        daily_trades = 0
        last_day = None
        last_trade_time = 0
        
        # Stats
        wins = 0
        losses = 0
        exit_reasons = {'profit_target': 0, 'trailing_stop': 0, 'stop_loss': 0, 'time_exit': 0}
        balance_history = [balance]
        
        # Main loop
        for i in range(20, len(data) - 50):
            current_time = data.iloc[i]['timestamp']
            current_day = current_time.date()
            
            # Daily reset
            if last_day != current_day:
                daily_trades = 0
                last_day = current_day
            
            # Check limits
            if daily_trades >= self.config['max_daily']:
                continue
            
            if i - last_trade_time < self.config['cooldown']:
                continue
            
            # Get indicators
            indicators = self.calculate_indicators(data, i)
            if not indicators:
                continue
            
            # Check entry
            entry = self.check_entry(indicators)
            if not entry['signal']:
                continue
            
            # Calculate position
            position_size = self.calculate_position_size(balance, entry['confidence'])
            entry_price = data.iloc[i]['close']
            
            # Simulate trade
            result = self.simulate_trade(i, entry_price, entry['direction'], 
                                       position_size, data)
            
            # Update balance
            balance += result['pnl']
            balance_history.append(balance)
            
            # Record trade
            trade = {
                'entry_time': current_time,
                'entry_price': entry_price,
                'direction': entry['direction'],
                'position_size': position_size,
                'confidence': entry['confidence'],
                **result
            }
            trades.append(trade)
            
            # Update stats
            if result['success']:
                wins += 1
            else:
                losses += 1
            
            exit_reasons[result['exit_reason']] += 1
            daily_trades += 1
            last_trade_time = i
            
            # Progress
            if len(trades) % 5 == 0:
                wr = wins / len(trades) * 100 if trades else 0
                ret = (balance - self.initial_balance) / self.initial_balance * 100
                print(f"📊 #{len(trades)}: {entry['direction'].upper()} → "
                      f"${result['pnl']:+.2f} | WR: {wr:.1f}% | Return: {ret:+.1f}%")
        
        # Calculate results
        total_trades = len(trades)
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        total_return = (balance - self.initial_balance) / self.initial_balance * 100
        
        # Additional metrics
        if total_trades > 0:
            winning_trades = [t for t in trades if t['success']]
            losing_trades = [t for t in trades if not t['success']]
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
            
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else float('inf')
            
            # Max drawdown
            peak = self.initial_balance
            max_dd = 0
            for bal in balance_history:
                if bal > peak:
                    peak = bal
                dd = (peak - bal) / peak * 100
                max_dd = max(max_dd, dd)
        else:
            avg_win = avg_loss = profit_factor = max_dd = 0
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_balance': balance,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_dd,
            'exit_reasons': exit_reasons,
            'trades': trades
        }
    
    def display_results(self, results: dict):
        """Display mathematical results"""
        print("\n" + "="*60)
        print("🧮 MATHEMATICAL GRADE A BOT - RESULTS")
        print("="*60)
        
        print(f"📊 PERFORMANCE:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1f}% ({results['wins']}W/{results['losses']}L)")
        print(f"   Total Return: {results['total_return']:+.1f}%")
        print(f"   Final Balance: ${results['final_balance']:.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.1f}%")
        
        print(f"\n📈 TRADE METRICS:")
        print(f"   Average Win: ${results['avg_win']:.2f}")
        print(f"   Average Loss: ${results['avg_loss']:.2f}")
        
        print(f"\n📤 EXIT ANALYSIS:")
        total = results['total_trades']
        if total > 0:
            for reason, count in results['exit_reasons'].items():
                pct = count / total * 100
                print(f"   {reason.title()}: {count} ({pct:.1f}%)")
        
        # Grade assessment
        print(f"\n🎯 GRADE ASSESSMENT:")
        grades = []
        
        # Win Rate
        if results['win_rate'] >= 60:
            print("   ✅ Win Rate: A+ (60%+)")
            grades.append(4)
        elif results['win_rate'] >= 55:
            print("   ✅ Win Rate: B+ (55%+)")
            grades.append(3)
        else:
            print("   ❌ Win Rate: Below Grade")
            grades.append(1)
        
        # Returns
        if results['total_return'] >= 15:
            print("   ✅ Returns: A+ (15%+)")
            grades.append(4)
        elif results['total_return'] >= 10:
            print("   ✅ Returns: B+ (10%+)")
            grades.append(3)
        else:
            print("   ❌ Returns: Below Grade")
            grades.append(1)
        
        # Profit Factor
        if results['profit_factor'] >= 1.8:
            print("   ✅ Risk/Reward: A+ (1.8+)")
            grades.append(4)
        elif results['profit_factor'] >= 1.4:
            print("   ✅ Risk/Reward: B+ (1.4+)")
            grades.append(3)
        else:
            print("   ❌ Risk/Reward: Below Grade")
            grades.append(1)
        
        # Frequency
        freq = results['total_trades'] / 60
        if freq >= 1.0:
            print("   ✅ Frequency: A+ (1.0+/day)")
            grades.append(4)
        elif freq >= 0.7:
            print("   ✅ Frequency: B+ (0.7+/day)")
            grades.append(3)
        else:
            print("   ❌ Frequency: Below Grade")
            grades.append(1)
        
        # Overall
        avg_grade = sum(grades) / len(grades)
        if avg_grade >= 3.5:
            overall = "A (EXCELLENT)"
        elif avg_grade >= 2.5:
            overall = "B+ (GOOD)"
        else:
            overall = "F (NEEDS WORK)"
        
        print(f"\n🏆 OVERALL GRADE: {overall}")
        print("="*60)

def main():
    print("🧮 MATHEMATICAL GRADE A BOT")
    print("📊 PURE STATISTICAL APPROACH")
    print("=" * 50)
    
    balance = 200.0
    days = 60
    
    bot = MathematicalGradeABot(balance)
    data = bot.generate_market_data(days)
    results = bot.run_backtest(data)
    bot.display_results(results)

if __name__ == "__main__":
    main() 