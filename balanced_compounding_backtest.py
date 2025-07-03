#!/usr/bin/env python3
"""
Balanced Compounding Bot - Backtesting System
Focus: Optimal balance between win rate, frequency, and profitability
Features:
- Balanced entry conditions (not too strict, not too loose)
- Smart position sizing that grows with balance
- Flexible profit targets based on market conditions
- Effective risk management without being overly conservative
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BalancedCompoundingBacktest:
    """Backtest the Balanced Compounding Bot"""
    
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # BALANCED CONFIG FOR OPTIMAL PERFORMANCE
        self.config = {
            # LEVERAGE - Moderate for balance of safety and profit
            "leverage_min": 18,
            "leverage_max": 28,
            
            # POSITION SIZING - Smart scaling
            "position_pct_min": 18,  # 18% minimum of balance
            "position_pct_max": 30,  # 30% maximum of balance  
            "min_position": 40.0,    # Lower minimum for more trades
            
            # PROFIT TARGETS - Realistic and flexible
            "profit_target_min": 1.2,   # 1.2% minimum (easier to hit)
            "profit_target_max": 2.8,   # 2.8% maximum 
            "trailing_stop_pct": 0.7,   # 0.7% trailing stop
            "trailing_activation": 0.4, # Activate at 0.4% profit
            "emergency_stop": 1.8,      # 1.8% emergency stop
            
            # ENTRY CONDITIONS - Balanced for quality and frequency
            "min_confidence": 50,       # 50% minimum confidence
            "rsi_oversold": 35,         # RSI < 35 
            "rsi_overbought": 65,       # RSI > 65
            "volume_multiplier": 1.3,   # 1.3x average volume
            
            # RISK MANAGEMENT - Balanced approach
            "max_daily_trades": 12,     # More opportunities
            "max_hold_hours": 8,        # Reasonable hold time
            "max_consecutive_losses": 3, # Allow some losses
            "daily_loss_limit": 4.0,    # 4% daily loss limit
        }
        
        print("🚀 BALANCED COMPOUNDING BOT - BACKTESTING")
        print("⚖️ OPTIMAL BALANCE: WIN RATE + FREQUENCY + PROFIT")
        print("📊 SMART POSITION SIZING & REALISTIC TARGETS")
        print("=" * 60)
    
    def generate_realistic_market_data(self, days: int = 60) -> pd.DataFrame:
        """Generate realistic market data with good trading opportunities"""
        print(f"📊 Generating {days} days of realistic market data...")
        
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        current_price = start_price
        current_time = datetime.now() - timedelta(days=days)
        
        for i in range(total_minutes):
            # Time-based volatility with more opportunities
            hour = (i // 60) % 24
            if 0 <= hour < 8:  # Asian session
                volatility = 0.0018
            elif 8 <= hour < 16:  # European session  
                volatility = 0.0025
            else:  # US session
                volatility = 0.0032
            
            # Market cycles that create trading opportunities
            weekly_trend = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.0008
            daily_cycle = np.sin(i / (24 * 60) * 2 * np.pi) * 0.0004
            hourly_cycle = np.sin(i / 60 * 2 * np.pi) * 0.0002
            
            # Market shocks for volatility
            if np.random.random() < 0.0012:  # Slightly more frequent
                shock = np.random.choice([-1, 1]) * np.random.uniform(0.018, 0.045)
                volatility *= 2.2
            else:
                shock = 0
            
            # Price movement with more realistic patterns
            noise = np.random.normal(0, volatility)
            price_change = weekly_trend + daily_cycle + hourly_cycle + noise + shock
            current_price *= (1 + price_change)
            current_price = max(95, min(210, current_price))
            
            # OHLC generation
            intrabar_vol = volatility * 0.6
            high = current_price * (1 + abs(np.random.normal(0, intrabar_vol)))
            low = current_price * (1 - abs(np.random.normal(0, intrabar_vol)))
            open_price = current_price * (1 + np.random.normal(0, intrabar_vol * 0.4))
            
            high = max(high, open_price, current_price)
            low = min(low, open_price, current_price)
            
            # Volume correlated with price action
            base_volume = 1300000
            volume_factor = 1 + abs(price_change) * 65 + np.random.uniform(0.5, 2.0)
            if abs(price_change) > 0.01:
                volume_factor *= 1.4
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
        """Calculate technical indicators with balanced sensitivity"""
        if idx < 50:
            return None
        
        window = data.iloc[max(0, idx-50):idx+1]
        
        # RSI calculation
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Moving averages - multiple timeframes
        ma_5 = window['close'].rolling(5).mean().iloc[-1]
        ma_10 = window['close'].rolling(10).mean().iloc[-1]
        ma_20 = window['close'].rolling(20).mean().iloc[-1]
        ma_50 = window['close'].rolling(50).mean().iloc[-1] if len(window) >= 50 else window['close'].mean()
        
        # Volume analysis
        volume_ma = window['volume'].rolling(15).mean().iloc[-1]
        volume_trend = window['volume'].rolling(3).mean().iloc[-1] / volume_ma if volume_ma > 0 else 1
        
        # Momentum indicators
        if len(window) >= 8:
            momentum_fast = (window['close'].iloc[-1] - window['close'].iloc[-4]) / window['close'].iloc[-4] * 100
            momentum_slow = (window['close'].iloc[-1] - window['close'].iloc[-8]) / window['close'].iloc[-8] * 100
        else:
            momentum_fast = momentum_slow = 0
        
        # Volatility and trend strength
        volatility = window['close'].pct_change().rolling(10).std().iloc[-1] * 100
        trend_strength = abs(ma_5 - ma_20) / ma_20 * 100 if ma_20 > 0 else 0
        
        return {
            'rsi': current_rsi,
            'ma_5': ma_5,
            'ma_10': ma_10,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'volume_trend': volume_trend,
            'momentum_fast': momentum_fast,
            'momentum_slow': momentum_slow,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'current_price': window['close'].iloc[-1]
        }
    
    def analyze_opportunity(self, indicators: dict) -> dict:
        """Balanced opportunity analysis for optimal trade frequency and quality"""
        if not indicators:
            return {"confidence": 0, "direction": "hold"}
        
        confidence = 0
        direction = "hold"
        reasons = []
        
        current_price = indicators['current_price']
        rsi = indicators['rsi']
        
        # RSI signals with balanced thresholds (45% weight)
        if rsi < self.config['rsi_oversold']:
            confidence += 45
            direction = "long"
            reasons.append("oversold")
        elif rsi > self.config['rsi_overbought']:
            confidence += 45
            direction = "short"
            reasons.append("overbought")
        elif rsi < 42:
            confidence += 25
            direction = "long"
            reasons.append("getting_oversold")
        elif rsi > 58:
            confidence += 25
            direction = "short"
            reasons.append("getting_overbought")
        
        # Trend analysis with multiple confirmations (35% weight)
        if current_price > indicators['ma_5'] > indicators['ma_10']:
            if direction == "long" or direction == "hold":
                confidence += 25
                direction = "long"
                reasons.append("uptrend")
                # Bonus for strong trend
                if indicators['trend_strength'] > 1.5:
                    confidence += 10
                    reasons.append("strong_uptrend")
        elif current_price < indicators['ma_5'] < indicators['ma_10']:
            if direction == "short" or direction == "hold":
                confidence += 25
                direction = "short"
                reasons.append("downtrend")
                if indicators['trend_strength'] > 1.5:
                    confidence += 10
                    reasons.append("strong_downtrend")
        
        # Volume confirmation (15% weight)
        if indicators['volume_trend'] > self.config['volume_multiplier']:
            confidence += 15
            reasons.append("high_volume")
        elif indicators['volume_trend'] > 1.1:
            confidence += 8
            reasons.append("good_volume")
        
        # Momentum alignment (5% weight)
        if direction == "long" and indicators['momentum_fast'] > 0.3:
            confidence += 5
            reasons.append("positive_momentum")
        elif direction == "short" and indicators['momentum_fast'] < -0.3:
            confidence += 5
            reasons.append("negative_momentum")
        
        # Volatility considerations - boost confidence in moderate volatility
        if 1.5 <= indicators['volatility'] <= 3.5:
            confidence *= 1.1  # Boost confidence in good volatility
            reasons.append("good_volatility")
        elif indicators['volatility'] > 4.0:
            confidence *= 0.9  # Slight discount for high volatility
            reasons.append("high_volatility")
        
        return {
            "confidence": min(confidence, 95),
            "direction": direction,
            "reasons": reasons
        }
    
    def calculate_smart_position_size(self, confidence: float, balance: float) -> tuple:
        """Smart position sizing that scales with balance and confidence"""
        confidence_ratio = confidence / 100
        
        # Dynamic position percentage
        position_pct = (self.config['position_pct_min'] + 
                       (self.config['position_pct_max'] - self.config['position_pct_min']) * confidence_ratio)
        
        # Position size grows with balance (compounding effect)
        base_position = max(self.config['min_position'], balance * position_pct / 100)
        
        # Adjust for account size (larger accounts use slightly lower %)
        if balance > 500:
            position_pct *= 0.95
        elif balance > 1000:
            position_pct *= 0.9
        
        position_size = balance * position_pct / 100
        position_size = max(self.config['min_position'], position_size)
        
        # Smart leverage based on confidence and balance
        leverage = int(self.config['leverage_min'] + 
                      (self.config['leverage_max'] - self.config['leverage_min']) * confidence_ratio)
        
        # Dynamic profit target - higher confidence = higher target
        profit_target_pct = (self.config['profit_target_min'] + 
                            (self.config['profit_target_max'] - self.config['profit_target_min']) * confidence_ratio)
        
        return position_size, leverage, profit_target_pct
    
    def simulate_balanced_trade(self, entry_price: float, direction: str, 
                              leverage: int, position_size: float, profit_target_pct: float,
                              data: pd.DataFrame, start_idx: int) -> dict:
        """Simulate trade with balanced exit logic"""
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
                
                # Activate trailing stop
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
        """Run balanced backtest for optimal performance"""
        print("\n🧪 RUNNING BALANCED COMPOUNDING BACKTEST")
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
        
        # Tracking
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'emergency_stop': 0, 'time_exit': 0}
        trailing_activations = 0
        compounding_growth = [balance]
        
        i = 50  # Start earlier for more opportunities
        while i < len(data) - 400:
            current_time = data.iloc[i]['timestamp']
            current_price = data.iloc[i]['close']
            
            # Daily reset
            today = current_time.date()
            if last_trade_day != today:
                daily_trades = 0
                daily_loss = 0
                last_trade_day = today
            
            # Trade conditions
            daily_loss_pct = (daily_loss / balance) * 100 if balance > 0 else 0
            
            if (daily_trades < self.config['max_daily_trades'] and 
                consecutive_losses < self.config['max_consecutive_losses'] and
                daily_loss_pct < self.config['daily_loss_limit']):
                
                indicators = self.calculate_indicators(data, i)
                if indicators:
                    analysis = self.analyze_opportunity(indicators)
                    
                    if (analysis['confidence'] >= self.config['min_confidence'] and 
                        analysis['direction'] != "hold"):
                        
                        # Calculate trade parameters
                        position_size, leverage, profit_target_pct = self.calculate_smart_position_size(
                            analysis['confidence'], balance)
                        
                        # Simulate trade
                        trade_result = self.simulate_balanced_trade(
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
                            'balance_growth': (balance - old_balance) / old_balance * 100,
                            'reasons': analysis['reasons']
                        }
                        trades.append(trade)
                        daily_trades += 1
                        
                        # Progress updates
                        if len(trades) % 5 == 0 or len(trades) <= 10:
                            win_rate = (wins / len(trades) * 100) if trades else 0
                            growth = ((balance - self.initial_balance) / self.initial_balance) * 100
                            print(f"📊 Trade #{len(trades)}: {analysis['direction'].upper()} @ ${current_price:.2f} → "
                                  f"${trade_result['pnl_amount']:+.2f} ({trade_result['profit_pct']:+.1f}%) | "
                                  f"WR: {win_rate:.1f}% | Growth: {growth:+.1f}% | Balance: ${balance:.2f}")
                        
                        # Skip ahead
                        skip_minutes = max(trade_result['hold_minutes'], 10)
                        i += skip_minutes
                    else:
                        i += 2  # Faster scanning
                else:
                    i += 1
            else:
                i += 15  # Skip when limits reached
        
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
            
            # Drawdown calculation
            if len(compounding_growth) > 1:
                peak = compounding_growth[0]
                max_drawdown = 0
                for balance_point in compounding_growth:
                    if balance_point > peak:
                        peak = balance_point
                    drawdown = (peak - balance_point) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                max_drawdown = 0
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
        """Display balanced results focused on all key metrics"""
        print("\n" + "="*70)
        print("🚀 BALANCED COMPOUNDING BOT - BACKTEST RESULTS")
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
        
        # Compounding projections
        if results['final_balance'] > 0 and results['total_return'] > 0:
            print(f"\n💎 COMPOUNDING PROJECTIONS:")
            daily_return = (results['final_balance'] / self.initial_balance) ** (1/60) - 1
            monthly_return = ((1 + daily_return) ** 30 - 1) * 100
            annual_return = ((1 + daily_return) ** 365 - 1) * 100
            
            print(f"   📅 Daily Return: {daily_return*100:+.2f}%")
            print(f"   📅 Monthly Projection: {monthly_return:+.1f}%")
            print(f"   📅 Annual Projection: {annual_return:+.1f}%")
            
            # 6-month projection
            six_month_balance = self.initial_balance * ((1 + daily_return) ** 180)
            print(f"   🎯 6-Month Projection: ${six_month_balance:.2f} ({((six_month_balance/self.initial_balance-1)*100):+.0f}%)")
        
        # Performance grading
        print(f"\n💡 PERFORMANCE ASSESSMENT:")
        score = 0
        if results['win_rate'] >= 65:
            print("   🔥 Win Rate: EXCELLENT")
            score += 3
        elif results['win_rate'] >= 55:
            print("   ✅ Win Rate: GOOD")
            score += 2
        elif results['win_rate'] >= 45:
            print("   ⚠️ Win Rate: MODERATE")
            score += 1
        else:
            print("   ❌ Win Rate: POOR")
        
        if results['total_return'] >= 25:
            print("   🔥 Returns: EXCELLENT")
            score += 3
        elif results['total_return'] >= 15:
            print("   ✅ Returns: GOOD")
            score += 2
        elif results['total_return'] >= 5:
            print("   ⚠️ Returns: MODERATE") 
            score += 1
        else:
            print("   ❌ Returns: POOR")
        
        if results['profit_factor'] >= 2.0:
            print("   🔥 Profit Factor: EXCELLENT")
            score += 3
        elif results['profit_factor'] >= 1.5:
            print("   ✅ Profit Factor: GOOD")
            score += 2
        elif results['profit_factor'] >= 1.2:
            print("   ⚠️ Profit Factor: MODERATE")
            score += 1
        else:
            print("   ❌ Profit Factor: POOR")
        
        trades_per_day = results['total_trades'] / 60
        if trades_per_day >= 1.5:
            print("   ✅ Frequency: GOOD")
            score += 1
        elif trades_per_day >= 0.8:
            print("   ⚠️ Frequency: MODERATE")
        else:
            print("   ❌ Frequency: LOW")
        
        print(f"\n🎯 OVERALL GRADE: {score}/10")
        
        print("="*70)

def main():
    """Main balanced backtesting function"""
    print("🚀 BALANCED COMPOUNDING BOT - BACKTESTING SYSTEM")
    print("⚖️ OPTIMAL BALANCE: WIN RATE + FREQUENCY + PROFITABILITY")
    print("=" * 60)
    
    try:
        balance = float(input("💵 Enter starting balance (default $200): ") or "200")
    except ValueError:
        balance = 200.0
    
    try:
        days = int(input("📅 Backtest period in days (default 60): ") or "60")
    except ValueError:
        days = 60
    
    backtester = BalancedCompoundingBacktest(initial_balance=balance)
    
    # Generate market data
    data = backtester.generate_realistic_market_data(days=days)
    
    # Run backtest
    results = backtester.run_backtest(data)
    
    # Display results
    backtester.display_results(results)
    
    # Final recommendation
    score = 0
    if results['win_rate'] >= 60: score += 2
    if results['total_return'] >= 20: score += 2
    if results['profit_factor'] >= 1.8: score += 2
    if results['max_drawdown'] <= 15: score += 1
    if results['total_trades'] >= 60: score += 1
    
    if score >= 7:
        print(f"\n🎯 RECOMMENDATION: EXCELLENT FOR LIVE DEPLOYMENT!")
        print(f"   🔥 Outstanding performance across all metrics")
        print(f"   💰 Strong compounding potential")
        print(f"   ✅ Ready for live trading with proper risk management")
    elif score >= 5:
        print(f"\n⚠️ RECOMMENDATION: GOOD POTENTIAL - START SMALL")
        print(f"   ✅ Solid performance with room for optimization")
        print(f"   💡 Consider live testing with reduced position sizes")
    elif score >= 3:
        print(f"\n⚠️ RECOMMENDATION: NEEDS IMPROVEMENT")
        print(f"   📊 Some good metrics but requires optimization")
        print(f"   🔧 Focus on improving weaker areas before live deployment")
    else:
        print(f"\n❌ RECOMMENDATION: SIGNIFICANT OPTIMIZATION NEEDED")
        print(f"   🔴 Performance below acceptable thresholds")
        print(f"   🛠️ Major strategy revision required")

if __name__ == "__main__":
    main()