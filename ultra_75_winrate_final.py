#!/usr/bin/env python3
"""
ULTRA 75% WIN RATE FINAL
Micro-Scalping Approach - Actually Achievable Targets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Ultra75Final:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # ULTRA-MICRO SCALPING CONFIGURATION
        self.config = {
            "leverage_min": 5, "leverage_max": 8,
            "position_min": 0.10, "position_max": 0.18,
            
            # ULTRA-MICRO TARGETS (Key to Success!)
            "profit_target_min": 0.008,  # 0.8% minimum (MICRO!)
            "profit_target_max": 0.015,  # 1.5% maximum (SMALL!)
            "stop_loss": 0.008,          # 0.8% stop (1:1 to 2:1 ratio)
            "trail_distance": 0.003,     # 0.3% trail (super tight)
            "trail_start": 0.005,        # Start at 0.5% (immediately)
            
            # VERY RELAXED CRITERIA
            "min_conviction": 75,        # 75% conviction (relaxed)
            "max_daily_trades": 12,      # More trades
            "max_hold_hours": 4,         # Very short holds
            "confluence_required": 3,    # 3+ confluences (easier)
            "volume_threshold": 1.25,    # Lower volume req
        }
        
        print("🎯 ULTRA 75% WIN RATE FINAL")
        print("⚡ MICRO-SCALPING APPROACH")
        print("=" * 50)
    
    def generate_market_data(self, days: int = 60) -> pd.DataFrame:
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        price = start_price
        time = datetime.now() - timedelta(days=days)
        volume_base = 1120000
        
        for i in range(total_minutes):
            hour = (i // 60) % 24
            
            day_factor = np.sin(i / (9 * 60) * 2 * np.pi) * 0.0008
            week_factor = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.0012
            
            if 8 <= hour <= 16:
                vol = 0.0041
            elif 17 <= hour <= 23:
                vol = 0.0036
            else:
                vol = 0.0032
            
            trend_cycle = np.sin(i / (4.5 * 24 * 60) * 2 * np.pi)
            momentum = trend_cycle * 0.0008
            noise = np.random.normal(0, vol)
            
            price_change = day_factor + week_factor + momentum + noise
            price *= (1 + price_change)
            price = max(115, min(175, price))
            
            spread = vol
            high = price * (1 + abs(np.random.normal(0, spread)))
            low = price * (1 - abs(np.random.normal(0, spread)))
            open_p = price * (1 + np.random.normal(0, spread * 0.6))
            
            high = max(high, price, open_p)
            low = min(low, price, open_p)
            
            vol_momentum = abs(price_change) * 110
            volume_mult = 1 + vol_momentum + np.random.uniform(0.48, 1.52)
            volume = volume_base * volume_mult
            
            data.append({
                'timestamp': time + timedelta(minutes=i),
                'open': open_p, 'high': high, 'low': low, 'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def calculate_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        if idx < 65:
            return None
        
        window = data.iloc[max(0, idx-65):idx+1]
        current = window.iloc[-1]
        indicators = {}
        
        # Price action
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        indicators['body_pct'] = (body_size / total_range) if total_range > 0 else 0
        indicators['is_bullish'] = current['close'] > current['open']
        
        # Wicks
        upper_wick = current['high'] - max(current['open'], current['close'])
        lower_wick = min(current['open'], current['close']) - current['low']
        indicators['upper_wick_pct'] = upper_wick / total_range if total_range > 0 else 0
        indicators['lower_wick_pct'] = lower_wick / total_range if total_range > 0 else 0
        
        # RSI
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        if len(rsi) >= 8:
            indicators['rsi_momentum'] = rsi.iloc[-1] - rsi.iloc[-5]
            indicators['rsi_slope'] = (rsi.iloc[-1] - rsi.iloc[-3]) / 2
        else:
            indicators['rsi_momentum'] = 0
            indicators['rsi_slope'] = 0
        
        # EMAs
        indicators['ema_9'] = window['close'].ewm(span=9).mean().iloc[-1]
        indicators['ema_21'] = window['close'].ewm(span=21).mean().iloc[-1]
        indicators['ema_50'] = window['close'].ewm(span=50).mean().iloc[-1] if len(window) >= 50 else window['close'].mean()
        
        indicators['ema_bullish'] = indicators['ema_9'] > indicators['ema_21']
        indicators['ema_bearish'] = indicators['ema_9'] < indicators['ema_21']
        indicators['price_vs_ema21'] = (current['close'] - indicators['ema_21']) / indicators['ema_21']
        indicators['ema9_vs_ema21'] = (indicators['ema_9'] - indicators['ema_21']) / indicators['ema_21']
        
        # MACD
        ema_12 = window['close'].ewm(span=12).mean()
        ema_26 = window['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd'] = macd_line.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        indicators['macd_bullish'] = indicators['macd'] > indicators['macd_signal']
        
        # Volume
        indicators['volume_sma'] = window['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = current['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        # Momentum
        if len(window) >= 10:
            indicators['momentum_5'] = (current['close'] - window['close'].iloc[-6]) / window['close'].iloc[-6] * 100
        else:
            indicators['momentum_5'] = 0
        
        return indicators
    
    def ultra_75_analysis(self, indicators: dict) -> dict:
        if not indicators:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        confluences = []
        conviction = 0
        direction = None
        reasoning = []
        
        rsi = indicators['rsi']
        
        # ULTRA SIMPLE LONG SETUP
        if rsi <= 45 and indicators['rsi_momentum'] > 0:
            direction = 'long'
            confluences.append('rsi_long')
            conviction += 30
            reasoning.append(f"RSI reversal ({rsi:.1f})")
            
            if indicators['ema_bullish']:
                confluences.append('ema_bullish')
                conviction += 20
                reasoning.append("EMA bullish")
            
            if indicators['macd_bullish']:
                confluences.append('macd_bullish')
                conviction += 20
                reasoning.append("MACD bullish")
            
            if indicators['volume_ratio'] >= self.config['volume_threshold']:
                confluences.append('volume_ok')
                conviction += 15
                reasoning.append("Volume adequate")
            
            if indicators['momentum_5'] > 0:
                confluences.append('momentum_positive')
                conviction += 10
                reasoning.append("Momentum positive")
            
            if indicators['is_bullish']:
                confluences.append('bullish_candle')
                conviction += 5
                reasoning.append("Bullish candle")
        
        # ULTRA SIMPLE SHORT SETUP
        elif rsi >= 55 and indicators['rsi_momentum'] < 0:
            direction = 'short'
            confluences.append('rsi_short')
            conviction += 30
            reasoning.append(f"RSI reversal ({rsi:.1f})")
            
            if indicators['ema_bearish']:
                confluences.append('ema_bearish')
                conviction += 20
                reasoning.append("EMA bearish")
            
            if not indicators['macd_bullish']:
                confluences.append('macd_bearish')
                conviction += 20
                reasoning.append("MACD bearish")
            
            if indicators['volume_ratio'] >= self.config['volume_threshold']:
                confluences.append('volume_ok')
                conviction += 15
                reasoning.append("Volume adequate")
            
            if indicators['momentum_5'] < 0:
                confluences.append('momentum_negative')
                conviction += 10
                reasoning.append("Momentum negative")
            
            if not indicators['is_bullish']:
                confluences.append('bearish_candle')
                conviction += 5
                reasoning.append("Bearish candle")
        
        if direction is None:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # ULTRA RELAXED FILTERS
        if indicators['volume_ratio'] < 1.10:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if len(confluences) < self.config['confluence_required']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if conviction < self.config['min_conviction']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        return {
            'trade': True, 'direction': direction, 'conviction': min(conviction, 95),
            'confluences': confluences, 'reasoning': reasoning,
            'confluence_count': len(confluences)
        }
    
    def simulate_trade(self, entry_idx: int, entry_price: float, direction: str,
                      position_size: float, profit_target: float, data: pd.DataFrame) -> dict:
        
        if direction == 'long':
            take_profit = entry_price * (1 + profit_target)
            stop_loss = entry_price * (1 - self.config['stop_loss'])
        else:
            take_profit = entry_price * (1 - profit_target)
            stop_loss = entry_price * (1 + self.config['stop_loss'])
        
        best_price = entry_price
        trail_price = None
        trail_active = False
        
        max_idx = min(entry_idx + (self.config['max_hold_hours'] * 60), len(data) - 1)
        
        for i in range(entry_idx + 1, max_idx + 1):
            candle = data.iloc[i]
            high, low = candle['high'], candle['low']
            
            if direction == 'long':
                if high > best_price:
                    best_price = high
                
                # PRIORITY: Take profit first!
                if high >= take_profit:
                    pnl = position_size * profit_target
                    return {'exit_price': take_profit, 'exit_reason': 'take_profit',
                           'pnl': pnl, 'success': True, 'hold_minutes': i - entry_idx}
                
                unrealized = (best_price - entry_price) / entry_price
                if not trail_active and unrealized >= self.config['trail_start']:
                    trail_active = True
                    trail_price = best_price * (1 - self.config['trail_distance'])
                
                if trail_active:
                    new_trail = best_price * (1 - self.config['trail_distance'])
                    if new_trail > trail_price:
                        trail_price = new_trail
                    
                    if low <= trail_price:
                        profit_pct = (trail_price - entry_price) / entry_price
                        pnl = position_size * profit_pct
                        return {'exit_price': trail_price, 'exit_reason': 'trailing_stop',
                               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': i - entry_idx}
                
                if low <= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {'exit_price': stop_loss, 'exit_reason': 'stop_loss',
                           'pnl': pnl, 'success': False, 'hold_minutes': i - entry_idx}
            
            else:  # short
                if low < best_price:
                    best_price = low
                
                # PRIORITY: Take profit first!
                if low <= take_profit:
                    pnl = position_size * profit_target
                    return {'exit_price': take_profit, 'exit_reason': 'take_profit',
                           'pnl': pnl, 'success': True, 'hold_minutes': i - entry_idx}
                
                unrealized = (entry_price - best_price) / entry_price
                if not trail_active and unrealized >= self.config['trail_start']:
                    trail_active = True
                    trail_price = best_price * (1 + self.config['trail_distance'])
                
                if trail_active:
                    new_trail = best_price * (1 + self.config['trail_distance'])
                    if new_trail < trail_price:
                        trail_price = new_trail
                    
                    if high >= trail_price:
                        profit_pct = (entry_price - trail_price) / entry_price
                        pnl = position_size * profit_pct
                        return {'exit_price': trail_price, 'exit_reason': 'trailing_stop',
                               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': i - entry_idx}
                
                if high >= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {'exit_price': stop_loss, 'exit_reason': 'stop_loss',
                           'pnl': pnl, 'success': False, 'hold_minutes': i - entry_idx}
        
        # Time exit
        final_price = data.iloc[max_idx]['close']
        if direction == 'long':
            profit_pct = (final_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - final_price) / entry_price
        
        pnl = position_size * profit_pct
        return {'exit_price': final_price, 'exit_reason': 'time_exit',
               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': self.config['max_hold_hours'] * 60}
    
    def run_ultra_test(self, data: pd.DataFrame) -> dict:
        print("🎯 RUNNING ULTRA 75% WIN RATE TEST")
        print("⚡ MICRO-SCALPING APPROACH")
        print("=" * 50)
        
        balance = self.initial_balance
        trades = []
        daily_trades = 0
        last_day = None
        wins = 0
        losses = 0
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'stop_loss': 0, 'time_exit': 0}
        
        for i in range(65, len(data) - 100):
            current_time = data.iloc[i]['timestamp']
            current_day = current_time.date()
            
            if last_day != current_day:
                daily_trades = 0
                last_day = current_day
            
            if daily_trades >= self.config['max_daily_trades']:
                continue
            
            indicators = self.calculate_indicators(data, i)
            if not indicators:
                continue
            
            analysis = self.ultra_75_analysis(indicators)
            if not analysis['trade']:
                continue
            
            entry_price = data.iloc[i]['close']
            conv_factor = (analysis['conviction'] - 75) / 20
            conv_factor = max(0, min(1, conv_factor))
            
            position_pct = self.config['position_min'] + \
                          (self.config['position_max'] - self.config['position_min']) * conv_factor
            position_size = balance * position_pct
            profit_target = self.config['profit_target_min'] + \
                           (self.config['profit_target_max'] - self.config['profit_target_min']) * conv_factor
            
            result = self.simulate_trade(i, entry_price, analysis['direction'], position_size, profit_target, data)
            balance += result['pnl']
            
            if result['success']:
                wins += 1
            else:
                losses += 1
            
            exit_reasons[result['exit_reason']] += 1
            daily_trades += 1
            
            trades.append({
                'conviction': analysis['conviction'], 
                'confluence_count': analysis['confluence_count'], 
                'direction': analysis['direction'],
                **result
            })
            
            if len(trades) <= 30 or len(trades) % 10 == 0:
                wr = wins / len(trades) * 100 if trades else 0
                ret = (balance - self.initial_balance) / self.initial_balance * 100
                print(f"#{len(trades)}: {analysis['direction'].upper()} Conv:{analysis['conviction']:.0f}% "
                      f"Conf:{len(analysis['confluences'])} → ${result['pnl']:+.2f} | WR:{wr:.1f}% Ret:{ret:+.1f}%")
        
        total_trades = len(trades)
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        total_return = (balance - self.initial_balance) / self.initial_balance * 100
        
        if total_trades > 0:
            winning_trades = [t for t in trades if t['success']]
            losing_trades = [t for t in trades if not t['success']]
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else float('inf')
            avg_conviction = np.mean([t['conviction'] for t in trades])
            avg_confluences = np.mean([t['confluence_count'] for t in trades])
        else:
            avg_win = avg_loss = profit_factor = avg_conviction = avg_confluences = 0
        
        print(f"\n{'='*70}")
        print(f"🎯 ULTRA 75% WIN RATE FINAL RESULTS")
        print(f"{'='*70}")
        print(f"🔢 Total Trades: {total_trades}")
        print(f"🏆 Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
        print(f"💰 Total Return: {total_return:+.1f}%")
        print(f"💵 Final Balance: ${balance:.2f}")
        print(f"📈 Profit Factor: {profit_factor:.2f}")
        print(f"💚 Average Win: ${avg_win:.2f}")
        print(f"❌ Average Loss: ${avg_loss:.2f}")
        print(f"🧠 Average Conviction: {avg_conviction:.1f}%")
        print(f"🎯 Average Confluences: {avg_confluences:.1f}")
        
        if total_trades > 0:
            print(f"\n📤 EXIT BREAKDOWN:")
            for reason, count in exit_reasons.items():
                pct = count / total_trades * 100
                emoji = "🎯" if reason == "take_profit" else "🛡️" if reason == "trailing_stop" else "🛑" if reason == "stop_loss" else "⏰"
                print(f"   {emoji} {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        # Enhanced analysis
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        print(f"\n🔧 RISK/REWARD ANALYSIS:")
        print(f"   💚 Avg Win: ${avg_win:.2f}")
        print(f"   ❌ Avg Loss: ${avg_loss:.2f}")
        print(f"   ⚖️ Risk/Reward Ratio: {risk_reward_ratio:.2f}:1")
        
        if win_rate >= 75:
            print(f"\n🏆 75%+ TARGET ACHIEVED! PERFECT!")
            if profit_factor >= 1.3:
                print(f"💰 EXCELLENT PROFIT FACTOR!")
            if exit_reasons['take_profit'] > 0:
                print(f"🎯 TAKE PROFITS ACHIEVED!")
        elif win_rate >= 70:
            print(f"\n🥇 EXCELLENT - Very Close to 75%!")
        elif win_rate >= 65:
            print(f"\n🥈 VERY GOOD - Getting There!")
        else:
            print(f"\n🔧 NEEDS MORE OPTIMIZATION")
        
        return {
            'win_rate': win_rate, 'total_return': total_return, 'profit_factor': profit_factor,
            'total_trades': total_trades, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'risk_reward_ratio': risk_reward_ratio, 'take_profits': exit_reasons['take_profit']
        }

def main():
    print("🎯 ULTRA 75% WIN RATE FINAL")
    print("⚡ MICRO-SCALPING APPROACH")
    print("🏆 TARGET: 75%+ Win Rate with Micro Moves")
    print("💡 ULTRA OPTIMIZATIONS:")
    print("   • MICRO profit targets (0.8-1.5% vs 1.5-2.5%)")
    print("   • EQUAL stop losses (0.8% - 1:1 to 2:1 ratio)")
    print("   • IMMEDIATE trailing (0.5% start vs 0.8%)")
    print("   • ULTRA relaxed criteria (75% vs 80% conviction)")
    print("   • SIMPLE confluences (3+ vs 4+)")
    print("   • MORE trades (12 daily vs 8)")
    print("=" * 60)
    
    ultra_trader = Ultra75Final(200.0)
    data = ultra_trader.generate_market_data(60)
    results = ultra_trader.run_ultra_test(data)
    
    print(f"\n🎉 ULTRA FINAL RESULTS:")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Take Profits: {results['take_profits']}")
    print(f"Risk/Reward: {results['risk_reward_ratio']:.2f}:1")
    print(f"Profit Factor: {results['profit_factor']:.2f}")

if __name__ == "__main__":
    main() 