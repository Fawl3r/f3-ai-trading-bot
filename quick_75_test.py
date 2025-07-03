#!/usr/bin/env python3
"""
QUICK 75% WIN RATE TEST
Key fixes: Better risk/reward + stricter entries
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Quick75Test:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # KEY FIXES FOR 75% WIN RATE
        self.config = {
            "leverage_min": 3, "leverage_max": 5,
            "position_min": 0.08, "position_max": 0.14,
            
            # IMPROVED RISK/REWARD
            "profit_target_min": 0.022,  # 2.2% minimum (HIGHER)
            "profit_target_max": 0.045,  # 4.5% maximum
            "stop_loss": 0.020,          # 2.0% stop (TIGHTER)
            "trail_distance": 0.005,     # 0.5% trail (TIGHTER)
            "trail_start": 0.010,        # Start at 1.0%
            
            # ULTRA-STRICT THRESHOLDS
            "min_conviction": 90,        # 90% conviction
            "max_daily_trades": 3,       # Only best
            "max_hold_hours": 10,        # Shorter holds
            "confluence_required": 6,    # 6+ confluences
            "volume_threshold": 1.6,     # Higher volume
        }
        
        print("🎯 QUICK 75% WIN RATE TEST")
        print("🔧 KEY FIXES: BETTER RISK/REWARD + STRICTER ENTRIES")
        print("=" * 50)
    
    def generate_market_data(self, days: int = 60) -> pd.DataFrame:
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        price = start_price
        time = datetime.now() - timedelta(days=days)
        volume_base = 1150000
        
        for i in range(total_minutes):
            hour = (i // 60) % 24
            
            day_factor = np.sin(i / (9 * 60) * 2 * np.pi) * 0.0008
            week_factor = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.0012
            
            if 8 <= hour <= 16:
                vol = 0.004
            elif 17 <= hour <= 23:
                vol = 0.0035
            else:
                vol = 0.003
            
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
            volume_mult = 1 + vol_momentum + np.random.uniform(0.5, 1.5)
            volume = volume_base * volume_mult
            
            data.append({
                'timestamp': time + timedelta(minutes=i),
                'open': open_p, 'high': high, 'low': low, 'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def calculate_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        if idx < 60:
            return None
        
        window = data.iloc[max(0, idx-60):idx+1]
        current = window.iloc[-1]
        indicators = {}
        
        # Price action
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        indicators['body_pct'] = (body_size / total_range) if total_range > 0 else 0
        indicators['is_bullish'] = current['close'] > current['open']
        
        # Enhanced wicks
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
        
        indicators['ema_bullish'] = indicators['ema_9'] > indicators['ema_21'] > indicators['ema_50']
        indicators['ema_bearish'] = indicators['ema_9'] < indicators['ema_21'] < indicators['ema_50']
        indicators['price_vs_ema21'] = (current['close'] - indicators['ema_21']) / indicators['ema_21']
        
        # MACD
        ema_12 = window['close'].ewm(span=12).mean()
        ema_26 = window['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd'] = macd_line.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        indicators['macd_bullish'] = indicators['macd'] > indicators['macd_signal']
        indicators['macd_strength'] = abs(macd_line.iloc[-1] - signal_line.iloc[-1])
        
        # Volume
        indicators['volume_sma'] = window['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = current['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        if len(window) >= 12:
            recent_vol = window['volume'].tail(6).mean()
            prev_vol = window['volume'].iloc[-12:-6].mean()
            indicators['volume_trend'] = (recent_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
        else:
            indicators['volume_trend'] = 0
        
        # OBV
        obv = 0
        obv_values = []
        for i in range(1, len(window)):
            if window['close'].iloc[i] > window['close'].iloc[i-1]:
                obv += window['volume'].iloc[i]
            elif window['close'].iloc[i] < window['close'].iloc[i-1]:
                obv -= window['volume'].iloc[i]
            obv_values.append(obv)
        
        if len(obv_values) >= 10:
            recent_obv = np.mean(obv_values[-5:])
            prev_obv = np.mean(obv_values[-10:-5])
            indicators['obv_trend'] = (recent_obv - prev_obv) / abs(prev_obv) if prev_obv != 0 else 0
        else:
            indicators['obv_trend'] = 0
        
        # MFI
        typical_price = (window['high'] + window['low'] + window['close']) / 3
        money_flow = typical_price * window['volume']
        
        positive_flow = 0
        negative_flow = 0
        for i in range(1, min(len(window), 14)):
            if typical_price.iloc[-i] > typical_price.iloc[-i-1]:
                positive_flow += money_flow.iloc[-i]
            else:
                negative_flow += money_flow.iloc[-i]
        
        if negative_flow > 0:
            money_ratio = positive_flow / negative_flow
            indicators['mfi'] = 100 - (100 / (1 + money_ratio))
        else:
            indicators['mfi'] = 100
        
        # Support/Resistance
        recent_30 = window.tail(30) if len(window) >= 30 else window
        indicators['resistance'] = recent_30['high'].max()
        indicators['support'] = recent_30['low'].min()
        
        price_range = indicators['resistance'] - indicators['support']
        if price_range > 0:
            indicators['price_position'] = (current['close'] - indicators['support']) / price_range
            indicators['support_distance'] = (current['close'] - indicators['support']) / current['close']
            indicators['resistance_distance'] = (indicators['resistance'] - current['close']) / current['close']
        else:
            indicators['price_position'] = 0.5
            indicators['support_distance'] = 0
            indicators['resistance_distance'] = 0
        
        return indicators
    
    def ultra_strict_analysis(self, indicators: dict) -> dict:
        if not indicators:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        confluences = []
        conviction = 0
        direction = None
        reasoning = []
        
        rsi = indicators['rsi']
        
        # ULTRA-STRICT LONG (Perfect setups only)
        if (rsi <= 32 and indicators['rsi_momentum'] > 1.5 and indicators['rsi_slope'] > 1.0 and
            indicators['price_position'] <= 0.30 and indicators['support_distance'] <= 0.015 and
            indicators['lower_wick_pct'] >= 0.35 and indicators['body_pct'] >= 0.6):
            
            direction = 'long'
            confluences.append('perfect_rsi_reversal')
            conviction += 35
            
            if indicators['ema_bullish']:
                confluences.append('perfect_ema_bullish')
                conviction += 25
            
            if indicators['macd_bullish'] and indicators['macd_strength'] > 0.015:
                confluences.append('strong_macd_bullish')
                conviction += 22
            
            if indicators['volume_ratio'] >= 2.0 and indicators['volume_trend'] > 0.15:
                confluences.append('exceptional_volume')
                conviction += 20
            
            if indicators['obv_trend'] > 0.08:
                confluences.append('strong_obv_accumulation')
                conviction += 18
            
            if indicators['mfi'] <= 30:
                confluences.append('extreme_mfi_oversold')
                conviction += 18
            
            if indicators['support_distance'] <= 0.01:
                confluences.append('perfect_support')
                conviction += 15
        
        # ULTRA-STRICT SHORT (Perfect setups only)
        elif (rsi >= 68 and indicators['rsi_momentum'] < -1.5 and indicators['rsi_slope'] < -1.0 and
              indicators['price_position'] >= 0.70 and indicators['resistance_distance'] <= 0.015 and
              indicators['upper_wick_pct'] >= 0.35 and indicators['body_pct'] >= 0.6):
            
            direction = 'short'
            confluences.append('perfect_rsi_reversal')
            conviction += 35
            
            if indicators['ema_bearish']:
                confluences.append('perfect_ema_bearish')
                conviction += 25
            
            if not indicators['macd_bullish'] and indicators['macd_strength'] > 0.015:
                confluences.append('strong_macd_bearish')
                conviction += 22
            
            if indicators['volume_ratio'] >= 2.0 and indicators['volume_trend'] > 0.15:
                confluences.append('exceptional_volume')
                conviction += 20
            
            if indicators['obv_trend'] < -0.08:
                confluences.append('strong_obv_distribution')
                conviction += 18
            
            if indicators['mfi'] >= 70:
                confluences.append('extreme_mfi_overbought')
                conviction += 18
            
            if indicators['resistance_distance'] <= 0.01:
                confluences.append('perfect_resistance')
                conviction += 15
        
        if direction is None:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # ULTRA-STRICT FILTERS
        if indicators['volume_ratio'] < self.config['volume_threshold']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if len(confluences) < self.config['confluence_required']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if conviction < self.config['min_conviction']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        return {
            'trade': True, 'direction': direction, 'conviction': min(conviction, 99),
            'confluences': confluences, 'confluence_count': len(confluences)
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
    
    def run_test(self, data: pd.DataFrame) -> dict:
        print("🎯 RUNNING QUICK 75% TEST")
        print("=" * 40)
        
        balance = self.initial_balance
        trades = []
        daily_trades = 0
        last_day = None
        wins = 0
        losses = 0
        
        for i in range(60, len(data) - 100):
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
            
            analysis = self.ultra_strict_analysis(indicators)
            if not analysis['trade']:
                continue
            
            entry_price = data.iloc[i]['close']
            conv_factor = (analysis['conviction'] - 90) / 9
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
            
            daily_trades += 1
            trades.append({'conviction': analysis['conviction'], 'confluence_count': analysis['confluence_count'], **result})
            
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
        else:
            avg_win = avg_loss = profit_factor = 0
        
        print(f"\n{'='*60}")
        print(f"🎯 QUICK 75% TEST RESULTS")
        print(f"{'='*60}")
        print(f"🔢 Total Trades: {total_trades}")
        print(f"🏆 Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
        print(f"💰 Total Return: {total_return:+.1f}%")
        print(f"💵 Final Balance: ${balance:.2f}")
        print(f"📈 Profit Factor: {profit_factor:.2f}")
        print(f"💚 Average Win: ${avg_win:.2f}")
        print(f"❌ Average Loss: ${avg_loss:.2f}")
        
        if win_rate >= 75:
            print(f"\n🏆 75% TARGET ACHIEVED!")
        elif win_rate >= 70:
            print(f"\n🥇 EXCELLENT - Very Close!")
        elif win_rate >= 65:
            print(f"\n🥈 VERY GOOD")
        else:
            print(f"\n🔧 NEEDS MORE WORK")
        
        return {'win_rate': win_rate, 'total_return': total_return, 'profit_factor': profit_factor}

def main():
    print("🎯 QUICK 75% WIN RATE TEST")
    print("🔧 KEY FIXES: TIGHTER STOPS + HIGHER TARGETS + ULTRA-STRICT")
    print("=" * 60)
    
    test = Quick75Test(200.0)
    data = test.generate_market_data(60)
    results = test.run_test(data)

if __name__ == "__main__":
    main() 