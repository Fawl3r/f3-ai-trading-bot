#!/usr/bin/env python3
"""
AGGRESSIVE SCALED BOT - TARGETING $5-12 PER TRADE
High-Risk, High-Reward Version
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AggressiveScaledBot:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # AGGRESSIVE SCALING CONFIGURATION
        self.config = {
            "leverage_min": 10, "leverage_max": 20,  # Higher leverage!
            "position_min": 0.25, "position_max": 0.50,  # Much larger positions (25-50%!)
            
            # AGGRESSIVE PROFIT TARGETS
            "profit_target_min": 0.015,  # 1.5% minimum 
            "profit_target_max": 0.035,  # 3.5% maximum (much higher!)
            "stop_loss": 0.008,          # 0.8% stop (still tight)
            "trail_distance": 0.002,     # 0.2% trail
            "trail_start": 0.005,        # Start at 0.5%
            
            # SELECTIVE CRITERIA (Higher quality trades)
            "min_conviction": 75,        # 75% conviction (more selective)
            "max_daily_trades": 12,      # Fewer trades, bigger size
            "max_hold_hours": 3,         # Slightly longer holds
            "confluence_required": 4,    # 4+ confluences (higher quality)
            "volume_threshold": 1.20,    # Higher volume requirement
            
            # CONFLUENCE WEIGHTS (same as enhanced)
            "confluence_weights": {
                "rsi_signal": 30,
                "ema_alignment": 25,
                "macd_confirmation": 25,
                "volume_surge": 20,
                "momentum_follow": 15,
                "candle_pattern": 10,
                "market_structure": 15
            }
        }
        
        print("🚀 AGGRESSIVE SCALED BOT - TARGETING $5-12 PER TRADE")
        print("⚠️  HIGH RISK - HIGH REWARD VERSION")
        print("💪 LARGER POSITIONS + HIGHER TARGETS")
        print("=" * 60)
    
    def generate_market_data(self, days: int = 60, seed: int = 42) -> pd.DataFrame:
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(seed)
        
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
    
    def calculate_enhanced_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        if idx < 65:
            return None
        
        window = data.iloc[max(0, idx-65):idx+1]
        current = window.iloc[-1]
        indicators = {}
        
        # Enhanced Price action
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        indicators['body_pct'] = (body_size / total_range) if total_range > 0 else 0
        indicators['is_bullish'] = current['close'] > current['open']
        
        # Enhanced RSI
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        # RSI momentum and slope
        if len(rsi) >= 10:
            indicators['rsi_momentum'] = rsi.iloc[-1] - rsi.iloc[-6]
            indicators['rsi_slope'] = (rsi.iloc[-1] - rsi.iloc[-4]) / 3
            indicators['rsi_acceleration'] = rsi.iloc[-1] - 2*rsi.iloc[-2] + rsi.iloc[-3]
        else:
            indicators['rsi_momentum'] = 0
            indicators['rsi_slope'] = 0
            indicators['rsi_acceleration'] = 0
        
        # Enhanced EMAs
        indicators['ema_9'] = window['close'].ewm(span=9).mean().iloc[-1]
        indicators['ema_21'] = window['close'].ewm(span=21).mean().iloc[-1]
        indicators['ema_50'] = window['close'].ewm(span=50).mean().iloc[-1] if len(window) >= 50 else window['close'].mean()
        
        indicators['ema_bullish'] = indicators['ema_9'] > indicators['ema_21']
        indicators['ema_strong_bullish'] = indicators['ema_9'] > indicators['ema_21'] > indicators['ema_50']
        indicators['ema_crossover'] = indicators['ema_9'] > indicators['ema_21'] and len(window) >= 2 and \
                                    window['close'].ewm(span=9).mean().iloc[-2] <= window['close'].ewm(span=21).mean().iloc[-2]
        
        # Enhanced MACD
        ema_12 = window['close'].ewm(span=12).mean()
        ema_26 = window['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd'] = macd_line.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        indicators['macd_bullish'] = indicators['macd'] > indicators['macd_signal']
        indicators['macd_strength'] = abs(indicators['macd_histogram'])
        
        # Enhanced Volume
        indicators['volume_sma'] = window['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = current['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        # Volume momentum
        if len(window) >= 10:
            recent_vol = window['volume'].tail(5).mean()
            prev_vol = window['volume'].iloc[-10:-5].mean()
            indicators['volume_momentum'] = (recent_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
        else:
            indicators['volume_momentum'] = 0
        
        # Enhanced momentum
        if len(window) >= 10:
            indicators['momentum_3'] = (current['close'] - window['close'].iloc[-4]) / window['close'].iloc[-4] * 100
            indicators['momentum_5'] = (current['close'] - window['close'].iloc[-6]) / window['close'].iloc[-6] * 100
            indicators['momentum_consistent'] = indicators['momentum_3'] * indicators['momentum_5'] > 0
        else:
            indicators['momentum_3'] = 0
            indicators['momentum_5'] = 0
            indicators['momentum_consistent'] = False
        
        # Market structure
        if len(window) >= 20:
            recent_highs = window['high'].tail(10).max()
            recent_lows = window['low'].tail(10).min()
            prev_highs = window['high'].iloc[-20:-10].max()
            prev_lows = window['low'].iloc[-20:-10].min()
            
            indicators['higher_highs'] = recent_highs > prev_highs
            indicators['higher_lows'] = recent_lows > prev_lows
            indicators['lower_highs'] = recent_highs < prev_highs
            indicators['lower_lows'] = recent_lows < prev_lows
            
            indicators['uptrend'] = indicators['higher_highs'] and indicators['higher_lows']
            indicators['downtrend'] = indicators['lower_highs'] and indicators['lower_lows']
        else:
            indicators['uptrend'] = False
            indicators['downtrend'] = False
        
        return indicators
    
    def aggressive_analysis(self, indicators: dict) -> dict:
        if not indicators:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        confluences = []
        conviction = 0
        direction = None
        reasoning = []
        weights = self.config['confluence_weights']
        
        rsi = indicators['rsi']
        
        # AGGRESSIVE LONG SETUP (More selective)
        if rsi <= 48:  # Slightly more selective
            rsi_score = 0
            if rsi <= 40 and indicators['rsi_momentum'] > -1 and indicators['rsi_slope'] > 0:
                rsi_score = weights['rsi_signal']
                confluences.append('strong_rsi_long')
                reasoning.append(f"Strong RSI reversal ({rsi:.1f})")
            elif rsi <= 45 and indicators['rsi_momentum'] > 0:
                rsi_score = weights['rsi_signal'] * 0.8
                confluences.append('rsi_long')
                reasoning.append(f"RSI favorable ({rsi:.1f})")
            
            if rsi_score > 0:
                direction = 'long'
                conviction += rsi_score
                
                # STRICTER EMA requirements for bigger trades
                if indicators['ema_strong_bullish'] and indicators['ema_crossover']:
                    confluences.append('perfect_ema_setup')
                    conviction += weights['ema_alignment'] * 1.2
                    reasoning.append("Perfect EMA setup")
                elif indicators['ema_strong_bullish']:
                    confluences.append('strong_ema_bullish')
                    conviction += weights['ema_alignment']
                    reasoning.append("Strong EMA alignment")
                elif indicators['ema_bullish']:
                    confluences.append('ema_bullish')
                    conviction += weights['ema_alignment'] * 0.6
                    reasoning.append("EMA bullish")
                
                # STRICTER MACD requirements
                if indicators['macd_bullish'] and indicators['macd_strength'] > 0.01:
                    confluences.append('strong_macd_bullish')
                    conviction += weights['macd_confirmation']
                    reasoning.append("Strong MACD bullish")
                elif indicators['macd_bullish'] and indicators['macd_strength'] > 0.005:
                    confluences.append('macd_bullish')
                    conviction += weights['macd_confirmation'] * 0.7
                    reasoning.append("MACD bullish")
                
                # STRICTER Volume requirements
                if indicators['volume_ratio'] >= 2.0 and indicators['volume_momentum'] > 0.2:
                    confluences.append('massive_volume_surge')
                    conviction += weights['volume_surge'] * 1.3
                    reasoning.append(f"Massive volume surge ({indicators['volume_ratio']:.1f}x)")
                elif indicators['volume_ratio'] >= 1.5 and indicators['volume_momentum'] > 0.1:
                    confluences.append('volume_surge')
                    conviction += weights['volume_surge']
                    reasoning.append(f"Volume surge ({indicators['volume_ratio']:.1f}x)")
                elif indicators['volume_ratio'] >= self.config['volume_threshold']:
                    confluences.append('volume_ok')
                    conviction += weights['volume_surge'] * 0.5
                    reasoning.append("Volume adequate")
                
                # STRICTER Momentum requirements
                if indicators['momentum_consistent'] and indicators['momentum_3'] > 0.5:
                    confluences.append('powerful_momentum')
                    conviction += weights['momentum_follow'] * 1.2
                    reasoning.append("Powerful momentum")
                elif indicators['momentum_consistent'] and indicators['momentum_3'] > 0.2:
                    confluences.append('strong_momentum')
                    conviction += weights['momentum_follow']
                    reasoning.append("Strong momentum")
                elif indicators['momentum_3'] > 0:
                    confluences.append('momentum_positive')
                    conviction += weights['momentum_follow'] * 0.5
                    reasoning.append("Momentum positive")
                
                # Enhanced Candle Pattern
                if indicators['is_bullish'] and indicators['body_pct'] >= 0.6:
                    confluences.append('very_strong_bullish_candle')
                    conviction += weights['candle_pattern'] * 1.5
                    reasoning.append("Very strong bullish candle")
                elif indicators['is_bullish'] and indicators['body_pct'] >= 0.4:
                    confluences.append('strong_bullish_candle')
                    conviction += weights['candle_pattern']
                    reasoning.append("Strong bullish candle")
                elif indicators['is_bullish']:
                    confluences.append('bullish_candle')
                    conviction += weights['candle_pattern'] * 0.5
                    reasoning.append("Bullish candle")
                
                # Market Structure
                if indicators['uptrend']:
                    confluences.append('uptrend_structure')
                    conviction += weights['market_structure']
                    reasoning.append("Uptrend structure")
        
        # AGGRESSIVE SHORT SETUP (More selective)
        elif rsi >= 52:  # Slightly more selective
            rsi_score = 0
            if rsi >= 60 and indicators['rsi_momentum'] < 1 and indicators['rsi_slope'] < 0:
                rsi_score = weights['rsi_signal']
                confluences.append('strong_rsi_short')
                reasoning.append(f"Strong RSI reversal ({rsi:.1f})")
            elif rsi >= 55 and indicators['rsi_momentum'] < 0:
                rsi_score = weights['rsi_signal'] * 0.8
                confluences.append('rsi_short')
                reasoning.append(f"RSI favorable ({rsi:.1f})")
            
            if rsi_score > 0:
                direction = 'short'
                conviction += rsi_score
                
                # STRICTER EMA requirements
                if not indicators['ema_bullish'] and not indicators['ema_strong_bullish']:
                    confluences.append('strong_ema_bearish')
                    conviction += weights['ema_alignment']
                    reasoning.append("Strong EMA bearish")
                elif not indicators['ema_bullish']:
                    confluences.append('ema_bearish')
                    conviction += weights['ema_alignment'] * 0.6
                    reasoning.append("EMA bearish")
                
                # STRICTER MACD requirements
                if not indicators['macd_bullish'] and indicators['macd_strength'] > 0.01:
                    confluences.append('strong_macd_bearish')
                    conviction += weights['macd_confirmation']
                    reasoning.append("Strong MACD bearish")
                elif not indicators['macd_bullish'] and indicators['macd_strength'] > 0.005:
                    confluences.append('macd_bearish')
                    conviction += weights['macd_confirmation'] * 0.7
                    reasoning.append("MACD bearish")
                
                # STRICTER Volume requirements
                if indicators['volume_ratio'] >= 2.0 and indicators['volume_momentum'] > 0.2:
                    confluences.append('massive_volume_surge')
                    conviction += weights['volume_surge'] * 1.3
                    reasoning.append(f"Massive volume surge ({indicators['volume_ratio']:.1f}x)")
                elif indicators['volume_ratio'] >= 1.5 and indicators['volume_momentum'] > 0.1:
                    confluences.append('volume_surge')
                    conviction += weights['volume_surge']
                    reasoning.append(f"Volume surge ({indicators['volume_ratio']:.1f}x)")
                elif indicators['volume_ratio'] >= self.config['volume_threshold']:
                    confluences.append('volume_ok')
                    conviction += weights['volume_surge'] * 0.5
                    reasoning.append("Volume adequate")
                
                # STRICTER Momentum requirements
                if indicators['momentum_consistent'] and indicators['momentum_3'] < -0.5:
                    confluences.append('powerful_momentum')
                    conviction += weights['momentum_follow'] * 1.2
                    reasoning.append("Powerful bearish momentum")
                elif indicators['momentum_consistent'] and indicators['momentum_3'] < -0.2:
                    confluences.append('strong_momentum')
                    conviction += weights['momentum_follow']
                    reasoning.append("Strong bearish momentum")
                elif indicators['momentum_3'] < 0:
                    confluences.append('momentum_negative')
                    conviction += weights['momentum_follow'] * 0.5
                    reasoning.append("Momentum negative")
                
                # Enhanced Candle Pattern
                if not indicators['is_bullish'] and indicators['body_pct'] >= 0.6:
                    confluences.append('very_strong_bearish_candle')
                    conviction += weights['candle_pattern'] * 1.5
                    reasoning.append("Very strong bearish candle")
                elif not indicators['is_bullish'] and indicators['body_pct'] >= 0.4:
                    confluences.append('strong_bearish_candle')
                    conviction += weights['candle_pattern']
                    reasoning.append("Strong bearish candle")
                elif not indicators['is_bullish']:
                    confluences.append('bearish_candle')
                    conviction += weights['candle_pattern'] * 0.5
                    reasoning.append("Bearish candle")
                
                # Market Structure
                if indicators['downtrend']:
                    confluences.append('downtrend_structure')
                    conviction += weights['market_structure']
                    reasoning.append("Downtrend structure")
        
        if direction is None:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # STRICTER FILTERS for bigger trades
        if indicators['volume_ratio'] < self.config['volume_threshold']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if len(confluences) < self.config['confluence_required']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if conviction < self.config['min_conviction']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # Aggressive bonuses for high-quality setups
        if len(confluences) >= 6:
            conviction += 8
        if len(confluences) >= 7:
            conviction += 5
        if conviction >= 90:
            conviction += 3  # Bonus for very high conviction
        
        return {
            'trade': True, 'direction': direction, 'conviction': min(conviction, 99),
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
    
    def run_aggressive_test(self, data: pd.DataFrame, verbose: bool = True) -> dict:
        if verbose:
            print("🚀 RUNNING AGGRESSIVE SCALED TEST")
            print("💰 TARGETING $5-12 PER TRADE")
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
            
            indicators = self.calculate_enhanced_indicators(data, i)
            if not indicators:
                continue
            
            analysis = self.aggressive_analysis(indicators)
            if not analysis['trade']:
                continue
            
            entry_price = data.iloc[i]['close']
            conv_factor = (analysis['conviction'] - 75) / 25
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
            
            if verbose and (len(trades) <= 20 or len(trades) % 10 == 0):
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
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🚀 AGGRESSIVE SCALED BOT RESULTS")
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
            
            # Per-trade analysis
            print(f"\n💰 PER-TRADE ANALYSIS:")
            print(f"   💚 Average Win: ${avg_win:.2f}")
            print(f"   ❌ Average Loss: ${avg_loss:.2f}")
            risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            print(f"   ⚖️ Risk/Reward Ratio: {risk_reward_ratio:.2f}:1")
            
            # Goal achievement check
            if avg_win >= 5.0:
                print(f"\n🎯 TARGET ACHIEVED! Average win ≥ $5.00!")
                if avg_win >= 8.0:
                    print(f"🚀 EXCEEDED TARGET! Average win ≥ $8.00!")
                if avg_win >= 12.0:
                    print(f"💎 DIAMOND HANDS! Average win ≥ $12.00!")
            else:
                print(f"\n⚠️  Target not achieved - Average win: ${avg_win:.2f} (Goal: $5+)")
            
            if win_rate >= 65 and avg_win >= 5.0:
                print(f"\n🏆 AGGRESSIVE BOT SUCCESS!")
                print(f"✅ High win rate AND high profit per trade!")
                print(f"🚀 Ready for scaled deployment!")
            elif win_rate >= 60:
                print(f"\n🥇 GOOD PERFORMANCE")
                print(f"✅ Decent win rate, higher profit per trade")
            else:
                print(f"\n⚠️  NEEDS OPTIMIZATION")
                print(f"❌ Win rate too low for aggressive strategy")
        
        return {
            'win_rate': win_rate, 'total_return': total_return, 'profit_factor': profit_factor,
            'total_trades': total_trades, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'risk_reward_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf'),
            'take_profits': exit_reasons['take_profit'], 'final_balance': balance,
            'exit_reasons': exit_reasons, 'wins': wins, 'losses': losses
        }

def main():
    print("🚀 AGGRESSIVE SCALED BOT - TARGETING $5-12 PER TRADE")
    print("⚠️  HIGH RISK - HIGH REWARD VERSION")
    print("💪 LARGER POSITIONS (25-50%) + HIGHER TARGETS (1.5-3.5%)")
    print("🎯 FEWER TRADES, BIGGER SIZE, HIGHER QUALITY")
    print("=" * 70)
    
    aggressive_trader = AggressiveScaledBot(200.0)
    data = aggressive_trader.generate_market_data(60)
    results = aggressive_trader.run_aggressive_test(data)
    
    print(f"\n🎉 AGGRESSIVE FINAL RESULTS:")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Average Win: ${results['avg_win']:.2f}")
    print(f"Average Loss: ${results['avg_loss']:.2f}")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")

if __name__ == "__main__":
    main() 