#!/usr/bin/env python3
"""
BALANCED AI VISUAL TRADER - PERFECT BALANCE
Smart multi-indicator analysis balancing quality and opportunity
Target: 65%+ Win Rate with 20-40 trades over 60 days
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BalancedAIVisualTrader:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # BALANCED CONFIGURATION
        self.config = {
            # BALANCED SIZING
            "leverage_min": 6,
            "leverage_max": 12,
            "position_min": 0.06,      # 6% minimum position
            "position_max": 0.15,      # 15% maximum position
            
            # BALANCED PROFIT/LOSS
            "profit_target_min": 0.015,  # 1.5% minimum
            "profit_target_max": 0.030,  # 3.0% maximum  
            "stop_loss": 0.008,          # 0.8% stop loss
            "trail_distance": 0.004,     # 0.4% trail
            "trail_start": 0.012,        # Start at 1.2%
            
            # BALANCED THRESHOLDS
            "min_conviction": 85,        # 85% AI conviction
            "max_daily_trades": 5,       # Balanced frequency
            "max_hold_hours": 6,         # Reasonable holds
            "confluence_required": 5,    # Need 5+ confluences
            "volume_threshold": 1.5,     # Good volume
        }
        
        print("⚖️ BALANCED AI VISUAL TRADER")
        print("🎯 SMART BALANCE: Quality + Opportunity")
        print("🏆 TARGET: 65%+ WIN RATE, 20-40 TRADES")
        print("=" * 60)
    
    def generate_market_data(self, days: int = 60) -> pd.DataFrame:
        """Generate realistic market data with clear opportunities"""
        print(f"📊 Generating {days} days of balanced market data...")
        
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        price = start_price
        time = datetime.now() - timedelta(days=days)
        volume_base = 1300000
        
        for i in range(total_minutes):
            # Balanced market cycles
            hour = (i // 60) % 24
            day_factor = np.sin(i / (18 * 60) * 2 * np.pi) * 0.0005
            week_factor = np.sin(i / (4 * 24 * 60) * 2 * np.pi) * 0.0008
            
            # Time-based volatility
            if 8 <= hour <= 16:
                vol = 0.0022    # Market hours
            elif 17 <= hour <= 23:
                vol = 0.0018    # Evening
            else:
                vol = 0.0012    # Night
            
            # Trend cycles with clear reversals
            trend_cycle = np.sin(i / (2 * 24 * 60) * 2 * np.pi)
            momentum = trend_cycle * 0.0004
            noise = np.random.normal(0, vol * 0.7)
            
            # Price movement
            price_change = day_factor + week_factor + momentum + noise
            price *= (1 + price_change)
            price = max(128, min(162, price))
            
            # OHLC
            spread = vol * 0.6
            high = price * (1 + abs(np.random.normal(0, spread * 0.7)))
            low = price * (1 - abs(np.random.normal(0, spread * 0.7)))
            open_p = price * (1 + np.random.normal(0, spread * 0.3))
            
            high = max(high, price, open_p)
            low = min(low, price, open_p)
            
            # Volume
            vol_momentum = abs(price_change) * 180
            volume_mult = 1 + vol_momentum + np.random.uniform(0.6, 1.7)
            volume = volume_base * volume_mult
            
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
        print(f"📈 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        return df
    
    def calculate_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        """Calculate indicators with balanced precision"""
        if idx < 80:
            return None
        
        window = data.iloc[max(0, idx-80):idx+1]
        current = window.iloc[-1]
        
        indicators = {}
        
        # === PRICE ACTION ===
        indicators['price'] = current['close']
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        indicators['body_pct'] = (body_size / total_range) if total_range > 0 else 0
        indicators['is_bullish'] = current['close'] > current['open']
        
        # === RSI ===
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        # RSI momentum
        if len(rsi) >= 3:
            indicators['rsi_momentum'] = rsi.iloc[-1] - rsi.iloc[-3]
        else:
            indicators['rsi_momentum'] = 0
        
        # === MOVING AVERAGES ===
        indicators['ema_9'] = window['close'].ewm(span=9).mean().iloc[-1]
        indicators['ema_21'] = window['close'].ewm(span=21).mean().iloc[-1]
        indicators['ema_50'] = window['close'].ewm(span=50).mean().iloc[-1] if len(window) >= 50 else window['close'].mean()
        
        # EMA trends
        indicators['ema_bullish'] = indicators['ema_9'] > indicators['ema_21'] > indicators['ema_50']
        indicators['ema_bearish'] = indicators['ema_9'] < indicators['ema_21'] < indicators['ema_50']
        
        # === MACD ===
        ema_12 = window['close'].ewm(span=12).mean()
        ema_26 = window['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd'] = macd_line.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        indicators['macd_histogram'] = macd_line.iloc[-1] - signal_line.iloc[-1]
        
        # === VOLUME ===
        indicators['volume'] = current['volume']
        indicators['volume_sma'] = window['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = current['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        # === OBV ===
        obv = 0
        obv_values = []
        for i in range(1, len(window)):
            if window['close'].iloc[i] > window['close'].iloc[i-1]:
                obv += window['volume'].iloc[i]
            elif window['close'].iloc[i] < window['close'].iloc[i-1]:
                obv -= window['volume'].iloc[i]
            obv_values.append(obv)
        
        indicators['obv'] = obv
        
        # OBV trend
        if len(obv_values) >= 10:
            recent_obv = np.mean(obv_values[-5:])
            prev_obv = np.mean(obv_values[-10:-5])
            indicators['obv_trend'] = 1 if recent_obv > prev_obv else -1
        else:
            indicators['obv_trend'] = 0
        
        # === MFI ===
        typical_price = (window['high'] + window['low'] + window['close']) / 3
        money_flow = typical_price * window['volume']
        
        positive_flow = 0
        negative_flow = 0
        
        for i in range(1, min(len(window), 15)):
            if typical_price.iloc[-i] > typical_price.iloc[-i-1]:
                positive_flow += money_flow.iloc[-i]
            else:
                negative_flow += money_flow.iloc[-i]
        
        if negative_flow > 0:
            money_ratio = positive_flow / negative_flow
            indicators['mfi'] = 100 - (100 / (1 + money_ratio))
        else:
            indicators['mfi'] = 100
        
        # === MOMENTUM ===
        if len(window) >= 10:
            indicators['momentum_10'] = (current['close'] - window['close'].iloc[-11]) / window['close'].iloc[-11] * 100
        else:
            indicators['momentum_10'] = 0
        
        # === SUPPORT/RESISTANCE ===
        recent_30 = window.tail(30) if len(window) >= 30 else window
        indicators['resistance'] = recent_30['high'].max()
        indicators['support'] = recent_30['low'].min()
        indicators['price_position'] = ((current['close'] - indicators['support']) / 
                                       (indicators['resistance'] - indicators['support'])) if indicators['resistance'] > indicators['support'] else 0.5
        
        return indicators
    
    def balanced_ai_analysis(self, indicators: dict) -> dict:
        """Balanced AI analysis for optimal win rate"""
        if not indicators:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        confluences = []
        conviction = 0
        direction = None
        reasoning = []
        
        # === BALANCED RSI ANALYSIS ===
        rsi = indicators['rsi']
        rsi_mom = indicators['rsi_momentum']
        
        if rsi <= 25:
            confluences.append('rsi_oversold')
            conviction += 30
            direction = 'long'
            reasoning.append(f"RSI oversold ({rsi:.1f})")
            
            if rsi_mom > 0:
                confluences.append('rsi_momentum_positive')
                conviction += 12
                reasoning.append("RSI showing upward momentum")
                
        elif rsi >= 75:
            confluences.append('rsi_overbought')
            conviction += 30
            direction = 'short'
            reasoning.append(f"RSI overbought ({rsi:.1f})")
            
            if rsi_mom < 0:
                confluences.append('rsi_momentum_negative')
                conviction += 12
                reasoning.append("RSI showing downward momentum")
        
        # Skip if RSI not extreme enough
        if direction is None:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # === EMA TREND CONFIRMATION ===
        ema_bullish = indicators['ema_bullish']
        ema_bearish = indicators['ema_bearish']
        
        if direction == 'long' and ema_bullish:
            confluences.append('ema_trend_bullish')
            conviction += 18
            reasoning.append("EMA trend bullish")
        elif direction == 'short' and ema_bearish:
            confluences.append('ema_trend_bearish')
            conviction += 18
            reasoning.append("EMA trend bearish")
        elif direction == 'long' and not ema_bearish:
            confluences.append('ema_neutral_bullish')
            conviction += 8
            reasoning.append("EMA trend neutral/bullish")
        elif direction == 'short' and not ema_bullish:
            confluences.append('ema_neutral_bearish')
            conviction += 8
            reasoning.append("EMA trend neutral/bearish")
        
        # === MACD CONFIRMATION ===
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_hist = indicators['macd_histogram']
        
        if direction == 'long' and macd > macd_signal:
            confluences.append('macd_bullish')
            conviction += 20
            reasoning.append("MACD bullish signal")
        elif direction == 'short' and macd < macd_signal:
            confluences.append('macd_bearish')
            conviction += 20
            reasoning.append("MACD bearish signal")
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = indicators['volume_ratio']
        
        if volume_ratio >= self.config['volume_threshold']:
            confluences.append('volume_confirmation')
            conviction += 18
            reasoning.append(f"Strong volume ({volume_ratio:.1f}x)")
        elif volume_ratio >= 1.2:
            confluences.append('volume_adequate')
            conviction += 10
            reasoning.append(f"Adequate volume ({volume_ratio:.1f}x)")
        
        # === OBV CONFIRMATION ===
        obv_trend = indicators['obv_trend']
        
        if direction == 'long' and obv_trend > 0:
            confluences.append('obv_bullish')
            conviction += 15
            reasoning.append("OBV bullish trend")
        elif direction == 'short' and obv_trend < 0:
            confluences.append('obv_bearish')
            conviction += 15
            reasoning.append("OBV bearish trend")
        
        # === MFI CONFIRMATION ===
        mfi = indicators['mfi']
        
        if direction == 'long' and mfi <= 25:
            confluences.append('mfi_oversold')
            conviction += 16
            reasoning.append(f"MFI oversold ({mfi:.1f})")
        elif direction == 'short' and mfi >= 75:
            confluences.append('mfi_overbought')
            conviction += 16
            reasoning.append(f"MFI overbought ({mfi:.1f})")
        
        # === MOMENTUM CONFIRMATION ===
        momentum = indicators['momentum_10']
        
        if direction == 'long' and momentum > 0.5:
            confluences.append('momentum_bullish')
            conviction += 12
            reasoning.append("Positive momentum")
        elif direction == 'short' and momentum < -0.5:
            confluences.append('momentum_bearish')
            conviction += 12
            reasoning.append("Negative momentum")
        
        # === SUPPORT/RESISTANCE ===
        price_pos = indicators['price_position']
        
        if direction == 'long' and price_pos <= 0.25:
            confluences.append('near_support')
            conviction += 14
            reasoning.append("Price near support")
        elif direction == 'short' and price_pos >= 0.75:
            confluences.append('near_resistance')
            conviction += 14
            reasoning.append("Price near resistance")
        
        # === FINAL BALANCED DECISION ===
        enough_confluences = len(confluences) >= self.config['confluence_required']
        good_conviction = conviction >= self.config['min_conviction']
        
        trade_signal = enough_confluences and good_conviction
        
        return {
            'trade': trade_signal,
            'direction': direction,
            'conviction': min(conviction, 95),
            'confluences': confluences,
            'reasoning': reasoning,
            'confluence_count': len(confluences),
            'indicators_summary': {
                'rsi': rsi,
                'mfi': mfi,
                'volume_ratio': volume_ratio,
                'momentum': momentum
            }
        }
    
    def simulate_balanced_trade(self, entry_idx: int, entry_price: float, direction: str,
                               position_size: float, profit_target: float, data: pd.DataFrame) -> dict:
        """Simulate trade with balanced management"""
        
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
            high, low, close = candle['high'], candle['low'], candle['close']
            
            if direction == 'long':
                if high > best_price:
                    best_price = high
                
                # Take profit
                if high >= take_profit:
                    pnl = position_size * profit_target
                    return {
                        'exit_price': take_profit,
                        'exit_reason': 'take_profit',
                        'pnl': pnl,
                        'success': True,
                        'hold_minutes': i - entry_idx
                    }
                
                # Trailing stop
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
                        return {
                            'exit_price': trail_price,
                            'exit_reason': 'trailing_stop',
                            'pnl': pnl,
                            'success': pnl > 0,
                            'hold_minutes': i - entry_idx
                        }
                
                # Stop loss
                if low <= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss',
                        'pnl': pnl,
                        'success': False,
                        'hold_minutes': i - entry_idx
                    }
            
            else:  # short
                if low < best_price:
                    best_price = low
                
                if low <= take_profit:
                    pnl = position_size * profit_target
                    return {
                        'exit_price': take_profit,
                        'exit_reason': 'take_profit',
                        'pnl': pnl,
                        'success': True,
                        'hold_minutes': i - entry_idx
                    }
                
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
                        return {
                            'exit_price': trail_price,
                            'exit_reason': 'trailing_stop',
                            'pnl': pnl,
                            'success': pnl > 0,
                            'hold_minutes': i - entry_idx
                        }
                
                if high >= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss',
                        'pnl': pnl,
                        'success': False,
                        'hold_minutes': i - entry_idx
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
            'success': pnl > 0,
            'hold_minutes': self.config['max_hold_hours'] * 60
        }
    
    def run_balanced_backtest(self, data: pd.DataFrame) -> dict:
        """Run balanced backtest"""
        print("\n⚖️ RUNNING BALANCED AI TRADER")
        print("🎯 SMART BALANCE: Quality + Opportunity")
        print("=" * 50)
        
        balance = self.initial_balance
        trades = []
        daily_trades = 0
        last_day = None
        
        wins = 0
        losses = 0
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'stop_loss': 0, 'time_exit': 0}
        
        for i in range(80, len(data) - 100):
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
            
            analysis = self.balanced_ai_analysis(indicators)
            
            if not analysis['trade']:
                continue
            
            # Calculate position size
            entry_price = data.iloc[i]['close']
            conv_factor = (analysis['conviction'] - 80) / 15
            conv_factor = max(0, min(1, conv_factor))
            
            position_pct = self.config['position_min'] + \
                          (self.config['position_max'] - self.config['position_min']) * conv_factor
            position_size = balance * position_pct
            leverage = int(self.config['leverage_min'] + \
                          (self.config['leverage_max'] - self.config['leverage_min']) * conv_factor)
            profit_target = self.config['profit_target_min'] + \
                           (self.config['profit_target_max'] - self.config['profit_target_min']) * conv_factor
            
            # Simulate trade
            result = self.simulate_balanced_trade(
                i, entry_price, analysis['direction'], 
                position_size, profit_target, data)
            
            balance += result['pnl']
            
            if result['success']:
                wins += 1
            else:
                losses += 1
            
            exit_reasons[result['exit_reason']] += 1
            daily_trades += 1
            
            trades.append({
                'entry_time': current_time,
                'direction': analysis['direction'],
                'conviction': analysis['conviction'],
                'confluences': analysis['confluences'],
                'confluence_count': analysis['confluence_count'],
                **result
            })
            
            if len(trades) % 5 == 0:
                wr = wins / len(trades) * 100 if trades else 0
                ret = (balance - self.initial_balance) / self.initial_balance * 100
                print(f"⚖️ #{len(trades)}: {analysis['direction'].upper()} "
                      f"Conv:{analysis['conviction']:.0f}% Conf:{len(analysis['confluences'])} → "
                      f"${result['pnl']:+.2f} | WR:{wr:.1f}% Ret:{ret:+.1f}%")
        
        # Calculate results
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
            'avg_conviction': avg_conviction,
            'avg_confluences': avg_confluences,
            'exit_reasons': exit_reasons,
            'trades': trades
        }
    
    def display_balanced_results(self, results: dict):
        """Display balanced results"""
        print("\n" + "="*80)
        print("⚖️ BALANCED AI VISUAL TRADER - RESULTS")
        print("🎯 SMART BALANCE: QUALITY + OPPORTUNITY")
        print("="*80)
        
        print(f"📊 PERFORMANCE METRICS:")
        print(f"   🔢 Total Trades: {results['total_trades']}")
        print(f"   🏆 Win Rate: {results['win_rate']:.1f}% ({results['wins']}W/{results['losses']}L)")
        print(f"   💰 Total Return: {results['total_return']:+.1f}%")
        print(f"   💵 Final Balance: ${results['final_balance']:.2f}")
        print(f"   📈 Profit Factor: {results['profit_factor']:.2f}")
        
        print(f"\n📈 TRADE ANALYSIS:")
        print(f"   💚 Average Win: ${results['avg_win']:.2f}")
        print(f"   ❌ Average Loss: ${results['avg_loss']:.2f}")
        print(f"   🧠 Average Conviction: {results['avg_conviction']:.1f}%")
        print(f"   🎯 Average Confluences: {results['avg_confluences']:.1f}")
        
        print(f"\n📤 EXIT ANALYSIS:")
        total = results['total_trades']
        if total > 0:
            for reason, count in results['exit_reasons'].items():
                pct = count / total * 100
                emoji = "🎯" if reason == "take_profit" else "🛡️" if reason == "trailing_stop" else "🛑" if reason == "stop_loss" else "⏰"
                print(f"   {emoji} {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        print(f"\n🏆 BALANCED ASSESSMENT:")
        
        if results['win_rate'] >= 65:
            print("   ✅ Win Rate: A+ (65%+ TARGET ACHIEVED!)")
            wr_grade = "A+"
        elif results['win_rate'] >= 58:
            print("   ✅ Win Rate: A- (58%+)")
            wr_grade = "A-"
        elif results['win_rate'] >= 50:
            print("   ✅ Win Rate: B+ (50%+)")
            wr_grade = "B+"
        else:
            print("   ❌ Win Rate: Needs Improvement")
            wr_grade = "F"
        
        if results['total_return'] >= 15:
            print("   ✅ Returns: A+ (15%+)")
            ret_grade = "A+"
        elif results['total_return'] >= 10:
            print("   ✅ Returns: A- (10%+)")
            ret_grade = "A-"
        elif results['total_return'] >= 5:
            print("   ✅ Returns: B+ (5%+)")
            ret_grade = "B+"
        else:
            print("   ❌ Returns: Needs Improvement")
            ret_grade = "F"
        
        if results['profit_factor'] >= 2.0:
            print("   ✅ Risk/Reward: A+ (2.0+)")
            pf_grade = "A+"
        elif results['profit_factor'] >= 1.7:
            print("   ✅ Risk/Reward: A- (1.7+)")
            pf_grade = "A-"
        elif results['profit_factor'] >= 1.4:
            print("   ✅ Risk/Reward: B+ (1.4+)")
            pf_grade = "B+"
        else:
            print("   ❌ Risk/Reward: Needs Improvement")
            pf_grade = "F"
        
        freq = results['total_trades'] / 60
        if 0.3 <= freq <= 0.7:
            print("   ✅ Frequency: A+ (Optimal range)")
            freq_grade = "A+"
        elif 0.2 <= freq <= 0.9:
            print("   ✅ Frequency: B+ (Good range)")
            freq_grade = "B+"
        else:
            print("   ❌ Frequency: Suboptimal")
            freq_grade = "F"
        
        # Overall grade
        grade_values = {"A+": 4, "A-": 3.7, "B+": 3.3, "F": 0}
        avg_grade = (grade_values[wr_grade] + grade_values[ret_grade] + 
                    grade_values[pf_grade] + grade_values[freq_grade]) / 4
        
        if avg_grade >= 3.7:
            overall = "A+ (EXCEPTIONAL BALANCE!)"
            emoji = "🏆"
        elif avg_grade >= 3.3:
            overall = "A (EXCELLENT BALANCE)"
            emoji = "🥇"
        elif avg_grade >= 2.8:
            overall = "B+ (GOOD BALANCE)"
            emoji = "✅"
        else:
            overall = "NEEDS OPTIMIZATION"
            emoji = "❌"
        
        print(f"\n{emoji} OVERALL GRADE: {overall}")
        
        if avg_grade >= 3.3:
            print("\n🎉 BALANCED AI ACHIEVED TARGET!")
            print("⚖️ Perfect balance of quality and opportunity!")
            print("🚀 Ready for live trading deployment!")
        
        print("="*80)

def main():
    print("⚖️ BALANCED AI VISUAL TRADER")
    print("🎯 SMART BALANCE: Quality + Opportunity")
    print("👁️ ANALYZES: All Indicators with Balance")
    print("🏆 TARGET: 65%+ Win Rate, 20-40 Trades")
    print("=" * 60)
    
    ai_trader = BalancedAIVisualTrader(200.0)
    data = ai_trader.generate_market_data(60)
    results = ai_trader.run_balanced_backtest(data)
    ai_trader.display_balanced_results(results)

if __name__ == "__main__":
    main() 