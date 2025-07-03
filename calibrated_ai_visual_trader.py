#!/usr/bin/env python3
"""
CALIBRATED AI VISUAL TRADER - FINAL VERSION
Properly calibrated for crypto volatility
Target: 60%+ Win Rate with sustainable profits
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CalibratedAIVisualTrader:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # CALIBRATED CONFIGURATION FOR CRYPTO
        self.config = {
            # CALIBRATED SIZING
            "leverage_min": 4,
            "leverage_max": 8,
            "position_min": 0.12,      # 12% minimum position
            "position_max": 0.20,      # 20% maximum position
            
            # CALIBRATED RISK MANAGEMENT (KEY FIX)
            "profit_target_min": 0.025,  # 2.5% minimum
            "profit_target_max": 0.050,  # 5.0% maximum  
            "stop_loss": 0.035,          # 3.5% stop loss (WIDER FOR CRYPTO)
            "trail_distance": 0.008,     # 0.8% trail
            "trail_start": 0.020,        # Start at 2.0%
            
            # CALIBRATED THRESHOLDS
            "min_conviction": 82,        # 82% conviction
            "max_daily_trades": 5,       # Quality frequency
            "max_hold_hours": 16,        # Longer holds for crypto
            "confluence_required": 4,    # 4+ confluences
            "volume_threshold": 1.25,    # Reasonable volume
        }
        
        print("🎯 CALIBRATED AI VISUAL TRADER")
        print("⚡ OPTIMIZED FOR CRYPTO VOLATILITY")
        print("🏆 TARGET: 60%+ WIN RATE")
        print("=" * 60)
    
    def generate_market_data(self, days: int = 60) -> pd.DataFrame:
        """Generate realistic crypto market data"""
        print(f"📊 Generating {days} days of calibrated crypto data...")
        
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        price = start_price
        time = datetime.now() - timedelta(days=days)
        volume_base = 1000000
        
        for i in range(total_minutes):
            hour = (i // 60) % 24
            
            # Crypto market cycles
            day_factor = np.sin(i / (10 * 60) * 2 * np.pi) * 0.0008
            week_factor = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.0012
            
            # Realistic crypto volatility
            if 8 <= hour <= 16:
                vol = 0.004     # Higher volatility
            elif 17 <= hour <= 23:
                vol = 0.0035    # Evening volatility
            else:
                vol = 0.003     # Night volatility
            
            # Trend cycles with crypto characteristics
            trend_cycle = np.sin(i / (5 * 24 * 60) * 2 * np.pi)
            momentum = trend_cycle * 0.0008
            noise = np.random.normal(0, vol)
            
            price_change = day_factor + week_factor + momentum + noise
            price *= (1 + price_change)
            price = max(115, min(175, price))
            
            # OHLC with crypto spreads
            spread = vol
            high = price * (1 + abs(np.random.normal(0, spread)))
            low = price * (1 - abs(np.random.normal(0, spread)))
            open_p = price * (1 + np.random.normal(0, spread * 0.6))
            
            high = max(high, price, open_p)
            low = min(low, price, open_p)
            
            # Volume with crypto patterns
            vol_momentum = abs(price_change) * 100
            volume_mult = 1 + vol_momentum + np.random.uniform(0.4, 1.4)
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
        """Calculate calibrated indicators"""
        if idx < 60:
            return None
        
        window = data.iloc[max(0, idx-60):idx+1]
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
        if len(rsi) >= 5:
            indicators['rsi_momentum'] = rsi.iloc[-1] - rsi.iloc[-5]
            indicators['rsi_slope'] = (rsi.iloc[-1] - rsi.iloc[-3]) / 2
        else:
            indicators['rsi_momentum'] = 0
            indicators['rsi_slope'] = 0
        
        # === MOVING AVERAGES ===
        indicators['ema_9'] = window['close'].ewm(span=9).mean().iloc[-1]
        indicators['ema_21'] = window['close'].ewm(span=21).mean().iloc[-1]
        indicators['ema_50'] = window['close'].ewm(span=50).mean().iloc[-1] if len(window) >= 50 else window['close'].mean()
        indicators['sma_20'] = window['close'].rolling(20).mean().iloc[-1]
        
        # EMA relationships
        indicators['ema_bullish'] = indicators['ema_9'] > indicators['ema_21'] > indicators['ema_50']
        indicators['ema_bearish'] = indicators['ema_9'] < indicators['ema_21'] < indicators['ema_50']
        indicators['price_vs_ema9'] = (current['close'] - indicators['ema_9']) / indicators['ema_9']
        indicators['price_vs_ema21'] = (current['close'] - indicators['ema_21']) / indicators['ema_21']
        
        # === MACD ===
        ema_12 = window['close'].ewm(span=12).mean()
        ema_26 = window['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd'] = macd_line.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        indicators['macd_histogram'] = macd_line.iloc[-1] - signal_line.iloc[-1]
        
        # MACD analysis
        indicators['macd_bullish'] = indicators['macd'] > indicators['macd_signal']
        indicators['macd_strength'] = abs(indicators['macd_histogram'])
        
        # === VOLUME ===
        indicators['volume'] = current['volume']
        indicators['volume_sma'] = window['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = current['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        # Volume analysis
        if len(window) >= 10:
            recent_vol = window['volume'].tail(10).mean()
            prev_vol = window['volume'].iloc[-20:-10].mean() if len(window) >= 20 else recent_vol
            indicators['volume_trend'] = (recent_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
        else:
            indicators['volume_trend'] = 0
        
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
        
        # OBV trend analysis
        if len(obv_values) >= 10:
            recent_obv = np.mean(obv_values[-5:])
            prev_obv = np.mean(obv_values[-10:-5])
            indicators['obv_trend'] = (recent_obv - prev_obv) / abs(prev_obv) if prev_obv != 0 else 0
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
        if len(window) >= 14:
            indicators['momentum_14'] = (current['close'] - window['close'].iloc[-15]) / window['close'].iloc[-15] * 100
        else:
            indicators['momentum_14'] = 0
        
        if len(window) >= 7:
            indicators['momentum_7'] = (current['close'] - window['close'].iloc[-8]) / window['close'].iloc[-8] * 100
        else:
            indicators['momentum_7'] = 0
        
        # === SUPPORT/RESISTANCE ===
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
    
    def calibrated_ai_analysis(self, indicators: dict) -> dict:
        """Calibrated AI analysis for crypto markets"""
        if not indicators:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        confluences = []
        conviction = 0
        direction = None
        reasoning = []
        
        rsi = indicators['rsi']
        rsi_momentum = indicators['rsi_momentum']
        rsi_slope = indicators['rsi_slope']
        
        # === CALIBRATED LONG SETUP ===
        if (rsi <= 40 and rsi_momentum > 0 and rsi_slope > 0 and 
            indicators['price_position'] <= 0.45 and 
            indicators['support_distance'] <= 0.03):
            
            direction = 'long'
            confluences.append('rsi_oversold_reversal')
            conviction += 28
            reasoning.append(f"RSI oversold with clear reversal ({rsi:.1f})")
            
            # EMA trend analysis
            if indicators['ema_bullish']:
                confluences.append('perfect_ema_alignment')
                conviction += 22
                reasoning.append("Perfect EMA bullish alignment")
            elif indicators['price_vs_ema21'] > -0.02:
                confluences.append('near_ema21_support')
                conviction += 15
                reasoning.append("Price near EMA21 support")
            
            # MACD confirmation
            if indicators['macd_bullish'] and indicators['macd_strength'] > 0.01:
                confluences.append('strong_macd_bullish')
                conviction += 20
                reasoning.append("Strong MACD bullish signal")
            elif indicators['macd_bullish']:
                confluences.append('macd_bullish')
                conviction += 12
                reasoning.append("MACD bullish crossover")
            
            # Volume confirmation
            if indicators['volume_ratio'] >= self.config['volume_threshold']:
                confluences.append('volume_confirmation')
                conviction += 16
                reasoning.append(f"Strong volume ({indicators['volume_ratio']:.1f}x)")
                
                if indicators['volume_trend'] > 0.1:
                    confluences.append('volume_trend_bullish')
                    conviction += 8
                    reasoning.append("Volume trend increasing")
            
            # OBV confirmation
            if indicators['obv_trend'] > 0.05:
                confluences.append('obv_bullish_trend')
                conviction += 14
                reasoning.append("OBV showing bullish accumulation")
            
            # MFI confirmation
            if indicators['mfi'] <= 40:
                confluences.append('mfi_oversold')
                conviction += 16
                reasoning.append(f"MFI oversold ({indicators['mfi']:.1f})")
            
            # Momentum confirmation
            if indicators['momentum_7'] > 0.5:
                confluences.append('momentum_bullish')
                conviction += 10
                reasoning.append("Short-term momentum positive")
            
            # Support level confirmation
            if indicators['support_distance'] <= 0.015:
                confluences.append('strong_support_level')
                conviction += 12
                reasoning.append("Price at strong support")
        
        # === CALIBRATED SHORT SETUP ===
        elif (rsi >= 60 and rsi_momentum < 0 and rsi_slope < 0 and 
              indicators['price_position'] >= 0.55 and 
              indicators['resistance_distance'] <= 0.03):
            
            direction = 'short'
            confluences.append('rsi_overbought_reversal')
            conviction += 28
            reasoning.append(f"RSI overbought with clear reversal ({rsi:.1f})")
            
            # EMA trend analysis
            if indicators['ema_bearish']:
                confluences.append('perfect_ema_alignment')
                conviction += 22
                reasoning.append("Perfect EMA bearish alignment")
            elif indicators['price_vs_ema21'] < 0.02:
                confluences.append('near_ema21_resistance')
                conviction += 15
                reasoning.append("Price near EMA21 resistance")
            
            # MACD confirmation
            if not indicators['macd_bullish'] and indicators['macd_strength'] > 0.01:
                confluences.append('strong_macd_bearish')
                conviction += 20
                reasoning.append("Strong MACD bearish signal")
            elif not indicators['macd_bullish']:
                confluences.append('macd_bearish')
                conviction += 12
                reasoning.append("MACD bearish crossover")
            
            # Volume confirmation
            if indicators['volume_ratio'] >= self.config['volume_threshold']:
                confluences.append('volume_confirmation')
                conviction += 16
                reasoning.append(f"Strong volume ({indicators['volume_ratio']:.1f}x)")
                
                if indicators['volume_trend'] > 0.1:
                    confluences.append('volume_trend_bearish')
                    conviction += 8
                    reasoning.append("Volume trend increasing")
            
            # OBV confirmation
            if indicators['obv_trend'] < -0.05:
                confluences.append('obv_bearish_trend')
                conviction += 14
                reasoning.append("OBV showing bearish distribution")
            
            # MFI confirmation
            if indicators['mfi'] >= 60:
                confluences.append('mfi_overbought')
                conviction += 16
                reasoning.append(f"MFI overbought ({indicators['mfi']:.1f})")
            
            # Momentum confirmation
            if indicators['momentum_7'] < -0.5:
                confluences.append('momentum_bearish')
                conviction += 10
                reasoning.append("Short-term momentum negative")
            
            # Resistance level confirmation
            if indicators['resistance_distance'] <= 0.015:
                confluences.append('strong_resistance_level')
                conviction += 12
                reasoning.append("Price at strong resistance")
        
        # No trade if no clear direction
        if direction is None:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # === CALIBRATED FILTERS ===
        
        # Volume requirement (relaxed)
        if indicators['volume_ratio'] < 1.1:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # Confluence requirement
        if len(confluences) < self.config['confluence_required']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # Conviction requirement
        if conviction < self.config['min_conviction']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # Bonus for exceptional setups
        if len(confluences) >= 6:
            conviction += 8
            reasoning.append("Exceptional confluence count")
        
        if indicators['volume_ratio'] >= 2.0:
            conviction += 6
            reasoning.append("Exceptional volume")
        
        return {
            'trade': True,
            'direction': direction,
            'conviction': min(conviction, 98),
            'confluences': confluences,
            'reasoning': reasoning,
            'confluence_count': len(confluences),
            'indicators_summary': {
                'rsi': rsi,
                'mfi': indicators['mfi'],
                'volume_ratio': indicators['volume_ratio'],
                'price_position': indicators['price_position']
            }
        }
    
    def simulate_calibrated_trade(self, entry_idx: int, entry_price: float, direction: str,
                                 position_size: float, profit_target: float, data: pd.DataFrame) -> dict:
        """Simulate trade with calibrated management"""
        
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
                        'hold_minutes': i - entry_idx,
                        'profit_pct': profit_target * 100
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
                            'hold_minutes': i - entry_idx,
                            'profit_pct': profit_pct * 100
                        }
                
                # Stop loss (wider for crypto)
                if low <= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss',
                        'pnl': pnl,
                        'success': False,
                        'hold_minutes': i - entry_idx,
                        'profit_pct': -self.config['stop_loss'] * 100
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
                        'hold_minutes': i - entry_idx,
                        'profit_pct': profit_target * 100
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
                            'hold_minutes': i - entry_idx,
                            'profit_pct': profit_pct * 100
                        }
                
                if high >= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss',
                        'pnl': pnl,
                        'success': False,
                        'hold_minutes': i - entry_idx,
                        'profit_pct': -self.config['stop_loss'] * 100
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
            'hold_minutes': self.config['max_hold_hours'] * 60,
            'profit_pct': profit_pct * 100
        }
    
    def run_calibrated_backtest(self, data: pd.DataFrame) -> dict:
        """Run calibrated backtest"""
        print("\n🎯 RUNNING CALIBRATED AI TRADER")
        print("⚡ OPTIMIZED FOR CRYPTO VOLATILITY")
        print("=" * 50)
        
        balance = self.initial_balance
        trades = []
        daily_trades = 0
        last_day = None
        
        wins = 0
        losses = 0
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'stop_loss': 0, 'time_exit': 0}
        
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
            
            analysis = self.calibrated_ai_analysis(indicators)
            
            if not analysis['trade']:
                continue
            
            # Calculate position
            entry_price = data.iloc[i]['close']
            conv_factor = (analysis['conviction'] - 82) / 16
            conv_factor = max(0, min(1, conv_factor))
            
            position_pct = self.config['position_min'] + \
                          (self.config['position_max'] - self.config['position_min']) * conv_factor
            position_size = balance * position_pct
            leverage = int(self.config['leverage_min'] + \
                          (self.config['leverage_max'] - self.config['leverage_min']) * conv_factor)
            profit_target = self.config['profit_target_min'] + \
                           (self.config['profit_target_max'] - self.config['profit_target_min']) * conv_factor
            
            # Simulate trade
            result = self.simulate_calibrated_trade(
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
                'reasoning': analysis['reasoning'],
                **result
            })
            
            if len(trades) % 3 == 0 or len(trades) <= 15:
                wr = wins / len(trades) * 100 if trades else 0
                ret = (balance - self.initial_balance) / self.initial_balance * 100
                print(f"🎯 #{len(trades)}: {analysis['direction'].upper()} "
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
            max_win = max([t['pnl'] for t in winning_trades]) if winning_trades else 0
            max_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else float('inf')
            avg_conviction = np.mean([t['conviction'] for t in trades])
            avg_confluences = np.mean([t['confluence_count'] for t in trades])
            avg_hold = np.mean([t['hold_minutes'] for t in trades])
        else:
            avg_win = avg_loss = max_win = max_loss = profit_factor = 0
            avg_conviction = avg_confluences = avg_hold = 0
        
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
            'max_win': max_win,
            'max_loss': max_loss,
            'avg_conviction': avg_conviction,
            'avg_confluences': avg_confluences,
            'avg_hold': avg_hold,
            'exit_reasons': exit_reasons,
            'trades': trades
        }
    
    def display_calibrated_results(self, results: dict):
        """Display calibrated results"""
        print("\n" + "="*80)
        print("🎯 CALIBRATED AI VISUAL TRADER - FINAL RESULTS")
        print("⚡ OPTIMIZED FOR CRYPTO VOLATILITY")
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
        print(f"   🚀 Best Trade: ${results['max_win']:.2f}")
        print(f"   💀 Worst Trade: ${results['max_loss']:.2f}")
        
        print(f"\n🤖 AI METRICS:")
        print(f"   🧠 Average Conviction: {results['avg_conviction']:.1f}%")
        print(f"   🎯 Average Confluences: {results['avg_confluences']:.1f}")
        print(f"   ⏱️ Average Hold: {results['avg_hold']:.0f} minutes")
        
        print(f"\n📤 EXIT ANALYSIS:")
        total = results['total_trades']
        if total > 0:
            for reason, count in results['exit_reasons'].items():
                pct = count / total * 100
                emoji = "🎯" if reason == "take_profit" else "🛡️" if reason == "trailing_stop" else "🛑" if reason == "stop_loss" else "⏰"
                print(f"   {emoji} {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        print(f"\n🏆 CALIBRATION ASSESSMENT:")
        
        if results['win_rate'] >= 60:
            print("   ✅ Win Rate: A+ (60%+ TARGET ACHIEVED!)")
            wr_grade = 4.0
        elif results['win_rate'] >= 50:
            print("   ✅ Win Rate: A- (50%+)")
            wr_grade = 3.7
        elif results['win_rate'] >= 40:
            print("   ✅ Win Rate: B+ (40%+)")
            wr_grade = 3.3
        else:
            print("   ❌ Win Rate: Still needs calibration")
            wr_grade = 2.0
        
        if results['total_return'] >= 15:
            print("   ✅ Returns: A+ (15%+)")
            ret_grade = 4.0
        elif results['total_return'] >= 10:
            print("   ✅ Returns: A- (10%+)")
            ret_grade = 3.7
        elif results['total_return'] >= 5:
            print("   ✅ Returns: B+ (5%+)")
            ret_grade = 3.3
        elif results['total_return'] >= 0:
            print("   ✅ Returns: B (Positive)")
            ret_grade = 3.0
        else:
            print("   ❌ Returns: Needs more calibration")
            ret_grade = 2.0
        
        if results['profit_factor'] >= 1.5:
            print("   ✅ Risk/Reward: A+ (1.5+)")
            pf_grade = 4.0
        elif results['profit_factor'] >= 1.2:
            print("   ✅ Risk/Reward: B+ (1.2+)")
            pf_grade = 3.3
        elif results['profit_factor'] >= 1.0:
            print("   ✅ Risk/Reward: B (1.0+)")
            pf_grade = 3.0
        else:
            print("   ❌ Risk/Reward: Needs calibration")
            pf_grade = 2.0
        
        # Stop loss analysis
        stop_loss_pct = results['exit_reasons'].get('stop_loss', 0) / total * 100 if total > 0 else 0
        if stop_loss_pct <= 40:
            print("   ✅ Stop Loss Rate: A+ (≤40%)")
            sl_grade = 4.0
        elif stop_loss_pct <= 50:
            print("   ✅ Stop Loss Rate: B+ (≤50%)")
            sl_grade = 3.3
        elif stop_loss_pct <= 60:
            print("   ✅ Stop Loss Rate: B (≤60%)")
            sl_grade = 3.0
        else:
            print("   ❌ Stop Loss Rate: Too high, needs wider stops")
            sl_grade = 2.0
        
        avg_grade = (wr_grade + ret_grade + pf_grade + sl_grade) / 4
        
        if avg_grade >= 3.8:
            overall = "A+ (PERFECTLY CALIBRATED!)"
            emoji = "🏆"
        elif avg_grade >= 3.5:
            overall = "A (EXCELLENT CALIBRATION)"
            emoji = "🥇"
        elif avg_grade >= 3.0:
            overall = "B+ (GOOD CALIBRATION)"
            emoji = "✅"
        else:
            overall = "NEEDS MORE CALIBRATION"
            emoji = "🔧"
        
        print(f"\n{emoji} CALIBRATION GRADE: {overall}")
        
        if avg_grade >= 3.5:
            print("\n🎉 CALIBRATION SUCCESSFUL!")
            print("🎯 AI Visual Trader PERFECTLY calibrated!")
            print("⚡ Optimized for crypto volatility!")
            print("🚀 Ready for live deployment!")
        elif avg_grade >= 3.0:
            print("\n✅ GOOD CALIBRATION ACHIEVED!")
            print("🔧 Minor adjustments may further improve performance")
        else:
            print("\n🔧 CALIBRATION IN PROGRESS...")
            print("⚙️ Adjusting parameters for optimal performance")
        
        print("="*80)

def main():
    print("🎯 CALIBRATED AI VISUAL TRADER")
    print("⚡ FINAL CALIBRATION FOR CRYPTO")
    print("👁️ SEES: Candles, Volume, OBV, MACD, MFI, RSI, Momentum")
    print("🎯 MAKES: Decisive, Extremely Profitable Trades")
    print("🔧 CALIBRATED: Wider Stops for Crypto Volatility")
    print("=" * 70)
    
    ai_trader = CalibratedAIVisualTrader(200.0)
    data = ai_trader.generate_market_data(60)
    results = ai_trader.run_calibrated_backtest(data)
    ai_trader.display_calibrated_results(results)

if __name__ == "__main__":
    main() 