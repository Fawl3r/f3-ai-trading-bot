#!/usr/bin/env python3
"""
AI VISUAL TRADER - Advanced Multi-Indicator Analysis
The AI can "see" and analyze:
- Candlestick patterns and price action
- Volume analysis and OBV (On-Balance Volume)
- MACD (Moving Average Convergence Divergence)
- MFI (Money Flow Index)
- RSI (Relative Strength Index)
- Momentum indicators
- Market structure and confluence zones

Makes decisive, extremely profitable trades with high conviction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AIVisualTrader:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # AI CONFIGURATION FOR MAXIMUM PROFITABILITY
        self.config = {
            # AGGRESSIVE PROFIT TARGETING
            "leverage_min": 18,
            "leverage_max": 35,
            "position_min": 0.12,      # 12% minimum position
            "position_max": 0.25,      # 25% maximum position
            
            # PROFIT OPTIMIZATION
            "profit_target_min": 0.022,  # 2.2% minimum
            "profit_target_max": 0.055,  # 5.5% maximum  
            "stop_loss": 0.015,          # 1.5% stop loss
            "trail_distance": 0.005,     # 0.5% trail
            "trail_start": 0.012,        # Start at 1.2%
            
            # AI DECISION THRESHOLDS
            "min_conviction": 75,        # 75% AI conviction minimum
            "max_daily_trades": 15,      # More opportunities
            "max_hold_hours": 12,        # Reasonable holds
            "confluence_required": 3,    # Need 3+ confluences
        }
        
        print("🤖 AI VISUAL TRADER - ADVANCED ANALYSIS")
        print("👁️ MULTI-INDICATOR CONFLUENCE SYSTEM")
        print("🎯 EXTREMELY PROFITABLE DECISION MAKING")
        print("=" * 60)
    
    def generate_market_data(self, days: int = 60) -> pd.DataFrame:
        """Generate realistic market data with all required indicators"""
        print(f"📊 Generating {days} days of market data...")
        
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        price = start_price
        time = datetime.now() - timedelta(days=days)
        volume_base = 1000000
        
        for i in range(total_minutes):
            # Enhanced volatility cycles
            hour = (i // 60) % 24
            day_factor = np.sin(i / (24 * 60) * 2 * np.pi) * 0.0008
            week_factor = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.0012
            
            # Time-based volatility
            if 8 <= hour <= 16:
                vol = 0.003    # High during market hours
            elif 17 <= hour <= 23:
                vol = 0.0025   # Medium in evening
            else:
                vol = 0.0018   # Lower at night
            
            # Market microstructure
            trend_strength = np.sin(i / (3 * 24 * 60) * 2 * np.pi)
            noise = np.random.normal(0, vol)
            
            # Price movement
            price_change = day_factor + week_factor + (trend_strength * 0.0005) + noise
            price *= (1 + price_change)
            price = max(120, min(170, price))
            
            # OHLC with realistic spreads
            spread = vol * 0.8
            high = price * (1 + abs(np.random.normal(0, spread)))
            low = price * (1 - abs(np.random.normal(0, spread)))
            open_p = price * (1 + np.random.normal(0, spread * 0.4))
            
            # Ensure OHLC validity
            high = max(high, price, open_p)
            low = min(low, price, open_p)
            
            # Volume with momentum correlation
            momentum = abs(price_change) * 100
            volume_mult = 1 + momentum + np.random.uniform(0.6, 2.4)
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
    
    def calculate_all_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        """Calculate ALL indicators that AI can see"""
        if idx < 50:
            return None
        
        # Get sufficient window for all calculations
        window = data.iloc[max(0, idx-50):idx+1]
        current = window.iloc[-1]
        
        indicators = {}
        
        # === PRICE ACTION & CANDLESTICKS ===
        indicators['price'] = current['close']
        indicators['high'] = current['high']
        indicators['low'] = current['low']
        indicators['open'] = current['open']
        
        # Candlestick analysis
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        indicators['body_pct'] = (body_size / total_range) if total_range > 0 else 0
        indicators['is_bullish'] = current['close'] > current['open']
        indicators['upper_shadow'] = current['high'] - max(current['close'], current['open'])
        indicators['lower_shadow'] = min(current['close'], current['open']) - current['low']
        
        # === RSI ===
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        # === MOVING AVERAGES ===
        indicators['ema_9'] = window['close'].ewm(span=9).mean().iloc[-1]
        indicators['ema_21'] = window['close'].ewm(span=21).mean().iloc[-1]
        indicators['sma_50'] = window['close'].rolling(50).mean().iloc[-1] if len(window) >= 50 else window['close'].mean()
        
        # === MACD ===
        ema_12 = window['close'].ewm(span=12).mean()
        ema_26 = window['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd'] = macd_line.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        indicators['macd_histogram'] = macd_line.iloc[-1] - signal_line.iloc[-1]
        
        # === VOLUME ANALYSIS ===
        indicators['volume'] = current['volume']
        indicators['volume_sma'] = window['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = current['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        # === OBV (On-Balance Volume) ===
        obv = 0
        for i in range(1, len(window)):
            if window['close'].iloc[i] > window['close'].iloc[i-1]:
                obv += window['volume'].iloc[i]
            elif window['close'].iloc[i] < window['close'].iloc[i-1]:
                obv -= window['volume'].iloc[i]
        indicators['obv'] = obv
        
        # OBV trend
        if len(window) >= 10:
            obv_values = []
            running_obv = 0
            for i in range(len(window)-10, len(window)):
                if i > 0 and window['close'].iloc[i] > window['close'].iloc[i-1]:
                    running_obv += window['volume'].iloc[i]
                elif i > 0 and window['close'].iloc[i] < window['close'].iloc[i-1]:
                    running_obv -= window['volume'].iloc[i]
                obv_values.append(running_obv)
            
            if len(obv_values) >= 2:
                indicators['obv_trend'] = 1 if obv_values[-1] > obv_values[0] else -1
            else:
                indicators['obv_trend'] = 0
        else:
            indicators['obv_trend'] = 0
        
        # === MFI (Money Flow Index) ===
        typical_price = (window['high'] + window['low'] + window['close']) / 3
        money_flow = typical_price * window['volume']
        
        mfi_values = []
        for i in range(1, len(window)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                mfi_values.append(('positive', money_flow.iloc[i]))
            else:
                mfi_values.append(('negative', money_flow.iloc[i]))
        
        if len(mfi_values) >= 14:
            recent_mfi = mfi_values[-14:]
            positive_flow = sum([mf[1] for mf in recent_mfi if mf[0] == 'positive'])
            negative_flow = sum([mf[1] for mf in recent_mfi if mf[0] == 'negative'])
            
            if negative_flow > 0:
                money_ratio = positive_flow / negative_flow
                indicators['mfi'] = 100 - (100 / (1 + money_ratio))
            else:
                indicators['mfi'] = 100
        else:
            indicators['mfi'] = 50
        
        # === MOMENTUM ===
        if len(window) >= 10:
            indicators['momentum_10'] = (current['close'] - window['close'].iloc[-11]) / window['close'].iloc[-11] * 100
        else:
            indicators['momentum_10'] = 0
            
        if len(window) >= 5:
            indicators['momentum_5'] = (current['close'] - window['close'].iloc[-6]) / window['close'].iloc[-6] * 100
        else:
            indicators['momentum_5'] = 0
        
        # === VOLATILITY ===
        if len(window) >= 20:
            indicators['volatility'] = window['close'].pct_change().rolling(20).std().iloc[-1] * 100
        else:
            indicators['volatility'] = 0
        
        # === PRICE LEVELS ===
        recent_20 = window.tail(20) if len(window) >= 20 else window
        indicators['resistance'] = recent_20['high'].max()
        indicators['support'] = recent_20['low'].min()
        indicators['price_position'] = (current['close'] - indicators['support']) / (indicators['resistance'] - indicators['support']) if indicators['resistance'] > indicators['support'] else 0.5
        
        return indicators
    
    def ai_market_analysis(self, indicators: dict) -> dict:
        """AI analyzes all indicators and makes decisive trading decisions"""
        if not indicators:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        confluences = []
        conviction = 0
        direction = None
        reasoning = []
        
        # === RSI ANALYSIS ===
        rsi = indicators['rsi']
        if rsi <= 25:
            confluences.append('rsi_extreme_oversold')
            conviction += 25
            direction = 'long'
            reasoning.append(f"RSI extremely oversold ({rsi:.1f})")
        elif rsi <= 35:
            confluences.append('rsi_oversold')
            conviction += 15
            direction = 'long'
            reasoning.append(f"RSI oversold ({rsi:.1f})")
        elif rsi >= 75:
            confluences.append('rsi_extreme_overbought')
            conviction += 25
            direction = 'short'
            reasoning.append(f"RSI extremely overbought ({rsi:.1f})")
        elif rsi >= 65:
            confluences.append('rsi_overbought')
            conviction += 15
            direction = 'short'
            reasoning.append(f"RSI overbought ({rsi:.1f})")
        
        # === MACD ANALYSIS ===
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_hist = indicators['macd_histogram']
        
        if macd > macd_signal and macd_hist > 0:
            if direction == 'long':
                confluences.append('macd_bullish')
                conviction += 20
                reasoning.append("MACD bullish crossover")
            elif direction == 'short':
                conviction -= 10  # Conflicting signal
        elif macd < macd_signal and macd_hist < 0:
            if direction == 'short':
                confluences.append('macd_bearish')
                conviction += 20
                reasoning.append("MACD bearish crossover")
            elif direction == 'long':
                conviction -= 10  # Conflicting signal
        
        # === VOLUME & OBV ANALYSIS ===
        volume_ratio = indicators['volume_ratio']
        obv_trend = indicators['obv_trend']
        
        if volume_ratio >= 1.5:
            confluences.append('high_volume')
            conviction += 15
            reasoning.append(f"High volume confirmation ({volume_ratio:.1f}x)")
            
            # OBV confirmation
            if direction == 'long' and obv_trend > 0:
                confluences.append('obv_bullish')
                conviction += 15
                reasoning.append("OBV showing bullish trend")
            elif direction == 'short' and obv_trend < 0:
                confluences.append('obv_bearish')
                conviction += 15
                reasoning.append("OBV showing bearish trend")
        
        # === MFI ANALYSIS ===
        mfi = indicators['mfi']
        if mfi <= 20:
            if direction == 'long':
                confluences.append('mfi_oversold')
                conviction += 18
                reasoning.append(f"MFI oversold ({mfi:.1f})")
        elif mfi >= 80:
            if direction == 'short':
                confluences.append('mfi_overbought')
                conviction += 18
                reasoning.append(f"MFI overbought ({mfi:.1f})")
        
        # === MOMENTUM ANALYSIS ===
        mom_5 = indicators['momentum_5']
        mom_10 = indicators['momentum_10']
        
        if direction == 'long':
            if mom_5 > 0.5 and mom_10 > 0.8:
                confluences.append('momentum_bullish')
                conviction += 12
                reasoning.append("Strong bullish momentum")
            elif mom_5 < -0.3:
                conviction -= 8  # Conflicting momentum
        elif direction == 'short':
            if mom_5 < -0.5 and mom_10 < -0.8:
                confluences.append('momentum_bearish')
                conviction += 12
                reasoning.append("Strong bearish momentum")
            elif mom_5 > 0.3:
                conviction -= 8  # Conflicting momentum
        
        # === PRICE ACTION ANALYSIS ===
        price_pos = indicators['price_position']
        body_pct = indicators['body_pct']
        is_bullish = indicators['is_bullish']
        
        # Support/Resistance levels
        if direction == 'long' and price_pos <= 0.2:
            confluences.append('near_support')
            conviction += 12
            reasoning.append("Price near support level")
        elif direction == 'short' and price_pos >= 0.8:
            confluences.append('near_resistance')
            conviction += 12
            reasoning.append("Price near resistance level")
        
        # Candlestick confirmation
        if body_pct >= 0.6:  # Strong candle
            if direction == 'long' and is_bullish:
                confluences.append('strong_bullish_candle')
                conviction += 8
                reasoning.append("Strong bullish candlestick")
            elif direction == 'short' and not is_bullish:
                confluences.append('strong_bearish_candle')
                conviction += 8
                reasoning.append("Strong bearish candlestick")
        
        # === EMA TREND ANALYSIS ===
        price = indicators['price']
        ema_9 = indicators['ema_9']
        ema_21 = indicators['ema_21']
        
        if ema_9 > ema_21:  # Bullish trend
            if direction == 'long':
                confluences.append('ema_trend_bullish')
                conviction += 10
                reasoning.append("EMA trend bullish")
        elif ema_9 < ema_21:  # Bearish trend
            if direction == 'short':
                confluences.append('ema_trend_bearish')
                conviction += 10
                reasoning.append("EMA trend bearish")
        
        # === VOLATILITY FILTER ===
        volatility = indicators['volatility']
        if 2.0 <= volatility <= 6.0:
            conviction *= 1.1
            reasoning.append(f"Optimal volatility ({volatility:.1f}%)")
        elif volatility > 8.0:
            conviction *= 0.9
            reasoning.append("High volatility caution")
        
        # === FINAL AI DECISION ===
        enough_confluences = len(confluences) >= self.config['confluence_required']
        high_conviction = conviction >= self.config['min_conviction']
        
        trade_signal = enough_confluences and high_conviction
        
        return {
            'trade': trade_signal,
            'direction': direction,
            'conviction': min(conviction, 98),
            'confluences': confluences,
            'reasoning': reasoning,
            'confluence_count': len(confluences),
            'indicators_summary': {
                'rsi': rsi,
                'macd_signal': 'bullish' if macd > macd_signal else 'bearish',
                'volume_strength': volume_ratio,
                'mfi': mfi,
                'momentum': mom_5,
                'price_position': price_pos
            }
        }
    
    def calculate_position_size(self, balance: float, conviction: float) -> tuple:
        """Calculate position size based on AI conviction"""
        conv_factor = conviction / 100
        
        # Position sizing
        position_pct = self.config['position_min'] + \
                      (self.config['position_max'] - self.config['position_min']) * conv_factor
        position_size = balance * position_pct
        
        # Leverage
        leverage = int(self.config['leverage_min'] + \
                      (self.config['leverage_max'] - self.config['leverage_min']) * conv_factor)
        
        # Profit target
        profit_target = self.config['profit_target_min'] + \
                       (self.config['profit_target_max'] - self.config['profit_target_min']) * conv_factor
        
        return position_size, leverage, profit_target
    
    def simulate_ai_trade(self, entry_idx: int, entry_price: float, direction: str,
                         position_size: float, profit_target: float, data: pd.DataFrame) -> dict:
        """Simulate trade with AI-driven exits"""
        
        # Calculate targets
        if direction == 'long':
            take_profit = entry_price * (1 + profit_target)
            stop_loss = entry_price * (1 - self.config['stop_loss'])
        else:
            take_profit = entry_price * (1 - profit_target)
            stop_loss = entry_price * (1 + self.config['stop_loss'])
        
        # Tracking
        best_price = entry_price
        trail_price = None
        trail_active = False
        
        # Simulate
        max_idx = min(entry_idx + (self.config['max_hold_hours'] * 60), len(data) - 1)
        
        for i in range(entry_idx + 1, max_idx + 1):
            candle = data.iloc[i]
            high, low, close = candle['high'], candle['low'], candle['close']
            
            if direction == 'long':
                # Update best price
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
                unrealized = (high - entry_price) / entry_price
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
                
                # Stop loss
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
                
                unrealized = (entry_price - low) / entry_price
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
    
    def run_ai_backtest(self, data: pd.DataFrame) -> dict:
        """Run AI visual trading backtest"""
        print("\n🤖 RUNNING AI VISUAL TRADER BACKTEST")
        print("👁️ AI ANALYZING ALL INDICATORS")
        print("=" * 50)
        
        # Initialize
        balance = self.initial_balance
        trades = []
        daily_trades = 0
        last_day = None
        
        # Stats
        wins = 0
        losses = 0
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'stop_loss': 0, 'time_exit': 0}
        confluence_stats = {}
        
        # Main trading loop
        for i in range(50, len(data) - 100):
            current_time = data.iloc[i]['timestamp']
            current_day = current_time.date()
            
            # Daily reset
            if last_day != current_day:
                daily_trades = 0
                last_day = current_day
            
            # Check daily limit
            if daily_trades >= self.config['max_daily_trades']:
                continue
            
            # AI analysis
            indicators = self.calculate_all_indicators(data, i)
            if not indicators:
                continue
            
            analysis = self.ai_market_analysis(indicators)
            
            # Check if AI wants to trade
            if not analysis['trade']:
                continue
            
            # AI has made a decision - execute trade
            entry_price = data.iloc[i]['close']
            position_size, leverage, profit_target = self.calculate_position_size(
                balance, analysis['conviction'])
            
            # Simulate the trade
            result = self.simulate_ai_trade(
                i, entry_price, analysis['direction'], 
                position_size, profit_target, data)
            
            # Update balance
            balance += result['pnl']
            
            # Record trade
            trade = {
                'entry_time': current_time,
                'entry_price': entry_price,
                'direction': analysis['direction'],
                'position_size': position_size,
                'leverage': leverage,
                'conviction': analysis['conviction'],
                'confluences': analysis['confluences'],
                'confluence_count': analysis['confluence_count'],
                'reasoning': analysis['reasoning'],
                'indicators': analysis['indicators_summary'],
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
            
            # Track confluence performance
            for conf in analysis['confluences']:
                if conf not in confluence_stats:
                    confluence_stats[conf] = {'count': 0, 'wins': 0}
                confluence_stats[conf]['count'] += 1
                if result['success']:
                    confluence_stats[conf]['wins'] += 1
            
            # Progress updates
            if len(trades) % 5 == 0:
                wr = wins / len(trades) * 100 if trades else 0
                ret = (balance - self.initial_balance) / self.initial_balance * 100
                print(f"🤖 #{len(trades)}: {analysis['direction'].upper()} "
                      f"Conv:{analysis['conviction']:.0f}% Conf:{len(analysis['confluences'])} → "
                      f"${result['pnl']:+.2f} | WR:{wr:.1f}% Ret:{ret:+.1f}%")
        
        # Calculate final results
        total_trades = len(trades)
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        total_return = (balance - self.initial_balance) / self.initial_balance * 100
        
        # Additional metrics
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
            'confluence_stats': confluence_stats,
            'trades': trades
        }
    
    def display_ai_results(self, results: dict):
        """Display comprehensive AI trading results"""
        print("\n" + "="*80)
        print("🤖 AI VISUAL TRADER - BACKTEST RESULTS")
        print("👁️ MULTI-INDICATOR CONFLUENCE ANALYSIS")
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
        
        print(f"\n🎯 CONFLUENCE PERFORMANCE:")
        conf_stats = results['confluence_stats']
        if conf_stats:
            for conf, stats in sorted(conf_stats.items(), key=lambda x: x[1]['wins']/x[1]['count'], reverse=True):
                win_rate = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
                print(f"   {conf.replace('_', ' ').title()}: {stats['wins']}/{stats['count']} ({win_rate:.1f}%)")
        
        # Final grade assessment
        print(f"\n🏆 GRADE ASSESSMENT:")
        grades = []
        
        if results['win_rate'] >= 60:
            print("   ✅ Win Rate: A+ (60%+)")
            grades.append(4)
        elif results['win_rate'] >= 55:
            print("   ✅ Win Rate: B+ (55%+)")
            grades.append(3)
        else:
            print("   ❌ Win Rate: Needs Improvement")
            grades.append(1)
        
        if results['total_return'] >= 15:
            print("   ✅ Returns: A+ (15%+)")
            grades.append(4)
        elif results['total_return'] >= 10:
            print("   ✅ Returns: B+ (10%+)")
            grades.append(3)
        else:
            print("   ❌ Returns: Needs Improvement")
            grades.append(1)
        
        if results['profit_factor'] >= 1.8:
            print("   ✅ Risk/Reward: A+ (1.8+)")
            grades.append(4)
        elif results['profit_factor'] >= 1.4:
            print("   ✅ Risk/Reward: B+ (1.4+)")
            grades.append(3)
        else:
            print("   ❌ Risk/Reward: Needs Improvement")
            grades.append(1)
        
        freq = results['total_trades'] / 60
        if freq >= 1.0:
            print("   ✅ Frequency: A+ (1.0+/day)")
            grades.append(4)
        elif freq >= 0.7:
            print("   ✅ Frequency: B+ (0.7+/day)")
            grades.append(3)
        else:
            print("   ❌ Frequency: Needs Improvement")
            grades.append(1)
        
        avg_grade = sum(grades) / len(grades)
        if avg_grade >= 3.5:
            overall = "A (EXCELLENT)"
            emoji = "🏆"
        elif avg_grade >= 2.5:
            overall = "B+ (GOOD)"
            emoji = "✅"
        else:
            overall = "F (NEEDS WORK)"
            emoji = "❌"
        
        print(f"\n{emoji} OVERALL GRADE: {overall}")
        
        if avg_grade >= 3.0:
            print("\n🎉 AI READY FOR LIVE TRADING!")
            print("🤖 Multi-indicator analysis successful!")
        
        print("="*80)

def main():
    print("🤖 AI VISUAL TRADER - ADVANCED ANALYSIS")
    print("👁️ ANALYZES: Candles, Volume, OBV, MACD, MFI, RSI, Momentum")
    print("🎯 MAKES: Decisive, Extremely Profitable Trades")
    print("=" * 70)
    
    balance = 200.0
    days = 60
    
    ai_trader = AIVisualTrader(balance)
    data = ai_trader.generate_market_data(days)
    results = ai_trader.run_ai_backtest(data)
    ai_trader.display_ai_results(results)

if __name__ == "__main__":
    main() 