#!/usr/bin/env python3
"""
OPTIMIZED AI VISUAL TRADER - Extreme Performance
Enhanced multi-indicator analysis with optimized parameters
Target: 65%+ Win Rate, 20%+ Returns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OptimizedAIVisualTrader:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # OPTIMIZED CONFIGURATION FOR HIGH WIN RATE
        self.config = {
            # CONSERVATIVE POSITION SIZING
            "leverage_min": 8,
            "leverage_max": 15,
            "position_min": 0.08,      # 8% minimum position
            "position_max": 0.18,      # 18% maximum position
            
            # OPTIMIZED PROFIT/LOSS
            "profit_target_min": 0.015,  # 1.5% minimum
            "profit_target_max": 0.035,  # 3.5% maximum  
            "stop_loss": 0.008,          # 0.8% stop loss (tighter)
            "trail_distance": 0.003,     # 0.3% trail
            "trail_start": 0.008,        # Start at 0.8%
            
            # STRICTER AI THRESHOLDS
            "min_conviction": 82,        # 82% AI conviction minimum
            "max_daily_trades": 8,       # Quality over quantity
            "max_hold_hours": 6,         # Shorter holds
            "confluence_required": 4,    # Need 4+ confluences
            "volume_threshold": 1.3,     # Volume confirmation
        }
        
        print("🚀 OPTIMIZED AI VISUAL TRADER")
        print("🎯 TARGET: 65%+ WIN RATE, 20%+ RETURNS")
        print("🧠 ENHANCED CONFLUENCE ANALYSIS")
        print("=" * 60)
    
    def generate_market_data(self, days: int = 60) -> pd.DataFrame:
        """Generate realistic market data with all required indicators"""
        print(f"📊 Generating {days} days of optimized market data...")
        
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        price = start_price
        time = datetime.now() - timedelta(days=days)
        volume_base = 1200000
        
        for i in range(total_minutes):
            # Market cycles with clearer trends
            hour = (i // 60) % 24
            day_factor = np.sin(i / (24 * 60) * 2 * np.pi) * 0.0006
            week_factor = np.sin(i / (5 * 24 * 60) * 2 * np.pi) * 0.001
            
            # Time-based volatility
            if 8 <= hour <= 16:
                vol = 0.0025    # Market hours
            elif 17 <= hour <= 23:
                vol = 0.002     # Evening
            else:
                vol = 0.0015    # Night
            
            # Trend with momentum
            trend_cycle = np.sin(i / (2 * 24 * 60) * 2 * np.pi)
            momentum = trend_cycle * 0.0004
            noise = np.random.normal(0, vol * 0.8)
            
            # Price movement
            price_change = day_factor + week_factor + momentum + noise
            price *= (1 + price_change)
            price = max(125, min(165, price))
            
            # OHLC with realistic behavior
            spread = vol * 0.6
            high = price * (1 + abs(np.random.normal(0, spread * 0.7)))
            low = price * (1 - abs(np.random.normal(0, spread * 0.7)))
            open_p = price * (1 + np.random.normal(0, spread * 0.3))
            
            # Ensure OHLC validity
            high = max(high, price, open_p)
            low = min(low, price, open_p)
            
            # Volume with better correlation
            vol_momentum = abs(price_change) * 150
            volume_mult = 1 + vol_momentum + np.random.uniform(0.7, 1.8)
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
        """Calculate ALL indicators with enhanced accuracy"""
        if idx < 60:
            return None
        
        # Get sufficient window
        window = data.iloc[max(0, idx-60):idx+1]
        current = window.iloc[-1]
        
        indicators = {}
        
        # === ENHANCED PRICE ACTION ===
        indicators['price'] = current['close']
        indicators['high'] = current['high']
        indicators['low'] = current['low']
        indicators['open'] = current['open']
        
        # Advanced candlestick analysis
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        indicators['body_pct'] = (body_size / total_range) if total_range > 0 else 0
        indicators['is_bullish'] = current['close'] > current['open']
        indicators['upper_shadow'] = current['high'] - max(current['close'], current['open'])
        indicators['lower_shadow'] = min(current['close'], current['open']) - current['low']
        
        # Doji detection
        indicators['is_doji'] = indicators['body_pct'] < 0.1
        indicators['is_hammer'] = indicators['lower_shadow'] > body_size * 2 and indicators['upper_shadow'] < body_size * 0.5
        indicators['is_shooting_star'] = indicators['upper_shadow'] > body_size * 2 and indicators['lower_shadow'] < body_size * 0.5
        
        # === ENHANCED RSI ===
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
        indicators['sma_20'] = window['close'].rolling(20).mean().iloc[-1]
        
        # EMA alignment
        indicators['ema_bullish_align'] = indicators['ema_9'] > indicators['ema_21'] > indicators['ema_50']
        indicators['ema_bearish_align'] = indicators['ema_9'] < indicators['ema_21'] < indicators['ema_50']
        
        # === ENHANCED MACD ===
        ema_12 = window['close'].ewm(span=12).mean()
        ema_26 = window['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd'] = macd_line.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        indicators['macd_histogram'] = macd_line.iloc[-1] - signal_line.iloc[-1]
        
        # MACD momentum
        if len(macd_line) >= 3:
            indicators['macd_momentum'] = macd_line.iloc[-1] - macd_line.iloc[-3]
        else:
            indicators['macd_momentum'] = 0
        
        # === ENHANCED VOLUME ===
        indicators['volume'] = current['volume']
        indicators['volume_sma'] = window['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = current['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        # Volume trend
        if len(window) >= 5:
            recent_vol = window['volume'].tail(5).mean()
            prev_vol = window['volume'].iloc[-10:-5].mean() if len(window) >= 10 else recent_vol
            indicators['volume_trend'] = 1 if recent_vol > prev_vol else -1
        else:
            indicators['volume_trend'] = 0
        
        # === ENHANCED OBV ===
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
            indicators['obv_trend'] = 1 if recent_obv > prev_obv else -1
        else:
            indicators['obv_trend'] = 0
        
        # === ENHANCED MFI ===
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
        
        # === ENHANCED MOMENTUM ===
        if len(window) >= 14:
            indicators['momentum_14'] = (current['close'] - window['close'].iloc[-15]) / window['close'].iloc[-15] * 100
        else:
            indicators['momentum_14'] = 0
            
        if len(window) >= 7:
            indicators['momentum_7'] = (current['close'] - window['close'].iloc[-8]) / window['close'].iloc[-8] * 100
        else:
            indicators['momentum_7'] = 0
        
        # Rate of change
        if len(window) >= 10:
            indicators['roc_10'] = ((current['close'] - window['close'].iloc[-11]) / window['close'].iloc[-11]) * 100
        else:
            indicators['roc_10'] = 0
        
        # === ENHANCED VOLATILITY ===
        if len(window) >= 20:
            indicators['volatility'] = window['close'].pct_change().rolling(20).std().iloc[-1] * 100
            indicators['atr'] = ((window['high'] - window['low']).rolling(14).mean().iloc[-1] / current['close']) * 100
        else:
            indicators['volatility'] = 0
            indicators['atr'] = 0
        
        # === ENHANCED SUPPORT/RESISTANCE ===
        recent_30 = window.tail(30) if len(window) >= 30 else window
        indicators['resistance'] = recent_30['high'].max()
        indicators['support'] = recent_30['low'].min()
        indicators['price_position'] = (current['close'] - indicators['support']) / (indicators['resistance'] - indicators['support']) if indicators['resistance'] > indicators['support'] else 0.5
        
        # Pivot points
        yesterday = window.iloc[-2] if len(window) >= 2 else window.iloc[-1]
        pivot = (yesterday['high'] + yesterday['low'] + yesterday['close']) / 3
        indicators['pivot'] = pivot
        indicators['r1'] = 2 * pivot - yesterday['low']
        indicators['s1'] = 2 * pivot - yesterday['high']
        
        return indicators
    
    def enhanced_ai_analysis(self, indicators: dict) -> dict:
        """Enhanced AI analysis with optimized confluences"""
        if not indicators:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        confluences = []
        conviction = 0
        direction = None
        reasoning = []
        
        # === OPTIMIZED RSI ANALYSIS ===
        rsi = indicators['rsi']
        rsi_mom = indicators['rsi_momentum']
        
        if rsi <= 22:
            confluences.append('rsi_extreme_oversold')
            conviction += 28
            direction = 'long'
            reasoning.append(f"RSI extremely oversold ({rsi:.1f})")
            
            # RSI momentum confirmation
            if rsi_mom > 0:
                confluences.append('rsi_momentum_reversal')
                conviction += 12
                reasoning.append("RSI showing reversal momentum")
                
        elif rsi <= 32:
            confluences.append('rsi_oversold')
            conviction += 18
            direction = 'long'
            reasoning.append(f"RSI oversold ({rsi:.1f})")
            
        elif rsi >= 78:
            confluences.append('rsi_extreme_overbought')
            conviction += 28
            direction = 'short'
            reasoning.append(f"RSI extremely overbought ({rsi:.1f})")
            
            if rsi_mom < 0:
                confluences.append('rsi_momentum_reversal')
                conviction += 12
                reasoning.append("RSI showing reversal momentum")
                
        elif rsi >= 68:
            confluences.append('rsi_overbought')
            conviction += 18
            direction = 'short'
            reasoning.append(f"RSI overbought ({rsi:.1f})")
        
        # === ENHANCED MACD ANALYSIS ===
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_hist = indicators['macd_histogram']
        macd_mom = indicators['macd_momentum']
        
        # Strong MACD signals
        if macd > macd_signal and macd_hist > 0:
            if direction == 'long':
                confluences.append('macd_bullish')
                conviction += 22
                reasoning.append("MACD bullish crossover")
                
                if macd_mom > 0:
                    confluences.append('macd_momentum_strong')
                    conviction += 8
                    reasoning.append("MACD momentum accelerating")
                    
        elif macd < macd_signal and macd_hist < 0:
            if direction == 'short':
                confluences.append('macd_bearish')
                conviction += 22
                reasoning.append("MACD bearish crossover")
                
                if macd_mom < 0:
                    confluences.append('macd_momentum_strong')
                    conviction += 8
                    reasoning.append("MACD momentum accelerating")
        
        # === ENHANCED VOLUME ANALYSIS ===
        volume_ratio = indicators['volume_ratio']
        volume_trend = indicators['volume_trend']
        
        if volume_ratio >= self.config['volume_threshold']:
            confluences.append('volume_confirmation')
            conviction += 20
            reasoning.append(f"Strong volume confirmation ({volume_ratio:.1f}x)")
            
            # Volume trend confirmation
            if direction == 'long' and volume_trend > 0:
                confluences.append('volume_trend_bullish')
                conviction += 10
                reasoning.append("Volume trend supporting bullish move")
            elif direction == 'short' and volume_trend > 0:
                confluences.append('volume_trend_bearish')
                conviction += 10
                reasoning.append("Volume trend supporting bearish move")
        
        # === ENHANCED OBV ANALYSIS ===
        obv_trend = indicators['obv_trend']
        if direction == 'long' and obv_trend > 0:
            confluences.append('obv_bullish_divergence')
            conviction += 16
            reasoning.append("OBV confirming bullish sentiment")
        elif direction == 'short' and obv_trend < 0:
            confluences.append('obv_bearish_divergence')
            conviction += 16
            reasoning.append("OBV confirming bearish sentiment")
        
        # === ENHANCED MFI ANALYSIS ===
        mfi = indicators['mfi']
        if mfi <= 18:
            if direction == 'long':
                confluences.append('mfi_oversold')
                conviction += 20
                reasoning.append(f"MFI extreme oversold ({mfi:.1f})")
        elif mfi >= 82:
            if direction == 'short':
                confluences.append('mfi_overbought')
                conviction += 20
                reasoning.append(f"MFI extreme overbought ({mfi:.1f})")
        
        # === ENHANCED MOMENTUM ANALYSIS ===
        mom_7 = indicators['momentum_7']
        mom_14 = indicators['momentum_14']
        roc = indicators['roc_10']
        
        if direction == 'long':
            if mom_7 > 0.3 and mom_14 > 0.5:
                confluences.append('momentum_bullish_strong')
                conviction += 15
                reasoning.append("Strong bullish momentum")
            elif roc > 0.8:
                confluences.append('rate_of_change_bullish')
                conviction += 10
                reasoning.append("Positive rate of change")
                
        elif direction == 'short':
            if mom_7 < -0.3 and mom_14 < -0.5:
                confluences.append('momentum_bearish_strong')
                conviction += 15
                reasoning.append("Strong bearish momentum")
            elif roc < -0.8:
                confluences.append('rate_of_change_bearish')
                conviction += 10
                reasoning.append("Negative rate of change")
        
        # === ENHANCED PRICE ACTION ===
        price_pos = indicators['price_position']
        body_pct = indicators['body_pct']
        is_bullish = indicators['is_bullish']
        is_hammer = indicators['is_hammer']
        is_shooting_star = indicators['is_shooting_star']
        
        # Support/Resistance with tighter zones
        if direction == 'long' and price_pos <= 0.15:
            confluences.append('price_near_support')
            conviction += 18
            reasoning.append("Price at strong support zone")
        elif direction == 'short' and price_pos >= 0.85:
            confluences.append('price_near_resistance')
            conviction += 18
            reasoning.append("Price at strong resistance zone")
        
        # Enhanced candlestick patterns
        if direction == 'long':
            if is_hammer:
                confluences.append('hammer_reversal')
                conviction += 15
                reasoning.append("Hammer reversal pattern")
            elif body_pct >= 0.7 and is_bullish:
                confluences.append('strong_bullish_candle')
                conviction += 12
                reasoning.append("Strong bullish candlestick")
                
        elif direction == 'short':
            if is_shooting_star:
                confluences.append('shooting_star_reversal')
                conviction += 15
                reasoning.append("Shooting star reversal pattern")
            elif body_pct >= 0.7 and not is_bullish:
                confluences.append('strong_bearish_candle')
                conviction += 12
                reasoning.append("Strong bearish candlestick")
        
        # === EMA TREND ANALYSIS ===
        ema_bullish = indicators['ema_bullish_align']
        ema_bearish = indicators['ema_bearish_align']
        
        if direction == 'long' and ema_bullish:
            confluences.append('ema_trend_aligned')
            conviction += 14
            reasoning.append("EMA trend perfectly aligned bullish")
        elif direction == 'short' and ema_bearish:
            confluences.append('ema_trend_aligned')
            conviction += 14
            reasoning.append("EMA trend perfectly aligned bearish")
        
        # === VOLATILITY OPTIMIZATION ===
        volatility = indicators['volatility']
        atr = indicators['atr']
        
        if 1.5 <= volatility <= 4.5 and 0.8 <= atr <= 2.5:
            conviction *= 1.15
            reasoning.append(f"Optimal volatility environment ({volatility:.1f}%)")
        elif volatility > 7.0:
            conviction *= 0.85
            reasoning.append("High volatility caution applied")
        
        # === CONFLUENCE QUALITY BOOST ===
        if len(confluences) >= 5:
            conviction *= 1.1
            reasoning.append("Exceptional confluence alignment")
        elif len(confluences) >= 4:
            conviction *= 1.05
            reasoning.append("Strong confluence alignment")
        
        # === FINAL OPTIMIZED DECISION ===
        enough_confluences = len(confluences) >= self.config['confluence_required']
        high_conviction = conviction >= self.config['min_conviction']
        
        trade_signal = enough_confluences and high_conviction
        
        return {
            'trade': trade_signal,
            'direction': direction,
            'conviction': min(conviction, 95),
            'confluences': confluences,
            'reasoning': reasoning,
            'confluence_count': len(confluences),
            'indicators_summary': {
                'rsi': rsi,
                'macd_signal': 'bullish' if macd > macd_signal else 'bearish',
                'volume_strength': volume_ratio,
                'mfi': mfi,
                'momentum_7': mom_7,
                'price_position': price_pos,
                'volatility': volatility
            }
        }
    
    def calculate_position_size(self, balance: float, conviction: float) -> tuple:
        """Optimized position sizing"""
        conv_factor = (conviction - 80) / 15  # Scale from 80-95% conviction
        conv_factor = max(0, min(1, conv_factor))
        
        # Conservative position sizing
        position_pct = self.config['position_min'] + \
                      (self.config['position_max'] - self.config['position_min']) * conv_factor
        position_size = balance * position_pct
        
        # Moderate leverage
        leverage = int(self.config['leverage_min'] + \
                      (self.config['leverage_max'] - self.config['leverage_min']) * conv_factor)
        
        # Optimized profit targets
        profit_target = self.config['profit_target_min'] + \
                       (self.config['profit_target_max'] - self.config['profit_target_min']) * conv_factor
        
        return position_size, leverage, profit_target
    
    def simulate_optimized_trade(self, entry_idx: int, entry_price: float, direction: str,
                                position_size: float, profit_target: float, data: pd.DataFrame) -> dict:
        """Optimized trade simulation with better exits"""
        
        # Calculate targets
        if direction == 'long':
            take_profit = entry_price * (1 + profit_target)
            stop_loss = entry_price * (1 - self.config['stop_loss'])
        else:
            take_profit = entry_price * (1 - profit_target)
            stop_loss = entry_price * (1 + self.config['stop_loss'])
        
        # Enhanced tracking
        best_price = entry_price
        trail_price = None
        trail_active = False
        
        # Simulate with shorter max hold
        max_idx = min(entry_idx + (self.config['max_hold_hours'] * 60), len(data) - 1)
        
        for i in range(entry_idx + 1, max_idx + 1):
            candle = data.iloc[i]
            high, low, close = candle['high'], candle['low'], candle['close']
            
            if direction == 'long':
                if high > best_price:
                    best_price = high
                
                # Take profit hit
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
                
                # Enhanced trailing stop
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
                
                # Tight stop loss
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
    
    def run_optimized_backtest(self, data: pd.DataFrame) -> dict:
        """Run optimized AI backtest"""
        print("\n🚀 RUNNING OPTIMIZED AI VISUAL TRADER")
        print("🎯 TARGET: 65%+ WIN RATE, 20%+ RETURNS")
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
        for i in range(60, len(data) - 100):
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
            
            analysis = self.enhanced_ai_analysis(indicators)
            
            # Check if AI wants to trade
            if not analysis['trade']:
                continue
            
            # AI has made a decision - execute trade
            entry_price = data.iloc[i]['close']
            position_size, leverage, profit_target = self.calculate_position_size(
                balance, analysis['conviction'])
            
            # Simulate the trade
            result = self.simulate_optimized_trade(
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
            if len(trades) % 3 == 0:
                wr = wins / len(trades) * 100 if trades else 0
                ret = (balance - self.initial_balance) / self.initial_balance * 100
                print(f"🚀 #{len(trades)}: {analysis['direction'].upper()} "
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
    
    def display_optimized_results(self, results: dict):
        """Display comprehensive optimized results"""
        print("\n" + "="*80)
        print("🚀 OPTIMIZED AI VISUAL TRADER - RESULTS")
        print("🎯 EXTREME PERFORMANCE MULTI-INDICATOR SYSTEM")
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
        
        print(f"\n🎯 TOP CONFLUENCE PERFORMANCE:")
        conf_stats = results['confluence_stats']
        if conf_stats:
            sorted_confs = sorted(conf_stats.items(), key=lambda x: x[1]['wins']/x[1]['count'], reverse=True)[:10]
            for conf, stats in sorted_confs:
                win_rate = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
                print(f"   {conf.replace('_', ' ').title()}: {stats['wins']}/{stats['count']} ({win_rate:.1f}%)")
        
        # Enhanced grade assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
        grades = []
        
        if results['win_rate'] >= 65:
            print("   ✅ Win Rate: A+ (65%+ TARGET MET!)")
            grades.append(4)
        elif results['win_rate'] >= 58:
            print("   ✅ Win Rate: A- (58%+)")
            grades.append(3.5)
        elif results['win_rate'] >= 50:
            print("   ✅ Win Rate: B+ (50%+)")
            grades.append(3)
        else:
            print("   ❌ Win Rate: Needs Improvement")
            grades.append(1)
        
        if results['total_return'] >= 20:
            print("   ✅ Returns: A+ (20%+ TARGET MET!)")
            grades.append(4)
        elif results['total_return'] >= 15:
            print("   ✅ Returns: A- (15%+)")
            grades.append(3.5)
        elif results['total_return'] >= 10:
            print("   ✅ Returns: B+ (10%+)")
            grades.append(3)
        else:
            print("   ❌ Returns: Needs Improvement")
            grades.append(1)
        
        if results['profit_factor'] >= 2.2:
            print("   ✅ Risk/Reward: A+ (2.2+)")
            grades.append(4)
        elif results['profit_factor'] >= 1.8:
            print("   ✅ Risk/Reward: A- (1.8+)")
            grades.append(3.5)
        elif results['profit_factor'] >= 1.4:
            print("   ✅ Risk/Reward: B+ (1.4+)")
            grades.append(3)
        else:
            print("   ❌ Risk/Reward: Needs Improvement")
            grades.append(1)
        
        freq = results['total_trades'] / 60
        if freq >= 0.8:
            print("   ✅ Frequency: A+ (0.8+/day)")
            grades.append(4)
        elif freq >= 0.6:
            print("   ✅ Frequency: A- (0.6+/day)")
            grades.append(3.5)
        elif freq >= 0.4:
            print("   ✅ Frequency: B+ (0.4+/day)")
            grades.append(3)
        else:
            print("   ❌ Frequency: Needs Improvement")
            grades.append(1)
        
        avg_grade = sum(grades) / len(grades)
        if avg_grade >= 3.8:
            overall = "A+ (EXCEPTIONAL!)"
            emoji = "🏆"
        elif avg_grade >= 3.3:
            overall = "A (EXCELLENT)"
            emoji = "🥇"
        elif avg_grade >= 2.8:
            overall = "B+ (VERY GOOD)"
            emoji = "✅"
        else:
            overall = "NEEDS OPTIMIZATION"
            emoji = "❌"
        
        print(f"\n{emoji} OVERALL GRADE: {overall}")
        
        if avg_grade >= 3.3:
            print("\n🎉 OPTIMIZED AI READY FOR LIVE TRADING!")
            print("🤖 Multi-indicator analysis EXTREMELY successful!")
            print("🚀 Target performance metrics achieved!")
        
        print("="*80)

def main():
    print("🚀 OPTIMIZED AI VISUAL TRADER")
    print("👁️ SEES: Candles, Volume, OBV, MACD, MFI, RSI, Momentum")
    print("🎯 TARGET: 65%+ Win Rate, 20%+ Returns")
    print("🧠 ENHANCED: Multi-Confluence Analysis")
    print("=" * 70)
    
    balance = 200.0
    days = 60
    
    ai_trader = OptimizedAIVisualTrader(balance)
    data = ai_trader.generate_market_data(days)
    results = ai_trader.run_optimized_backtest(data)
    ai_trader.display_optimized_results(results)

if __name__ == "__main__":
    main() 