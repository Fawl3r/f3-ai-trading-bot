#!/usr/bin/env python3
"""
AI LEARNING BOT - LEARNS FROM FAILURES
Adaptive Compounding + Machine Learning from Losses
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AILearningBot:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # BASE CONFIGURATION
        self.config = {
            "leverage_min": 5, "leverage_max": 8,
            "position_base": 0.08, "position_min": 0.05, "position_max": 0.25,
            "win_multiplier": 1.05, "loss_multiplier": 0.98,
            "profit_target_min": 0.005, "profit_target_max": 0.010,
            "stop_loss": 0.005, "trail_distance": 0.0015, "trail_start": 0.002,
            "max_daily_trades": 18, "max_hold_hours": 2,
            "volume_threshold": 1.10,
        }
        
        # AI LEARNING SYSTEM
        self.learning = {
            # ADAPTIVE THRESHOLDS (Learn optimal levels)
            "min_conviction": 68.0,
            "confluence_required": 3,
            "conviction_adjustment": 0.0,
            "confluence_adjustment": 0,
            
            # INDICATOR EFFECTIVENESS TRACKING
            "indicator_weights": {
                "rsi_signal": 30, "ema_alignment": 25, "macd_confirmation": 25,
                "volume_surge": 20, "momentum_follow": 15, "candle_pattern": 10,
                "market_structure": 15
            },
            "indicator_success_rates": {
                "rsi_signal": 0.75, "ema_alignment": 0.75, "macd_confirmation": 0.75,
                "volume_surge": 0.75, "momentum_follow": 0.75, "candle_pattern": 0.75,
                "market_structure": 0.75
            },
            
            # FAILURE PATTERN ANALYSIS
            "loss_patterns": [],
            "pattern_blacklist": set(),
            "market_condition_performance": {},
            
            # LEARNING PARAMETERS
            "learning_rate": 0.02,
            "min_sample_size": 20,
            "adaptation_frequency": 25,  # Adapt every 25 trades
        }
        
        # STATE TRACKING
        self.current_position_pct = self.config['position_base']
        self.trade_history = []
        self.learning_cycles = 0
        
        print("🧠 AI LEARNING BOT - LEARNS FROM FAILURES")
        print("📚 ADAPTIVE THRESHOLDS + PATTERN RECOGNITION")
        print("🎯 GETS SMARTER AFTER EVERY LOSS")
        print("=" * 55)
    
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
        
        # RSI
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        if len(rsi) >= 10:
            indicators['rsi_momentum'] = rsi.iloc[-1] - rsi.iloc[-6]
            indicators['rsi_slope'] = (rsi.iloc[-1] - rsi.iloc[-4]) / 3
        else:
            indicators['rsi_momentum'] = 0
            indicators['rsi_slope'] = 0
        
        # EMAs
        indicators['ema_9'] = window['close'].ewm(span=9).mean().iloc[-1]
        indicators['ema_21'] = window['close'].ewm(span=21).mean().iloc[-1]
        indicators['ema_50'] = window['close'].ewm(span=50).mean().iloc[-1] if len(window) >= 50 else window['close'].mean()
        
        indicators['ema_bullish'] = indicators['ema_9'] > indicators['ema_21']
        indicators['ema_strong_bullish'] = indicators['ema_9'] > indicators['ema_21'] > indicators['ema_50']
        
        # MACD
        ema_12 = window['close'].ewm(span=12).mean()
        ema_26 = window['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd'] = macd_line.iloc[-1]
        indicators['macd_signal'] = signal_line.iloc[-1]
        indicators['macd_bullish'] = indicators['macd'] > indicators['macd_signal']
        indicators['macd_strength'] = abs(indicators['macd'] - indicators['macd_signal'])
        
        # Volume
        indicators['volume_sma'] = window['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = current['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        if len(window) >= 10:
            recent_vol = window['volume'].tail(5).mean()
            prev_vol = window['volume'].iloc[-10:-5].mean()
            indicators['volume_momentum'] = (recent_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
        else:
            indicators['volume_momentum'] = 0
        
        # Momentum
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
            
            indicators['uptrend'] = recent_highs > prev_highs and recent_lows > prev_lows
            indicators['downtrend'] = recent_highs < prev_highs and recent_lows < prev_lows
        else:
            indicators['uptrend'] = False
            indicators['downtrend'] = False
        
        return indicators
    
    def analyze_loss_patterns(self, trade_result: dict, indicators: dict):
        """AI LEARNING: Analyze what led to the loss"""
        if not trade_result['success']:
            # Create loss pattern fingerprint
            pattern = {
                'rsi': round(indicators['rsi'], 1),
                'rsi_momentum': round(indicators['rsi_momentum'], 1),
                'volume_ratio': round(indicators['volume_ratio'], 1),
                'momentum_3': round(indicators['momentum_3'], 2),
                'macd_strength': round(indicators['macd_strength'], 3),
                'direction': trade_result['direction'],
                'conviction': trade_result['conviction'],
                'confluence_count': trade_result['confluence_count']
            }
            
            self.learning['loss_patterns'].append(pattern)
            
            # If pattern occurs frequently, blacklist it
            if len(self.learning['loss_patterns']) >= self.learning['min_sample_size']:
                pattern_key = f"{pattern['direction']}_{pattern['rsi']//5*5}_{pattern['volume_ratio']//0.5*0.5}"
                pattern_count = sum(1 for p in self.learning['loss_patterns'][-50:] 
                                  if f"{p['direction']}_{p['rsi']//5*5}_{p['volume_ratio']//0.5*0.5}" == pattern_key)
                
                if pattern_count >= 3:  # If 3+ losses with similar pattern
                    self.learning['pattern_blacklist'].add(pattern_key)
    
    def update_indicator_weights(self, trade_result: dict, confluences: list):
        """AI LEARNING: Adjust indicator weights based on success"""
        success = trade_result['success']
        learning_rate = self.learning['learning_rate']
        
        for confluence in confluences:
            # Map confluence to indicator
            if 'rsi' in confluence:
                indicator = 'rsi_signal'
            elif 'ema' in confluence:
                indicator = 'ema_alignment'
            elif 'macd' in confluence:
                indicator = 'macd_confirmation'
            elif 'volume' in confluence:
                indicator = 'volume_surge'
            elif 'momentum' in confluence:
                indicator = 'momentum_follow'
            elif 'candle' in confluence:
                indicator = 'candle_pattern'
            elif 'trend' in confluence or 'structure' in confluence:
                indicator = 'market_structure'
            else:
                continue
            
            # Update success rate
            current_rate = self.learning['indicator_success_rates'][indicator]
            if success:
                new_rate = current_rate + learning_rate * (1.0 - current_rate)
            else:
                new_rate = current_rate - learning_rate * current_rate
            
            self.learning['indicator_success_rates'][indicator] = max(0.1, min(0.95, new_rate))
            
            # Adjust weight based on success rate
            base_weight = 30 if indicator == 'rsi_signal' else 25 if indicator in ['ema_alignment', 'macd_confirmation'] else 20 if indicator == 'volume_surge' else 15
            success_multiplier = self.learning['indicator_success_rates'][indicator] / 0.75  # 0.75 is baseline
            self.learning['indicator_weights'][indicator] = base_weight * success_multiplier
    
    def adapt_thresholds(self):
        """AI LEARNING: Adapt conviction and confluence requirements"""
        if len(self.trade_history) < self.learning['min_sample_size']:
            return
        
        recent_trades = self.trade_history[-self.learning['min_sample_size']:]
        recent_win_rate = sum(1 for t in recent_trades if t['success']) / len(recent_trades)
        
        # Adjust conviction threshold
        if recent_win_rate < 0.70:  # If win rate too low
            self.learning['conviction_adjustment'] += 1.0  # Require higher conviction
            self.learning['confluence_adjustment'] = min(2, self.learning['confluence_adjustment'] + 1)
        elif recent_win_rate > 0.80:  # If win rate very high
            self.learning['conviction_adjustment'] = max(-5.0, self.learning['conviction_adjustment'] - 0.5)
            self.learning['confluence_adjustment'] = max(0, self.learning['confluence_adjustment'] - 1)
        
        # Update thresholds
        self.learning['min_conviction'] = 68.0 + self.learning['conviction_adjustment']
        self.learning['confluence_required'] = 3 + self.learning['confluence_adjustment']
    
    def ai_analysis(self, indicators: dict) -> dict:
        if not indicators:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # Check blacklisted patterns
        pattern_key = f"long_{indicators['rsi']//5*5}_{indicators['volume_ratio']//0.5*0.5}"
        if pattern_key in self.learning['pattern_blacklist']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': [], 'reason': 'blacklisted_pattern'}
        
        confluences = []
        conviction = 0
        direction = None
        weights = self.learning['indicator_weights']
        
        rsi = indicators['rsi']
        
        # LONG SETUP with AI-adjusted weights
        if rsi <= 52:
            if rsi <= 45 and indicators['rsi_momentum'] > -2 and indicators['rsi_slope'] >= 0:
                confluences.append('strong_rsi_long')
                conviction += weights['rsi_signal']
            elif rsi <= 50 and indicators['rsi_momentum'] >= 0:
                confluences.append('rsi_long')
                conviction += weights['rsi_signal'] * 0.7
            
            if confluences:
                direction = 'long'
                
                if indicators['ema_strong_bullish']:
                    confluences.append('strong_ema_bullish')
                    conviction += weights['ema_alignment']
                elif indicators['ema_bullish']:
                    confluences.append('ema_bullish')
                    conviction += weights['ema_alignment'] * 0.7
                
                if indicators['macd_bullish'] and indicators['macd_strength'] > 0.005:
                    confluences.append('strong_macd_bullish')
                    conviction += weights['macd_confirmation']
                elif indicators['macd_bullish']:
                    confluences.append('macd_bullish')
                    conviction += weights['macd_confirmation'] * 0.6
                
                if indicators['volume_ratio'] >= 1.5 and indicators['volume_momentum'] > 0.1:
                    confluences.append('volume_surge')
                    conviction += weights['volume_surge']
                elif indicators['volume_ratio'] >= self.config['volume_threshold']:
                    confluences.append('volume_ok')
                    conviction += weights['volume_surge'] * 0.5
                
                if indicators['momentum_consistent'] and indicators['momentum_3'] > 0.2:
                    confluences.append('strong_momentum')
                    conviction += weights['momentum_follow']
                elif indicators['momentum_3'] > 0:
                    confluences.append('momentum_positive')
                    conviction += weights['momentum_follow'] * 0.6
                
                if indicators['is_bullish'] and indicators['body_pct'] >= 0.4:
                    confluences.append('strong_bullish_candle')
                    conviction += weights['candle_pattern']
                elif indicators['is_bullish']:
                    confluences.append('bullish_candle')
                    conviction += weights['candle_pattern'] * 0.5
                
                if indicators['uptrend']:
                    confluences.append('uptrend_structure')
                    conviction += weights['market_structure']
        
        # SHORT SETUP (similar logic)
        elif rsi >= 48:
            pattern_key = f"short_{indicators['rsi']//5*5}_{indicators['volume_ratio']//0.5*0.5}"
            if pattern_key in self.learning['pattern_blacklist']:
                return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': [], 'reason': 'blacklisted_pattern'}
            
            if rsi >= 55 and indicators['rsi_momentum'] < 2 and indicators['rsi_slope'] <= 0:
                confluences.append('strong_rsi_short')
                conviction += weights['rsi_signal']
            elif rsi >= 50 and indicators['rsi_momentum'] <= 0:
                confluences.append('rsi_short')
                conviction += weights['rsi_signal'] * 0.7
            
            if confluences:
                direction = 'short'
                # Similar confluence logic for shorts...
                if not indicators['ema_bullish']:
                    confluences.append('ema_bearish')
                    conviction += weights['ema_alignment'] * 0.7
                
                if not indicators['macd_bullish'] and indicators['macd_strength'] > 0.005:
                    confluences.append('macd_bearish')
                    conviction += weights['macd_confirmation'] * 0.7
                
                if indicators['volume_ratio'] >= self.config['volume_threshold']:
                    confluences.append('volume_ok')
                    conviction += weights['volume_surge'] * 0.5
                
                if indicators['momentum_3'] < 0:
                    confluences.append('momentum_negative')
                    conviction += weights['momentum_follow'] * 0.6
        
        if direction is None:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        # AI-ADJUSTED FILTERS
        if indicators['volume_ratio'] < 1.05:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if len(confluences) < self.learning['confluence_required']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if conviction < self.learning['min_conviction']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        return {
            'trade': True, 'direction': direction, 'conviction': min(conviction, 98),
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
                           'pnl': pnl, 'success': True, 'hold_minutes': i - entry_idx, 'direction': direction}
                
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
                               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': i - entry_idx, 'direction': direction}
                
                if low <= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {'exit_price': stop_loss, 'exit_reason': 'stop_loss',
                           'pnl': pnl, 'success': False, 'hold_minutes': i - entry_idx, 'direction': direction}
            
            else:  # short
                if low < best_price:
                    best_price = low
                
                if low <= take_profit:
                    pnl = position_size * profit_target
                    return {'exit_price': take_profit, 'exit_reason': 'take_profit',
                           'pnl': pnl, 'success': True, 'hold_minutes': i - entry_idx, 'direction': direction}
                
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
                               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': i - entry_idx, 'direction': direction}
                
                if high >= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {'exit_price': stop_loss, 'exit_reason': 'stop_loss',
                           'pnl': pnl, 'success': False, 'hold_minutes': i - entry_idx, 'direction': direction}
        
        # Time exit
        final_price = data.iloc[max_idx]['close']
        if direction == 'long':
            profit_pct = (final_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - final_price) / entry_price
        
        pnl = position_size * profit_pct
        return {'exit_price': final_price, 'exit_reason': 'time_exit',
               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': self.config['max_hold_hours'] * 60, 'direction': direction}
    
    def update_position_size(self, trade_success: bool):
        if trade_success:
            self.current_position_pct *= self.config['win_multiplier']
        else:
            self.current_position_pct *= self.config['loss_multiplier']
        
        self.current_position_pct = max(self.config['position_min'], 
                                       min(self.config['position_max'], 
                                           self.current_position_pct))
    
    def run_ai_test(self, data: pd.DataFrame, verbose: bool = True) -> dict:
        if verbose:
            print("🧠 RUNNING AI LEARNING TEST")
            print("📚 LEARNING FROM EVERY FAILURE")
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
            
            analysis = self.ai_analysis(indicators)
            if not analysis['trade']:
                continue
            
            entry_price = data.iloc[i]['close']
            conv_factor = (analysis['conviction'] - 68) / 30
            conv_factor = max(0, min(1, conv_factor))
            
            position_size = balance * self.current_position_pct
            profit_target = self.config['profit_target_min'] + \
                           (self.config['profit_target_max'] - self.config['profit_target_min']) * conv_factor
            
            result = self.simulate_trade(i, entry_price, analysis['direction'], position_size, profit_target, data)
            result['conviction'] = analysis['conviction']
            result['confluence_count'] = analysis['confluence_count']
            
            balance += result['pnl']
            
            # AI LEARNING: Analyze this trade
            self.analyze_loss_patterns(result, indicators)
            self.update_indicator_weights(result, analysis['confluences'])
            
            # Periodic adaptation
            if len(trades) % self.learning['adaptation_frequency'] == 0:
                self.adapt_thresholds()
                self.learning_cycles += 1
            
            self.update_position_size(result['success'])
            
            if result['success']:
                wins += 1
            else:
                losses += 1
            
            exit_reasons[result['exit_reason']] += 1
            daily_trades += 1
            
            trades.append(result)
            self.trade_history.append(result)
            
            if verbose and (len(trades) <= 25 or len(trades) % 30 == 0):
                wr = wins / len(trades) * 100 if trades else 0
                ret = (balance - self.initial_balance) / self.initial_balance * 100
                blacklisted = len(self.learning['pattern_blacklist'])
                learning_note = f"L{self.learning_cycles}" if self.learning_cycles > 0 else ""
                print(f"#{len(trades)}: {analysis['direction'].upper()} Conv:{analysis['conviction']:.0f}% "
                      f"→ ${result['pnl']:+.2f} | WR:{wr:.1f}% Ret:{ret:+.1f}% BL:{blacklisted} {learning_note}")
        
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
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🧠 AI LEARNING BOT RESULTS")
            print(f"{'='*70}")
            print(f"🔢 Total Trades: {total_trades}")
            print(f"🏆 Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
            print(f"💰 Total Return: {total_return:+.1f}%")
            print(f"💵 Final Balance: ${balance:.2f}")
            print(f"📈 Profit Factor: {profit_factor:.2f}")
            print(f"💚 Average Win: ${avg_win:.2f}")
            print(f"❌ Average Loss: ${avg_loss:.2f}")
            
            print(f"\n🧠 AI LEARNING ANALYSIS:")
            print(f"   📚 Learning Cycles: {self.learning_cycles}")
            print(f"   🚫 Blacklisted Patterns: {len(self.learning['pattern_blacklist'])}")
            print(f"   📊 Adaptive Conviction: {self.learning['min_conviction']:.1f}% (vs 68% base)")
            print(f"   🎯 Adaptive Confluences: {self.learning['confluence_required']} (vs 3 base)")
            
            print(f"\n🔬 INDICATOR SUCCESS RATES:")
            for indicator, rate in self.learning['indicator_success_rates'].items():
                weight = self.learning['indicator_weights'][indicator]
                print(f"   📈 {indicator}: {rate:.1%} (Weight: {weight:.1f})")
            
            if len(self.learning['pattern_blacklist']) > 0:
                print(f"\n🚫 BLACKLISTED PATTERNS:")
                for pattern in list(self.learning['pattern_blacklist'])[:5]:
                    print(f"   ❌ {pattern}")
        
        return {
            'win_rate': win_rate, 'total_return': total_return, 'profit_factor': profit_factor,
            'total_trades': total_trades, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'final_balance': balance, 'learning_cycles': self.learning_cycles,
            'blacklisted_patterns': len(self.learning['pattern_blacklist']),
            'adaptive_conviction': self.learning['min_conviction'],
            'adaptive_confluences': self.learning['confluence_required']
        }

def main():
    print("🧠 AI LEARNING BOT - LEARNS FROM FAILURES")
    print("📚 ADAPTIVE THRESHOLDS + PATTERN RECOGNITION")
    print("🎯 GETS SMARTER AFTER EVERY LOSS")
    print("🔬 INDICATOR WEIGHT OPTIMIZATION")
    print("🚫 AUTOMATIC PATTERN BLACKLISTING")
    print("=" * 60)
    
    ai_trader = AILearningBot(200.0)
    data = ai_trader.generate_market_data(60)
    results = ai_trader.run_ai_test(data)
    
    print(f"\n🎉 AI LEARNING FINAL RESULTS:")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Learning Cycles: {results['learning_cycles']}")
    print(f"Blacklisted Patterns: {results['blacklisted_patterns']}")
    print(f"Final Balance: ${results['final_balance']:.2f}")

if __name__ == "__main__":
    main() 