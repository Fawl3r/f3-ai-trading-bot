#!/usr/bin/env python3
"""
COMPREHENSIVE OPPORTUNITY HUNTER VALIDATION
Tests AI across multiple market scenarios for consistency verification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OpportunityHunterAI:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # BASE CONFIGURATION
        self.config = {
            "leverage_min": 5, "leverage_max": 20,
            "position_base": 0.08, "position_min": 0.05, "position_max": 0.50,
            "win_multiplier": 1.05, "loss_multiplier": 0.98,
            "profit_target_min": 0.005, "profit_target_max": 0.050,
            "stop_loss": 0.005, "trail_distance": 0.0015, "trail_start": 0.002,
            "max_daily_trades": 20, "max_hold_hours": 4,
            "volume_threshold": 1.10,
        }
        
        # AI OPPORTUNITY LEARNING SYSTEM
        self.opportunity_ai = {
            "pattern_profits": {},
            "parabolic_indicators": {},
            "breakout_patterns": {},
            "momentum_thresholds": {},
            "opportunity_multipliers": {
                "low": 1.0, "medium": 1.5, "high": 2.5, "extreme": 4.0
            },
            "target_learning": {
                "conservative": 0.008, "moderate": 0.015,
                "aggressive": 0.025, "parabolic": 0.040
            },
            "min_sample_size": 15, "learning_rate": 0.03, "profit_threshold": 0.015,
        }
        
        # STANDARD AI LEARNING
        self.learning = {
            "min_conviction": 68.0, "confluence_required": 3,
            "conviction_adjustment": 0.0, "confluence_adjustment": 0,
            "indicator_weights": {
                "rsi_signal": 30, "ema_alignment": 25, "macd_confirmation": 25,
                "volume_surge": 20, "momentum_follow": 15, "candle_pattern": 10,
                "market_structure": 15, "parabolic_signal": 35
            },
            "indicator_success_rates": {
                "rsi_signal": 0.75, "ema_alignment": 0.75, "macd_confirmation": 0.75,
                "volume_surge": 0.75, "momentum_follow": 0.75, "candle_pattern": 0.75,
                "market_structure": 0.75, "parabolic_signal": 0.85
            },
            "pattern_blacklist": set(),
            "learning_rate": 0.02,
        }
        
        # STATE TRACKING
        self.current_position_pct = self.config['position_base']
        self.trade_history = []
        self.big_wins = []
        self.learning_cycles = 0
    
    def reset_state(self):
        """Reset AI state for fresh scenario testing"""
        self.current_position_pct = self.config['position_base']
        self.trade_history = []
        self.big_wins = []
        self.learning_cycles = 0
        self.opportunity_ai["pattern_profits"] = {}
        self.opportunity_ai["parabolic_indicators"] = {}
        self.learning["pattern_blacklist"] = set()
    
    def detect_parabolic_setup(self, indicators: dict, data: pd.DataFrame, idx: int) -> dict:
        """AI OPPORTUNITY DETECTION: Identify parabolic/breakout potential"""
        if idx < 100:
            return {"opportunity_level": "low", "parabolic_score": 0, "signals": []}
        
        window = data.iloc[max(0, idx-50):idx+1]
        signals = []
        score = 0
        
        # 1. VOLUME EXPLOSION
        if indicators['volume_ratio'] >= 3.0:
            signals.append("massive_volume_explosion")
            score += 40
        elif indicators['volume_ratio'] >= 2.0:
            signals.append("strong_volume_surge")
            score += 25
        elif indicators['volume_ratio'] >= 1.5:
            signals.append("volume_surge")
            score += 15
        
        # 2. MOMENTUM ACCELERATION
        if hasattr(indicators, 'momentum_3') and hasattr(indicators, 'momentum_5'):
            if abs(indicators['momentum_3']) > 1.0 and abs(indicators['momentum_5']) > 0.8:
                if indicators['momentum_3'] * indicators['momentum_5'] > 0:
                    signals.append("momentum_acceleration")
                    score += 30
        
        # 3. PRICE COMPRESSION TO EXPANSION
        recent_range = window['high'].tail(20).max() - window['low'].tail(20).min()
        prev_range = window['high'].iloc[-40:-20].max() - window['low'].iloc[-40:-20].min()
        if recent_range > prev_range * 1.5:
            signals.append("range_expansion")
            score += 25
        
        # 4. MULTI-TIMEFRAME ALIGNMENT
        if (indicators['ema_strong_bullish'] and indicators['macd_bullish'] and 
            indicators['momentum_3'] > 0.5 and indicators['rsi'] < 70):
            signals.append("multi_timeframe_bullish")
            score += 30
        elif (not indicators['ema_bullish'] and not indicators['macd_bullish'] and 
              indicators['momentum_3'] < -0.5 and indicators['rsi'] > 30):
            signals.append("multi_timeframe_bearish")
            score += 30
        
        # 5. VOLATILITY CONTRACTION TO EXPANSION
        recent_volatility = window['close'].tail(10).std()
        prev_volatility = window['close'].iloc[-30:-10].std()
        if recent_volatility > prev_volatility * 1.3:
            signals.append("volatility_expansion")
            score += 20
        
        # 6. BREAKOUT PATTERNS
        resistance = window['high'].tail(20).max()
        support = window['low'].tail(20).min()
        current_price = window['close'].iloc[-1]
        
        if current_price > resistance * 0.998:
            signals.append("resistance_breakout")
            score += 35
        elif current_price < support * 1.002:
            signals.append("support_breakdown")
            score += 35
        
        # DETERMINE OPPORTUNITY LEVEL
        if score >= 100:
            opportunity_level = "extreme"
        elif score >= 70:
            opportunity_level = "high"
        elif score >= 40:
            opportunity_level = "medium"
        else:
            opportunity_level = "low"
        
        return {
            "opportunity_level": opportunity_level,
            "parabolic_score": score,
            "signals": signals
        }
    
    def learn_from_big_wins(self, trade_result: dict, indicators: dict, opportunity_data: dict):
        """AI LEARNING: Analyze what made this trade highly profitable"""
        if trade_result['pnl'] > 0 and abs(trade_result['pnl']) / (trade_result.get('position_size', 100)) > self.opportunity_ai['profit_threshold']:
            
            win_pattern = {
                'rsi': round(indicators['rsi'], 1),
                'volume_ratio': round(indicators['volume_ratio'], 1),
                'momentum_3': round(indicators.get('momentum_3', 0), 2),
                'macd_strength': round(indicators['macd_strength'], 3),
                'opportunity_level': opportunity_data['opportunity_level'],
                'parabolic_score': opportunity_data['parabolic_score'],
                'signals': opportunity_data['signals'],
                'profit_pct': abs(trade_result['pnl']) / (trade_result.get('position_size', 100)),
                'direction': trade_result['direction']
            }
            
            self.big_wins.append(win_pattern)
            
            pattern_key = f"{win_pattern['direction']}_{win_pattern['opportunity_level']}"
            if pattern_key not in self.opportunity_ai['pattern_profits']:
                self.opportunity_ai['pattern_profits'][pattern_key] = []
            
            self.opportunity_ai['pattern_profits'][pattern_key].append(win_pattern['profit_pct'])
            
            for signal in opportunity_data['signals']:
                if signal not in self.opportunity_ai['parabolic_indicators']:
                    self.opportunity_ai['parabolic_indicators'][signal] = {'count': 0, 'profit_sum': 0}
                
                self.opportunity_ai['parabolic_indicators'][signal]['count'] += 1
                self.opportunity_ai['parabolic_indicators'][signal]['profit_sum'] += win_pattern['profit_pct']
    
    def calculate_dynamic_position_size(self, base_balance: float, opportunity_data: dict, conviction: float) -> float:
        """AI CAPITAL ALLOCATION: Size positions based on opportunity quality"""
        base_pct = self.current_position_pct
        
        opportunity_multiplier = self.opportunity_ai['opportunity_multipliers'][opportunity_data['opportunity_level']]
        conviction_multiplier = 1.0 + (conviction - 70) / 100
        conviction_multiplier = max(0.5, min(2.0, conviction_multiplier))
        
        pattern_key = f"long_{opportunity_data['opportunity_level']}"
        if pattern_key in self.opportunity_ai['pattern_profits']:
            avg_profit = np.mean(self.opportunity_ai['pattern_profits'][pattern_key][-10:])
            if avg_profit > 0.02:
                pattern_multiplier = 1.3
            elif avg_profit > 0.015:
                pattern_multiplier = 1.15
            else:
                pattern_multiplier = 1.0
        else:
            pattern_multiplier = 1.0
        
        final_pct = base_pct * opportunity_multiplier * conviction_multiplier * pattern_multiplier
        final_pct = max(self.config['position_min'], min(self.config['position_max'], final_pct))
        
        return base_balance * final_pct
    
    def calculate_dynamic_profit_target(self, opportunity_data: dict, indicators: dict) -> float:
        """AI TARGET OPTIMIZATION: Set profit targets based on opportunity quality"""
        base_targets = self.opportunity_ai['target_learning']
        
        if opportunity_data['opportunity_level'] == "extreme":
            base_target = base_targets['parabolic']
        elif opportunity_data['opportunity_level'] == "high":
            base_target = base_targets['aggressive']
        elif opportunity_data['opportunity_level'] == "medium":
            base_target = base_targets['moderate']
        else:
            base_target = base_targets['conservative']
        
        volume_multiplier = 1.0
        if indicators['volume_ratio'] >= 3.0:
            volume_multiplier = 1.4
        elif indicators['volume_ratio'] >= 2.0:
            volume_multiplier = 1.2
        elif indicators['volume_ratio'] >= 1.5:
            volume_multiplier = 1.1
        
        momentum_multiplier = 1.0
        if abs(indicators.get('momentum_3', 0)) > 1.0:
            momentum_multiplier = 1.3
        elif abs(indicators.get('momentum_3', 0)) > 0.5:
            momentum_multiplier = 1.15
        
        final_target = base_target * volume_multiplier * momentum_multiplier
        return max(self.config['profit_target_min'], min(self.config['profit_target_max'], final_target))
    
    def generate_market_data(self, days: int = 60, seed: int = 42, scenario: str = "normal") -> pd.DataFrame:
        """Generate different market scenarios for testing"""
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(seed)
        
        data = []
        price = start_price
        time = datetime.now() - timedelta(days=days)
        volume_base = 1120000
        
        # SCENARIO PARAMETERS
        if scenario == "bull_market":
            trend_strength = 0.0015  # Strong uptrend
            volatility_base = 0.0035
            parabolic_frequency = 0.3  # More parabolic moves
        elif scenario == "bear_market":
            trend_strength = -0.0012  # Strong downtrend
            volatility_base = 0.0045
            parabolic_frequency = 0.25
        elif scenario == "sideways":
            trend_strength = 0.0002  # Minimal trend
            volatility_base = 0.0025
            parabolic_frequency = 0.15  # Fewer big moves
        elif scenario == "high_volatility":
            trend_strength = 0.0005
            volatility_base = 0.0065  # Very volatile
            parabolic_frequency = 0.4  # Many opportunities
        elif scenario == "low_volatility":
            trend_strength = 0.0003
            volatility_base = 0.0020  # Low volatility
            parabolic_frequency = 0.1  # Few opportunities
        else:  # normal
            trend_strength = 0.0006
            volatility_base = 0.0041
            parabolic_frequency = 0.2
        
        for i in range(total_minutes):
            hour = (i // 60) % 24
            day_factor = np.sin(i / (9 * 60) * 2 * np.pi) * 0.0008
            week_factor = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.0012
            
            # Market trend
            trend_cycle = np.sin(i / (4.5 * 24 * 60) * 2 * np.pi)
            trend_momentum = trend_cycle * trend_strength
            
            # Parabolic movement periods
            parabolic_cycle = np.sin(i / (2 * 24 * 60) * 2 * np.pi) * 0.002
            if abs(parabolic_cycle) > (0.0015 * (1 - parabolic_frequency)):
                vol_multiplier = 2.5 * (1 + parabolic_frequency)
                momentum_boost = parabolic_cycle * 0.5 * (1 + parabolic_frequency)
            else:
                vol_multiplier = 1.0
                momentum_boost = 0
            
            # Time-based volatility
            if 8 <= hour <= 16:
                vol = volatility_base * vol_multiplier
            elif 17 <= hour <= 23:
                vol = volatility_base * 0.9 * vol_multiplier
            else:
                vol = volatility_base * 0.8 * vol_multiplier
            
            momentum = trend_momentum + momentum_boost
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
            
            vol_momentum = abs(price_change) * 110 * vol_multiplier
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
        
        # Enhanced Momentum
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
    
    def ai_opportunity_analysis(self, indicators: dict, data: pd.DataFrame, idx: int) -> dict:
        if not indicators:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        opportunity_data = self.detect_parabolic_setup(indicators, data, idx)
        
        confluences = []
        conviction = 0
        direction = None
        weights = self.learning['indicator_weights']
        
        rsi = indicators['rsi']
        
        # LONG SETUP
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
                
                if indicators['volume_ratio'] >= 1.5:
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
                
                if opportunity_data['parabolic_score'] >= 70:
                    confluences.append('parabolic_setup')
                    conviction += weights['parabolic_signal']
                elif opportunity_data['parabolic_score'] >= 40:
                    confluences.append('opportunity_setup')
                    conviction += weights['parabolic_signal'] * 0.6
        
        # SHORT SETUP
        elif rsi >= 48:
            if rsi >= 55 and indicators['rsi_momentum'] < 2 and indicators['rsi_slope'] <= 0:
                confluences.append('strong_rsi_short')
                conviction += weights['rsi_signal']
            elif rsi >= 50 and indicators['rsi_momentum'] <= 0:
                confluences.append('rsi_short')
                conviction += weights['rsi_signal'] * 0.7
            
            if confluences:
                direction = 'short'
                
                if not indicators['ema_bullish']:
                    confluences.append('ema_bearish')
                    conviction += weights['ema_alignment'] * 0.7
                
                if not indicators['macd_bullish'] and indicators['macd_strength'] > 0.005:
                    confluences.append('macd_bearish')
                    conviction += weights['macd_confirmation'] * 0.7
                
                if indicators['volume_ratio'] >= 1.5:
                    confluences.append('volume_surge')
                    conviction += weights['volume_surge']
                
                if indicators['momentum_3'] < 0:
                    confluences.append('momentum_negative')
                    conviction += weights['momentum_follow'] * 0.6
                
                if opportunity_data['parabolic_score'] >= 70:
                    confluences.append('parabolic_setup')
                    conviction += weights['parabolic_signal']
        
        if direction is None:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if indicators['volume_ratio'] < 1.05:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if len(confluences) < self.learning['confluence_required']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        if conviction < self.learning['min_conviction']:
            return {'trade': False, 'direction': None, 'conviction': 0, 'confluences': []}
        
        return {
            'trade': True, 'direction': direction, 'conviction': min(conviction, 98),
            'confluences': confluences, 'confluence_count': len(confluences),
            'opportunity_data': opportunity_data
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
                           'pnl': pnl, 'success': True, 'hold_minutes': i - entry_idx, 
                           'direction': direction, 'position_size': position_size}
                
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
                               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': i - entry_idx,
                               'direction': direction, 'position_size': position_size}
                
                if low <= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {'exit_price': stop_loss, 'exit_reason': 'stop_loss',
                           'pnl': pnl, 'success': False, 'hold_minutes': i - entry_idx,
                           'direction': direction, 'position_size': position_size}
            
            else:  # short
                if low < best_price:
                    best_price = low
                
                if low <= take_profit:
                    pnl = position_size * profit_target
                    return {'exit_price': take_profit, 'exit_reason': 'take_profit',
                           'pnl': pnl, 'success': True, 'hold_minutes': i - entry_idx,
                           'direction': direction, 'position_size': position_size}
                
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
                               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': i - entry_idx,
                               'direction': direction, 'position_size': position_size}
                
                if high >= stop_loss:
                    pnl = -position_size * self.config['stop_loss']
                    return {'exit_price': stop_loss, 'exit_reason': 'stop_loss',
                           'pnl': pnl, 'success': False, 'hold_minutes': i - entry_idx,
                           'direction': direction, 'position_size': position_size}
        
        # Time exit
        final_price = data.iloc[max_idx]['close']
        if direction == 'long':
            profit_pct = (final_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - final_price) / entry_price
        
        pnl = position_size * profit_pct
        return {'exit_price': final_price, 'exit_reason': 'time_exit',
               'pnl': pnl, 'success': pnl > 0, 'hold_minutes': self.config['max_hold_hours'] * 60,
               'direction': direction, 'position_size': position_size}
    
    def update_position_size(self, trade_success: bool):
        if trade_success:
            self.current_position_pct *= self.config['win_multiplier']
        else:
            self.current_position_pct *= self.config['loss_multiplier']
        
        self.current_position_pct = max(self.config['position_min'], 
                                       min(self.config['position_max'], 
                                           self.current_position_pct))
    
    def run_scenario_test(self, data: pd.DataFrame, scenario_name: str, verbose: bool = False) -> dict:
        balance = self.initial_balance
        trades = []
        daily_trades = 0
        last_day = None
        wins = 0
        losses = 0
        
        opportunity_stats = {"low": [], "medium": [], "high": [], "extreme": []}
        max_single_profit = 0
        
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
            
            analysis = self.ai_opportunity_analysis(indicators, data, i)
            if not analysis['trade']:
                continue
            
            entry_price = data.iloc[i]['close']
            opportunity_data = analysis['opportunity_data']
            
            position_size = self.calculate_dynamic_position_size(balance, opportunity_data, analysis['conviction'])
            profit_target = self.calculate_dynamic_profit_target(opportunity_data, indicators)
            
            result = self.simulate_trade(i, entry_price, analysis['direction'], position_size, profit_target, data)
            result['conviction'] = analysis['conviction']
            result['confluence_count'] = analysis['confluence_count']
            result['opportunity_level'] = opportunity_data['opportunity_level']
            result['parabolic_score'] = opportunity_data['parabolic_score']
            result['profit_target_used'] = profit_target
            result['position_pct'] = position_size / balance
            
            balance += result['pnl']
            
            opportunity_stats[opportunity_data['opportunity_level']].append(result['pnl'])
            
            if result['pnl'] > max_single_profit:
                max_single_profit = result['pnl']
            
            self.learn_from_big_wins(result, indicators, opportunity_data)
            self.update_position_size(result['success'])
            
            if result['success']:
                wins += 1
            else:
                losses += 1
            
            daily_trades += 1
            trades.append(result)
            self.trade_history.append(result)
        
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
        
        return {
            'scenario': scenario_name,
            'win_rate': win_rate, 'total_return': total_return, 'profit_factor': profit_factor,
            'total_trades': total_trades, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'final_balance': balance, 'max_single_profit': max_single_profit,
            'big_wins_count': len(self.big_wins), 'opportunity_stats': opportunity_stats
        }

def run_comprehensive_validation():
    print("🎯 COMPREHENSIVE OPPORTUNITY HUNTER VALIDATION")
    print("🔬 TESTING AI ACROSS MULTIPLE MARKET SCENARIOS")
    print("=" * 70)
    
    # TEST SCENARIOS
    test_scenarios = [
        {"name": "Normal Market", "scenario": "normal", "seed": 42},
        {"name": "Bull Market", "scenario": "bull_market", "seed": 123},
        {"name": "Bear Market", "scenario": "bear_market", "seed": 456},
        {"name": "Sideways Market", "scenario": "sideways", "seed": 789},
        {"name": "High Volatility", "scenario": "high_volatility", "seed": 101},
        {"name": "Low Volatility", "scenario": "low_volatility", "seed": 202},
        {"name": "Random Seed 1", "scenario": "normal", "seed": 999},
        {"name": "Random Seed 2", "scenario": "normal", "seed": 777},
        {"name": "Random Seed 3", "scenario": "normal", "seed": 555},
        {"name": "Random Seed 4", "scenario": "normal", "seed": 333},
    ]
    
    all_results = []
    
    for i, test in enumerate(test_scenarios):
        print(f"\n📊 SCENARIO {i+1}/10: {test['name']}")
        print("-" * 50)
        
        hunter_ai = OpportunityHunterAI(200.0)
        data = hunter_ai.generate_market_data(60, test['seed'], test['scenario'])
        result = hunter_ai.run_scenario_test(data, test['name'])
        
        all_results.append(result)
        
        print(f"🎯 Win Rate: {result['win_rate']:.1f}%")
        print(f"💰 Return: {result['total_return']:+.1f}%")
        print(f"🔢 Trades: {result['total_trades']}")
        print(f"💎 Max Profit: ${result['max_single_profit']:+.2f}")
        print(f"📈 Profit Factor: {result['profit_factor']:.2f}")
        print(f"🧠 Big Wins: {result['big_wins_count']}")
    
    # COMPREHENSIVE ANALYSIS
    print(f"\n{'='*80}")
    print("🔬 COMPREHENSIVE VALIDATION ANALYSIS")
    print(f"{'='*80}")
    
    win_rates = [r['win_rate'] for r in all_results]
    returns = [r['total_return'] for r in all_results]
    trade_counts = [r['total_trades'] for r in all_results]
    max_profits = [r['max_single_profit'] for r in all_results]
    profit_factors = [r['profit_factor'] for r in all_results if r['profit_factor'] != float('inf')]
    big_wins = [r['big_wins_count'] for r in all_results]
    
    print(f"📊 WIN RATE ANALYSIS:")
    print(f"   Average: {np.mean(win_rates):.1f}%")
    print(f"   Range: {min(win_rates):.1f}% - {max(win_rates):.1f}%")
    print(f"   Standard Deviation: {np.std(win_rates):.1f}%")
    print(f"   Scenarios >75%: {sum(1 for wr in win_rates if wr > 75)}/10")
    
    print(f"\n💰 RETURN ANALYSIS:")
    print(f"   Average: {np.mean(returns):+.1f}%")
    print(f"   Range: {min(returns):+.1f}% - {max(returns):+.1f}%")
    print(f"   Standard Deviation: {np.std(returns):.1f}%")
    print(f"   Positive Returns: {sum(1 for r in returns if r > 0)}/10")
    
    print(f"\n🔢 TRADE FREQUENCY:")
    print(f"   Average Trades: {np.mean(trade_counts):.0f}")
    print(f"   Range: {min(trade_counts)} - {max(trade_counts)}")
    
    print(f"\n💎 MAXIMUM SINGLE PROFITS:")
    print(f"   Average: ${np.mean(max_profits):+.2f}")
    print(f"   Best: ${max(max_profits):+.2f}")
    print(f"   Profits >$30: {sum(1 for p in max_profits if p > 30)}/10")
    
    if profit_factors:
        print(f"\n📈 PROFIT FACTOR:")
        print(f"   Average: {np.mean(profit_factors):.2f}")
        print(f"   Range: {min(profit_factors):.2f} - {max(profit_factors):.2f}")
    
    print(f"\n🧠 AI LEARNING:")
    print(f"   Average Big Wins: {np.mean(big_wins):.0f}")
    print(f"   Range: {min(big_wins)} - {max(big_wins)}")
    
    # DETAILED RESULTS TABLE
    print(f"\n📋 DETAILED RESULTS TABLE:")
    print(f"{'Scenario':<20} {'Win%':<8} {'Return%':<10} {'Trades':<8} {'Max$':<10} {'PF':<6}")
    print("-" * 70)
    for result in all_results:
        pf_str = f"{result['profit_factor']:.2f}" if result['profit_factor'] != float('inf') else "∞"
        print(f"{result['scenario']:<20} {result['win_rate']:<7.1f}% "
              f"{result['total_return']:<9.1f}% {result['total_trades']:<8} "
              f"${result['max_single_profit']:<9.2f} {pf_str:<6}")
    
    # CONSISTENCY ASSESSMENT
    consistency_score = 0
    if min(win_rates) > 70:
        consistency_score += 25
    if all(r > 0 for r in returns):
        consistency_score += 25
    if np.std(win_rates) < 5:
        consistency_score += 25
    if min(profit_factors) > 2.0:
        consistency_score += 25
    
    print(f"\n🏆 CONSISTENCY SCORE: {consistency_score}/100")
    if consistency_score >= 90:
        print("🌟 EXCELLENT - Highly consistent performance across all scenarios!")
    elif consistency_score >= 75:
        print("✅ GOOD - Strong consistency with minor variations")
    elif consistency_score >= 60:
        print("⚠️  MODERATE - Some inconsistency detected")
    else:
        print("❌ POOR - Significant inconsistency across scenarios")
    
    return all_results

if __name__ == "__main__":
    results = run_comprehensive_validation() 