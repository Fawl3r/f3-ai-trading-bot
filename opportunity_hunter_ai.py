#!/usr/bin/env python3
"""
OPPORTUNITY HUNTER AI - LEARNS TO MAXIMIZE RETURNS
AI Learning + Parabolic Detection + Dynamic Capital Allocation
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
            "leverage_min": 5, "leverage_max": 20,  # Higher max leverage for opportunities
            "position_base": 0.08, "position_min": 0.05, "position_max": 0.50,  # Up to 50% for big moves
            "win_multiplier": 1.05, "loss_multiplier": 0.98,
            "profit_target_min": 0.005, "profit_target_max": 0.050,  # Up to 5% targets!
            "stop_loss": 0.005, "trail_distance": 0.0015, "trail_start": 0.002,
            "max_daily_trades": 20, "max_hold_hours": 4,  # Longer holds for big moves
            "volume_threshold": 1.10,
        }
        
        # AI OPPORTUNITY LEARNING SYSTEM
        self.opportunity_ai = {
            # PATTERN PROFITABILITY TRACKING
            "pattern_profits": {},  # Track which patterns make the most money
            "parabolic_indicators": {},  # Learn parabolic movement signatures
            "breakout_patterns": {},  # Track breakout success rates
            "momentum_thresholds": {},  # Learn optimal momentum levels
            
            # DYNAMIC CAPITAL ALLOCATION
            "opportunity_multipliers": {
                "low": 1.0,      # Normal position size
                "medium": 1.5,   # 50% larger
                "high": 2.5,     # 2.5x larger  
                "extreme": 4.0   # 4x larger for parabolic moves
            },
            
            # PROFIT TARGET OPTIMIZATION
            "target_learning": {
                "conservative": 0.008,  # 0.8%
                "moderate": 0.015,      # 1.5%
                "aggressive": 0.025,    # 2.5%
                "parabolic": 0.040      # 4.0% for big moves
            },
            
            # LEARNING PARAMETERS
            "min_sample_size": 15,
            "learning_rate": 0.03,
            "profit_threshold": 0.015,  # Profits >1.5% are "big wins"
        }
        
        # STANDARD AI LEARNING
        self.learning = {
            "min_conviction": 68.0, "confluence_required": 3,
            "conviction_adjustment": 0.0, "confluence_adjustment": 0,
            "indicator_weights": {
                "rsi_signal": 30, "ema_alignment": 25, "macd_confirmation": 25,
                "volume_surge": 20, "momentum_follow": 15, "candle_pattern": 10,
                "market_structure": 15, "parabolic_signal": 35  # New indicator!
            },
            "indicator_success_rates": {
                "rsi_signal": 0.75, "ema_alignment": 0.75, "macd_confirmation": 0.75,
                "volume_surge": 0.75, "momentum_follow": 0.75, "candle_pattern": 0.75,
                "market_structure": 0.75, "parabolic_signal": 0.85  # Starts high
            },
            "pattern_blacklist": set(),
            "learning_rate": 0.02,
        }
        
        # STATE TRACKING
        self.current_position_pct = self.config['position_base']
        self.trade_history = []
        self.big_wins = []  # Track trades with >1.5% profit
        self.learning_cycles = 0
        
        print("🎯 OPPORTUNITY HUNTER AI - LEARNS TO MAXIMIZE RETURNS")
        print("🚀 PARABOLIC DETECTION + DYNAMIC CAPITAL ALLOCATION")
        print("💎 AI LEARNS WHICH PATTERNS MAKE THE MOST MONEY")
        print("📈 SCALES UP CAPITAL ON HIGH-OPPORTUNITY SETUPS")
        print("=" * 65)
    
    def detect_parabolic_setup(self, indicators: dict, data: pd.DataFrame, idx: int) -> dict:
        """AI OPPORTUNITY DETECTION: Identify parabolic/breakout potential"""
        if idx < 100:
            return {"opportunity_level": "low", "parabolic_score": 0, "signals": []}
        
        window = data.iloc[max(0, idx-50):idx+1]
        signals = []
        score = 0
        
        # 1. VOLUME EXPLOSION (Key parabolic indicator)
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
                if indicators['momentum_3'] * indicators['momentum_5'] > 0:  # Same direction
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
        
        if current_price > resistance * 0.998:  # Near resistance breakout
            signals.append("resistance_breakout")
            score += 35
        elif current_price < support * 1.002:  # Near support breakdown
            signals.append("support_breakdown")
            score += 35
        
        # 7. RSI DIVERGENCE REVERSAL
        if len(window) >= 30:
            price_trend = window['close'].iloc[-10:].mean() - window['close'].iloc[-20:-10].mean()
            rsi_values = []
            for i in range(len(window)-20, len(window)):
                if i >= 14:
                    delta = window['close'].iloc[max(0,i-14):i].diff()
                    gain = (delta.where(delta > 0, 0)).mean()
                    loss = (-delta.where(delta < 0, 0)).mean()
                    if loss > 0:
                        rs = gain / loss
                        rsi_val = 100 - (100 / (1 + rs))
                        rsi_values.append(rsi_val)
            
            if len(rsi_values) >= 10:
                rsi_trend = np.mean(rsi_values[-5:]) - np.mean(rsi_values[-10:-5])
                if (price_trend > 0 and rsi_trend < 0) or (price_trend < 0 and rsi_trend > 0):
                    signals.append("rsi_divergence")
                    score += 25
        
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
            
            # This was a big win - learn from it!
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
            
            # Update pattern profitability tracking
            pattern_key = f"{win_pattern['direction']}_{win_pattern['opportunity_level']}"
            if pattern_key not in self.opportunity_ai['pattern_profits']:
                self.opportunity_ai['pattern_profits'][pattern_key] = []
            
            self.opportunity_ai['pattern_profits'][pattern_key].append(win_pattern['profit_pct'])
            
            # Learn parabolic indicators
            for signal in opportunity_data['signals']:
                if signal not in self.opportunity_ai['parabolic_indicators']:
                    self.opportunity_ai['parabolic_indicators'][signal] = {'count': 0, 'profit_sum': 0}
                
                self.opportunity_ai['parabolic_indicators'][signal]['count'] += 1
                self.opportunity_ai['parabolic_indicators'][signal]['profit_sum'] += win_pattern['profit_pct']
    
    def calculate_dynamic_position_size(self, base_balance: float, opportunity_data: dict, conviction: float) -> float:
        """AI CAPITAL ALLOCATION: Size positions based on opportunity quality"""
        base_pct = self.current_position_pct
        
        # Get opportunity multiplier
        opportunity_multiplier = self.opportunity_ai['opportunity_multipliers'][opportunity_data['opportunity_level']]
        
        # Conviction bonus (higher conviction = larger size)
        conviction_multiplier = 1.0 + (conviction - 70) / 100  # +10% per 10 conviction points above 70
        conviction_multiplier = max(0.5, min(2.0, conviction_multiplier))
        
        # Pattern success bonus
        pattern_key = f"long_{opportunity_data['opportunity_level']}"  # Simplified for example
        if pattern_key in self.opportunity_ai['pattern_profits']:
            avg_profit = np.mean(self.opportunity_ai['pattern_profits'][pattern_key][-10:])  # Last 10 trades
            if avg_profit > 0.02:  # If pattern averages >2% profit
                pattern_multiplier = 1.3
            elif avg_profit > 0.015:  # If pattern averages >1.5% profit
                pattern_multiplier = 1.15
            else:
                pattern_multiplier = 1.0
        else:
            pattern_multiplier = 1.0
        
        # Calculate final position percentage
        final_pct = base_pct * opportunity_multiplier * conviction_multiplier * pattern_multiplier
        
        # Apply bounds
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
        
        # Adjust based on volume (higher volume = higher targets)
        volume_multiplier = 1.0
        if indicators['volume_ratio'] >= 3.0:
            volume_multiplier = 1.4
        elif indicators['volume_ratio'] >= 2.0:
            volume_multiplier = 1.2
        elif indicators['volume_ratio'] >= 1.5:
            volume_multiplier = 1.1
        
        # Adjust based on momentum
        momentum_multiplier = 1.0
        if abs(indicators.get('momentum_3', 0)) > 1.0:
            momentum_multiplier = 1.3
        elif abs(indicators.get('momentum_3', 0)) > 0.5:
            momentum_multiplier = 1.15
        
        final_target = base_target * volume_multiplier * momentum_multiplier
        return max(self.config['profit_target_min'], min(self.config['profit_target_max'], final_target))
    
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
            
            # Add parabolic movement periods
            parabolic_cycle = np.sin(i / (2 * 24 * 60) * 2 * np.pi) * 0.002
            if abs(parabolic_cycle) > 0.0015:  # Parabolic periods
                vol_multiplier = 2.5
                momentum_boost = parabolic_cycle * 0.5
            else:
                vol_multiplier = 1.0
                momentum_boost = 0
            
            if 8 <= hour <= 16:
                vol = 0.0041 * vol_multiplier
            elif 17 <= hour <= 23:
                vol = 0.0036 * vol_multiplier
            else:
                vol = 0.0032 * vol_multiplier
            
            trend_cycle = np.sin(i / (4.5 * 24 * 60) * 2 * np.pi)
            momentum = trend_cycle * 0.0008 + momentum_boost
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
        
        # STEP 1: Detect opportunity level
        opportunity_data = self.detect_parabolic_setup(indicators, data, idx)
        
        # STEP 2: Standard confluence analysis with opportunity boost
        confluences = []
        conviction = 0
        direction = None
        weights = self.learning['indicator_weights']
        
        rsi = indicators['rsi']
        
        # LONG SETUP with opportunity multiplier
        if rsi <= 52:
            if rsi <= 45 and indicators['rsi_momentum'] > -2 and indicators['rsi_slope'] >= 0:
                confluences.append('strong_rsi_long')
                conviction += weights['rsi_signal']
            elif rsi <= 50 and indicators['rsi_momentum'] >= 0:
                confluences.append('rsi_long')
                conviction += weights['rsi_signal'] * 0.7
            
            if confluences:
                direction = 'long'
                
                # Standard confluences
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
                
                # PARABOLIC SIGNAL BOOST
                if opportunity_data['parabolic_score'] >= 70:
                    confluences.append('parabolic_setup')
                    conviction += weights['parabolic_signal']
                elif opportunity_data['parabolic_score'] >= 40:
                    confluences.append('opportunity_setup')
                    conviction += weights['parabolic_signal'] * 0.6
        
        # SHORT SETUP (similar logic)
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
                
                # PARABOLIC SIGNAL BOOST for shorts
                if opportunity_data['parabolic_score'] >= 70:
                    confluences.append('parabolic_setup')
                    conviction += weights['parabolic_signal']
        
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
    
    def run_opportunity_hunter_test(self, data: pd.DataFrame, verbose: bool = True) -> dict:
        if verbose:
            print("🎯 RUNNING OPPORTUNITY HUNTER AI TEST")
            print("🚀 PARABOLIC DETECTION + DYNAMIC CAPITAL ALLOCATION")
            print("=" * 60)
        
        balance = self.initial_balance
        trades = []
        daily_trades = 0
        last_day = None
        wins = 0
        losses = 0
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'stop_loss': 0, 'time_exit': 0}
        
        # Track opportunity performance
        opportunity_stats = {"low": [], "medium": [], "high": [], "extreme": []}
        max_single_profit = 0
        best_trade = None
        
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
            
            # DYNAMIC POSITION SIZING based on opportunity
            position_size = self.calculate_dynamic_position_size(balance, opportunity_data, analysis['conviction'])
            
            # DYNAMIC PROFIT TARGET based on opportunity
            profit_target = self.calculate_dynamic_profit_target(opportunity_data, indicators)
            
            result = self.simulate_trade(i, entry_price, analysis['direction'], position_size, profit_target, data)
            result['conviction'] = analysis['conviction']
            result['confluence_count'] = analysis['confluence_count']
            result['opportunity_level'] = opportunity_data['opportunity_level']
            result['parabolic_score'] = opportunity_data['parabolic_score']
            result['profit_target_used'] = profit_target
            result['position_pct'] = position_size / balance
            
            balance += result['pnl']
            
            # Track opportunity performance
            opportunity_stats[opportunity_data['opportunity_level']].append(result['pnl'])
            
            # Track best trade
            if result['pnl'] > max_single_profit:
                max_single_profit = result['pnl']
                best_trade = result.copy()
            
            # AI LEARNING: Learn from big wins
            self.learn_from_big_wins(result, indicators, opportunity_data)
            
            self.update_position_size(result['success'])
            
            if result['success']:
                wins += 1
            else:
                losses += 1
            
            exit_reasons[result['exit_reason']] += 1
            daily_trades += 1
            
            trades.append(result)
            self.trade_history.append(result)
            
            if verbose and (len(trades) <= 20 or len(trades) % 25 == 0):
                wr = wins / len(trades) * 100 if trades else 0
                ret = (balance - self.initial_balance) / self.initial_balance * 100
                opp_emoji = "🚀" if opportunity_data['opportunity_level'] == "extreme" else "📈" if opportunity_data['opportunity_level'] == "high" else "📊" if opportunity_data['opportunity_level'] == "medium" else "📉"
                print(f"#{len(trades)}: {analysis['direction'].upper()} {opp_emoji}{opportunity_data['opportunity_level'].upper()} "
                      f"Score:{opportunity_data['parabolic_score']} Size:{result['position_pct']*100:.1f}% "
                      f"Target:{profit_target*100:.1f}% → ${result['pnl']:+.2f} | WR:{wr:.1f}% Ret:{ret:+.1f}%")
        
        total_trades = len(trades)
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        total_return = (balance - self.initial_balance) / self.initial_balance * 100
        
        if total_trades > 0:
            winning_trades = [t for t in trades if t['success']]
            losing_trades = [t for t in trades if not t['success']]
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else float('inf')
            avg_position = np.mean([t['position_pct'] for t in trades])
        else:
            avg_win = avg_loss = profit_factor = avg_position = 0
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"🎯 OPPORTUNITY HUNTER AI RESULTS")
            print(f"{'='*80}")
            print(f"🔢 Total Trades: {total_trades}")
            print(f"🏆 Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
            print(f"💰 Total Return: {total_return:+.1f}%")
            print(f"💵 Final Balance: ${balance:.2f}")
            print(f"📈 Profit Factor: {profit_factor:.2f}")
            print(f"💚 Average Win: ${avg_win:.2f}")
            print(f"❌ Average Loss: ${avg_loss:.2f}")
            print(f"📊 Average Position Size: {avg_position*100:.1f}%")
            
            print(f"\n🎯 OPPORTUNITY LEVEL PERFORMANCE:")
            for level, profits in opportunity_stats.items():
                if profits:
                    count = len(profits)
                    avg_profit = np.mean(profits)
                    total_profit = sum(profits)
                    emoji = "🚀" if level == "extreme" else "📈" if level == "high" else "📊" if level == "medium" else "📉"
                    print(f"   {emoji} {level.upper()}: {count} trades, Avg: ${avg_profit:+.2f}, Total: ${total_profit:+.2f}")
            
            if best_trade:
                print(f"\n💎 BEST TRADE:")
                print(f"   🏆 Profit: ${best_trade['pnl']:+.2f}")
                print(f"   📊 Opportunity: {best_trade['opportunity_level'].upper()}")
                print(f"   📈 Score: {best_trade['parabolic_score']}")
                print(f"   💪 Position: {best_trade['position_pct']*100:.1f}%")
                print(f"   🎯 Target: {best_trade['profit_target_used']*100:.1f}%")
            
            print(f"\n🧠 AI LEARNING STATS:")
            print(f"   💎 Big Wins Learned: {len(self.big_wins)}")
            print(f"   📈 Pattern Profits Tracked: {len(self.opportunity_ai['pattern_profits'])}")
            print(f"   🚀 Parabolic Indicators: {len(self.opportunity_ai['parabolic_indicators'])}")
            
            if len(self.big_wins) > 0:
                print(f"\n🔬 TOP PROFITABLE PATTERNS:")
                for pattern_key, profits in list(self.opportunity_ai['pattern_profits'].items())[:3]:
                    avg_profit = np.mean(profits[-5:]) * 100  # Last 5 trades, as percentage
                    print(f"   💰 {pattern_key}: {avg_profit:.1f}% average profit")
        
        return {
            'win_rate': win_rate, 'total_return': total_return, 'profit_factor': profit_factor,
            'total_trades': total_trades, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'final_balance': balance, 'max_single_profit': max_single_profit,
            'big_wins_count': len(self.big_wins), 'avg_position': avg_position,
            'opportunity_stats': opportunity_stats, 'best_trade': best_trade
        }

def main():
    print("🎯 OPPORTUNITY HUNTER AI - LEARNS TO MAXIMIZE RETURNS")
    print("🚀 PARABOLIC DETECTION + DYNAMIC CAPITAL ALLOCATION")
    print("💎 AI LEARNS WHICH PATTERNS MAKE THE MOST MONEY")
    print("📈 SCALES UP CAPITAL ON HIGH-OPPORTUNITY SETUPS")
    print("🔬 ADAPTIVE PROFIT TARGETS + PATTERN LEARNING")
    print("=" * 70)
    
    hunter_ai = OpportunityHunterAI(200.0)
    data = hunter_ai.generate_market_data(60)
    results = hunter_ai.run_opportunity_hunter_test(data)
    
    print(f"\n🎉 OPPORTUNITY HUNTER FINAL RESULTS:")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Max Single Profit: ${results['max_single_profit']:+.2f}")
    print(f"Big Wins Learned: {results['big_wins_count']}")
    print(f"Final Balance: ${results['final_balance']:.2f}")

if __name__ == "__main__":
    main() 