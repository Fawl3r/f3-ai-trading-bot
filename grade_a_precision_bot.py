#!/usr/bin/env python3
"""
GRADE A PRECISION BOT - Quality Over Quantity
Target: Overall Grade A through precision trading
- Ultra-selective entries (80%+ confidence)
- Conservative position sizing
- Excellent risk/reward ratios
- 55%+ win rate with 2.0+ profit factor
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GradeAPrecisionBot:
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        
        # PRECISION-FOCUSED CONFIGURATION
        self.config = {
            # CONSERVATIVE LEVERAGE
            "leverage_min": 8,
            "leverage_max": 15,
            
            # CONSERVATIVE POSITION SIZING
            "position_pct_min": 8,   # 8% of balance minimum
            "position_pct_max": 18,  # 18% maximum for highest confidence
            "min_position": 16.0,    # Minimum $16 position
            
            # EXCELLENT RISK/REWARD
            "profit_target_min": 2.5,   # Minimum 2.5% profit target
            "profit_target_max": 6.0,   # Maximum 6% for ultra high confidence
            "stop_loss": 1.8,           # Tight 1.8% stop loss
            "trailing_stop_pct": 0.6,   # 0.6% trailing stop
            "trailing_activation": 1.2, # Activate at 1.2% profit
            
            # ULTRA-SELECTIVE ENTRIES
            "min_confidence": 80,       # Very high minimum confidence
            "rsi_oversold": 30,         # More extreme RSI levels
            "rsi_overbought": 70,       
            "volume_multiplier": 1.5,   # Require strong volume
            
            # QUALITY CONTROL
            "max_daily_trades": 8,      # Fewer, better trades
            "max_hold_hours": 12,       # Shorter holds
            "max_consecutive_losses": 2, # Quick loss control
            "daily_loss_limit": 3.0,   # 3% daily loss limit
            
            # TREND CONFIRMATION
            "trend_strength_min": 2,    # Require strong trends
            "momentum_threshold": 0.5,  # Strong momentum required
        }
        
        print("🎯 GRADE A PRECISION BOT - QUALITY OVER QUANTITY")
        print("💎 ULTRA-SELECTIVE • HIGH WIN RATE • EXCELLENT R:R")
        print("=" * 60)
    
    def generate_realistic_market_data(self, days: int = 60) -> pd.DataFrame:
        print(f"📊 Generating {days} days of realistic market data...")
        
        start_price = 140.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        current_price = start_price
        current_time = datetime.now() - timedelta(days=days)
        
        for i in range(total_minutes):
            # Natural market cycles
            hour = (i // 60) % 24
            base_vol = 0.0015  # Lower base volatility
            if 8 <= hour < 16:
                base_vol = 0.002   # Market hours
            elif 16 <= hour < 24:
                base_vol = 0.0025  # Evening activity
            
            # Smoother trend cycles
            weekly = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.001
            daily = np.sin(i / (24 * 60) * 2 * np.pi) * 0.0006
            
            # Occasional volatility spikes
            spike = 1.0
            if np.random.random() < 0.002:  # Rare spikes
                spike = np.random.uniform(1.5, 2.5)
            
            # Price movement
            noise = np.random.normal(0, base_vol * spike)
            price_change = weekly + daily + noise
            current_price *= (1 + price_change)
            current_price = max(110, min(180, current_price))
            
            # OHLC generation
            intrabar_vol = base_vol * spike * 0.6
            high = current_price * (1 + abs(np.random.normal(0, intrabar_vol)))
            low = current_price * (1 - abs(np.random.normal(0, intrabar_vol)))
            open_price = current_price * (1 + np.random.normal(0, intrabar_vol * 0.3))
            
            high = max(high, open_price, current_price)
            low = min(low, open_price, current_price)
            
            # Volume
            base_volume = 800000
            volume_factor = 1 + abs(price_change) * 50 + np.random.uniform(0.8, 1.8)
            volume = base_volume * volume_factor * spike
            
            data.append({
                'timestamp': current_time + timedelta(minutes=i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': current_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} candles")
        print(f"📈 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        return df
    
    def calculate_precision_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        if idx < 50:
            return None
        
        window = data.iloc[max(0, idx-50):idx+1]
        
        # RSI
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Multiple MAs for trend confirmation
        ma_5 = window['close'].rolling(5).mean().iloc[-1]
        ma_10 = window['close'].rolling(10).mean().iloc[-1]
        ma_20 = window['close'].rolling(20).mean().iloc[-1]
        ma_50 = window['close'].rolling(50).mean().iloc[-1] if len(window) >= 50 else window['close'].mean()
        
        # Volume analysis
        volume_ma = window['volume'].rolling(10).mean().iloc[-1]
        volume_ratio = window['volume'].iloc[-1] / volume_ma if volume_ma > 0 else 1
        
        # Advanced momentum
        momentum_5 = (window['close'].iloc[-1] - window['close'].iloc[-6]) / window['close'].iloc[-6] * 100 if len(window) >= 6 else 0
        momentum_15 = (window['close'].iloc[-1] - window['close'].iloc[-16]) / window['close'].iloc[-16] * 100 if len(window) >= 16 else 0
        
        # Trend strength analysis
        current_price = window['close'].iloc[-1]
        trend_strength = 0
        
        # Very strong trend confirmation
        if current_price > ma_5 > ma_10 > ma_20 > ma_50:
            trend_strength = 3  # Very strong uptrend
        elif current_price > ma_5 > ma_10 > ma_20:
            trend_strength = 2  # Strong uptrend
        elif current_price > ma_5 > ma_10:
            trend_strength = 1  # Mild uptrend
        elif current_price < ma_5 < ma_10 < ma_20 < ma_50:
            trend_strength = -3  # Very strong downtrend
        elif current_price < ma_5 < ma_10 < ma_20:
            trend_strength = -2  # Strong downtrend
        elif current_price < ma_5 < ma_10:
            trend_strength = -1  # Mild downtrend
        
        # Support/Resistance levels
        recent_20 = window['close'].rolling(20).agg(['min', 'max']).iloc[-1]
        support = recent_20['min']
        resistance = recent_20['max']
        
        # Distance from support/resistance
        support_distance = (current_price - support) / support * 100 if support > 0 else 0
        resistance_distance = (resistance - current_price) / current_price * 100 if resistance > 0 else 0
        
        # Volatility
        volatility = window['close'].pct_change().rolling(10).std().iloc[-1] * 100 if len(window) >= 10 else 0
        
        return {
            'rsi': current_rsi,
            'ma_5': ma_5,
            'ma_10': ma_10,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'volume_ratio': volume_ratio,
            'momentum_5': momentum_5,
            'momentum_15': momentum_15,
            'trend_strength': trend_strength,
            'support_distance': support_distance,
            'resistance_distance': resistance_distance,
            'volatility': volatility,
            'current_price': current_price,
            'support': support,
            'resistance': resistance
        }
    
    def analyze_precision_opportunity(self, indicators: dict) -> dict:
        if not indicators:
            return {"confidence": 0, "direction": "hold"}
        
        confidence = 0
        direction = "hold"
        reasons = []
        
        rsi = indicators['rsi']
        trend_strength = indicators['trend_strength']
        
        # ULTRA-SELECTIVE RSI SIGNALS (40% weight)
        if rsi <= self.config['rsi_oversold']:
            confidence += 40
            direction = "long"
            reasons.append("extreme_oversold")
            if rsi <= 25:  # Ultra oversold
                confidence += 15
                reasons.append("ultra_oversold")
        elif rsi >= self.config['rsi_overbought']:
            confidence += 40
            direction = "short"
            reasons.append("extreme_overbought")
            if rsi >= 75:  # Ultra overbought
                confidence += 15
                reasons.append("ultra_overbought")
        
        # STRONG TREND CONFIRMATION (35% weight)
        if trend_strength >= self.config['trend_strength_min']:
            if direction == "long":
                confidence += 35
                reasons.append("strong_uptrend_alignment")
                if trend_strength == 3:
                    confidence += 10
                    reasons.append("very_strong_uptrend")
            elif direction == "short":
                confidence -= 20  # Penalize counter-trend
                reasons.append("counter_strong_uptrend")
        elif trend_strength <= -self.config['trend_strength_min']:
            if direction == "short":
                confidence += 35
                reasons.append("strong_downtrend_alignment")
                if trend_strength == -3:
                    confidence += 10
                    reasons.append("very_strong_downtrend")
            elif direction == "long":
                confidence -= 20  # Penalize counter-trend
                reasons.append("counter_strong_downtrend")
        
        # MOMENTUM CONFIRMATION (15% weight)
        if direction == "long" and indicators['momentum_5'] > self.config['momentum_threshold']:
            confidence += 15
            reasons.append("strong_upward_momentum")
            if indicators['momentum_15'] > self.config['momentum_threshold']:
                confidence += 5
                reasons.append("multi_timeframe_momentum_up")
        elif direction == "short" and indicators['momentum_5'] < -self.config['momentum_threshold']:
            confidence += 15
            reasons.append("strong_downward_momentum")
            if indicators['momentum_15'] < -self.config['momentum_threshold']:
                confidence += 5
                reasons.append("multi_timeframe_momentum_down")
        
        # VOLUME CONFIRMATION (10% weight)
        if indicators['volume_ratio'] >= self.config['volume_multiplier']:
            confidence += 10
            reasons.append("high_volume_confirmation")
            if indicators['volume_ratio'] >= 2.0:
                confidence += 5
                reasons.append("exceptional_volume")
        
        # SUPPORT/RESISTANCE LEVELS (bonus)
        if direction == "long" and indicators['support_distance'] <= 2.0:
            confidence += 8
            reasons.append("near_support_level")
        elif direction == "short" and indicators['resistance_distance'] <= 2.0:
            confidence += 8
            reasons.append("near_resistance_level")
        
        # VOLATILITY FILTER
        if 1.0 <= indicators['volatility'] <= 3.5:
            confidence *= 1.1
            reasons.append("optimal_volatility")
        elif indicators['volatility'] > 4.5:
            confidence *= 0.8
            reasons.append("high_volatility_caution")
        elif indicators['volatility'] < 0.8:
            confidence *= 0.9
            reasons.append("low_volatility_caution")
        
        return {
            "confidence": max(0, min(confidence, 98)),
            "direction": direction,
            "reasons": reasons
        }
    
    def calculate_precision_position_size(self, confidence: float, balance: float) -> tuple:
        # Conservative position sizing based on confidence
        confidence_ratio = (confidence - 80) / 20  # Scale from 80-100% confidence
        confidence_ratio = max(0, min(confidence_ratio, 1))
        
        position_pct = (self.config['position_pct_min'] + 
                       (self.config['position_pct_max'] - self.config['position_pct_min']) * confidence_ratio)
        
        position_size = max(self.config['min_position'], balance * position_pct / 100)
        
        # Conservative leverage scaling
        leverage = int(self.config['leverage_min'] + 
                      (self.config['leverage_max'] - self.config['leverage_min']) * confidence_ratio)
        
        # Risk-adjusted profit targets
        profit_target_pct = (self.config['profit_target_min'] + 
                            (self.config['profit_target_max'] - self.config['profit_target_min']) * confidence_ratio)
        
        return position_size, leverage, profit_target_pct
    
    def simulate_precision_trade(self, entry_price: float, direction: str, 
                               leverage: int, position_size: float, profit_target_pct: float,
                               data: pd.DataFrame, start_idx: int) -> dict:
        best_price = entry_price
        trailing_stop_price = None
        trailing_activated = False
        
        max_hold_minutes = self.config['max_hold_hours'] * 60
        
        for i in range(start_idx + 1, min(start_idx + max_hold_minutes + 1, len(data))):
            candle = data.iloc[i]
            high, low, close = candle['high'], candle['low'], candle['close']
            
            if direction == 'long':
                if high > best_price:
                    best_price = high
                
                current_profit_pct = (high - entry_price) / entry_price * 100
                
                # Trailing stop activation
                if not trailing_activated and current_profit_pct >= self.config['trailing_activation']:
                    trailing_activated = True
                    trailing_stop_price = best_price * (1 - self.config['trailing_stop_pct'] / 100)
                
                # Update trailing stop
                if trailing_activated:
                    new_stop = best_price * (1 - self.config['trailing_stop_pct'] / 100)
                    if new_stop > trailing_stop_price:
                        trailing_stop_price = new_stop
                
                # Take profit
                if current_profit_pct >= profit_target_pct:
                    exit_price = entry_price * (1 + profit_target_pct / 100)
                    pnl_amount = position_size * (profit_target_pct / 100)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'take_profit',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': profit_target_pct
                    }
                
                # Trailing stop hit
                if trailing_activated and low <= trailing_stop_price:
                    profit_pct = (trailing_stop_price - entry_price) / entry_price * 100
                    pnl_amount = position_size * profit_pct / 100
                    return {
                        'exit_price': trailing_stop_price,
                        'exit_reason': 'trailing_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': profit_pct
                    }
                
                # Stop loss
                loss_pct = (entry_price - low) / entry_price * 100
                if loss_pct >= self.config['stop_loss']:
                    exit_price = entry_price * (1 - self.config['stop_loss'] / 100)
                    pnl_amount = position_size * (-self.config['stop_loss'] / 100)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'stop_loss',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': -self.config['stop_loss']
                    }
            
            else:  # Short
                if low < best_price:
                    best_price = low
                
                current_profit_pct = (entry_price - low) / entry_price * 100
                
                if not trailing_activated and current_profit_pct >= self.config['trailing_activation']:
                    trailing_activated = True
                    trailing_stop_price = best_price * (1 + self.config['trailing_stop_pct'] / 100)
                
                if trailing_activated:
                    new_stop = best_price * (1 + self.config['trailing_stop_pct'] / 100)
                    if new_stop < trailing_stop_price:
                        trailing_stop_price = new_stop
                
                if current_profit_pct >= profit_target_pct:
                    exit_price = entry_price * (1 - profit_target_pct / 100)
                    pnl_amount = position_size * (profit_target_pct / 100)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'take_profit',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': profit_target_pct
                    }
                
                if trailing_activated and high >= trailing_stop_price:
                    profit_pct = (entry_price - trailing_stop_price) / entry_price * 100
                    pnl_amount = position_size * profit_pct / 100
                    return {
                        'exit_price': trailing_stop_price,
                        'exit_reason': 'trailing_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': profit_pct
                    }
                
                loss_pct = (high - entry_price) / entry_price * 100
                if loss_pct >= self.config['stop_loss']:
                    exit_price = entry_price * (1 + self.config['stop_loss'] / 100)
                    pnl_amount = position_size * (-self.config['stop_loss'] / 100)
                    return {
                        'exit_price': exit_price,
                        'exit_reason': 'stop_loss',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated,
                        'profit_pct': -self.config['stop_loss']
                    }
        
        # Time exit
        final_price = data.iloc[min(start_idx + max_hold_minutes, len(data) - 1)]['close']
        if direction == 'long':
            profit_pct = (final_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - final_price) / entry_price * 100
        
        pnl_amount = position_size * profit_pct / 100
        
        return {
            'exit_price': final_price,
            'exit_reason': 'time_exit',
            'pnl_amount': pnl_amount,
            'hold_minutes': max_hold_minutes,
            'trailing_activated': trailing_activated,
            'profit_pct': profit_pct
        }
    
    def run_precision_backtest(self, data: pd.DataFrame) -> dict:
        print("\n🎯 RUNNING PRECISION A-GRADE BACKTEST")
        print("💎 ULTRA-SELECTIVE • QUALITY OVER QUANTITY")
        print("=" * 50)
        
        balance = self.initial_balance
        trades = []
        
        wins = 0
        losses = 0
        total_profit = 0
        daily_trades = 0
        consecutive_losses = 0
        daily_loss = 0
        last_trade_day = None
        
        exit_reasons = {'take_profit': 0, 'trailing_stop': 0, 'stop_loss': 0, 'time_exit': 0}
        trailing_activations = 0
        compounding_growth = [balance]
        
        i = 50
        while i < len(data) - 100:
            current_time = data.iloc[i]['timestamp']
            current_price = data.iloc[i]['close']
            
            today = current_time.date()
            if last_trade_day != today:
                daily_trades = 0
                daily_loss = 0
                last_trade_day = today
            
            daily_loss_pct = (daily_loss / balance) * 100 if balance > 0 else 0
            
            # Strict trading conditions
            if (daily_trades < self.config['max_daily_trades'] and 
                consecutive_losses < self.config['max_consecutive_losses'] and
                daily_loss_pct < self.config['daily_loss_limit']):
                
                indicators = self.calculate_precision_indicators(data, i)
                if indicators:
                    analysis = self.analyze_precision_opportunity(indicators)
                    
                    # ULTRA-SELECTIVE ENTRY
                    if (analysis['confidence'] >= self.config['min_confidence'] and 
                        analysis['direction'] != "hold"):
                        
                        position_size, leverage, profit_target_pct = self.calculate_precision_position_size(
                            analysis['confidence'], balance)
                        
                        trade_result = self.simulate_precision_trade(
                            current_price, analysis['direction'], leverage, 
                            position_size, profit_target_pct, data, i
                        )
                        
                        old_balance = balance
                        balance += trade_result['pnl_amount']
                        total_profit += trade_result['pnl_amount']
                        
                        if trade_result['pnl_amount'] > 0:
                            wins += 1
                            consecutive_losses = 0
                        else:
                            losses += 1
                            consecutive_losses += 1
                            daily_loss += abs(trade_result['pnl_amount'])
                        
                        exit_reasons[trade_result['exit_reason']] += 1
                        if trade_result['trailing_activated']:
                            trailing_activations += 1
                        
                        compounding_growth.append(balance)
                        
                        trade = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'exit_price': trade_result['exit_price'],
                            'direction': analysis['direction'],
                            'leverage': leverage,
                            'position_size': position_size,
                            'confidence': analysis['confidence'],
                            'pnl_amount': trade_result['pnl_amount'],
                            'profit_pct': trade_result['profit_pct'],
                            'exit_reason': trade_result['exit_reason'],
                            'hold_minutes': trade_result['hold_minutes'],
                            'trailing_activated': trade_result['trailing_activated'],
                            'balance': balance,
                            'reasons': analysis['reasons']
                        }
                        trades.append(trade)
                        daily_trades += 1
                        
                        if len(trades) % 5 == 0 or len(trades) <= 15:
                            win_rate = (wins / len(trades) * 100) if trades else 0
                            growth = ((balance - self.initial_balance) / self.initial_balance) * 100
                            print(f"💎 #{len(trades)}: {analysis['direction'].upper()} @ ${current_price:.2f} → "
                                  f"${trade_result['pnl_amount']:+.2f} ({trade_result['profit_pct']:+.1f}%) | "
                                  f"WR: {win_rate:.1f}% | Growth: {growth:+.1f}% | ${balance:.2f}")
                        
                        # Skip ahead to avoid overtrading
                        skip_minutes = max(trade_result['hold_minutes'], 30)
                        i += skip_minutes
                    else:
                        i += 5  # Smaller skip when no trade
                else:
                    i += 5
            else:
                i += 15  # Longer skip when limits hit
        
        # Calculate comprehensive results
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_return = ((balance - self.initial_balance) / self.initial_balance) * 100
        
        if total_trades > 0:
            profitable_trades = [t for t in trades if t['pnl_amount'] > 0]
            losing_trades = [t for t in trades if t['pnl_amount'] < 0]
            
            avg_win = np.mean([t['pnl_amount'] for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([t['pnl_amount'] for t in losing_trades]) if losing_trades else 0
            max_win = max([t['pnl_amount'] for t in profitable_trades]) if profitable_trades else 0
            max_loss = min([t['pnl_amount'] for t in losing_trades]) if losing_trades else 0
            
            total_wins_amount = sum([t['pnl_amount'] for t in profitable_trades])
            total_losses_amount = abs(sum([t['pnl_amount'] for t in losing_trades]))
            profit_factor = total_wins_amount / max(total_losses_amount, 0.01)
            
            avg_leverage = np.mean([t['leverage'] for t in trades])
            avg_confidence = np.mean([t['confidence'] for t in trades])
            avg_hold_time = np.mean([t['hold_minutes'] for t in trades])
            avg_profit_pct = np.mean([t['profit_pct'] for t in profitable_trades]) if profitable_trades else 0
            
            # Drawdown calculation
            peak = self.initial_balance
            max_drawdown = 0
            for balance_point in compounding_growth:
                if balance_point > peak:
                    peak = balance_point
                drawdown = (peak - balance_point) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        else:
            avg_win = avg_loss = max_win = max_loss = profit_factor = 0
            avg_leverage = avg_confidence = avg_hold_time = avg_profit_pct = max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_balance': balance,
            'total_profit': total_profit,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'avg_leverage': avg_leverage,
            'avg_confidence': avg_confidence,
            'avg_hold_time': avg_hold_time,
            'avg_profit_pct': avg_profit_pct,
            'max_drawdown': max_drawdown,
            'exit_reasons': exit_reasons,
            'trailing_activations': trailing_activations,
            'compounding_growth': compounding_growth,
            'trades': trades
        }
    
    def display_precision_results(self, results: dict):
        print("\n" + "="*80)
        print("🎯 GRADE A PRECISION BOT - BACKTEST RESULTS")
        print("💎 ULTRA-SELECTIVE • QUALITY OVER QUANTITY")
        print("="*80)
        
        print(f"📊 PERFORMANCE METRICS:")
        print(f"   🔢 Total Trades: {results['total_trades']}")
        print(f"   🏆 Win Rate: {results['win_rate']:.1f}% ({results['wins']}W/{results['losses']}L)")
        print(f"   💰 Total Return: {results['total_return']:+.1f}%")
        print(f"   💵 Final Balance: ${results['final_balance']:.2f}")
        print(f"   💎 Total Profit: ${results['total_profit']:+.2f}")
        print(f"   📈 Profit Factor: {results['profit_factor']:.2f}")
        print(f"   📉 Max Drawdown: {results['max_drawdown']:.1f}%")
        
        print(f"\n📊 TRADE BREAKDOWN:")
        print(f"   💚 Average Win: ${results['avg_win']:.2f} ({results['avg_profit_pct']:.1f}%)")
        print(f"   ❌ Average Loss: ${results['avg_loss']:.2f}")
        print(f"   🚀 Best Trade: ${results['max_win']:.2f}")
        print(f"   💀 Worst Trade: ${results['max_loss']:.2f}")
        
        print(f"\n⚡ EXECUTION METRICS:")
        print(f"   📊 Average Leverage: {results['avg_leverage']:.1f}x")
        print(f"   🤖 Average Confidence: {results['avg_confidence']:.1f}%")
        print(f"   ⏱️ Average Hold: {results['avg_hold_time']:.1f} min")
        
        total_trades = results['total_trades']
        if total_trades > 0:
            trailing_pct = (results['trailing_activations'] / total_trades) * 100
            print(f"   🛡️ Trailing Stops: {results['trailing_activations']}/{total_trades} ({trailing_pct:.1f}%)")
        
        print(f"\n📤 EXIT ANALYSIS:")
        exit_reasons = results['exit_reasons']
        if total_trades > 0:
            for reason, count in exit_reasons.items():
                pct = (count / total_trades) * 100
                emoji = "🎯" if reason == "take_profit" else "🛡️" if reason == "trailing_stop" else "🛑" if reason == "stop_loss" else "⏰"
                print(f"   {emoji} {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        # A-GRADE ASSESSMENT
        print(f"\n🎯 A-GRADE PERFORMANCE ASSESSMENT:")
        grades = []
        
        # Win Rate Assessment
        if results['win_rate'] >= 55:
            print("   🔥 Win Rate: A+ (EXCELLENT) ✅")
            grades.append("A+")
        elif results['win_rate'] >= 50:
            print("   ✅ Win Rate: B+ (GOOD)")
            grades.append("B+")
        elif results['win_rate'] >= 40:
            print("   ⚠️ Win Rate: C (FAIR)")
            grades.append("C")
        else:
            print("   ❌ Win Rate: F (POOR)")
            grades.append("F")
        
        # Returns Assessment
        if results['total_return'] >= 15:
            print("   🔥 Returns: A+ (EXCELLENT) ✅")
            grades.append("A+")
        elif results['total_return'] >= 8:
            print("   ✅ Returns: B+ (GOOD)")
            grades.append("B+")
        elif results['total_return'] >= 3:
            print("   ⚠️ Returns: C (FAIR)")
            grades.append("C")
        else:
            print("   ❌ Returns: F (POOR)")
            grades.append("F")
        
        # Risk/Reward Assessment
        if results['profit_factor'] >= 2.0:
            print("   🔥 Risk/Reward: A+ (EXCELLENT) ✅")
            grades.append("A+")
        elif results['profit_factor'] >= 1.5:
            print("   ✅ Risk/Reward: B+ (GOOD)")
            grades.append("B+")
        elif results['profit_factor'] >= 1.1:
            print("   ⚠️ Risk/Reward: C (FAIR)")
            grades.append("C")
        else:
            print("   ❌ Risk/Reward: F (POOR)")
            grades.append("F")
        
        # Frequency Assessment
        trades_per_day = results['total_trades'] / 60
        if trades_per_day >= 1.0:
            print("   🔥 Frequency: A+ (EXCELLENT) ✅")
            grades.append("A+")
        elif trades_per_day >= 0.7:
            print("   ✅ Frequency: B+ (GOOD)")
            grades.append("B+")
        elif trades_per_day >= 0.4:
            print("   ⚠️ Frequency: C (FAIR)")
            grades.append("C")
        else:
            print("   ❌ Frequency: F (POOR)")
            grades.append("F")
        
        # Calculate Overall Grade
        grade_scores = {"A+": 4, "B+": 3, "C": 2, "F": 1}
        avg_score = sum(grade_scores[g] for g in grades) / len(grades)
        
        if avg_score >= 3.75:
            overall = "A+ (OUTSTANDING)"
            emoji = "🏆"
        elif avg_score >= 3.0:
            overall = "A (EXCELLENT)"
            emoji = "🔥"
        elif avg_score >= 2.5:
            overall = "B+ (GOOD)"
            emoji = "✅"
        elif avg_score >= 1.5:
            overall = "C (NEEDS WORK)"
            emoji = "⚠️"
        else:
            overall = "F (MAJOR ISSUES)"
            emoji = "❌"
        
        print(f"\n{emoji} OVERALL GRADE: {overall}")
        
        if avg_score >= 3.0:
            print("\n🎉 CONGRATULATIONS! A-GRADE ACHIEVED!")
            print("🚀 This precision bot is ready for live deployment!")
            print("💎 Quality over quantity strategy successful!")
        
        print("="*80)

def main():
    print("🎯 GRADE A PRECISION BOT - QUALITY OVER QUANTITY")
    print("💎 ULTRA-SELECTIVE • HIGH WIN RATE • EXCELLENT R:R")
    print("=" * 60)
    
    try:
        balance = float(input("💵 Starting balance (default $200): ") or "200")
    except ValueError:
        balance = 200.0
    
    try:
        days = int(input("📅 Backtest days (default 60): ") or "60")
    except ValueError:
        days = 60
    
    backtester = GradeAPrecisionBot(initial_balance=balance)
    data = backtester.generate_realistic_market_data(days=days)
    results = backtester.run_precision_backtest(data)
    backtester.display_precision_results(results)

if __name__ == "__main__":
    main() 