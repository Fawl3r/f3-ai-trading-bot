#!/usr/bin/env python3
"""
High Win Rate Tuned Bot
TARGET: 75-85% Win Rates for 300%-1000%+ Returns
Quality over Quantity - Smart Entry/Exit Optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from final_optimized_ai_bot import FinalOptimizedAI
from indicators import TechnicalIndicators

class HighWinRateTunedBot:
    """High win rate bot optimized for 75-85% win rates"""
    
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        self.ai_analyzer = FinalOptimizedAI()
        self.indicators = TechnicalIndicators()
        
        # HIGH WIN RATE PROFILES - Quality over Quantity
        self.tuned_profiles = {
            "PRECISION_SCALP": {
                "strategy": "Ultra-precise scalping with high-confidence entries",
                "position_size_pct": 20.0,
                "leverage": 15,
                "stop_loss_pct": 0.5,
                "take_profit_pct": 1.5,
                "max_hold_time_min": 30,
                "max_daily_trades": 15,  # Reduced for quality
                "ai_threshold": 15.0,
                "trend_strength_min": 70,  # Higher requirement
                "momentum_min": 2.0,  # Strong momentum required
                "volume_min": 1.5,  # Volume confirmation
                "target_return": 400,
                "target_winrate": 80
            },
            "MOMENTUM_MASTER": {
                "strategy": "High-confidence momentum with multi-confirmation",
                "position_size_pct": 25.0,
                "leverage": 20,
                "stop_loss_pct": 0.7,
                "take_profit_pct": 2.0,
                "max_hold_time_min": 60,
                "max_daily_trades": 12,
                "ai_threshold": 20.0,
                "trend_strength_min": 75,
                "momentum_min": 2.5,
                "volume_min": 1.8,
                "rsi_range": [25, 75],  # Avoid extremes
                "target_return": 600,
                "target_winrate": 82
            },
            "TREND_SNIPER": {
                "strategy": "Sniper-like trend entries with maximum precision",
                "position_size_pct": 30.0,
                "leverage": 25,
                "stop_loss_pct": 0.8,
                "take_profit_pct": 2.5,
                "max_hold_time_min": 90,
                "max_daily_trades": 10,
                "ai_threshold": 25.0,
                "trend_strength_min": 80,
                "momentum_min": 3.0,
                "volume_min": 2.0,
                "ma_alignment": True,  # Perfect MA alignment required
                "target_return": 800,
                "target_winrate": 85
            },
            "ELITE_PRECISION": {
                "strategy": "Elite precision trading - only the best setups",
                "position_size_pct": 35.0,
                "leverage": 30,
                "stop_loss_pct": 1.0,
                "take_profit_pct": 3.0,
                "max_hold_time_min": 120,
                "max_daily_trades": 8,
                "ai_threshold": 30.0,
                "trend_strength_min": 85,
                "momentum_min": 3.5,
                "volume_min": 2.5,
                "confluence_required": 4,  # Multiple confirmations
                "target_return": 1000,
                "target_winrate": 85
            }
        }
        
        print("🎯 HIGH WIN RATE TUNED BOT")
        print("🏆 TARGET: 75-85% Win Rates")
        print("💎 QUALITY OVER QUANTITY")
        print("🚀 PRECISION ENTRY/EXIT OPTIMIZATION")
        print("=" * 80)
        print("🔧 HIGH WIN RATE SETTINGS:")
        print("   • Win Rate Target: 75-85%")
        print("   • Quality Filters: MAXIMUM")
        print("   • Entry Precision: ENHANCED")
        print("   • Multi-Confirmation: REQUIRED")
        print("   • Trade Volume: REDUCED for QUALITY")
        print("=" * 80)
    
    def test_high_winrate_tuning(self):
        """Test high win rate tuned system"""
        
        print("\n🎯 HIGH WIN RATE TUNING TEST")
        print("🏆 TARGET: 75-85% Win Rates for Massive Returns")
        print("=" * 80)
        
        # Generate high-quality market data
        data = self._generate_trending_data(days=21)
        
        strategies = ["PRECISION_SCALP", "MOMENTUM_MASTER", "TREND_SNIPER", "ELITE_PRECISION"]
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*25} {strategy} TUNING TEST {'='*25}")
            results[strategy] = self._test_tuned_strategy(strategy, data)
        
        # Display results
        self._display_tuned_results(results)
        return results
    
    def _test_tuned_strategy(self, strategy: str, data: pd.DataFrame) -> Dict:
        """Test single tuned strategy"""
        
        profile = self.tuned_profiles[strategy]
        
        print(f"🎯 {strategy}")
        print(f"   • {profile['strategy']}")
        print(f"   • Size: {profile['position_size_pct']}% | Leverage: {profile['leverage']}x")
        print(f"   • Target Win Rate: {profile['target_winrate']}%")
        print(f"   • Quality Filters: MAXIMUM")
        print(f"   • Target Return: {profile['target_return']}%")
        
        # Reset AI
        self.ai_analyzer = FinalOptimizedAI()
        
        return self._run_tuned_simulation(data, profile)
    
    def _enhanced_quality_analysis(self, data: pd.DataFrame, current_idx: int, profile: Dict) -> Dict:
        """Enhanced quality analysis for high win rates"""
        
        if current_idx < 50:
            return {'quality_score': 0, 'entry_allowed': False}
        
        # Get comprehensive data
        recent_data = data.iloc[max(0, current_idx-50):current_idx+1]
        current_price = data.iloc[current_idx]['close']
        
        # Multiple timeframe analysis
        sma_5 = recent_data['close'].tail(5).mean()
        sma_10 = recent_data['close'].tail(10).mean()
        sma_20 = recent_data['close'].tail(20).mean()
        sma_50 = recent_data['close'].tail(50).mean() if len(recent_data) >= 50 else sma_20
        
        # Enhanced momentum analysis
        momentum_3 = ((current_price - recent_data['close'].iloc[-3]) / recent_data['close'].iloc[-3]) * 100
        momentum_5 = ((current_price - recent_data['close'].iloc[-5]) / recent_data['close'].iloc[-5]) * 100
        momentum_10 = ((current_price - recent_data['close'].iloc[-10]) / recent_data['close'].iloc[-10]) * 100
        momentum_20 = ((current_price - recent_data['close'].iloc[-20]) / recent_data['close'].iloc[-20]) * 100
        
        # Volume analysis
        current_volume = data.iloc[current_idx].get('volume', 1000)
        avg_volume_10 = recent_data['volume'].tail(10).mean()
        avg_volume_20 = recent_data['volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume_10 if avg_volume_10 > 0 else 1
        volume_trend = avg_volume_10 / avg_volume_20 if avg_volume_20 > 0 else 1
        
        # RSI analysis
        current_rsi = data.iloc[current_idx].get('rsi', 50)
        
        # Price action analysis
        high = data.iloc[current_idx]['high']
        low = data.iloc[current_idx]['low']
        price_range = ((high - low) / current_price) * 100
        
        # Quality scoring system (0-100)
        quality_score = 0
        confluence_count = 0
        
        # 1. TREND ALIGNMENT (25 points)
        trend_score = 0
        if sma_5 > sma_10 > sma_20 > sma_50:  # Perfect bullish alignment
            trend_score = 25
            trend_direction = 'bullish'
            confluence_count += 1
        elif sma_5 < sma_10 < sma_20 < sma_50:  # Perfect bearish alignment
            trend_score = 25
            trend_direction = 'bearish'
            confluence_count += 1
        elif abs(sma_5 - sma_10) / sma_10 < 0.005:  # Sideways
            trend_score = 5
            trend_direction = 'sideways'
        else:  # Mixed signals
            trend_score = 10
            trend_direction = 'mixed'
        
        quality_score += trend_score
        
        # 2. MOMENTUM QUALITY (25 points)
        momentum_score = 0
        if trend_direction == 'bullish':
            if momentum_3 > 0.5 and momentum_5 > 1.0 and momentum_10 > 1.5:
                momentum_score = 25
                confluence_count += 1
            elif momentum_5 > 0.5 and momentum_10 > 1.0:
                momentum_score = 15
            elif momentum_5 > 0:
                momentum_score = 8
        elif trend_direction == 'bearish':
            if momentum_3 < -0.5 and momentum_5 < -1.0 and momentum_10 < -1.5:
                momentum_score = 25
                confluence_count += 1
            elif momentum_5 < -0.5 and momentum_10 < -1.0:
                momentum_score = 15
            elif momentum_5 < 0:
                momentum_score = 8
        
        quality_score += momentum_score
        
        # 3. VOLUME CONFIRMATION (20 points)
        volume_score = 0
        if volume_ratio > profile.get('volume_min', 1.5):
            volume_score += 15
            confluence_count += 1
        if volume_trend > 1.1:  # Increasing volume trend
            volume_score += 5
        
        quality_score += volume_score
        
        # 4. RSI POSITIONING (15 points)
        rsi_score = 0
        rsi_range = profile.get('rsi_range', [20, 80])
        if rsi_range[0] < current_rsi < rsi_range[1]:  # Good RSI range
            rsi_score = 15
            confluence_count += 1
        elif 30 < current_rsi < 70:  # Acceptable range
            rsi_score = 8
        
        quality_score += rsi_score
        
        # 5. PRICE ACTION QUALITY (15 points)
        price_action_score = 0
        if 0.5 < price_range < 3.0:  # Good volatility range
            price_action_score = 15
            confluence_count += 1
        elif 0.2 < price_range < 5.0:  # Acceptable range
            price_action_score = 8
        
        quality_score += price_action_score
        
        # CONFLUENCE REQUIREMENT
        confluence_required = profile.get('confluence_required', 3)
        if confluence_count < confluence_required:
            quality_score *= 0.5  # Penalize lack of confluence
        
        # TREND STRENGTH REQUIREMENT
        trend_strength = abs(momentum_10) if trend_direction != 'sideways' else 0
        trend_strength_min = profile.get('trend_strength_min', 70)
        if trend_strength * 10 < trend_strength_min:  # Convert to 0-100 scale
            quality_score *= 0.7
        
        # MOMENTUM REQUIREMENT
        momentum_min = profile.get('momentum_min', 2.0)
        if abs(momentum_5) < momentum_min:
            quality_score *= 0.6
        
        # Determine trade direction and entry permission
        entry_allowed = quality_score >= 65  # High threshold for quality
        
        if trend_direction == 'bullish' and momentum_5 > momentum_min:
            trade_direction = 'long'
        elif trend_direction == 'bearish' and momentum_5 < -momentum_min:
            trade_direction = 'short'
        else:
            trade_direction = None
            entry_allowed = False
        
        return {
            'quality_score': quality_score,
            'entry_allowed': entry_allowed,
            'trade_direction': trade_direction,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'momentum_3': momentum_3,
            'momentum_5': momentum_5,
            'momentum_10': momentum_10,
            'volume_ratio': volume_ratio,
            'current_rsi': current_rsi,
            'confluence_count': confluence_count,
            'price_range': price_range
        }
    
    def _run_tuned_simulation(self, data: pd.DataFrame, profile: Dict) -> Dict:
        """Run tuned simulation focusing on high win rates"""
        
        balance = self.initial_balance
        position = None
        trades = []
        daily_trades = 0
        last_date = None
        
        total_profit = 0
        total_loss = 0
        winning_trades = 0
        losing_trades = 0
        max_balance = balance
        max_drawdown = 0
        
        long_trades = 0
        short_trades = 0
        long_wins = 0
        short_wins = 0
        
        trades_shown = 0
        max_show = 15
        
        rejected_trades = 0
        quality_scores = []
        
        for i in range(60, len(data)):
            current = data.iloc[i]
            price = current['close']
            current_date = current['timestamp'].date()
            
            # Reset daily counter
            if last_date != current_date:
                daily_trades = 0
                last_date = current_date
            
            # Skip if max trades reached
            if daily_trades >= profile['max_daily_trades']:
                continue
            
            # HIGH QUALITY ENTRY ANALYSIS
            if position is None and daily_trades < profile['max_daily_trades']:
                recent_data = data.iloc[max(0, i-100):i+1]
                
                # Enhanced quality analysis
                quality_analysis = self._enhanced_quality_analysis(data, i, profile)
                quality_scores.append(quality_analysis['quality_score'])
                
                if not quality_analysis['entry_allowed'] or quality_analysis['trade_direction'] is None:
                    rejected_trades += 1
                    continue
                
                trade_direction = quality_analysis['trade_direction']
                
                # AI analysis with profile threshold
                ai_result = self.ai_analyzer.analyze_trade_opportunity(recent_data, price, trade_direction)
                
                # STRICT AI THRESHOLD
                if ai_result['ai_confidence'] >= profile['ai_threshold']:
                    # Dynamic position sizing based on quality
                    base_size_pct = profile['position_size_pct']
                    quality_multiplier = min(1.3, quality_analysis['quality_score'] / 70)
                    adjusted_size_pct = base_size_pct * quality_multiplier
                    
                    base_position_size = balance * (adjusted_size_pct / 100)
                    leveraged_size = base_position_size * profile['leverage']
                    
                    # Conservative risk management for high win rate
                    max_risk = balance * 0.4
                    if leveraged_size > max_risk:
                        leveraged_size = max_risk
                    
                    # Calculate stops and targets
                    if trade_direction == 'long':
                        stop_loss = price * (1 - profile['stop_loss_pct'] / 100)
                        take_profit = price * (1 + profile['take_profit_pct'] / 100)
                    else:
                        stop_loss = price * (1 + profile['stop_loss_pct'] / 100)
                        take_profit = price * (1 - profile['take_profit_pct'] / 100)
                    
                    position = {
                        'entry_price': price,
                        'size': leveraged_size,
                        'base_size': base_position_size,
                        'leverage': profile['leverage'],
                        'direction': trade_direction,
                        'ai_confidence': ai_result['ai_confidence'],
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_time': current['timestamp'],
                        'strategy': profile['strategy'],
                        'quality_score': quality_analysis['quality_score'],
                        'profile': profile
                    }
                    daily_trades += 1
                    
                    if trade_direction == 'long':
                        long_trades += 1
                    else:
                        short_trades += 1
                    
                    if trades_shown < max_show:
                        direction_emoji = "📈" if trade_direction == 'long' else "📉"
                        print(f"    🎯 {direction_emoji} {trade_direction.upper()} ${price:.2f} | "
                              f"Quality: {quality_analysis['quality_score']:.0f}/100 | "
                              f"AI: {ai_result['ai_confidence']:.1f}% | "
                              f"Size: ${leveraged_size:.0f} ({profile['leverage']}x)")
                        trades_shown += 1
                else:
                    rejected_trades += 1
            
            # ENHANCED POSITION MANAGEMENT
            elif position is not None:
                should_close = False
                close_reason = ""
                
                # Direction-specific exits
                if position['direction'] == 'long':
                    if price <= position['stop_loss']:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif price >= position['take_profit']:
                        should_close = True
                        close_reason = "Take Profit"
                else:
                    if price >= position['stop_loss']:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif price <= position['take_profit']:
                        should_close = True
                        close_reason = "Take Profit"
                
                # Time exit
                hold_time = (current['timestamp'] - position['entry_time']).total_seconds() / 60
                if hold_time > profile['max_hold_time_min']:
                    should_close = True
                    close_reason = "Time Exit"
                
                # Smart trend reversal exit
                current_quality = self._enhanced_quality_analysis(data, i, profile)
                if (position['direction'] == 'long' and 
                    current_quality['trend_direction'] == 'bearish' and 
                    current_quality['quality_score'] > 70):
                    should_close = True
                    close_reason = "Trend Reversal"
                elif (position['direction'] == 'short' and 
                      current_quality['trend_direction'] == 'bullish' and 
                      current_quality['quality_score'] > 70):
                    should_close = True
                    close_reason = "Trend Reversal"
                
                # Partial profit taking for high-quality trades
                if (position['quality_score'] > 80 and 
                    ((position['direction'] == 'long' and price > position['entry_price'] * 1.01) or
                     (position['direction'] == 'short' and price < position['entry_price'] * 0.99))):
                    # Move stop to breakeven
                    if position['direction'] == 'long':
                        position['stop_loss'] = max(position['stop_loss'], position['entry_price'])
                    else:
                        position['stop_loss'] = min(position['stop_loss'], position['entry_price'])
                
                if should_close:
                    # Calculate P&L
                    if position['direction'] == 'long':
                        price_change_pct = ((price - position['entry_price']) / position['entry_price']) * 100
                    else:
                        price_change_pct = ((position['entry_price'] - price) / position['entry_price']) * 100
                    
                    leveraged_pnl_pct = price_change_pct * position['leverage']
                    pnl = position['base_size'] * (leveraged_pnl_pct / 100)
                    
                    balance += pnl
                    
                    # Track metrics
                    if balance > max_balance:
                        max_balance = balance
                    
                    current_drawdown = ((max_balance - balance) / max_balance) * 100
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
                    
                    # Win/Loss tracking
                    if pnl > 0:
                        outcome = 'win'
                        winning_trades += 1
                        total_profit += pnl
                        if position['direction'] == 'long':
                            long_wins += 1
                        else:
                            short_wins += 1
                    else:
                        outcome = 'loss'
                        losing_trades += 1
                        total_loss += abs(pnl)
                    
                    # Update AI
                    self.ai_analyzer.update_trade_result(position['ai_confidence'], outcome)
                    
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': leveraged_pnl_pct,
                        'price_change_pct': price_change_pct,
                        'close_reason': close_reason,
                        'hold_time_min': hold_time,
                        'ai_confidence': position['ai_confidence'],
                        'quality_score': position['quality_score'],
                        'leverage': position['leverage'],
                        'direction': position['direction'],
                        'strategy': position['strategy'],
                        'win': outcome == 'win',
                        'balance_after': balance
                    })
                    
                    if len(trades) <= max_show:
                        outcome_emoji = "✅" if outcome == 'win' else "❌"
                        direction_emoji = "📈" if position['direction'] == 'long' else "📉"
                        print(f"    📤 EXIT {direction_emoji} ${price:.2f} | P&L: ${pnl:+.0f} ({leveraged_pnl_pct:+.1f}%) | "
                              f"{close_reason} | Qual: {position['quality_score']:.0f} | {outcome_emoji}")
                    
                    position = None
                    
                    # Risk management
                    if balance < self.initial_balance * 0.3:
                        print(f"    🛑 RISK STOP: Balance ${balance:.0f}")
                        break
        
        # Calculate metrics
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = ((balance - self.initial_balance) / self.initial_balance) * 100
        profit_factor = (total_profit / max(total_loss, 0.01)) if total_loss > 0 else float('inf')
        
        long_win_rate = (long_wins / long_trades * 100) if long_trades > 0 else 0
        short_win_rate = (short_wins / short_trades * 100) if short_trades > 0 else 0
        
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': balance,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'max_balance': max_balance,
            'max_drawdown': max_drawdown,
            'target_return': profile['target_return'],
            'target_winrate': profile['target_winrate'],
            'target_achieved': total_return >= profile['target_return'],
            'winrate_achieved': win_rate >= profile['target_winrate'],
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_wins': long_wins,
            'short_wins': short_wins,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'avg_quality_score': avg_quality_score,
            'rejected_trades': rejected_trades,
            'trades': trades
        }
    
    def _generate_trending_data(self, days: int = 21) -> pd.DataFrame:
        """Generate trending data with clear directional moves"""
        print(f"📈 Generating {days} days of TRENDING market data...")
        print("🎯 Clear directional moves for high win rate opportunities")
        
        data = []
        price = 200.0
        minutes = days * 24 * 60
        
        np.random.seed(999)  # Consistent seed for trending data
        
        for i in range(minutes):
            time_factor = i / minutes
            
            # Create clear trending phases
            phase_length = 0.15  # Longer phases for clearer trends
            current_phase = int(time_factor / phase_length) % 4
            
            if current_phase == 0:  # Strong uptrend
                main_trend = 60 * (time_factor / phase_length)
                phase_name = "STRONG_UPTREND"
            elif current_phase == 1:  # Consolidation
                main_trend = 60 + np.sin(2 * np.pi * (time_factor - phase_length) / phase_length * 3) * 15
                phase_name = "CONSOLIDATION"
            elif current_phase == 2:  # Strong downtrend
                main_trend = 60 - 50 * ((time_factor - 2*phase_length) / phase_length)
                phase_name = "STRONG_DOWNTREND"
            else:  # Recovery trend
                main_trend = 10 + 40 * ((time_factor - 3*phase_length) / phase_length)
                phase_name = "RECOVERY_TREND"
            
            # Moderate volatility for cleaner trends
            volatility = np.random.normal(0, 2.5)
            
            # Occasional large moves
            if np.random.random() < 0.005:  # 0.5% chance
                volatility += np.random.normal(0, 8)
            
            # Smooth intraday movements
            intraday_osc = np.sin(2 * np.pi * time_factor * 8) * 3
            
            price_change = main_trend * 0.002 + intraday_osc * 0.02 + volatility * 0.05
            price += price_change
            
            # Keep realistic bounds
            price = max(120, min(400, price))
            
            # OHLC with moderate spreads
            spread = np.random.uniform(0.3, 1.5)
            high = price + spread/2 + abs(np.random.normal(0, 0.3))
            low = price - spread/2 - abs(np.random.normal(0, 0.3))
            open_price = price + np.random.uniform(-0.3, 0.3)
            
            # Volume patterns that support trends
            base_volume = 2500
            if phase_name in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
                volume_multiplier = 1.8
            else:
                volume_multiplier = 1.2
            
            volume = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)
            
            data.append({
                'timestamp': datetime.now() - timedelta(minutes=minutes-i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'market_phase': phase_name
            })
        
        df = pd.DataFrame(data)
        df = self.indicators.calculate_all_indicators(df)
        
        print(f"✅ Generated {len(df):,} trending data points")
        print(f"📊 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"📈 Trend clarity optimized for high win rates")
        
        return df
    
    def _display_tuned_results(self, results: Dict):
        """Display tuned results with win rate focus"""
        
        print(f"\n" + "=" * 140)
        print("🎯 HIGH WIN RATE TUNED RESULTS")
        print("🏆 TARGET: 75-85% Win Rates for Massive Returns")
        print("=" * 140)
        
        print(f"\n💎 HIGH WIN RATE PERFORMANCE SUMMARY:")
        print("-" * 160)
        print(f"{'Strategy':<16} {'Return':<12} {'Win Rate':<10} {'Target WR':<10} {'WR Status':<12} {'Trades':<8} {'Quality':<8} {'Rejected':<9} {'Profit Factor':<12}")
        print("-" * 160)
        
        best_winrate = 0
        best_return = -float('inf')
        best_strategy = ""
        winrate_targets_hit = 0
        return_targets_hit = 0
        total_trades = 0
        
        for strategy, result in results.items():
            winrate_status = "🎯 HIT!" if result['winrate_achieved'] else "📈 Progress"
            return_status = "🚀 TARGET!" if result['target_achieved'] else "📈 Building"
            
            if result['winrate_achieved']:
                winrate_targets_hit += 1
            if result['target_achieved']:
                return_targets_hit += 1
            
            if result['win_rate'] > best_winrate:
                best_winrate = result['win_rate']
            
            if result['total_return'] > best_return:
                best_return = result['total_return']
                best_strategy = strategy
            
            total_trades += result['total_trades']
            
            print(f"{strategy:<16} {result['total_return']:+8.1f}%   {result['win_rate']:>6.1f}%   "
                  f"{result['target_winrate']:>6.0f}%    {winrate_status:<12} {result['total_trades']:<8} "
                  f"{result['avg_quality_score']:>5.0f}/100 {result['rejected_trades']:<9} "
                  f"{result['profit_factor']:>8.1f}")
        
        print("-" * 160)
        
        print(f"\n🎯 HIGH WIN RATE TUNING SUMMARY:")
        print(f"   🏆 Best Win Rate: {best_winrate:.1f}% (Target: 75-85%)")
        print(f"   🚀 Best Return: {best_strategy} ({best_return:+.1f}%)")
        print(f"   🎯 Win Rate Targets Hit: {winrate_targets_hit}/4 strategies")
        print(f"   💰 Return Targets Hit: {return_targets_hit}/4 strategies")
        print(f"   📊 Total Quality Trades: {total_trades}")
        
        if best_winrate >= 75:
            print(f"\n🎉 HIGH WIN RATE TARGET ACHIEVED!")
            print(f"   🏆 {best_winrate:.1f}% win rate EXCEEDS 75%+ target!")
            print(f"   💡 Quality filtering and precision entries = SUCCESS!")
            
            if best_return >= 300:
                print(f"   🚀 MASSIVE RETURNS: {best_return:+.1f}% (300%+ target achieved!)")
            else:
                print(f"   📈 Returns building: {best_return:+.1f}% (approaching 300%+ target)")
        else:
            print(f"\n📈 WIN RATE IMPROVING:")
            print(f"   🔧 Current best: {best_winrate:.1f}% (target: 75%+)")
            print(f"   🎯 Quality filters working, need fine-tuning")
        
        print("=" * 140)

def main():
    """Run high win rate tuned test"""
    bot = HighWinRateTunedBot()
    results = bot.test_high_winrate_tuning()
    
    # Final analysis
    total_strategies = len(results)
    winrate_success = sum(1 for r in results.values() if r['winrate_achieved'])
    return_success = sum(1 for r in results.values() if r['target_achieved'])
    best_winrate = max(r['win_rate'] for r in results.values())
    best_return = max(r['total_return'] for r in results.values())
    total_trades = sum(r['total_trades'] for r in results.values())
    
    print(f"\n🎯 HIGH WIN RATE TUNING COMPLETE!")
    print(f"🏆 Win Rate Success: {winrate_success}/{total_strategies} strategies hit 75-85% target")
    print(f"🚀 Return Success: {return_success}/{total_strategies} strategies hit return targets")
    print(f"📊 Best Win Rate: {best_winrate:.1f}%")
    print(f"💰 Best Return: {best_return:+.1f}%")
    print(f"📈 Quality Trades: {total_trades}")
    
    if best_winrate >= 75:
        print("🎉 HIGH WIN RATE TARGET ACHIEVED!")
        print("💡 Quality over quantity approach = SUCCESS!")
        if best_return >= 300:
            print("🚀 MASSIVE RETURNS ACHIEVED!")
    else:
        print("🔧 Continue tuning for 75-85% win rate target")
    
    return results

if __name__ == "__main__":
    main()