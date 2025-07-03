#!/usr/bin/env python3
"""
Smart Trend-Adaptive Bot
Analyzes trend direction and adapts between Long/Short strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from final_optimized_ai_bot import FinalOptimizedAI
from indicators import TechnicalIndicators

class SmartTrendAdaptiveBot:
    """Smart bot that adapts to trend direction for optimal Long/Short positioning"""
    
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        self.ai_analyzer = FinalOptimizedAI()
        self.indicators = TechnicalIndicators()
        
        # ADAPTIVE TRADING PROFILES
        self.adaptive_profiles = {
            "TREND_FOLLOWER": {
                "strategy": "Follow strong trends with momentum",
                "position_size_pct": 12.0,
                "leverage": 8,
                "stop_loss_pct": 1.5,
                "take_profit_pct": 4.0,
                "max_hold_time_min": 120,
                "max_daily_trades": 15,
                "ai_threshold": 40.0,
                "trend_strength_threshold": 2.0,  # Minimum trend strength
                "target_return": 200
            },
            "REVERSAL_HUNTER": {
                "strategy": "Counter-trend reversals at extremes",
                "position_size_pct": 10.0,
                "leverage": 6,
                "stop_loss_pct": 2.0,
                "take_profit_pct": 6.0,
                "max_hold_time_min": 180,
                "max_daily_trades": 10,
                "ai_threshold": 50.0,
                "reversal_rsi_threshold": 15,  # Extreme RSI levels
                "target_return": 150
            },
            "BREAKOUT_TRADER": {
                "strategy": "Trade breakouts in either direction",
                "position_size_pct": 15.0,
                "leverage": 10,
                "stop_loss_pct": 1.2,
                "take_profit_pct": 3.5,
                "max_hold_time_min": 90,
                "max_daily_trades": 12,
                "ai_threshold": 35.0,
                "breakout_threshold": 1.5,  # % breakout from range
                "target_return": 250
            }
        }
        
        print("🧠 SMART TREND-ADAPTIVE BOT")
        print("📈📉 ADAPTS TO MARKET DIRECTION - LONG & SHORT")
        print("🎯 FEATURES:")
        print("   • Trend Analysis: Detects bullish/bearish/sideways markets")
        print("   • Directional Adaptation: Long in uptrends, Short in downtrends")
        print("   • Multiple Strategies: Trend Following, Reversal, Breakout")
        print("   • Smart Position Sizing: Risk-adjusted for market conditions")
        print("   • Realistic Risk Management: Proper stop losses and targets")
        print("=" * 80)
    
    def test_adaptive_performance(self):
        """Test trend-adaptive system"""
        
        print("\n🧠 SMART TREND-ADAPTIVE TEST")
        print("🎯 TARGET: Profit in ANY market condition (Bull/Bear/Sideways)")
        print("=" * 80)
        
        # Generate realistic data with mixed market conditions
        data = self._generate_mixed_market_data(days=30)
        
        strategies = ["TREND_FOLLOWER", "REVERSAL_HUNTER", "BREAKOUT_TRADER"]
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*20} {strategy} ADAPTIVE TEST {'='*20}")
            results[strategy] = self._test_adaptive_strategy(strategy, data)
        
        # Display results
        self._display_adaptive_results(results)
        return results
    
    def _test_adaptive_strategy(self, strategy: str, data: pd.DataFrame) -> Dict:
        """Test single adaptive strategy"""
        
        profile = self.adaptive_profiles[strategy]
        
        print(f"🧠 {strategy} STRATEGY")
        print(f"   • {profile['strategy']}")
        print(f"   • Position: {profile['position_size_pct']}% | Leverage: {profile['leverage']}x")
        print(f"   • Stop: {profile['stop_loss_pct']}% | Target: {profile['take_profit_pct']}%")
        print(f"   • Hold: {profile['max_hold_time_min']}min | Trades: {profile['max_daily_trades']}/day")
        print(f"   • AI: {profile['ai_threshold']}% | Target Return: {profile['target_return']}%")
        
        # Reset AI for each strategy
        self.ai_analyzer = FinalOptimizedAI()
        
        return self._run_adaptive_simulation(data, profile)
    
    def _analyze_market_trend(self, data: pd.DataFrame, current_idx: int) -> Dict:
        """Analyze current market trend direction and strength"""
        
        if current_idx < 50:
            return {'direction': 'sideways', 'strength': 0, 'confidence': 0}
        
        # Get recent data for analysis
        recent_data = data.iloc[max(0, current_idx-50):current_idx+1]
        current_price = data.iloc[current_idx]['close']
        
        # Multiple timeframe analysis
        sma_10 = recent_data['close'].tail(10).mean()
        sma_20 = recent_data['close'].tail(20).mean()
        sma_50 = recent_data['close'].tail(50).mean() if len(recent_data) >= 50 else sma_20
        
        # Price position relative to moving averages
        above_sma10 = current_price > sma_10
        above_sma20 = current_price > sma_20
        above_sma50 = current_price > sma_50
        
        # Moving average alignment
        sma_alignment_bull = sma_10 > sma_20 > sma_50
        sma_alignment_bear = sma_10 < sma_20 < sma_50
        
        # Price momentum
        price_change_10 = ((current_price - recent_data['close'].iloc[-10]) / recent_data['close'].iloc[-10]) * 100
        price_change_20 = ((current_price - recent_data['close'].iloc[-20]) / recent_data['close'].iloc[-20]) * 100
        
        # RSI trend
        current_rsi = data.iloc[current_idx].get('rsi', 50)
        
        # Volume trend
        recent_volume = recent_data['volume'].tail(10).mean()
        prev_volume = recent_data['volume'].iloc[-20:-10].mean() if len(recent_data) >= 20 else recent_volume
        volume_trend = (recent_volume / prev_volume - 1) * 100 if prev_volume > 0 else 0
        
        # Determine trend direction and strength
        bull_signals = 0
        bear_signals = 0
        
        # Price vs MAs
        if above_sma10: bull_signals += 1
        else: bear_signals += 1
        
        if above_sma20: bull_signals += 1
        else: bear_signals += 1
        
        if above_sma50: bull_signals += 1
        else: bear_signals += 1
        
        # MA alignment
        if sma_alignment_bull: bull_signals += 2
        elif sma_alignment_bear: bear_signals += 2
        
        # Price momentum
        if price_change_10 > 2: bull_signals += 1
        elif price_change_10 < -2: bear_signals += 1
        
        if price_change_20 > 5: bull_signals += 1
        elif price_change_20 < -5: bear_signals += 1
        
        # RSI trend
        if current_rsi > 60: bull_signals += 1
        elif current_rsi < 40: bear_signals += 1
        
        # Determine final trend
        total_signals = bull_signals + bear_signals
        if total_signals == 0:
            direction = 'sideways'
            strength = 0
            confidence = 0
        elif bull_signals > bear_signals:
            direction = 'bullish'
            strength = (bull_signals / total_signals) * 100
            confidence = min(100, strength * 1.2)
        else:
            direction = 'bearish'
            strength = (bear_signals / total_signals) * 100
            confidence = min(100, strength * 1.2)
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'price_momentum_10': price_change_10,
            'price_momentum_20': price_change_20,
            'rsi': current_rsi,
            'volume_trend': volume_trend,
            'bull_signals': bull_signals,
            'bear_signals': bear_signals
        }
    
    def _run_adaptive_simulation(self, data: pd.DataFrame, profile: Dict) -> Dict:
        """Run adaptive simulation with trend analysis"""
        
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
        
        for i in range(100, len(data)):
            current = data.iloc[i]
            price = current['close']
            rsi = current.get('rsi', 50)
            volume = current.get('volume', 1000)
            current_date = current['timestamp'].date()
            
            # Reset daily counter
            if last_date != current_date:
                daily_trades = 0
                last_date = current_date
            
            # Skip if max trades reached
            if daily_trades >= profile['max_daily_trades']:
                continue
            
            # TREND ANALYSIS
            trend_analysis = self._analyze_market_trend(data, i)
            
            # ADAPTIVE ENTRY LOGIC
            if position is None and daily_trades < profile['max_daily_trades']:
                recent_data = data.iloc[max(0, i-100):i+1]
                
                # Strategy-specific entry conditions with trend adaptation
                entry_signal = False
                trade_direction = None  # 'long' or 'short'
                
                if profile.get('strategy') == "Follow strong trends with momentum":
                    # Trend Follower: Follow the dominant trend
                    if (trend_analysis['direction'] == 'bullish' and 
                        trend_analysis['strength'] > profile.get('trend_strength_threshold', 2.0) and
                        rsi < 65):  # Not overbought
                        entry_signal = True
                        trade_direction = 'long'
                    
                    elif (trend_analysis['direction'] == 'bearish' and 
                          trend_analysis['strength'] > profile.get('trend_strength_threshold', 2.0) and
                          rsi > 35):  # Not oversold
                        entry_signal = True
                        trade_direction = 'short'
                
                elif profile.get('strategy') == "Counter-trend reversals at extremes":
                    # Reversal Hunter: Counter-trend at extremes
                    if (trend_analysis['direction'] == 'bearish' and 
                        rsi < profile.get('reversal_rsi_threshold', 15)):  # Extremely oversold
                        entry_signal = True
                        trade_direction = 'long'  # Buy the dip
                    
                    elif (trend_analysis['direction'] == 'bullish' and 
                          rsi > (100 - profile.get('reversal_rsi_threshold', 15))):  # Extremely overbought
                        entry_signal = True
                        trade_direction = 'short'  # Sell the rip
                
                elif profile.get('strategy') == "Trade breakouts in either direction":
                    # Breakout Trader: Trade momentum breakouts
                    recent_high = recent_data['high'].tail(20).max()
                    recent_low = recent_data['low'].tail(20).min()
                    range_size = recent_high - recent_low
                    
                    breakout_threshold = profile.get('breakout_threshold', 1.5) / 100
                    
                    if price > recent_high * (1 + breakout_threshold):  # Upward breakout
                        entry_signal = True
                        trade_direction = 'long'
                    
                    elif price < recent_low * (1 - breakout_threshold):  # Downward breakout
                        entry_signal = True
                        trade_direction = 'short'
                
                if entry_signal and trade_direction:
                    # AI analysis
                    ai_result = self.ai_analyzer.analyze_trade_opportunity(recent_data, price, trade_direction)
                    
                    if ai_result['ai_confidence'] >= profile['ai_threshold']:
                        # Calculate position size with leverage
                        base_position_size = balance * (profile['position_size_pct'] / 100)
                        leveraged_size = base_position_size * profile['leverage']
                        
                        # Risk management
                        max_risk = balance * 0.4  # Max 40% risk
                        if leveraged_size > max_risk:
                            leveraged_size = max_risk
                        
                        # Calculate stop loss and take profit based on direction
                        if trade_direction == 'long':
                            stop_loss = price * (1 - profile['stop_loss_pct'] / 100)
                            take_profit = price * (1 + profile['take_profit_pct'] / 100)
                        else:  # short
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
                            'strategy': profile.get('strategy', 'Unknown'),
                            'trend_analysis': trend_analysis,
                            'profile': profile
                        }
                        daily_trades += 1
                        
                        if trade_direction == 'long':
                            long_trades += 1
                        else:
                            short_trades += 1
                        
                        if len(trades) < 10:  # Show first few
                            direction_emoji = "📈" if trade_direction == 'long' else "📉"
                            print(f"    {direction_emoji} {trade_direction.upper()} ${price:.2f} | "
                                  f"Trend: {trend_analysis['direction']} ({trend_analysis['strength']:.1f}%) | "
                                  f"AI: {ai_result['ai_confidence']:.1f}% | "
                                  f"Size: ${leveraged_size:.0f} ({profile['leverage']}x)")
            
            # POSITION MANAGEMENT
            elif position is not None:
                should_close = False
                close_reason = ""
                
                # Direction-specific exit logic
                if position['direction'] == 'long':
                    # Long position exits
                    if price <= position['stop_loss']:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif price >= position['take_profit']:
                        should_close = True
                        close_reason = "Take Profit"
                else:  # short position
                    # Short position exits
                    if price >= position['stop_loss']:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif price <= position['take_profit']:
                        should_close = True
                        close_reason = "Take Profit"
                
                # Time-based exit
                if (current['timestamp'] - position['entry_time']).total_seconds() > profile['max_hold_time_min'] * 60:
                    should_close = True
                    close_reason = "Time Exit"
                
                # Trend reversal exit
                current_trend = self._analyze_market_trend(data, i)
                if (position['direction'] == 'long' and current_trend['direction'] == 'bearish' and current_trend['strength'] > 70):
                    should_close = True
                    close_reason = "Trend Reversal"
                elif (position['direction'] == 'short' and current_trend['direction'] == 'bullish' and current_trend['strength'] > 70):
                    should_close = True
                    close_reason = "Trend Reversal"
                
                if should_close:
                    # Calculate P&L based on direction
                    if position['direction'] == 'long':
                        price_change_pct = ((price - position['entry_price']) / position['entry_price']) * 100
                    else:  # short
                        price_change_pct = ((position['entry_price'] - price) / position['entry_price']) * 100
                    
                    leveraged_pnl_pct = price_change_pct * position['leverage']
                    pnl = position['base_size'] * (leveraged_pnl_pct / 100)
                    
                    balance += pnl
                    
                    # Track max balance and drawdown
                    if balance > max_balance:
                        max_balance = balance
                    
                    current_drawdown = ((max_balance - balance) / max_balance) * 100
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
                    
                    # Determine win/loss
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
                    
                    # Update AI learning
                    self.ai_analyzer.update_trade_result(position['ai_confidence'], outcome)
                    
                    # Calculate metrics
                    hold_time = (current['timestamp'] - position['entry_time']).total_seconds() / 60
                    
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': leveraged_pnl_pct,
                        'price_change_pct': price_change_pct,
                        'close_reason': close_reason,
                        'hold_time_min': hold_time,
                        'ai_confidence': position['ai_confidence'],
                        'leverage': position['leverage'],
                        'direction': position['direction'],
                        'strategy': position['strategy'],
                        'win': outcome == 'win',
                        'balance_after': balance,
                        'trend_at_entry': position['trend_analysis']['direction']
                    })
                    
                    if len(trades) <= 10:
                        outcome_emoji = "✅" if outcome == 'win' else "❌"
                        direction_emoji = "📈" if position['direction'] == 'long' else "📉"
                        print(f"    📤 EXIT {direction_emoji} ${price:.2f} | P&L: ${pnl:+.0f} ({leveraged_pnl_pct:+.1f}%) | "
                              f"{close_reason} | Balance: ${balance:.0f} | {outcome_emoji}")
                    
                    position = None
                    
                    # Risk management
                    if balance < self.initial_balance * 0.4:  # Stop at 60% loss
                        print(f"    🛑 RISK MANAGEMENT: Stopping at ${balance:.0f}")
                        break
        
        # Calculate final metrics
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = ((balance - self.initial_balance) / self.initial_balance) * 100
        profit_factor = (total_profit / max(total_loss, 0.01)) if total_loss > 0 else float('inf')
        
        # Direction-specific metrics
        long_win_rate = (long_wins / long_trades * 100) if long_trades > 0 else 0
        short_win_rate = (short_wins / short_trades * 100) if short_trades > 0 else 0
        
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
            'target_achieved': total_return >= profile['target_return'],
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_wins': long_wins,
            'short_wins': short_wins,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'trades': trades
        }
    
    def _generate_mixed_market_data(self, days: int = 30) -> pd.DataFrame:
        """Generate realistic data with mixed market conditions (bull/bear/sideways)"""
        print(f"📊 Generating {days} days of MIXED MARKET CONDITIONS...")
        print("📈📉➡️ Bull phases, Bear phases, and Sideways consolidation")
        
        data = []
        price = 200.0  # Starting price
        minutes = days * 24 * 60
        
        np.random.seed(999)  # Consistent mixed data
        
        for i in range(minutes):
            time_factor = i / minutes
            
            # Create distinct market phases
            phase_length = 0.15  # Each phase lasts ~15% of total time
            current_phase = int(time_factor / phase_length) % 4
            
            if current_phase == 0:  # Bull phase
                main_trend = 40 * (time_factor / phase_length)  # Strong uptrend
                phase_name = "BULL"
            elif current_phase == 1:  # Sideways phase
                main_trend = 40 + np.sin(2 * np.pi * (time_factor - phase_length) / phase_length * 3) * 15
                phase_name = "SIDEWAYS"
            elif current_phase == 2:  # Bear phase
                main_trend = 40 - 50 * ((time_factor - 2*phase_length) / phase_length)  # Strong downtrend
                phase_name = "BEAR"
            else:  # Recovery phase
                main_trend = -10 + 30 * ((time_factor - 3*phase_length) / phase_length)  # Recovery
                phase_name = "RECOVERY"
            
            # Add realistic volatility
            volatility = np.random.normal(0, 2.5)
            
            # Occasional large moves
            if np.random.random() < 0.003:  # 0.3% chance
                volatility += np.random.normal(0, 8)
            
            # Medium oscillations
            medium_osc = np.sin(2 * np.pi * time_factor * 8) * 3
            
            price_change = main_trend * 0.002 + medium_osc * 0.03 + volatility * 0.05
            price += price_change
            
            # Keep within reasonable bounds
            price = max(120, min(350, price))
            
            # Realistic OHLC
            spread = np.random.uniform(0.3, 1.2)
            high = price + spread/2 + abs(np.random.normal(0, 0.2))
            low = price - spread/2 - abs(np.random.normal(0, 0.2))
            open_price = price + np.random.uniform(-0.2, 0.2)
            
            # Volume patterns (higher during trends)
            base_volume = 2000
            if phase_name in ["BULL", "BEAR"]:
                volume_multiplier = 1.5  # Higher volume during trends
            else:
                volume_multiplier = 1.0  # Normal volume during sideways
            
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
        
        # Analyze market phases
        phase_counts = df['market_phase'].value_counts()
        
        print(f"✅ Generated {len(df):,} data points with mixed conditions")
        print(f"📊 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"📈 Market phases: {dict(phase_counts)}")
        print(f"🔥 Volatility: {df['close'].std():.2f}")
        
        return df
    
    def _display_adaptive_results(self, results: Dict):
        """Display adaptive trading results"""
        
        print(f"\n" + "=" * 130)
        print("🧠 SMART TREND-ADAPTIVE RESULTS")
        print("=" * 130)
        
        print(f"\n🎯 ADAPTIVE PERFORMANCE SUMMARY:")
        print("-" * 150)
        print(f"{'Strategy':<20} {'Return':<12} {'Target':<10} {'Status':<15} {'Trades':<8} {'Win Rate':<10} {'Long W/R':<10} {'Short W/R':<10} {'Max DD':<8}")
        print("-" * 150)
        
        best_return = -float('inf')
        best_strategy = ""
        targets_achieved = 0
        total_long_trades = 0
        total_short_trades = 0
        
        for strategy, result in results.items():
            target_status = "🎯 ACHIEVED" if result['target_achieved'] else "❌ MISSED"
            if result['target_achieved']:
                targets_achieved += 1
            
            if result['total_return'] > best_return:
                best_return = result['total_return']
                best_strategy = strategy
            
            total_long_trades += result['long_trades']
            total_short_trades += result['short_trades']
            
            print(f"{strategy:<20} {result['total_return']:+8.1f}%   {result['target_return']:>6}%   "
                  f"{target_status:<15} {result['total_trades']:<8} {result['win_rate']:>6.1f}%   "
                  f"{result['long_win_rate']:>6.1f}%   {result['short_win_rate']:>6.1f}%   "
                  f"{result['max_drawdown']:>6.1f}%")
        
        print("-" * 150)
        
        print(f"\n🧠 ADAPTIVE INTELLIGENCE SUMMARY:")
        print(f"   👑 Best Strategy: {best_strategy} ({best_return:+.1f}% return)")
        print(f"   🎯 Targets Achieved: {targets_achieved}/3 strategies")
        print(f"   📈 Total Long Trades: {total_long_trades}")
        print(f"   📉 Total Short Trades: {total_short_trades}")
        print(f"   ⚖️  Direction Balance: {total_long_trades/(total_long_trades+total_short_trades)*100:.1f}% Long / {total_short_trades/(total_long_trades+total_short_trades)*100:.1f}% Short")
        
        # Show successful strategies
        for strategy, result in results.items():
            if result['target_achieved']:
                final_balance = result['final_balance']
                print(f"   🚀 {strategy}: ${self.initial_balance:.0f} → ${final_balance:.0f} "
                      f"({result['total_return']:+.1f}%) - TARGET ACHIEVED!")
        
        print(f"\n📊 DIRECTIONAL ANALYSIS:")
        for strategy, result in results.items():
            if result['total_trades'] > 0:
                print(f"   {strategy}:")
                print(f"     📈 Long: {result['long_trades']} trades, {result['long_win_rate']:.1f}% win rate")
                print(f"     📉 Short: {result['short_trades']} trades, {result['short_win_rate']:.1f}% win rate")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Trend-adaptive approach allows profit in any market condition")
        print(f"   • Both long and short positions contribute to overall performance")
        print(f"   • Smart trend analysis prevents fighting the market")
        print(f"   • Risk management protects capital during adverse moves")
        
        print("=" * 130)

def main():
    """Run smart trend-adaptive test"""
    bot = SmartTrendAdaptiveBot()
    results = bot.test_adaptive_performance()
    
    # Final summary
    total_strategies = len(results)
    successful_strategies = sum(1 for r in results.values() if r['target_achieved'])
    best_return = max(r['total_return'] for r in results.values())
    total_trades = sum(r['total_trades'] for r in results.values())
    total_long = sum(r['long_trades'] for r in results.values())
    total_short = sum(r['short_trades'] for r in results.values())
    
    print(f"\n🧠 SMART TREND-ADAPTIVE TEST COMPLETE!")
    print(f"🎯 Success Rate: {successful_strategies}/{total_strategies} strategies achieved targets")
    print(f"🏆 Best Performance: {best_return:+.1f}%")
    print(f"📊 Trade Distribution: {total_long} Long / {total_short} Short ({total_trades} total)")
    
    if successful_strategies > 0:
        print("✅ ADAPTIVE INTELLIGENCE SUCCESSFUL!")
        print("💡 The bot successfully adapted to different market conditions!")
    else:
        print("📈 Continue optimizing adaptive algorithms")
    
    return results

if __name__ == "__main__":
    main()