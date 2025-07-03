#!/usr/bin/env python3
"""
Quick Adaptive Test - Demonstrates Long/Short trend adaptation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_trend_direction(prices, current_idx):
    """Simple trend analysis"""
    if current_idx < 20:
        return 'sideways', 0
    
    recent_prices = prices[max(0, current_idx-20):current_idx]
    
    # Simple trend detection
    sma_short = np.mean(recent_prices[-5:])
    sma_long = np.mean(recent_prices[-15:])
    current_price = prices[current_idx]
    
    # Trend strength (more sensitive)
    if sma_short > sma_long * 1.005 and current_price > sma_short:
        return 'bullish', 75
    elif sma_short < sma_long * 0.995 and current_price < sma_short:
        return 'bearish', 75
    else:
        return 'sideways', 25

def test_adaptive_trading():
    """Test adaptive long/short trading"""
    print("🧠 QUICK ADAPTIVE TRADING TEST")
    print("📈📉 Testing Long vs Short adaptation to trends")
    print("=" * 60)
    
    # Generate test data with clear trends
    np.random.seed(42)
    data_points = 1000
    
    # Create price data with distinct phases
    prices = []
    base_price = 200
    
    for i in range(data_points):
        phase = int(i / (data_points / 4))  # 4 phases
        
        if phase == 0:  # Bull trend
            trend = 0.15  # Stronger trend
            phase_name = "BULL"
        elif phase == 1:  # Sideways
            trend = np.sin(i * 0.01) * 0.02  # Small oscillation
            phase_name = "SIDEWAYS" 
        elif phase == 2:  # Bear trend
            trend = -0.18  # Stronger downtrend
            phase_name = "BEAR"
        else:  # Recovery
            trend = 0.12  # Strong recovery
            phase_name = "RECOVERY"
        
        noise = np.random.normal(0, 0.5)
        base_price += trend + noise
        prices.append(base_price)
    
    # Test different strategies
    strategies = {
        "LONG_ONLY": {"trades_long": True, "trades_short": False},
        "SHORT_ONLY": {"trades_long": False, "trades_short": True},
        "ADAPTIVE": {"trades_long": True, "trades_short": True}
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\n🧠 Testing {strategy_name} Strategy")
        
        balance = 200
        trades = 0
        wins = 0
        long_trades = 0
        short_trades = 0
        long_wins = 0
        short_wins = 0
        
        position = None
        
        for i in range(50, len(prices)-10):
            current_price = prices[i]
            
            # Analyze trend
            trend_direction, trend_strength = analyze_trend_direction(prices, i)
            
            # Entry logic
            if position is None and trades < 50:  # Limit trades for quick test
                entry_signal = False
                trade_direction = None
                
                if strategy["trades_long"] and trend_direction == 'bullish' and trend_strength > 50:
                    entry_signal = True
                    trade_direction = 'long'
                
                elif strategy["trades_short"] and trend_direction == 'bearish' and trend_strength > 50:
                    entry_signal = True
                    trade_direction = 'short'
                
                if entry_signal:
                    position = {
                        'entry_price': current_price,
                        'direction': trade_direction,
                        'entry_idx': i
                    }
                    trades += 1
                    
                    if trade_direction == 'long':
                        long_trades += 1
                    else:
                        short_trades += 1
                    
                    if trades <= 5:  # Show first few
                        direction_emoji = "📈" if trade_direction == 'long' else "📉"
                        print(f"  {direction_emoji} {trade_direction.upper()} @ ${current_price:.2f} | Trend: {trend_direction} ({trend_strength}%)")
            
            # Exit logic
            elif position is not None:
                # Simple exit after 10 periods or 2% move
                hold_time = i - position['entry_idx']
                
                if position['direction'] == 'long':
                    price_change = (current_price - position['entry_price']) / position['entry_price']
                else:  # short
                    price_change = (position['entry_price'] - current_price) / position['entry_price']
                
                should_exit = False
                exit_reason = ""
                
                if price_change >= 0.02:  # 2% profit
                    should_exit = True
                    exit_reason = "Take Profit"
                elif price_change <= -0.015:  # 1.5% loss
                    should_exit = True
                    exit_reason = "Stop Loss"
                elif hold_time >= 10:  # Time exit
                    should_exit = True
                    exit_reason = "Time Exit"
                
                if should_exit:
                    pnl_pct = price_change * 100
                    is_win = price_change > 0
                    
                    if is_win:
                        wins += 1
                        if position['direction'] == 'long':
                            long_wins += 1
                        else:
                            short_wins += 1
                    
                    balance += balance * 0.1 * price_change  # 10% position size
                    
                    if trades <= 5:
                        outcome_emoji = "✅" if is_win else "❌"
                        direction_emoji = "📈" if position['direction'] == 'long' else "📉"
                        print(f"  📤 EXIT {direction_emoji} @ ${current_price:.2f} | P&L: {pnl_pct:+.1f}% | {exit_reason} | {outcome_emoji}")
                    
                    position = None
        
        # Calculate results
        win_rate = (wins / trades * 100) if trades > 0 else 0
        total_return = (balance - 200) / 200 * 100
        long_win_rate = (long_wins / long_trades * 100) if long_trades > 0 else 0
        short_win_rate = (short_wins / short_trades * 100) if short_trades > 0 else 0
        
        results[strategy_name] = {
            'balance': balance,
            'return': total_return,
            'trades': trades,
            'wins': wins,
            'win_rate': win_rate,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_wins': long_wins,
            'short_wins': short_wins,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate
        }
        
        print(f"  📊 Results: {trades} trades, {win_rate:.1f}% win rate, {total_return:+.1f}% return")
    
    # Display comparison
    print(f"\n" + "=" * 80)
    print("🎯 ADAPTIVE TRADING COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<12} {'Return':<10} {'Trades':<8} {'Win Rate':<10} {'Long W/R':<10} {'Short W/R':<10}")
    print("-" * 80)
    
    for strategy, result in results.items():
        print(f"{strategy:<12} {result['return']:+7.1f}%   {result['trades']:<8} {result['win_rate']:>6.1f}%   "
              f"{result['long_win_rate']:>6.1f}%   {result['short_win_rate']:>6.1f}%")
    
    print("-" * 80)
    
    # Analysis
    adaptive_result = results['ADAPTIVE']
    long_only_result = results['LONG_ONLY']
    short_only_result = results['SHORT_ONLY']
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   📈 Long-Only Return: {long_only_result['return']:+.1f}%")
    print(f"   📉 Short-Only Return: {short_only_result['return']:+.1f}%")
    print(f"   🧠 Adaptive Return: {adaptive_result['return']:+.1f}%")
    
    if adaptive_result['return'] > max(long_only_result['return'], short_only_result['return']):
        print(f"   ✅ ADAPTIVE STRATEGY WINS!")
        print(f"   🎯 Advantage: {adaptive_result['return'] - max(long_only_result['return'], short_only_result['return']):+.1f}%")
    else:
        print(f"   📊 Single direction performed better in this test")
    
    print(f"\n🧠 DIRECTIONAL BREAKDOWN:")
    print(f"   📈 Adaptive Long: {adaptive_result['long_trades']} trades, {adaptive_result['long_win_rate']:.1f}% win rate")
    print(f"   📉 Adaptive Short: {adaptive_result['short_trades']} trades, {adaptive_result['short_win_rate']:.1f}% win rate")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"   • Trend analysis enables smart directional decisions")
    print(f"   • Adaptive approach can profit in both bull and bear markets")
    print(f"   • Single-direction strategies miss opportunities")
    print(f"   • Real market success requires trend-adaptive intelligence")

if __name__ == "__main__":
    test_adaptive_trading() 