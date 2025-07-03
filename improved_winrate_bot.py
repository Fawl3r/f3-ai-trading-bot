#!/usr/bin/env python3
"""
Improved Win Rate Bot
Direct modifications to the working system for better win rates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from final_optimized_ai_bot import FinalOptimizedAI
from backtest_risk_manager import BacktestRiskManager
from indicators import TechnicalIndicators

class ImprovedWinRateBot:
    """Improved win rate through better risk management"""
    
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        self.ai_analyzer = FinalOptimizedAI()
        self.risk_manager = BacktestRiskManager()
        self.indicators = TechnicalIndicators()
        
        # IMPROVED WIN RATE PARAMETERS
        self.improved_profiles = {
            "SAFE": {
                "stop_loss_pct": 1.0,      # Tighter than original 1.5%
                "take_profit_pct": 1.5,    # Lower than original 2.0%
                "position_size_pct": 2.0,
                "max_daily_trades": 6,
                "rsi_oversold": 20,
                "rsi_overbought": 80,
                "ai_threshold": 45.0
            },
            "RISK": {
                "stop_loss_pct": 1.2,      # Tighter than original 2.0%
                "take_profit_pct": 1.8,    # Lower than original 3.0%
                "position_size_pct": 4.0,
                "max_daily_trades": 10,
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "ai_threshold": 35.0
            },
            "SUPER_RISKY": {
                "stop_loss_pct": 1.5,      # Much tighter than original 3.0%
                "take_profit_pct": 2.2,    # Much lower than original 5.0%
                "position_size_pct": 6.0,
                "max_daily_trades": 15,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "ai_threshold": 25.0
            },
            "INSANE": {
                "stop_loss_pct": 1.0,      # Much tighter than original 2.0%
                "take_profit_pct": 2.5,    # Much lower than original 8.0%
                "position_size_pct": 8.0,
                "max_daily_trades": 8,
                "rsi_oversold": 15,
                "rsi_overbought": 85,
                "ai_threshold": 40.0
            }
        }
        
        print("📈 IMPROVED WIN RATE TRADING BOT")
        print("🎯 FOCUS: Better risk/reward ratios for higher win rates")
        print("📊 KEY IMPROVEMENTS:")
        print("   • Tighter stop losses (1.0-1.5% vs 1.5-3.0%)")
        print("   • More achievable take profits (1.5-2.5% vs 2.0-8.0%)")
        print("   • Optimized AI thresholds")
        print("   • Enhanced exit timing")
        print("=" * 80)
    
    def test_improved_winrates(self):
        """Test improved win rate system"""
        
        print("\n📈 IMPROVED WIN RATE TEST")
        print("🎯 Goal: Increase win rates through better risk/reward management")
        print("=" * 80)
        
        # Generate test data
        data = self._generate_test_data(days=14)
        
        modes = ["SAFE", "RISK", "SUPER_RISKY", "INSANE"]
        results = {}
        
        for mode in modes:
            print(f"\n{'='*15} IMPROVED {mode} MODE TEST {'='*15}")
            results[mode] = self._test_improved_mode(mode, data)
        
        # Compare with original results
        self._display_improved_comparison(results)
        return results
    
    def _test_improved_mode(self, mode: str, data: pd.DataFrame) -> Dict:
        """Test single mode with improved parameters"""
        
        profile = self.improved_profiles[mode]
        
        print(f"📈 IMPROVED {mode} MODE")
        print(f"   • Stop Loss: {profile['stop_loss_pct']}% (IMPROVED)")
        print(f"   • Take Profit: {profile['take_profit_pct']}% (ACHIEVABLE)")
        print(f"   • AI Threshold: {profile['ai_threshold']}%")
        print(f"   • Risk/Reward: 1:{profile['take_profit_pct']/profile['stop_loss_pct']:.1f}")
        
        # Reset AI
        self.ai_analyzer = FinalOptimizedAI()
        
        return self._run_improved_simulation(data, profile)
    
    def _run_improved_simulation(self, data: pd.DataFrame, profile: Dict) -> Dict:
        """Run simulation with improved parameters"""
        
        balance = self.initial_balance
        position = None
        trades = []
        daily_trades = 0
        last_date = None
        
        winning_trades = 0
        peak_balance = balance
        max_drawdown = 0
        
        for i in range(50, len(data)):
            current = data.iloc[i]
            price = current['close']
            rsi = current.get('rsi', 50)
            current_date = current['timestamp'].date()
            
            # Reset daily counter
            if last_date != current_date:
                daily_trades = 0
                last_date = current_date
            
            # Check daily limit
            if daily_trades >= profile['max_daily_trades']:
                continue
            
            # BUY SIGNAL
            if rsi < profile['rsi_oversold'] and position is None:
                recent_data = data.iloc[max(0, i-50):i+1]
                ai_result = self.ai_analyzer.analyze_trade_opportunity(recent_data, price, 'buy')
                
                if ai_result['ai_confidence'] >= profile['ai_threshold']:
                    position_size = balance * (profile['position_size_pct'] / 100)
                    
                    position = {
                        'entry_price': price,
                        'size': position_size,
                        'ai_confidence': ai_result['ai_confidence'],
                        'stop_loss': price * (1 - profile['stop_loss_pct'] / 100),
                        'take_profit': price * (1 + profile['take_profit_pct'] / 100),
                        'entry_time': current['timestamp']
                    }
                    daily_trades += 1
                    
                    if len(trades) < 5:  # Show first few trades
                        print(f"    📈 BUY ${price:.4f} | AI: {ai_result['ai_confidence']:.1f}% | SL: ${position['stop_loss']:.4f} | TP: ${position['take_profit']:.4f}")
            
            # POSITION MANAGEMENT
            elif position is not None:
                should_close = False
                close_reason = ""
                
                # Tight stop loss
                if price <= position['stop_loss']:
                    should_close = True
                    close_reason = "Stop Loss"
                
                # Achievable take profit
                elif price >= position['take_profit']:
                    should_close = True
                    close_reason = "Take Profit"
                
                # Early profit taking for win rate protection
                elif price >= position['entry_price'] * (1 + profile['take_profit_pct'] * 0.8 / 100):
                    # If we hit 80% of target and RSI is extreme, consider exit
                    if rsi > profile['rsi_overbought']:
                        recent_data = data.iloc[max(0, i-30):i+1]
                        ai_result = self.ai_analyzer.analyze_trade_opportunity(recent_data, price, 'sell')
                        if ai_result['ai_confidence'] >= profile['ai_threshold']:
                            should_close = True
                            close_reason = "Early Profit"
                
                # Time-based exit (prevent holding losers too long)
                elif (current['timestamp'] - position['entry_time']).total_seconds() > 3600:  # 1 hour
                    # Only exit if not at significant loss
                    unrealized_pnl_pct = ((price - position['entry_price']) / position['entry_price']) * 100
                    if unrealized_pnl_pct > -profile['stop_loss_pct'] * 0.5:  # Less than half stop loss
                        should_close = True
                        close_reason = "Time Exit"
                
                if should_close:
                    pnl = (price - position['entry_price']) * (position['size'] / position['entry_price'])
                    balance += pnl
                    
                    outcome = 'win' if pnl > 0 else 'loss'
                    if pnl > 0:
                        winning_trades += 1
                    
                    # Update AI learning
                    self.ai_analyzer.update_trade_result(position['ai_confidence'], outcome)
                    
                    # Track drawdown
                    if balance > peak_balance:
                        peak_balance = balance
                    current_drawdown = ((peak_balance - balance) / peak_balance) * 100
                    max_drawdown = max(max_drawdown, current_drawdown)
                    
                    # Calculate metrics
                    pnl_pct = ((price - position['entry_price']) / position['entry_price']) * 100
                    hold_time = (current['timestamp'] - position['entry_time']).total_seconds() / 60
                    
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'close_reason': close_reason,
                        'hold_time_min': hold_time,
                        'ai_confidence': position['ai_confidence']
                    })
                    
                    if len(trades) <= 5:  # Show first few trades
                        print(f"    📉 SELL ${price:.4f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | {close_reason}")
                    
                    position = None
                    daily_trades += 1
        
        # Calculate results
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = ((balance - self.initial_balance) / self.initial_balance) * 100
        
        # Breakdown by close reason
        close_reasons = {}
        for trade in trades:
            reason = trade['close_reason']
            if reason not in close_reasons:
                close_reasons[reason] = {'count': 0, 'wins': 0}
            close_reasons[reason]['count'] += 1
            if trade['pnl'] > 0:
                close_reasons[reason]['wins'] += 1
        
        # Calculate win rates by close reason
        for reason in close_reasons:
            close_reasons[reason]['win_rate'] = (close_reasons[reason]['wins'] / close_reasons[reason]['count']) * 100
        
        return {
            'final_balance': balance,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'close_reasons': close_reasons,
            'trades': trades
        }
    
    def _generate_test_data(self, days: int = 14) -> pd.DataFrame:
        """Generate test data similar to previous tests"""
        print(f"📊 Generating {days} days of test data...")
        
        data = []
        price = 145.0
        minutes = days * 24 * 60
        
        np.random.seed(123)  # Same seed as previous tests for comparison
        
        for i in range(minutes):
            time_factor = i / minutes
            
            # Market cycles
            daily_cycle = np.sin(2 * np.pi * i / (24 * 60)) * 2.0
            weekly_cycle = np.sin(2 * np.pi * i / (7 * 24 * 60)) * 3.0
            trend = np.sin(2 * np.pi * time_factor) * 8.0
            
            # Noise
            volatility = 0.5 + 0.3 * np.sin(2 * np.pi * i / (12 * 60))
            noise = np.random.normal(0, volatility)
            
            price_change = daily_cycle * 0.3 + weekly_cycle * 0.2 + trend * 0.1 + noise
            price += price_change
            price = max(130, min(160, price))
            
            data.append({
                'timestamp': datetime.now() - timedelta(minutes=minutes-i),
                'open': price + np.random.uniform(-0.1, 0.1),
                'high': price + np.random.uniform(0, 0.4),
                'low': price - np.random.uniform(0, 0.4),
                'close': price,
                'volume': np.random.uniform(1000, 2000) * (1 + abs(price_change) * 0.5)
            })
        
        df = pd.DataFrame(data)
        df = self.indicators.calculate_all_indicators(df)
        
        print(f"✅ Generated {len(df):,} data points")
        return df
    
    def _display_improved_comparison(self, results: Dict):
        """Display improved results vs original expectations"""
        
        print(f"\n" + "=" * 100)
        print("📈 IMPROVED WIN RATE RESULTS")
        print("=" * 100)
        
        print(f"\n🎯 WIN RATE IMPROVEMENT COMPARISON:")
        print("-" * 120)
        print(f"{'Mode':<16} {'Win Rate':<12} {'Return':<12} {'Trades':<8} {'Risk/Reward':<12} {'Improvement'}")
        print("-" * 120)
        
        # Expected baseline (from previous tests)
        baseline_winrates = {"SAFE": 17, "RISK": 17, "SUPER_RISKY": 18, "INSANE": 21}
        
        best_winrate = 0
        best_mode = ""
        total_improvement = 0
        
        for mode, result in results.items():
            if result['win_rate'] > best_winrate:
                best_winrate = result['win_rate']
                best_mode = mode
            
            baseline = baseline_winrates.get(mode, 20)
            improvement = result['win_rate'] - baseline
            total_improvement += improvement
            
            if improvement > 15:
                improvement_status = "🟢 EXCELLENT"
            elif improvement > 10:
                improvement_status = "🟢 VERY GOOD"
            elif improvement > 5:
                improvement_status = "🟡 GOOD"
            elif improvement > 0:
                improvement_status = "🟠 SLIGHT"
            else:
                improvement_status = "🔴 WORSE"
            
            risk_reward = self.improved_profiles[mode]['take_profit_pct'] / self.improved_profiles[mode]['stop_loss_pct']
            
            print(f"{mode:<16} {result['win_rate']:8.1f}%    {result['total_return']:+8.2f}%   "
                  f"{result['total_trades']:<8} 1:{risk_reward:.1f}        {improvement:+.1f}% {improvement_status}")
        
        print("-" * 120)
        
        avg_improvement = total_improvement / len(results)
        
        print(f"\n📈 WIN RATE IMPROVEMENT ANALYSIS:")
        print(f"   🏆 Best Win Rate: {best_mode} ({best_winrate:.1f}%)")
        print(f"   📊 Average Improvement: {avg_improvement:+.1f}% vs baseline")
        
        if avg_improvement > 10:
            print("   ✅ EXCELLENT: Significant win rate improvement achieved!")
        elif avg_improvement > 5:
            print("   🟢 GOOD: Noticeable win rate improvement")
        elif avg_improvement > 0:
            print("   🟡 MODEST: Some improvement, room for more")
        else:
            print("   🔴 NEEDS WORK: Win rates not improved")
        
        print(f"\n🎯 EXIT STRATEGY ANALYSIS:")
        for mode, result in results.items():
            if result['total_trades'] > 0:
                print(f"\n   📈 {mode} MODE:")
                if 'close_reasons' in result:
                    for reason, stats in result['close_reasons'].items():
                        print(f"      • {reason}: {stats['count']} trades ({stats['win_rate']:.1f}% win rate)")
        
        print("=" * 100)

def main():
    """Run improved win rate test"""
    bot = ImprovedWinRateBot()
    results = bot.test_improved_winrates()
    
    # Summary
    total_trades = sum(r['total_trades'] for r in results.values())
    total_wins = sum(r['winning_trades'] for r in results.values())
    overall_winrate = (total_wins / max(total_trades, 1)) * 100
    
    print(f"\n🎉 IMPROVED WIN RATE TEST COMPLETE!")
    print(f"📊 Overall Results: {total_wins}/{total_trades} trades | {overall_winrate:.1f}% win rate")
    
    if overall_winrate >= 40:
        print("✅ SUCCESS: Significant win rate improvement achieved!")
    elif overall_winrate >= 30:
        print("🟡 GOOD: Decent win rate improvement")
    else:
        print("📈 Continue optimization for better results")

if __name__ == "__main__":
    main() 