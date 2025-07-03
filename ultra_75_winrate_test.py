#!/usr/bin/env python3
"""
Ultra 75% Win Rate Test
Ultra-aggressive approach with very low AI thresholds for trade execution
Based on successful ultra-aggressive bot that achieved 1,043 trades
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from final_optimized_ai_bot import FinalOptimizedAI

class Ultra75WinRateTest:
    """Ultra-aggressive test for 75% win rate achievement"""
    
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        self.ai_analyzer = FinalOptimizedAI()
        
        # ULTRA-AGGRESSIVE 75% WIN RATE STRATEGIES
        self.strategies = {
            "ULTRA_PRECISION": {
                "name": "Ultra Precision 75%",
                "position_size_pct": 3.0,
                "leverage": 8,
                "stop_loss_pct": 0.6,  # Very tight
                "take_profit_pct": 1.8,  # 3:1 reward/risk
                "max_hold_time_min": 30,  # Quick exits
                "max_daily_trades": 50,
                "ai_threshold": 5.0,  # ULTRA LOW for execution
                "target_winrate": 75,
                "breakeven_protection": True,
                "partial_profit": 0.7
            },
            "MOMENTUM_BEAST": {
                "name": "Momentum Beast 75%",
                "position_size_pct": 4.0,
                "leverage": 10,
                "stop_loss_pct": 0.8,
                "take_profit_pct": 2.0,  # 2.5:1 reward/risk
                "max_hold_time_min": 45,
                "max_daily_trades": 60,
                "ai_threshold": 8.0,  # LOW for execution
                "target_winrate": 75,
                "trailing_stop": True,
                "early_exit": 0.8
            },
            "SCALP_MASTER": {
                "name": "Scalp Master 75%",
                "position_size_pct": 5.0,
                "leverage": 12,
                "stop_loss_pct": 0.5,  # Ultra tight
                "take_profit_pct": 1.5,  # 3:1 reward/risk
                "max_hold_time_min": 20,  # Very quick
                "max_daily_trades": 80,
                "ai_threshold": 3.0,  # ULTRA LOW
                "target_winrate": 75,
                "instant_breakeven": True,
                "micro_profits": True
            },
            "TIME_MASTER": {
                "name": "Time Exit Master 75%",
                "position_size_pct": 6.0,
                "leverage": 15,
                "stop_loss_pct": 1.0,
                "take_profit_pct": 2.5,
                "max_hold_time_min": 60,  # Time exits proven best
                "max_daily_trades": 40,
                "ai_threshold": 10.0,
                "target_winrate": 75,
                "time_exit_bias": True,  # Focus on time exits
                "smart_timing": True
            }
        }
        
        print("🚀 ULTRA 75% WIN RATE TEST")
        print("⚡ ULTRA-AGGRESSIVE APPROACH")
        print("🏆 TARGET: 75% Win Rate")
        print("🎯 LOW AI THRESHOLDS FOR EXECUTION")
        print("=" * 70)
        print("🔧 ULTRA-AGGRESSIVE FEATURES:")
        print("   • AI Thresholds: 3-10% (ULTRA LOW)")
        print("   • Tight Stops: 0.5-1.0%")
        print("   • High Rewards: 1.5-2.5%")
        print("   • Quick Exits: 20-60 minutes")
        print("   • High Volume: 40-80 trades/day")
        print("   • Breakeven Protection")
        print("   • Time Exit Focus (Proven Best)")
        print("=" * 70)
    
    def run_ultra_test(self):
        """Run ultra-aggressive 75% win rate test"""
        
        print("\n🚀 RUNNING ULTRA 75% WIN RATE TEST")
        print("⚡ Multiple Strategies | Ultra-Aggressive Execution")
        
        # Generate realistic data
        data = self._generate_realistic_data(days=21)
        
        results = {}
        for strategy_name, strategy in self.strategies.items():
            print(f"\n{'='*25} {strategy_name} TEST {'='*25}")
            results[strategy_name] = self._test_strategy(strategy_name, strategy, data)
        
        # Display comprehensive results
        self._display_comprehensive_results(results)
        
        return results
    
    def _test_strategy(self, name: str, strategy: Dict, data: pd.DataFrame) -> Dict:
        """Test single ultra-aggressive strategy"""
        
        print(f"🎯 {strategy['name']}")
        print(f"   • AI Threshold: {strategy['ai_threshold']}% (ULTRA LOW)")
        print(f"   • Risk/Reward: {strategy['stop_loss_pct']}% / {strategy['take_profit_pct']}%")
        print(f"   • Max Hold: {strategy['max_hold_time_min']} min")
        print(f"   • Daily Trades: {strategy['max_daily_trades']}")
        
        # Reset AI for each test
        self.ai_analyzer = FinalOptimizedAI()
        
        return self._run_ultra_backtest(data, strategy)
    
    def _run_ultra_backtest(self, data: pd.DataFrame, strategy: Dict) -> Dict:
        """Run ultra-aggressive backtest"""
        
        balance = self.initial_balance
        trades = []
        daily_trades = 0
        last_day = None
        
        wins = 0
        losses = 0
        
        exit_reasons = {
            'take_profit': 0,
            'stop_loss': 0,
            'time_exit': 0,
            'breakeven': 0,
            'partial_profit': 0,
            'trailing_stop': 0
        }
        
        rejected_by_ai = 0
        rejected_by_quality = 0
        
        print(f"📊 Ultra-backtesting {len(data)} data points...")
        
        for i in range(30, len(data), 2):  # Skip every 2 for speed
            current_day = data.iloc[i]['datetime'].date()
            
            # Reset daily counter
            if last_day != current_day:
                daily_trades = 0
                last_day = current_day
            
            # Check daily limit
            if daily_trades >= strategy['max_daily_trades']:
                continue
            
            # Get analysis window
            window = data.iloc[max(0, i-30):i+1]
            current_price = data.iloc[i]['close']
            
            # Ultra-aggressive AI analysis
            ai_result = self.ai_analyzer.analyze_trade_opportunity(window, current_price, 'buy')
            ai_confidence = ai_result.get('confidence', 0)
            
            # Check ULTRA LOW AI threshold
            if ai_confidence < strategy['ai_threshold']:
                rejected_by_ai += 1
                continue
            
            # Minimal quality checks for execution
            if not self._minimal_quality_check(window):
                rejected_by_quality += 1
                continue
            
            # Position setup
            position_size = balance * strategy['position_size_pct'] / 100
            leverage = strategy['leverage']
            
            # Determine direction
            direction = self._smart_direction(window, strategy)
            entry_price = current_price
            
            # Calculate stops with strategy-specific logic
            if direction == 'long':
                stop_loss = entry_price * (1 - strategy['stop_loss_pct'] / 100)
                take_profit = entry_price * (1 + strategy['take_profit_pct'] / 100)
            else:
                stop_loss = entry_price * (1 + strategy['stop_loss_pct'] / 100)
                take_profit = entry_price * (1 - strategy['take_profit_pct'] / 100)
            
            # Ultra-aggressive position management
            exit_info = self._ultra_position_management(data, i, entry_price, stop_loss, take_profit, direction, strategy)
            
            # Calculate P&L
            exit_price = exit_info['exit_price']
            if direction == 'long':
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            
            pnl_pct *= leverage
            pnl_amount = position_size * (pnl_pct / 100)
            balance += pnl_amount
            
            # Track results
            is_win = pnl_amount > 0
            if is_win:
                wins += 1
            else:
                losses += 1
            
            exit_reasons[exit_info['exit_reason']] += 1
            
            # Record trade
            trades.append({
                'entry_time': data.iloc[i]['datetime'],
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_amount': pnl_amount,
                'exit_reason': exit_info['exit_reason'],
                'ai_confidence': ai_confidence,
                'balance': balance,
                'hold_time': exit_info.get('hold_time', 0)
            })
            
            daily_trades += 1
        
        # Calculate metrics
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_return = ((balance - self.initial_balance) / self.initial_balance) * 100
        
        print(f"   📊 Executed: {total_trades} trades")
        print(f"   🧠 AI Rejected: {rejected_by_ai}")
        print(f"   🔍 Quality Rejected: {rejected_by_quality}")
        print(f"   🏆 Win Rate: {win_rate:.1f}%")
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_balance': balance,
            'exit_reasons': exit_reasons,
            'rejected_by_ai': rejected_by_ai,
            'rejected_by_quality': rejected_by_quality,
            'trades': trades,
            'target_achieved': win_rate >= strategy['target_winrate']
        }
    
    def _minimal_quality_check(self, data: pd.DataFrame) -> bool:
        """Minimal quality check for execution"""
        if len(data) < 10:
            return False
        
        # Very basic checks for movement
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-3]
        movement = abs((current_price - prev_price) / prev_price) * 100
        
        # Just need some movement
        return movement > 0.1
    
    def _smart_direction(self, data: pd.DataFrame, strategy: Dict) -> str:
        """Smart direction determination"""
        current_price = data['close'].iloc[-1]
        
        # Multiple indicators
        sma_5 = data['close'].tail(5).mean()
        sma_10 = data['close'].tail(10).mean()
        rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else 50
        
        # Momentum
        momentum = ((current_price - data['close'].iloc[-5]) / data['close'].iloc[-5]) * 100
        
        # Direction scoring
        long_score = 0
        short_score = 0
        
        # Trend
        if current_price > sma_5 > sma_10:
            long_score += 2
        elif current_price < sma_5 < sma_10:
            short_score += 2
        
        # Momentum
        if momentum > 0.3:
            long_score += 1
        elif momentum < -0.3:
            short_score += 1
        
        # RSI
        if rsi < 45:
            long_score += 1
        elif rsi > 55:
            short_score += 1
        
        return 'long' if long_score >= short_score else 'short'
    
    def _ultra_position_management(self, data: pd.DataFrame, start_idx: int,
                                 entry_price: float, stop_loss: float,
                                 take_profit: float, direction: str,
                                 strategy: Dict) -> Dict:
        """Ultra-aggressive position management"""
        
        entry_time = data.iloc[start_idx]['datetime']
        max_hold = strategy['max_hold_time_min']
        
        # Strategy-specific features
        instant_breakeven = strategy.get('instant_breakeven', False)
        partial_profit = strategy.get('partial_profit', 0)
        trailing_stop = strategy.get('trailing_stop', False)
        time_exit_bias = strategy.get('time_exit_bias', False)
        
        # Tracking
        breakeven_triggered = False
        partial_taken = False
        trailing_stop_level = stop_loss
        
        for j in range(start_idx + 1, min(start_idx + max_hold + 1, len(data))):
            if j >= len(data):
                break
                
            current_price = data.iloc[j]['close']
            current_time = data.iloc[j]['datetime']
            hold_time = (current_time - entry_time).total_seconds() / 60
            
            # Check take profit first
            if direction == 'long' and current_price >= take_profit:
                return {
                    'exit_price': take_profit,
                    'exit_time': current_time,
                    'exit_reason': 'take_profit',
                    'hold_time': hold_time
                }
            elif direction == 'short' and current_price <= take_profit:
                return {
                    'exit_price': take_profit,
                    'exit_time': current_time,
                    'exit_reason': 'take_profit',
                    'hold_time': hold_time
                }
            
            # Instant breakeven protection
            if instant_breakeven and not breakeven_triggered:
                profit_threshold = 0.002  # 0.2% profit
                if direction == 'long' and current_price >= entry_price * (1 + profit_threshold):
                    trailing_stop_level = entry_price
                    breakeven_triggered = True
                elif direction == 'short' and current_price <= entry_price * (1 - profit_threshold):
                    trailing_stop_level = entry_price
                    breakeven_triggered = True
            
            # Trailing stop logic
            if trailing_stop and breakeven_triggered:
                if direction == 'long':
                    new_stop = current_price * (1 - strategy['stop_loss_pct'] / 300)  # Tighter trailing
                    trailing_stop_level = max(trailing_stop_level, new_stop)
                else:
                    new_stop = current_price * (1 + strategy['stop_loss_pct'] / 300)
                    trailing_stop_level = min(trailing_stop_level, new_stop)
            
            # Check stop loss
            if direction == 'long' and current_price <= trailing_stop_level:
                exit_reason = 'breakeven' if breakeven_triggered else 'stop_loss'
                return {
                    'exit_price': trailing_stop_level,
                    'exit_time': current_time,
                    'exit_reason': exit_reason,
                    'hold_time': hold_time
                }
            elif direction == 'short' and current_price >= trailing_stop_level:
                exit_reason = 'breakeven' if breakeven_triggered else 'stop_loss'
                return {
                    'exit_price': trailing_stop_level,
                    'exit_time': current_time,
                    'exit_reason': exit_reason,
                    'hold_time': hold_time
                }
            
            # Partial profit taking
            if partial_profit > 0 and not partial_taken:
                if direction == 'long':
                    profit_pct = (current_price - entry_price) / entry_price
                    target_pct = (take_profit - entry_price) / entry_price
                else:
                    profit_pct = (entry_price - current_price) / entry_price
                    target_pct = (entry_price - take_profit) / entry_price
                
                if profit_pct >= target_pct * partial_profit:
                    partial_taken = True
                    return {
                        'exit_price': current_price,
                        'exit_time': current_time,
                        'exit_reason': 'partial_profit',
                        'hold_time': hold_time
                    }
            
            # Time exit bias (proven most successful)
            if time_exit_bias and hold_time >= max_hold * 0.8:  # Exit at 80% of max time
                return {
                    'exit_price': current_price,
                    'exit_time': current_time,
                    'exit_reason': 'time_exit',
                    'hold_time': hold_time
                }
        
        # Final time exit
        final_idx = min(start_idx + max_hold, len(data) - 1)
        final_price = data.iloc[final_idx]['close']
        final_time = data.iloc[final_idx]['datetime']
        
        return {
            'exit_price': final_price,
            'exit_time': final_time,
            'exit_reason': 'time_exit',
            'hold_time': max_hold
        }
    
    def _generate_realistic_data(self, days: int = 21) -> pd.DataFrame:
        """Generate realistic market data"""
        
        print(f"📈 Generating {days} days of realistic market data...")
        
        minutes = days * 1440
        base_price = 142.0
        prices = [base_price]
        
        # Create realistic market phases
        for i in range(1, minutes):
            # Base volatility
            change_pct = np.random.normal(0, 0.6)
            
            # Add trending periods
            if i % 480 == 0:  # Every 8 hours
                trend = np.random.choice([-0.2, 0.2, 0])  # Up, down, or sideways
            else:
                trend = 0
            
            total_change = change_pct + trend
            new_price = prices[-1] * (1 + total_change / 100)
            new_price = max(50, min(400, new_price))
            prices.append(new_price)
        
        # Create DataFrame
        data = []
        start_time = datetime.now() - timedelta(days=days)
        
        for i, price in enumerate(prices):
            timestamp = start_time + timedelta(minutes=i)
            
            # OHLC
            high = price * (1 + abs(np.random.normal(0, 0.12)) / 100)
            low = price * (1 - abs(np.random.normal(0, 0.12)) / 100)
            open_price = prices[i-1] if i > 0 else price
            
            # RSI
            rsi = 50 + np.sin(i / 60) * 20 + np.random.normal(0, 8)
            rsi = max(5, min(95, rsi))
            
            data.append({
                'datetime': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': np.random.uniform(3000, 12000),
                'rsi': rsi
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} data points")
        print(f"📊 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
    
    def _display_comprehensive_results(self, results: Dict):
        """Display comprehensive results"""
        
        print("\n" + "="*80)
        print("🚀 ULTRA 75% WIN RATE TEST RESULTS")
        print("⚡ ULTRA-AGGRESSIVE EXECUTION ANALYSIS")
        print("="*80)
        
        print("\n💎 STRATEGY PERFORMANCE SUMMARY:")
        print("-" * 80)
        print(f"{'Strategy':<18} {'Win Rate':<10} {'Target':<8} {'Status':<12} {'Trades':<8} {'Return':<10}")
        print("-" * 80)
        
        total_trades = 0
        total_wins = 0
        strategies_hit_target = 0
        best_winrate = 0
        best_strategy = ""
        best_return = -100
        
        for strategy_name, result in results.items():
            win_rate = result['win_rate']
            target = self.strategies[strategy_name]['target_winrate']
            status = "✅ HIT" if result['target_achieved'] else f"❌ {win_rate:.1f}%"
            trades = result['total_trades']
            total_return = result['total_return']
            
            total_trades += trades
            total_wins += result['wins']
            
            if result['target_achieved']:
                strategies_hit_target += 1
            
            if win_rate > best_winrate:
                best_winrate = win_rate
                best_strategy = strategy_name
            
            if total_return > best_return:
                best_return = total_return
            
            print(f"{strategy_name:<18} {win_rate:>6.1f}% {target:>6}% {status:<12} {trades:>6} {total_return:>+6.1f}%")
        
        print("-" * 80)
        
        overall_winrate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n🏆 OVERALL ULTRA PERFORMANCE:")
        print(f"   📊 Overall Win Rate: {overall_winrate:.1f}%")
        print(f"   🎯 75% Target Hit: {strategies_hit_target}/4 strategies")
        print(f"   🥇 Best Win Rate: {best_strategy} ({best_winrate:.1f}%)")
        print(f"   💰 Best Return: {best_return:+.1f}%")
        print(f"   📈 Total Trades Executed: {total_trades}")
        
        # Execution analysis
        print(f"\n⚡ EXECUTION ANALYSIS:")
        for strategy_name, result in results.items():
            ai_rejected = result['rejected_by_ai']
            quality_rejected = result['rejected_by_quality']
            executed = result['total_trades']
            total_opportunities = executed + ai_rejected + quality_rejected
            
            execution_rate = (executed / total_opportunities * 100) if total_opportunities > 0 else 0
            print(f"   • {strategy_name}: {execution_rate:.1f}% execution rate ({executed} trades)")
        
        # 75% Target Analysis
        print(f"\n🎯 75% WIN RATE TARGET ANALYSIS:")
        if strategies_hit_target > 0:
            print(f"   🎉 SUCCESS: {strategies_hit_target} strategies achieved 75%+ win rate!")
            print(f"   🏆 Ultra-aggressive approach WORKING")
            print(f"   🚀 Ready for live implementation")
        else:
            print(f"   📊 Progress: Best achieved {best_winrate:.1f}%")
            gap = 75 - best_winrate
            if gap < 10:
                print(f"   💡 Close to target - fine-tuning needed")
            elif gap < 25:
                print(f"   🔧 Moderate gap - strategy optimization needed")
            else:
                print(f"   ⚠️ Large gap - major revision required")
        
        print("="*80)

def main():
    """Main execution"""
    print("🚀 ULTRA 75% WIN RATE TEST")
    print("⚡ Ultra-Aggressive Execution Strategy")
    
    test = Ultra75WinRateTest()
    results = test.run_ultra_test()
    
    print(f"\n🎯 ULTRA TEST COMPLETE!")
    
    # Check if any strategy hit 75%
    success_count = sum(1 for r in results.values() if r['target_achieved'])
    if success_count > 0:
        print(f"🎉 {success_count} STRATEGIES ACHIEVED 75%+ WIN RATE!")
        print(f"🚀 ULTRA-AGGRESSIVE APPROACH SUCCESSFUL!")
    else:
        best_wr = max(r['win_rate'] for r in results.values()) if results else 0
        print(f"📊 Best achieved: {best_wr:.1f}% - Continue optimization")

if __name__ == "__main__":
    main() 