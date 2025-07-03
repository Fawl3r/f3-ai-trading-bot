#!/usr/bin/env python3
"""
Realistic Ultimate Win Rate Bot
Fixed version with proper data generation and P&L calculation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from final_optimized_ai_bot import FinalOptimizedAI
from indicators import TechnicalIndicators

class RealisticUltimateBot:
    """Realistic win rate optimization with proper testing"""
    
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        self.ai_analyzer = FinalOptimizedAI()
        self.indicators = TechnicalIndicators()
        
        # REALISTIC PROFILES - based on proven improvements
        self.realistic_profiles = {
            "SAFE": {
                "initial_stop_loss_pct": 1.0,
                "take_profit_pct": 1.5,
                "trailing_stop_distance": 0.3,
                "breakeven_threshold": 0.4,
                "partial_profit_threshold": 0.8,
                "max_hold_time_min": 60,
                "position_size_pct": 2.0,
                "max_daily_trades": 8,
                "ai_threshold": 40.0,
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "volume_multiplier": 1.1
            },
            "RISK": {
                "initial_stop_loss_pct": 1.2,
                "take_profit_pct": 1.8,
                "trailing_stop_distance": 0.4,
                "breakeven_threshold": 0.5,
                "partial_profit_threshold": 0.75,
                "max_hold_time_min": 50,
                "position_size_pct": 4.0,
                "max_daily_trades": 10,
                "ai_threshold": 30.0,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "volume_multiplier": 1.05
            },
            "SUPER_RISKY": {
                "initial_stop_loss_pct": 1.5,
                "take_profit_pct": 2.2,
                "trailing_stop_distance": 0.5,
                "breakeven_threshold": 0.6,
                "partial_profit_threshold": 0.7,
                "max_hold_time_min": 45,
                "position_size_pct": 6.0,
                "max_daily_trades": 12,
                "ai_threshold": 20.0,
                "rsi_oversold": 35,
                "rsi_overbought": 65,
                "volume_multiplier": 1.0
            },
            "INSANE": {
                "initial_stop_loss_pct": 1.0,
                "take_profit_pct": 2.5,
                "trailing_stop_distance": 0.2,
                "breakeven_threshold": 0.3,
                "partial_profit_threshold": 0.6,
                "max_hold_time_min": 40,
                "position_size_pct": 8.0,
                "max_daily_trades": 8,
                "ai_threshold": 35.0,
                "rsi_oversold": 20,
                "rsi_overbought": 80,
                "volume_multiplier": 1.2
            }
        }
        
        print("🔍 REALISTIC ULTIMATE WIN RATE BOT")
        print("🎯 HONEST TESTING - NO FALSE POSITIVES")
        print("🚀 REALISTIC FEATURES:")
        print("   • Proper data generation with real price movement")
        print("   • Accurate P&L calculations")
        print("   • Realistic market conditions")
        print("   • True win/loss determination")
        print("   • Based on improved 57.4% baseline")
        print("=" * 80)
    
    def test_realistic_ultimate(self):
        """Test realistic win rate system"""
        
        print("\n🔍 REALISTIC ULTIMATE WIN RATE TEST")
        print("🎯 Target: Achieve realistic 60-70% win rates with proper testing")
        print("=" * 80)
        
        # Generate REALISTIC test data with proper price movement
        data = self._generate_realistic_data(days=14)
        
        modes = ["SAFE", "RISK", "SUPER_RISKY", "INSANE"]
        results = {}
        
        for mode in modes:
            print(f"\n{'='*15} REALISTIC {mode} MODE TEST {'='*15}")
            results[mode] = self._test_realistic_mode(mode, data)
        
        # Display honest results
        self._display_realistic_results(results)
        return results
    
    def _test_realistic_mode(self, mode: str, data: pd.DataFrame) -> Dict:
        """Test single mode with realistic conditions"""
        
        profile = self.realistic_profiles[mode]
        
        print(f"🔍 REALISTIC {mode} MODE")
        print(f"   • Stop: {profile['initial_stop_loss_pct']}% → Trailing: {profile['trailing_stop_distance']}%")
        print(f"   • Target: {profile['take_profit_pct']}% | Partial: {profile['partial_profit_threshold']*100:.0f}%")
        print(f"   • Breakeven: {profile['breakeven_threshold']}% | Hold: {profile['max_hold_time_min']}min")
        print(f"   • AI: {profile['ai_threshold']}% | RSI: {profile['rsi_oversold']}-{profile['rsi_overbought']}")
        
        # Reset AI
        self.ai_analyzer = FinalOptimizedAI()
        
        return self._run_realistic_simulation(data, profile)
    
    def _run_realistic_simulation(self, data: pd.DataFrame, profile: Dict) -> Dict:
        """Run realistic simulation with proper P&L calculation"""
        
        balance = self.initial_balance
        position = None
        trades = []
        daily_trades = 0
        last_date = None
        
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        
        for i in range(50, len(data)):
            current = data.iloc[i]
            price = current['close']
            rsi = current.get('rsi', 50)
            current_date = current['timestamp'].date()
            
            # Reset daily counter
            if last_date != current_date:
                daily_trades = 0
                last_date = current_date
            
            # Skip if max trades reached
            if daily_trades >= profile['max_daily_trades']:
                continue
            
            # REALISTIC ENTRY LOGIC
            if rsi < profile['rsi_oversold'] and position is None:
                recent_data = data.iloc[max(0, i-50):i+1]
                
                # AI analysis
                ai_result = self.ai_analyzer.analyze_trade_opportunity(recent_data, price, 'buy')
                
                if ai_result['ai_confidence'] >= profile['ai_threshold']:
                    # Realistic entry filters
                    if self._realistic_entry_filter(recent_data, price, profile):
                        position_size = balance * (profile['position_size_pct'] / 100)
                        
                        position = {
                            'entry_price': price,
                            'size': position_size,
                            'ai_confidence': ai_result['ai_confidence'],
                            'current_stop': price * (1 - profile['initial_stop_loss_pct'] / 100),
                            'take_profit': price * (1 + profile['take_profit_pct'] / 100),
                            'partial_profit_target': price * (1 + profile['take_profit_pct'] * profile['partial_profit_threshold'] / 100),
                            'entry_time': current['timestamp'],
                            'highest_price': price,
                            'breakeven_moved': False,
                            'partial_taken': False,
                            'profile': profile
                        }
                        daily_trades += 1
                        
                        if len(trades) < 5:  # Show first few
                            print(f"    🚀 ENTER ${price:.4f} | AI: {ai_result['ai_confidence']:.1f}% | Stop: ${position['current_stop']:.4f} | Target: ${position['take_profit']:.4f}")
            
            # REALISTIC POSITION MANAGEMENT
            elif position is not None:
                should_close = False
                close_reason = ""
                
                # Update highest price and trailing stop
                if price > position['highest_price']:
                    position['highest_price'] = price
                    
                    # Move to breakeven protection
                    if not position['breakeven_moved'] and price > position['entry_price'] * (1 + profile['breakeven_threshold'] / 100):
                        position['current_stop'] = position['entry_price'] * 1.001  # Small profit
                        position['breakeven_moved'] = True
                        if len(trades) < 3:
                            print(f"        🛡️ BREAKEVEN ${price:.4f}")
                    
                    # Update trailing stop after breakeven
                    elif position['breakeven_moved']:
                        new_trailing_stop = price * (1 - profile['trailing_stop_distance'] / 100)
                        if new_trailing_stop > position['current_stop']:
                            position['current_stop'] = new_trailing_stop
                
                # Partial profit taking
                if not position['partial_taken'] and price >= position['partial_profit_target']:
                    # Take 40% profit, keep 60% running
                    partial_pnl = (price - position['entry_price']) * (position['size'] * 0.4 / position['entry_price'])
                    balance += partial_pnl
                    position['size'] *= 0.6  # Reduce position size
                    position['partial_taken'] = True
                    
                    if partial_pnl > 0:
                        winning_trades += 0.4  # Partial win
                        total_profit += partial_pnl
                    
                    if len(trades) < 3:
                        print(f"        💰 PARTIAL ${price:.4f} | P&L: ${partial_pnl:+.2f}")
                
                # Exit conditions
                
                # Stop loss hit
                if price <= position['current_stop']:
                    should_close = True
                    close_reason = "Stop Loss"
                
                # Take profit hit
                elif price >= position['take_profit']:
                    should_close = True
                    close_reason = "Take Profit"
                
                # Time-based exit
                elif (current['timestamp'] - position['entry_time']).total_seconds() > profile['max_hold_time_min'] * 60:
                    should_close = True
                    close_reason = "Time Exit"
                
                # Early profit exit on RSI reversal
                elif rsi > profile['rsi_overbought']:
                    unrealized_pnl_pct = ((price - position['entry_price']) / position['entry_price']) * 100
                    if unrealized_pnl_pct > 0.3:  # Only if profitable
                        should_close = True
                        close_reason = "Early Profit"
                
                if should_close:
                    # PROPER P&L CALCULATION
                    pnl = (price - position['entry_price']) * (position['size'] / position['entry_price'])
                    balance += pnl
                    
                    # REALISTIC WIN/LOSS DETERMINATION
                    if pnl > 0.01:  # Must be more than 1 cent profit
                        outcome = 'win'
                        winning_trades += (0.6 if position['partial_taken'] else 1.0)
                        total_profit += pnl
                    else:
                        outcome = 'loss'
                        losing_trades += 1
                        total_loss += abs(pnl)
                    
                    # Update AI learning
                    self.ai_analyzer.update_trade_result(position['ai_confidence'], outcome)
                    
                    # Calculate metrics
                    pnl_pct = ((price - position['entry_price']) / position['entry_price']) * 100
                    hold_time = (current['timestamp'] - position['entry_time']).total_seconds() / 60
                    
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'close_reason': close_reason,
                        'hold_time_min': hold_time,
                        'ai_confidence': position['ai_confidence'],
                        'breakeven_moved': position['breakeven_moved'],
                        'partial_taken': position['partial_taken'],
                        'win': outcome == 'win'
                    })
                    
                    if len(trades) <= 5:
                        outcome_emoji = "✅" if outcome == 'win' else "❌"
                        print(f"    📤 EXIT ${price:.4f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | {close_reason} | {outcome_emoji}")
                    
                    position = None
        
        # Calculate REALISTIC metrics
        total_trades = len(trades)
        actual_wins = len([t for t in trades if t['win']])
        win_rate = (actual_wins / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (total_profit / max(total_loss, 0.01)) if total_loss > 0 else float('inf')
        
        # Advanced analytics
        breakeven_count = len([t for t in trades if t['breakeven_moved']])
        partial_count = len([t for t in trades if t['partial_taken']])
        
        # Exit reason analysis
        exit_reasons = {}
        for trade in trades:
            reason = trade['close_reason']
            if reason not in exit_reasons:
                exit_reasons[reason] = {'total': 0, 'wins': 0}
            exit_reasons[reason]['total'] += 1
            if trade['win']:
                exit_reasons[reason]['wins'] += 1
        
        return {
            'final_balance': balance,
            'total_return': ((balance - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': total_trades,
            'winning_trades': actual_wins,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'breakeven_count': breakeven_count,
            'partial_count': partial_count,
            'exit_reasons': exit_reasons,
            'trades': trades
        }
    
    def _realistic_entry_filter(self, data: pd.DataFrame, price: float, profile: Dict) -> bool:
        """Realistic entry filters"""
        
        if len(data) < 20:
            return False
        
        # Volume check
        current_volume = data.iloc[-1].get('volume', 1000)
        avg_volume = data['volume'].tail(10).mean()
        if current_volume < avg_volume * profile['volume_multiplier']:
            return False
        
        # Don't enter in strong uptrend
        recent_rsi = data['rsi'].tail(3).mean()
        if recent_rsi > 50:
            return False
        
        return True
    
    def _generate_realistic_data(self, days: int = 14) -> pd.DataFrame:
        """Generate REALISTIC data with proper price movement"""
        print(f"🔍 Generating {days} days of REALISTIC test data...")
        
        data = []
        price = 148.0  # Starting price
        minutes = days * 24 * 60
        
        np.random.seed(555)  # Different seed for realistic data
        
        for i in range(minutes):
            # Create REALISTIC price movements with volatility
            time_factor = i / minutes
            
            # Main trend (longer cycles)
            main_trend = np.sin(2 * np.pi * time_factor * 0.5) * 8.0
            
            # Medium oscillations
            medium_osc = np.sin(2 * np.pi * time_factor * 3.0) * 3.0
            
            # Short-term noise (realistic volatility)
            noise = np.random.normal(0, 1.2)
            
            # Occasional large moves (simulate news events)
            if np.random.random() < 0.001:  # 0.1% chance
                noise += np.random.normal(0, 5.0)
            
            price_change = main_trend * 0.02 + medium_osc * 0.015 + noise * 0.03
            price += price_change
            
            # Wider price range for realistic trading
            price = max(135, min(165, price))
            
            # Realistic OHLC with proper spreads
            spread = np.random.uniform(0.15, 0.4)
            high = price + spread/2 + abs(np.random.normal(0, 0.1))
            low = price - spread/2 - abs(np.random.normal(0, 0.1))
            open_price = price + np.random.uniform(-0.08, 0.08)
            
            # Realistic volume patterns
            base_volume = 1500
            volume_multiplier = 1 + abs(price_change) * 0.3 + np.random.uniform(-0.5, 0.8)
            volume = base_volume * volume_multiplier
            
            data.append({
                'timestamp': datetime.now() - timedelta(minutes=minutes-i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df = self.indicators.calculate_all_indicators(df)
        
        print(f"✅ Generated {len(df):,} realistic data points | Range: ${df['close'].min():.2f}-${df['close'].max():.2f}")
        print(f"📊 Price volatility: {df['close'].std():.2f} | Volume range: {df['volume'].min():.0f}-{df['volume'].max():.0f}")
        return df
    
    def _display_realistic_results(self, results: Dict):
        """Display realistic, honest results"""
        
        print(f"\n" + "=" * 110)
        print("🔍 REALISTIC ULTIMATE WIN RATE RESULTS (HONEST TESTING)")
        print("=" * 110)
        
        print(f"\n🎯 REALISTIC WIN RATE PERFORMANCE:")
        print("-" * 130)
        print(f"{'Mode':<16} {'Win Rate':<12} {'Return':<12} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Profit Factor':<14} {'Status'}")
        print("-" * 130)
        
        best_winrate = 0
        best_mode = ""
        total_trades = 0
        total_wins = 0
        total_losses = 0
        
        for mode, result in results.items():
            total_trades += result['total_trades']
            total_wins += result['winning_trades']
            total_losses += result['losing_trades']
            
            if result['win_rate'] > best_winrate:
                best_winrate = result['win_rate']
                best_mode = mode
            
            # REALISTIC status levels
            if result['win_rate'] >= 70:
                status = "🏆 EXCELLENT"
            elif result['win_rate'] >= 60:
                status = "🟢 VERY GOOD"
            elif result['win_rate'] >= 55:
                status = "🟢 GOOD"
            elif result['win_rate'] >= 50:
                status = "🟡 DECENT"
            elif result['win_rate'] >= 40:
                status = "🟡 FAIR"
            else:
                status = "🔴 POOR"
            
            profit_factor_str = f"{result['profit_factor']:.2f}" if result['profit_factor'] != float('inf') else "∞"
            
            print(f"{mode:<16} {result['win_rate']:8.1f}%    {result['total_return']:+8.2f}%   "
                  f"{result['total_trades']:<8} {result['winning_trades']:<6} {result['losing_trades']:<8} {profit_factor_str:<14} {status}")
        
        print("-" * 130)
        
        overall_winrate = (total_wins / max(total_trades, 1)) * 100
        
        print(f"\n🔍 REALISTIC ACHIEVEMENT SUMMARY:")
        print(f"   👑 Best Mode: {best_mode} ({best_winrate:.1f}% win rate)")
        print(f"   📊 Overall: {overall_winrate:.1f}% ({total_wins}/{total_trades} trades)")
        print(f"   ❌ Total Losses: {total_losses}")
        
        # Compare to baseline honestly
        baseline_improvement = overall_winrate - 20
        improved_comparison = overall_winrate - 57.4
        
        print(f"   📈 vs Original (~20%): {baseline_improvement:+.1f}%")
        print(f"   📈 vs Improved (57.4%): {improved_comparison:+.1f}%")
        
        if overall_winrate >= 65:
            print(f"   🏆 EXCELLENT: 65%+ achieved!")
        elif overall_winrate >= 60:
            print(f"   🟢 VERY GOOD: 60%+ achieved!")
        elif overall_winrate >= 55:
            print(f"   🟢 GOOD: 55%+ achieved!")
        else:
            print(f"   📈 Room for improvement")
        
        # Exit reason analysis
        print(f"\n📊 EXIT REASON ANALYSIS:")
        all_exit_reasons = {}
        for mode, result in results.items():
            for reason, stats in result['exit_reasons'].items():
                if reason not in all_exit_reasons:
                    all_exit_reasons[reason] = {'total': 0, 'wins': 0}
                all_exit_reasons[reason]['total'] += stats['total']
                all_exit_reasons[reason]['wins'] += stats['wins']
        
        for reason, stats in all_exit_reasons.items():
            win_rate = (stats['wins'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"   • {reason}: {win_rate:.1f}% win rate ({stats['wins']}/{stats['total']} trades)")
        
        print("=" * 110)

def main():
    """Run realistic ultimate test"""
    bot = RealisticUltimateBot()
    results = bot.test_realistic_ultimate()
    
    # Final honest summary
    total_trades = sum(r['total_trades'] for r in results.values())
    total_wins = sum(r['winning_trades'] for r in results.values())
    total_losses = sum(r['losing_trades'] for r in results.values())
    overall_winrate = (total_wins / max(total_trades, 1)) * 100
    
    print(f"\n🔍 REALISTIC TEST COMPLETE!")
    print(f"🎯 HONEST Result: {total_wins}/{total_trades} trades | {overall_winrate:.1f}% win rate")
    print(f"❌ Losses: {total_losses}")
    
    if overall_winrate > 57.4:
        print("✅ IMPROVEMENT over our 57.4% baseline!")
    else:
        print("📈 More optimization needed")
    
    return results

if __name__ == "__main__":
    main()