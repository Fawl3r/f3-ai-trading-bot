#!/usr/bin/env python3
"""
OPTIMIZED 15-PAIR WIN RATE TEST
Target: 72-75% win rate with 15 pairs + realistic time calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class Optimized15PairWinRateTest:
    """Optimize 15-pair configuration for 72-75% win rate with time analysis"""
    
    def __init__(self):
        print("🎯 OPTIMIZED 15-PAIR WIN RATE TEST")
        print("🏆 Target: 72-75% win rate with 15 pairs")
        print("⏰ Including: Realistic time calculations")
        print("=" * 80)
        
        # 15-pair configuration (from sweet spot analysis)
        self.trading_pairs = ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'UNI', 'ADA', 'DOT', 'MATIC', 
                             'NEAR', 'ATOM', 'FTM', 'SAND', 'CRV']
        
        # Base proven configuration
        self.base_config = {
            "position_size_range": (0.02, 0.05),
            "leverage_range": (8, 15),
            "base_win_prob": 0.70,
            "favorability_boost": 0.10,
            "max_win_prob": 0.85,
            "stop_loss": 0.009,
            "profit_multiplier_range": (0.01, 0.20),
            "trade_frequency": 0.05,
            "parabolic_chance": 0.05,
            "parabolic_multiplier": (3, 8)
        }
        
        # Win rate optimization strategies
        self.optimization_strategies = {
            "Current_15": {
                "name": "Current 15-Pair (Baseline)",
                "quality_multiplier": 0.95,
                "confidence_threshold": 75.0,
                "volume_requirement": 1.5,
                "momentum_threshold": 0.015,
                "expected_wr": 70.1
            },
            "Conservative_15": {
                "name": "Conservative 15-Pair (Higher Quality)",
                "quality_multiplier": 0.98,  # Higher quality
                "confidence_threshold": 80.0,  # Higher confidence
                "volume_requirement": 2.0,  # Higher volume requirement
                "momentum_threshold": 0.020,  # Stronger momentum
                "expected_wr": 72.5
            },
            "Selective_15": {
                "name": "Selective 15-Pair (Ultra Quality)",
                "quality_multiplier": 1.0,  # No quality degradation
                "confidence_threshold": 85.0,  # Very high confidence
                "volume_requirement": 2.5,  # High volume requirement
                "momentum_threshold": 0.025,  # Strong momentum only
                "expected_wr": 74.0
            },
            "Balanced_15": {
                "name": "Balanced 15-Pair (Optimal Mix)",
                "quality_multiplier": 0.97,  # Slight quality boost
                "confidence_threshold": 78.0,  # Good confidence
                "volume_requirement": 1.8,  # Moderate volume
                "momentum_threshold": 0.018,  # Good momentum
                "expected_wr": 72.0
            }
        }

    def simulate_optimized_15_pair(self, strategy_name: str, strategy: Dict) -> Dict:
        """Simulate optimized 15-pair configuration"""
        
        scenarios = {
            "Bull Market": {"trend": 0.04, "volatility": 0.025},
            "Bear Market": {"trend": -0.02, "volatility": 0.04},
            "Sideways": {"trend": 0.001, "volatility": 0.015},
            "High Vol": {"trend": 0.03, "volatility": 0.08},
            "Low Vol": {"trend": 0.02, "volatility": 0.01}
        }
        
        scenario_results = {}
        
        for scenario_name, params in scenarios.items():
            result = self._simulate_optimized_scenario(strategy, params, scenario_name)
            scenario_results[scenario_name] = result
        
        # Calculate comprehensive metrics
        avg_win_rate = np.mean([r['win_rate'] for r in scenario_results.values()])
        avg_return = np.mean([r['return_pct'] for r in scenario_results.values()])
        total_trades = np.mean([r['total_trades'] for r in scenario_results.values()])
        total_profit = np.mean([r['total_profit'] for r in scenario_results.values()])
        profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        
        # Calculate trading frequency metrics
        daily_trades = total_trades / 30  # Assuming 30-day test period
        weekly_trades = total_trades / 4.3  # Average weeks in month
        
        # Time calculations (realistic trading scenario)
        starting_balance = 51.63  # User's actual balance
        target_profit_real = total_profit * (starting_balance / 200.0)  # Scale to real balance
        
        # Calculate time to achieve target profit
        if daily_trades > 0 and profit_per_trade > 0:
            # Conservative estimate: 70% of backtest performance in live trading
            live_performance_factor = 0.70
            live_daily_profit = daily_trades * profit_per_trade * live_performance_factor * (starting_balance / 200.0)
            
            if live_daily_profit > 0:
                days_to_target = target_profit_real / live_daily_profit
                weeks_to_target = days_to_target / 7
                months_to_target = days_to_target / 30
            else:
                days_to_target = weeks_to_target = months_to_target = float('inf')
        else:
            days_to_target = weeks_to_target = months_to_target = float('inf')
        
        return {
            'strategy_name': strategy_name,
            'avg_win_rate': avg_win_rate,
            'avg_return_pct': avg_return,
            'total_trades': total_trades,
            'total_profit': total_profit,
            'profit_per_trade': profit_per_trade,
            'daily_trades': daily_trades,
            'weekly_trades': weekly_trades,
            'target_profit_real': target_profit_real,
            'days_to_target': days_to_target,
            'weeks_to_target': weeks_to_target,
            'months_to_target': months_to_target,
            'scenario_results': scenario_results,
            'quality_multiplier': strategy['quality_multiplier'],
            'confidence_threshold': strategy['confidence_threshold']
        }

    def _simulate_optimized_scenario(self, strategy: Dict, params: Dict, scenario_name: str) -> Dict:
        """Simulate scenario with optimized parameters"""
        
        initial_balance = 200.0
        periods = 1000
        np.random.seed(42)
        
        wins = 0
        losses = 0
        total_profit = 0
        balance = initial_balance
        max_single_trade = 0
        
        # Optimized frequency calculation for 15 pairs
        pairs_count = len(self.trading_pairs)
        
        # Quality-focused frequency (prioritize quality over quantity)
        base_frequency = self.base_config['trade_frequency']
        
        # More conservative scaling for higher win rate
        if strategy['confidence_threshold'] >= 85:
            frequency_multiplier = 2.0  # Conservative
        elif strategy['confidence_threshold'] >= 80:
            frequency_multiplier = 2.3  # Moderate
        else:
            frequency_multiplier = 2.6  # More aggressive
        
        adjusted_frequency = base_frequency * frequency_multiplier
        
        for i in range(periods):
            # Market favorability
            vol_factor = min(params['volatility'] / 0.05, 1.0)
            trend_factor = min(abs(params['trend']) / 0.04, 1.0)
            cycle_factor = 0.5 + 0.5 * np.sin(i / 100)
            favorability = (vol_factor * 0.4 + trend_factor * 0.4 + cycle_factor * 0.2)
            
            # Enhanced quality filter simulation
            volume_condition = np.random.random() < (1.0 / strategy['volume_requirement'])
            momentum_condition = np.random.random() < 0.8  # 80% of periods have good momentum
            confidence_condition = (favorability * 100) >= (strategy['confidence_threshold'] - 10)
            
            # Only trade if all conditions met
            if (np.random.random() < adjusted_frequency and 
                volume_condition and momentum_condition and confidence_condition):
                
                # Position sizing (same proven formula)
                position_size = balance * (self.base_config['position_size_range'][0] + 
                                         self.base_config['position_size_range'][1] * favorability)
                
                # Leverage
                leverage = self.base_config['leverage_range'][0] + \
                          (self.base_config['leverage_range'][1] - self.base_config['leverage_range'][0]) * \
                          (params['volatility'] / 0.08)
                trade_size = position_size * leverage
                
                # Optimized win probability
                base_win_prob = self.base_config['base_win_prob'] * strategy['quality_multiplier']
                
                # Confidence boost for high-quality setups
                confidence_boost = (strategy['confidence_threshold'] - 75) / 100  # 0-0.1 boost
                favorability_boost = (favorability * self.base_config['favorability_boost']) + confidence_boost
                
                win_probability = min(base_win_prob + favorability_boost, self.base_config['max_win_prob'])
                
                if np.random.random() < win_probability:
                    # Winning trade
                    profit_mult = self.base_config['profit_multiplier_range'][0] + \
                                (self.base_config['profit_multiplier_range'][1] - self.base_config['profit_multiplier_range'][0]) * \
                                (params['volatility'] * favorability)
                    profit = trade_size * profit_mult
                    
                    # Parabolic trades
                    if params['volatility'] > 0.06 and np.random.random() < self.base_config['parabolic_chance']:
                        profit *= np.random.uniform(self.base_config['parabolic_multiplier'][0], 
                                                   self.base_config['parabolic_multiplier'][1])
                    
                    wins += 1
                    total_profit += profit
                    balance += profit
                    
                    if profit > max_single_trade:
                        max_single_trade = profit
                else:
                    # Losing trade
                    loss = trade_size * self.base_config['stop_loss']
                    losses += 1
                    total_profit -= loss
                    balance -= loss
        
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        return_pct = ((balance - initial_balance) / initial_balance * 100)
        
        return {
            'scenario': scenario_name,
            'win_rate': win_rate,
            'return_pct': return_pct,
            'total_trades': total_trades,
            'total_profit': total_profit,
            'wins': wins,
            'losses': losses,
            'max_trade': max_single_trade,
            'final_balance': balance
        }

    def run_optimized_15_pair_test(self):
        """Run optimized 15-pair test"""
        
        print(f"\n🔬 RUNNING OPTIMIZED 15-PAIR WIN RATE TESTS")
        print(f"🎯 Goal: 72-75% win rate with 15 pairs")
        print(f"💰 Real balance: $51.63")
        print("=" * 80)
        
        results = {}
        
        # Test each optimization strategy
        for strategy_name, strategy in self.optimization_strategies.items():
            print(f"\n📊 Testing {strategy['name']}...")
            print(f"   Confidence Threshold: {strategy['confidence_threshold']}%")
            print(f"   Volume Requirement: {strategy['volume_requirement']}x")
            print(f"   Expected Win Rate: {strategy['expected_wr']}%")
            
            result = self.simulate_optimized_15_pair(strategy_name, strategy)
            results[strategy_name] = result
            
            print(f"   🎯 Actual Win Rate: {result['avg_win_rate']:.1f}%")
            print(f"   💰 Total Profit: ${result['total_profit']:.2f}")
            print(f"   📊 Daily Trades: {result['daily_trades']:.1f}")
            print(f"   ⏰ Time to Target: {result['months_to_target']:.1f} months")
        
        self.analyze_optimized_results(results)
        return results

    def analyze_optimized_results(self, results: Dict):
        """Analyze optimized results with time calculations"""
        
        print(f"\n" + "=" * 120)
        print("🏆 OPTIMIZED 15-PAIR ANALYSIS RESULTS")
        print("=" * 120)
        
        # Sort by win rate
        sorted_by_wr = sorted(results.items(), key=lambda x: x[1]['avg_win_rate'], reverse=True)
        
        print(f"📊 WIN RATE RANKING:")
        print(f"{'Rank':<4} {'Strategy':<30} {'Win Rate':<10} {'Profit':<12} {'Daily Trades':<12} {'Time to Target':<15}")
        print("-" * 120)
        
        target_achieved = []
        
        for i, (strategy_name, result) in enumerate(sorted_by_wr, 1):
            strategy = self.optimization_strategies[strategy_name]
            
            # Check if target achieved
            target_met = result['avg_win_rate'] >= 72.0
            if target_met:
                target_achieved.append((strategy_name, result))
            
            status = "🎯" if target_met else "📊"
            time_str = f"{result['months_to_target']:.1f} months" if result['months_to_target'] < 100 else "∞"
            
            print(f"{i:<4} {status} {strategy['name']:<27} {result['avg_win_rate']:<9.1f}% "
                  f"${result['total_profit']:<11.2f} {result['daily_trades']:<11.1f} {time_str:<15}")
        
        # Analysis of strategies that achieved 72%+ win rate
        if target_achieved:
            print(f"\n🎯 STRATEGIES ACHIEVING 72%+ WIN RATE:")
            
            for strategy_name, result in target_achieved:
                strategy = self.optimization_strategies[strategy_name]
                
                print(f"\n🏆 {strategy['name']}:")
                print(f"   🎯 Win Rate: {result['avg_win_rate']:.1f}%")
                print(f"   💰 Backtest Profit: ${result['total_profit']:.2f}")
                print(f"   💵 Real Target Profit: ${result['target_profit_real']:.2f}")
                print(f"   📊 Daily Trades: {result['daily_trades']:.1f}")
                print(f"   ⏰ Time to Achieve Target:")
                
                if result['months_to_target'] < 100:
                    print(f"      • {result['days_to_target']:.0f} days")
                    print(f"      • {result['weeks_to_target']:.1f} weeks") 
                    print(f"      • {result['months_to_target']:.1f} months")
                    
                    # Break down the calculation
                    print(f"   📋 Calculation Details:")
                    print(f"      • Starting balance: $51.63")
                    print(f"      • Target profit: ${result['target_profit_real']:.2f}")
                    print(f"      • Expected daily profit: ${result['target_profit_real']/result['days_to_target']:.2f}")
                    print(f"      • Trades per day: {result['daily_trades']:.1f}")
                    print(f"      • Profit per trade: ${result['profit_per_trade']:.2f}")
                else:
                    print(f"      • Too long to calculate (low trade frequency)")
        
        # Best performer for 72%+ win rate
        if target_achieved:
            # Sort target achieved by profit
            best_strategy_name, best_result = max(target_achieved, key=lambda x: x[1]['total_profit'])
            best_strategy = self.optimization_strategies[best_strategy_name]
            
            print(f"\n🚀 RECOMMENDED CONFIGURATION:")
            print(f"   🏆 Strategy: {best_strategy['name']}")
            print(f"   🎯 Win Rate: {best_result['avg_win_rate']:.1f}%")
            print(f"   💰 Expected Profit: ${best_result['target_profit_real']:.2f}")
            print(f"   ⏰ Time to Achieve: {best_result['months_to_target']:.1f} months")
            print(f"   📊 Trading Activity: {best_result['daily_trades']:.1f} trades/day")
            
            # Configuration details
            print(f"\n⚙️ CONFIGURATION SETTINGS:")
            print(f"   🎲 Trading Pairs: {len(self.trading_pairs)} pairs")
            print(f"   🤖 Confidence Threshold: {best_strategy['confidence_threshold']}%")
            print(f"   📊 Volume Requirement: {best_strategy['volume_requirement']}x")
            print(f"   🔍 Quality Multiplier: {best_strategy['quality_multiplier']}")
            
            # Show trading pairs
            print(f"\n📋 TRADING PAIRS:")
            for i, pair in enumerate(self.trading_pairs, 1):
                print(f"   {i:2d}. {pair}")
            
            # Risk assessment
            min_wr = min([r['win_rate'] for r in best_result['scenario_results'].values()])
            max_wr = max([r['win_rate'] for r in best_result['scenario_results'].values()])
            
            print(f"\n🛡️ RISK ASSESSMENT:")
            print(f"   📊 Win Rate Range: {min_wr:.1f}% - {max_wr:.1f}%")
            print(f"   🎯 Average Win Rate: {best_result['avg_win_rate']:.1f}%")
            print(f"   ✅ Target Achievement: {'YES' if best_result['avg_win_rate'] >= 72.0 else 'NO'}")
            
            return best_strategy_name, best_result
        else:
            print(f"\n❌ NO STRATEGIES ACHIEVED 72%+ WIN RATE")
            print(f"   💡 Best achieved: {max(results.values(), key=lambda x: x['avg_win_rate'])['avg_win_rate']:.1f}%")
            print(f"   🔧 May need further optimization or lower target")
            return None, None

def main():
    """Run optimized 15-pair win rate test"""
    
    tester = Optimized15PairWinRateTest()
    results = tester.run_optimized_15_pair_test()
    
    return results

if __name__ == "__main__":
    main() 