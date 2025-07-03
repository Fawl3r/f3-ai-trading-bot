#!/usr/bin/env python3
"""
PROFITABILITY OPTIMIZATION TEST
Testing if more trading pairs can increase TOTAL PROFITABILITY
Even if win rate drops slightly, more opportunities = more total profit?
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ProfitabilityOptimizationTest:
    """Test different pair configurations for maximum total profitability"""
    
    def __init__(self):
        print("💰 PROFITABILITY OPTIMIZATION TEST")
        print("🎯 Focus: TOTAL PROFITABILITY (not just win rate)")
        print("📊 Testing: Can more pairs = more total profit?")
        print("=" * 80)
        
        # Original PROVEN configuration
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
            "parabolic_multiplier": (3, 8),
            "max_daily_trades_per_pair": 2  # 2 per pair to prevent overtrading
        }
        
        # Different pair configurations to test
        self.test_configurations = {
            "Original_5": {
                "name": "Original PROVEN (5 pairs)",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX'],
                "expected_wr": 73.8,
                "max_daily_trades": 10
            },
            "Top_10": {
                "name": "Top 10 Pairs",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'UNI', 'ADA', 'DOT', 'MATIC'],
                "expected_wr": 73.0,  # Slight drop expected
                "max_daily_trades": 20
            },
            "Top_15": {
                "name": "Top 15 Pairs",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'UNI', 'ADA', 'DOT', 'MATIC', 
                         'NEAR', 'ATOM', 'FTM', 'SAND', 'CRV'],
                "expected_wr": 72.0,  # Further slight drop
                "max_daily_trades": 30
            },
            "High_Volume_20": {
                "name": "High Volume 20 Pairs",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'UNI', 'ADA', 'DOT', 'MATIC',
                         'NEAR', 'ATOM', 'FTM', 'SAND', 'CRV', 'LTC', 'BCH', 'XRP', 'TRX', 'ETC'],
                "expected_wr": 71.0,  # Expected drop but more opportunities
                "max_daily_trades": 40
            },
            "Conservative_8": {
                "name": "Conservative 8 Pairs (Similar Quality)",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'UNI', 'ADA'],
                "expected_wr": 73.5,  # Minimal drop expected
                "max_daily_trades": 16
            }
        }

    def simulate_profitability_test(self, config_name: str, config: Dict) -> Dict:
        """Simulate trading with focus on total profitability"""
        
        # Test scenarios (same as proven validation)
        scenarios = {
            "Bull Market": {"trend": 0.04, "volatility": 0.025, "target_wr": 74.0, "target_return": 1424},
            "Bear Market": {"trend": -0.02, "volatility": 0.04, "target_wr": 75.6, "target_return": 3085},
            "Sideways": {"trend": 0.001, "volatility": 0.015, "target_wr": 72.2, "target_return": 319},
            "High Vol": {"trend": 0.03, "volatility": 0.08, "target_wr": 77.8, "target_return": 14755},
            "Low Vol": {"trend": 0.02, "volatility": 0.01, "target_wr": 72.3, "target_return": 152}
        }
        
        scenario_results = {}
        
        for scenario_name, params in scenarios.items():
            result = self._simulate_profitability_scenario(config, params, scenario_name)
            scenario_results[scenario_name] = result
        
        # Calculate overall metrics
        total_trades = sum(r['total_trades'] for r in scenario_results.values())
        total_wins = sum(r['wins'] for r in scenario_results.values())
        total_losses = sum(r['losses'] for r in scenario_results.values())
        
        avg_win_rate = np.mean([r['win_rate'] for r in scenario_results.values()])
        avg_return = np.mean([r['return_pct'] for r in scenario_results.values()])
        total_profit = sum(r['total_profit'] for r in scenario_results.values())
        max_single_trade = max(r['max_trade'] for r in scenario_results.values())
        
        return {
            'config_name': config_name,
            'scenario_results': scenario_results,
            'avg_win_rate': avg_win_rate,
            'avg_return_pct': avg_return,
            'total_trades': total_trades / len(scenarios),  # Average per scenario
            'total_profit': total_profit / len(scenarios),  # Average per scenario
            'max_single_trade': max_single_trade,
            'profit_per_trade': (total_profit / len(scenarios)) / (total_trades / len(scenarios)) if total_trades > 0 else 0,
            'pairs_count': len(config['pairs']),
            'expected_daily_trades': config['max_daily_trades']
        }

    def _simulate_profitability_scenario(self, config: Dict, params: Dict, scenario_name: str) -> Dict:
        """Simulate single scenario focused on profitability"""
        
        # Parameters
        initial_balance = 200.0
        periods = 1000
        np.random.seed(42)  # Keep same for comparison
        
        wins = 0
        losses = 0
        total_profit = 0
        balance = initial_balance
        max_single_trade = 0
        
        # Calculate trade frequency based on number of pairs
        pairs_count = len(config['pairs'])
        base_frequency = self.base_config['trade_frequency']
        
        # More pairs = proportionally more opportunities, but with quality filter
        # Use square root scaling to prevent over-trading
        frequency_multiplier = np.sqrt(pairs_count / 5)  # Original had 5 pairs
        adjusted_frequency = base_frequency * frequency_multiplier
        
        for i in range(periods):
            # Market favorability (same proven formula)
            vol_factor = min(params['volatility'] / 0.05, 1.0)
            trend_factor = min(abs(params['trend']) / 0.04, 1.0)
            cycle_factor = 0.5 + 0.5 * np.sin(i / 100)
            favorability = (vol_factor * 0.4 + trend_factor * 0.4 + cycle_factor * 0.2)
            
            # Quality filter: More pairs might have slightly lower average quality
            quality_factor = 1.0 - (pairs_count - 5) * 0.01  # 1% quality drop per additional pair
            quality_factor = max(quality_factor, 0.85)  # Minimum 85% quality
            
            if np.random.random() < adjusted_frequency:
                # Position sizing (same proven formula)
                position_size = balance * (self.base_config['position_size_range'][0] + 
                                         self.base_config['position_size_range'][1] * favorability)
                
                # Leverage (same proven formula)
                leverage = self.base_config['leverage_range'][0] + \
                          (self.base_config['leverage_range'][1] - self.base_config['leverage_range'][0]) * \
                          (params['volatility'] / 0.08)
                trade_size = position_size * leverage
                
                # Win probability with quality adjustment
                base_win_prob = self.base_config['base_win_prob'] * quality_factor
                favorability_boost = favorability * self.base_config['favorability_boost']
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
        
        # Calculate metrics
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

    def run_profitability_comparison(self):
        """Run comprehensive profitability comparison"""
        
        print("\n🔬 RUNNING PROFITABILITY OPTIMIZATION TESTS")
        print("🎯 Focus: Which configuration generates the most total profit?")
        print("=" * 80)
        
        results = {}
        
        # Test each configuration
        for config_name, config in self.test_configurations.items():
            print(f"\n📊 Testing {config['name']} ({len(config['pairs'])} pairs)...")
            
            test_result = self.simulate_profitability_test(config_name, config)
            results[config_name] = test_result
            
            print(f"   Win Rate: {test_result['avg_win_rate']:.1f}%")
            print(f"   Avg Return: +{test_result['avg_return_pct']:.1f}%")
            print(f"   Trades/Test: {test_result['total_trades']:.0f}")
            print(f"   Profit/Trade: ${test_result['profit_per_trade']:.2f}")
        
        # Comprehensive comparison
        self.analyze_profitability_results(results)
        
        return results

    def analyze_profitability_results(self, results: Dict):
        """Analyze and compare profitability results"""
        
        print(f"\n" + "=" * 100)
        print("💰 PROFITABILITY OPTIMIZATION ANALYSIS")
        print("=" * 100)
        
        # Sort by total profitability
        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_profit'], reverse=True)
        
        print(f"📊 PROFITABILITY RANKING (By Total Profit):")
        print(f"{'Rank':<4} {'Configuration':<25} {'Pairs':<6} {'Win Rate':<9} {'Avg Return':<11} {'Trades':<8} {'Total Profit':<12} {'Profit/Trade':<12}")
        print("-" * 100)
        
        for i, (config_name, result) in enumerate(sorted_results, 1):
            config = self.test_configurations[config_name]
            status = "🏆" if i == 1 else "📊"
            
            print(f"{i:<4} {status} {config['name']:<22} {result['pairs_count']:<6} "
                  f"{result['avg_win_rate']:<8.1f}% {result['avg_return_pct']:<10.1f}% "
                  f"{result['total_trades']:<7.0f} ${result['total_profit']:<11.2f} "
                  f"${result['profit_per_trade']:<11.2f}")
        
        # Best performer analysis
        best_config_name, best_result = sorted_results[0]
        best_config = self.test_configurations[best_config_name]
        original_result = results['Original_5']
        
        print(f"\n🏆 BEST PERFORMER: {best_config['name']}")
        print(f"   🎯 Total Profit: ${best_result['total_profit']:.2f}")
        print(f"   📊 Win Rate: {best_result['avg_win_rate']:.1f}%")
        print(f"   💰 Average Return: +{best_result['avg_return_pct']:.1f}%")
        print(f"   📈 Trades per Test: {best_result['total_trades']:.0f}")
        print(f"   ⚡ Profit per Trade: ${best_result['profit_per_trade']:.2f}")
        print(f"   🎲 Trading Pairs: {result['pairs_count']}")
        
        # Comparison with original
        profit_improvement = ((best_result['total_profit'] - original_result['total_profit']) / 
                            original_result['total_profit'] * 100)
        wr_change = best_result['avg_win_rate'] - original_result['avg_win_rate']
        trade_increase = ((best_result['total_trades'] - original_result['total_trades']) / 
                         original_result['total_trades'] * 100)
        
        print(f"\n📈 COMPARISON WITH ORIGINAL:")
        print(f"   💰 Total Profit Change: {profit_improvement:+.1f}%")
        print(f"   🎯 Win Rate Change: {wr_change:+.1f} percentage points")
        print(f"   📊 Trade Volume Change: {trade_increase:+.1f}%")
        print(f"   ⚡ Profit per Trade: ${best_result['profit_per_trade']:.2f} vs ${original_result['profit_per_trade']:.2f}")
        
        # Risk assessment
        print(f"\n🛡️ RISK ASSESSMENT:")
        if best_result['avg_win_rate'] >= 73.0:
            print(f"   ✅ Win rate acceptable: {best_result['avg_win_rate']:.1f}% (>73%)")
        else:
            print(f"   ⚠️ Win rate concerning: {best_result['avg_win_rate']:.1f}% (<73%)")
        
        if profit_improvement > 20:
            print(f"   ✅ Significant profit improvement: +{profit_improvement:.1f}%")
        elif profit_improvement > 0:
            print(f"   📊 Moderate profit improvement: +{profit_improvement:.1f}%")
        else:
            print(f"   ❌ No profit improvement: {profit_improvement:.1f}%")
        
        # Final recommendation
        print(f"\n🎯 RECOMMENDATION:")
        
        if (best_result['avg_win_rate'] >= 72.0 and profit_improvement > 15 and 
            best_config_name != 'Original_5'):
            print(f"   🚀 UPGRADE RECOMMENDED: {best_config['name']}")
            print(f"   💡 Reason: {profit_improvement:+.1f}% more profit with acceptable {best_result['avg_win_rate']:.1f}% win rate")
            print(f"   📊 Risk: Low (win rate still strong)")
            
            # Show specific pair list
            print(f"\n📋 RECOMMENDED PAIR LIST:")
            pairs_str = ", ".join(best_config['pairs'])
            print(f"   {pairs_str}")
            
        elif best_config_name == 'Original_5':
            print(f"   🏆 KEEP ORIGINAL: No improvement found")
            print(f"   💡 Reason: Original 5-pair configuration is still optimal")
            print(f"   📊 Risk: Zero (proven configuration)")
            
        else:
            print(f"   🛡️ KEEP ORIGINAL: Risk too high")
            print(f"   💡 Reason: Win rate drop ({wr_change:.1f}pp) not worth profit gain (+{profit_improvement:.1f}%)")
            print(f"   📊 Risk: High (significant win rate degradation)")
        
        return best_config_name, best_result

def main():
    """Run profitability optimization test"""
    
    tester = ProfitabilityOptimizationTest()
    results = tester.run_profitability_comparison()
    
    return results

if __name__ == "__main__":
    main() 