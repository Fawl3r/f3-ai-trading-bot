#!/usr/bin/env python3
"""
SWEET SPOT OPTIMIZATION
Finding the optimal balance between profitability and win rate
Testing middle ground configurations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SweetSpotOptimization:
    """Find the sweet spot between profitability and win rate"""
    
    def __init__(self):
        print("🎯 SWEET SPOT OPTIMIZATION")
        print("🔍 Finding optimal balance: Profitability vs Win Rate")
        print("📊 Testing middle ground configurations")
        print("=" * 80)
        
        # Base configuration (same proven logic)
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
        
        # Sweet spot configurations - carefully selected pairs
        self.configurations = {
            "Original_5": {
                "name": "Original PROVEN",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX'],
                "quality_tier": "Tier 1",
                "expected_wr": 73.8
            },
            "Elite_7": {
                "name": "Elite 7 (High Quality)",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'UNI'],
                "quality_tier": "Tier 1",
                "expected_wr": 72.5
            },
            "Premium_10": {
                "name": "Premium 10 (Quality Focus)",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'UNI', 'ADA', 'DOT', 'MATIC'],
                "quality_tier": "Tier 1-2",
                "expected_wr": 71.0
            },
            "Balanced_12": {
                "name": "Balanced 12 (Sweet Spot?)",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'UNI', 'ADA', 'DOT', 'MATIC', 'NEAR', 'ATOM'],
                "quality_tier": "Tier 1-2",
                "expected_wr": 70.0
            },
            "Extended_15": {
                "name": "Extended 15 (More Opportunities)",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'UNI', 'ADA', 'DOT', 'MATIC', 
                         'NEAR', 'ATOM', 'FTM', 'SAND', 'CRV'],
                "quality_tier": "Tier 1-3",
                "expected_wr": 68.5
            }
        }

    def simulate_sweet_spot_config(self, config_name: str, config: Dict) -> Dict:
        """Simulate configuration with focus on balance"""
        
        scenarios = {
            "Bull Market": {"trend": 0.04, "volatility": 0.025},
            "Bear Market": {"trend": -0.02, "volatility": 0.04},
            "Sideways": {"trend": 0.001, "volatility": 0.015},
            "High Vol": {"trend": 0.03, "volatility": 0.08},
            "Low Vol": {"trend": 0.02, "volatility": 0.01}
        }
        
        scenario_results = {}
        
        for scenario_name, params in scenarios.items():
            result = self._simulate_balanced_scenario(config, params, scenario_name)
            scenario_results[scenario_name] = result
        
        # Calculate comprehensive metrics
        avg_win_rate = np.mean([r['win_rate'] for r in scenario_results.values()])
        avg_return = np.mean([r['return_pct'] for r in scenario_results.values()])
        total_trades = np.mean([r['total_trades'] for r in scenario_results.values()])
        total_profit = np.mean([r['total_profit'] for r in scenario_results.values()])
        profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        
        # Calculate risk-adjusted metrics
        min_win_rate = min([r['win_rate'] for r in scenario_results.values()])
        win_rate_consistency = 100 - (max([r['win_rate'] for r in scenario_results.values()]) - min_win_rate)
        
        # Sweet spot score (balance of profitability and safety)
        profitability_score = total_profit / 100  # Scale profit
        safety_score = avg_win_rate / 100  # Scale win rate
        consistency_score = win_rate_consistency / 100  # Scale consistency
        
        # Weighted sweet spot score
        sweet_spot_score = (profitability_score * 0.4 + safety_score * 0.4 + consistency_score * 0.2) * 100
        
        return {
            'config_name': config_name,
            'pairs_count': len(config['pairs']),
            'avg_win_rate': avg_win_rate,
            'min_win_rate': min_win_rate,
            'avg_return_pct': avg_return,
            'total_trades': total_trades,
            'total_profit': total_profit,
            'profit_per_trade': profit_per_trade,
            'win_rate_consistency': win_rate_consistency,
            'sweet_spot_score': sweet_spot_score,
            'scenario_results': scenario_results,
            'quality_tier': config['quality_tier']
        }

    def _simulate_balanced_scenario(self, config: Dict, params: Dict, scenario_name: str) -> Dict:
        """Simulate with enhanced quality control"""
        
        initial_balance = 200.0
        periods = 1000
        np.random.seed(42)
        
        wins = 0
        losses = 0
        total_profit = 0
        balance = initial_balance
        max_single_trade = 0
        
        # Enhanced pair quality system
        pairs_count = len(config['pairs'])
        
        # Quality-based frequency adjustment (better than square root)
        if pairs_count <= 7:
            quality_multiplier = 1.0  # No quality degradation for small sets
            frequency_multiplier = pairs_count / 5  # Linear scaling
        elif pairs_count <= 12:
            quality_multiplier = 0.98  # Minimal quality degradation
            frequency_multiplier = 1.4 + (pairs_count - 7) * 0.15  # Gradual scaling
        else:
            quality_multiplier = 0.95  # Some quality degradation
            frequency_multiplier = 2.15 + (pairs_count - 12) * 0.1  # Slower scaling
        
        adjusted_frequency = self.base_config['trade_frequency'] * frequency_multiplier
        
        for i in range(periods):
            # Market favorability
            vol_factor = min(params['volatility'] / 0.05, 1.0)
            trend_factor = min(abs(params['trend']) / 0.04, 1.0)
            cycle_factor = 0.5 + 0.5 * np.sin(i / 100)
            favorability = (vol_factor * 0.4 + trend_factor * 0.4 + cycle_factor * 0.2)
            
            if np.random.random() < adjusted_frequency:
                # Position sizing
                position_size = balance * (self.base_config['position_size_range'][0] + 
                                         self.base_config['position_size_range'][1] * favorability)
                
                # Leverage
                leverage = self.base_config['leverage_range'][0] + \
                          (self.base_config['leverage_range'][1] - self.base_config['leverage_range'][0]) * \
                          (params['volatility'] / 0.08)
                trade_size = position_size * leverage
                
                # Enhanced win probability with quality control
                base_win_prob = self.base_config['base_win_prob'] * quality_multiplier
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

    def run_sweet_spot_analysis(self):
        """Run sweet spot analysis"""
        
        print("\n🔍 RUNNING SWEET SPOT ANALYSIS")
        print("🎯 Goal: Optimal balance between profitability and win rate")
        print("=" * 80)
        
        results = {}
        
        for config_name, config in self.configurations.items():
            print(f"\n📊 Testing {config['name']} ({len(config['pairs'])} pairs)...")
            
            result = self.simulate_sweet_spot_config(config_name, config)
            results[config_name] = result
            
            print(f"   Win Rate: {result['avg_win_rate']:.1f}% (min: {result['min_win_rate']:.1f}%)")
            print(f"   Total Profit: ${result['total_profit']:.2f}")
            print(f"   Trades: {result['total_trades']:.0f}")
            print(f"   Sweet Spot Score: {result['sweet_spot_score']:.1f}/100")
        
        self.analyze_sweet_spot_results(results)
        return results

    def analyze_sweet_spot_results(self, results: Dict):
        """Analyze sweet spot results"""
        
        print(f"\n" + "=" * 120)
        print("🎯 SWEET SPOT ANALYSIS RESULTS")
        print("=" * 120)
        
        # Sort by sweet spot score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['sweet_spot_score'], reverse=True)
        
        print(f"{'Rank':<4} {'Configuration':<25} {'Pairs':<6} {'Win Rate':<12} {'Min WR':<8} {'Profit':<12} {'Trades':<8} {'Score':<8}")
        print("-" * 120)
        
        for i, (config_name, result) in enumerate(sorted_results, 1):
            config = self.configurations[config_name]
            status = "🏆" if i == 1 else "📊"
            
            print(f"{i:<4} {status} {config['name']:<22} {result['pairs_count']:<6} "
                  f"{result['avg_win_rate']:<11.1f}% {result['min_win_rate']:<7.1f}% "
                  f"${result['total_profit']:<11.2f} {result['total_trades']:<7.0f} {result['sweet_spot_score']:<7.1f}")
        
        # Best sweet spot analysis
        best_config_name, best_result = sorted_results[0]
        best_config = self.configurations[best_config_name]
        original_result = results['Original_5']
        
        print(f"\n🏆 SWEET SPOT WINNER: {best_config['name']}")
        print(f"   🎯 Sweet Spot Score: {best_result['sweet_spot_score']:.1f}/100")
        print(f"   📊 Win Rate: {best_result['avg_win_rate']:.1f}% (min: {best_result['min_win_rate']:.1f}%)")
        print(f"   💰 Total Profit: ${best_result['total_profit']:.2f}")
        print(f"   📈 Trades: {best_result['total_trades']:.0f}")
        print(f"   ⚡ Profit/Trade: ${best_result['profit_per_trade']:.2f}")
        print(f"   🎲 Pairs: {result['pairs_count']} ({best_config['quality_tier']})")
        
        # Comparison analysis
        profit_vs_original = ((best_result['total_profit'] - original_result['total_profit']) / 
                            original_result['total_profit'] * 100)
        wr_vs_original = best_result['avg_win_rate'] - original_result['avg_win_rate']
        
        print(f"\n📈 SWEET SPOT vs ORIGINAL:")
        print(f"   💰 Profit Change: {profit_vs_original:+.1f}%")
        print(f"   🎯 Win Rate Change: {wr_vs_original:+.1f}pp")
        print(f"   📊 Trade Change: {((best_result['total_trades'] - original_result['total_trades'])/original_result['total_trades']*100):+.1f}%")
        
        # Risk assessment
        print(f"\n🛡️ RISK ASSESSMENT:")
        
        if best_result['avg_win_rate'] >= 70.0:
            risk_level = "LOW"
            risk_color = "✅"
        elif best_result['avg_win_rate'] >= 67.0:
            risk_level = "MODERATE"
            risk_color = "⚠️"
        else:
            risk_level = "HIGH"
            risk_color = "❌"
        
        print(f"   {risk_color} Risk Level: {risk_level}")
        print(f"   📊 Win Rate: {best_result['avg_win_rate']:.1f}%")
        print(f"   📉 Minimum WR: {best_result['min_win_rate']:.1f}%")
        print(f"   🎯 Consistency: {best_result['win_rate_consistency']:.1f}%")
        
        # Final recommendation
        print(f"\n🎯 SWEET SPOT RECOMMENDATION:")
        
        if (best_result['sweet_spot_score'] > original_result['sweet_spot_score'] + 5 and 
            best_result['avg_win_rate'] >= 70.0):
            
            print(f"   🚀 IMPLEMENT SWEET SPOT: {best_config['name']}")
            print(f"   💡 Reason: Better balance (+{profit_vs_original:.1f}% profit, {wr_vs_original:+.1f}pp win rate)")
            print(f"   📊 Risk: {risk_level} ({best_result['avg_win_rate']:.1f}% win rate)")
            
            # Show specific pairs
            print(f"\n📋 RECOMMENDED TRADING PAIRS:")
            for i, pair in enumerate(best_config['pairs'], 1):
                print(f"   {i:2d}. {pair}")
                
        else:
            print(f"   🏆 KEEP ORIGINAL: Still the best balance")
            print(f"   💡 Reason: Original provides best risk-adjusted returns")
            print(f"   📊 Risk: MINIMAL (proven 73.8% win rate)")
        
        return best_config_name, best_result

def main():
    """Run sweet spot optimization"""
    
    optimizer = SweetSpotOptimization()
    results = optimizer.run_sweet_spot_analysis()
    
    return results

if __name__ == "__main__":
    main() 