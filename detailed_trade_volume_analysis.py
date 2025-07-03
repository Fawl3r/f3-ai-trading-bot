#!/usr/bin/env python3
"""
DETAILED TRADE VOLUME & PROFIT PROJECTION ANALYSIS
Comprehensive comparison including full trade volumes and realistic projections
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DetailedTradeVolumeAnalysis:
    """Detailed analysis of trade volumes and profit projections"""
    
    def __init__(self):
        print("📊 DETAILED TRADE VOLUME & PROFIT PROJECTION ANALYSIS")
        print("🔍 Comprehensive comparison of configurations")
        print("💰 Realistic profit projections with $51.63")
        print("=" * 80)
        
        # Configuration data from previous tests
        self.configurations = {
            "Original_5": {
                "name": "Original Proven 5",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX'],
                "win_rate": 75.6,
                "total_profit_backtest": 98.47,
                "trades_per_scenario": 56,
                "daily_trades": 1.9,
                "quality_score": 0.98
            },
            "Quality_7": {
                "name": "Quality 7 (Recommended)",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'UNI'],
                "win_rate": 76.3,
                "total_profit_backtest": 109.76,
                "trades_per_scenario": 62,
                "daily_trades": 2.0,
                "quality_score": 0.96
            },
            "Extended_15": {
                "name": "Extended 15 (High Volume)",
                "pairs": ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'UNI', 'ADA', 'DOT', 'MATIC', 
                         'NEAR', 'ATOM', 'FTM', 'SAND', 'CRV'],
                "win_rate": 70.1,
                "total_profit_backtest": 427.11,
                "trades_per_scenario": 135,
                "daily_trades": 4.5,
                "quality_score": 0.92
            }
        }
        
        # User's actual balance
        self.starting_balance = 51.63
        self.backtest_balance = 200.0

    def calculate_trade_projections(self, config_name: str, config: Dict):
        """Calculate detailed trade volume projections"""
        
        # Backtest metrics
        backtest_trades = config['trades_per_scenario']
        backtest_period_days = 30  # Assuming 30-day backtest periods
        daily_trades = config['daily_trades']
        
        # Scale to real balance
        profit_scale = self.starting_balance / self.backtest_balance
        real_profit_per_trade = (config['total_profit_backtest'] / backtest_trades) * profit_scale
        
        # Projection periods
        projections = {}
        
        # Different timeframes
        timeframes = {
            "1_week": 7,
            "1_month": 30,
            "3_months": 90,
            "6_months": 180,
            "1_year": 365
        }
        
        for period_name, days in timeframes.items():
            total_trades = daily_trades * days
            total_profit = total_trades * real_profit_per_trade
            end_balance = self.starting_balance + total_profit
            
            # Calculate returns with compounding effect
            daily_return = real_profit_per_trade * daily_trades / self.starting_balance
            compounded_balance = self.starting_balance * ((1 + daily_return) ** days)
            compounded_profit = compounded_balance - self.starting_balance
            
            projections[period_name] = {
                'days': days,
                'total_trades': total_trades,
                'simple_profit': total_profit,
                'simple_balance': end_balance,
                'compounded_profit': compounded_profit,
                'compounded_balance': compounded_balance,
                'simple_return_pct': (total_profit / self.starting_balance) * 100,
                'compounded_return_pct': (compounded_profit / self.starting_balance) * 100
            }
        
        return {
            'config_name': config_name,
            'config': config,
            'real_profit_per_trade': real_profit_per_trade,
            'projections': projections
        }

    def run_detailed_analysis(self):
        """Run detailed trade volume and profit analysis"""
        
        print(f"\n🔬 RUNNING DETAILED TRADE VOLUME ANALYSIS")
        print(f"💰 Starting Balance: ${self.starting_balance}")
        print("=" * 80)
        
        results = {}
        
        for config_name, config in self.configurations.items():
            print(f"\n📊 Analyzing {config['name']}...")
            result = self.calculate_trade_projections(config_name, config)
            results[config_name] = result
            
            print(f"   🎯 Win Rate: {config['win_rate']:.1f}%")
            print(f"   🎲 Pairs: {len(config['pairs'])}")
            print(f"   📈 Daily Trades: {config['daily_trades']:.1f}")
            print(f"   💰 Profit/Trade: ${result['real_profit_per_trade']:.2f}")
        
        self.display_comprehensive_comparison(results)
        return results

    def display_comprehensive_comparison(self, results: Dict):
        """Display comprehensive comparison"""
        
        print(f"\n" + "=" * 120)
        print("📊 COMPREHENSIVE TRADE VOLUME & PROFIT COMPARISON")
        print("=" * 120)
        
        # Basic metrics comparison
        print(f"\n📈 BASIC METRICS COMPARISON:")
        print(f"{'Configuration':<25} {'Pairs':<6} {'Win Rate':<10} {'Daily Trades':<12} {'Profit/Trade':<12} {'Quality':<8}")
        print("-" * 120)
        
        for config_name, result in results.items():
            config = result['config']
            print(f"{config['name']:<25} {len(config['pairs']):<6} {config['win_rate']:<9.1f}% "
                  f"{config['daily_trades']:<11.1f} ${result['real_profit_per_trade']:<11.2f} {config['quality_score']:<7.0%}")
        
        # Monthly trade volume comparison
        print(f"\n📊 MONTHLY TRADE VOLUME COMPARISON:")
        print(f"{'Configuration':<25} {'Monthly Trades':<15} {'Monthly Profit':<15} {'Monthly Return':<15}")
        print("-" * 120)
        
        for config_name, result in results.items():
            config = result['config']
            monthly = result['projections']['1_month']
            print(f"{config['name']:<25} {monthly['total_trades']:<14.0f} "
                  f"${monthly['simple_profit']:<14.2f} {monthly['simple_return_pct']:<14.1f}%")
        
        # Detailed profit projections
        self.display_profit_projections(results)
        
        # Trade volume analysis
        self.display_trade_volume_analysis(results)

    def display_profit_projections(self, results: Dict):
        """Display detailed profit projections"""
        
        print(f"\n" + "=" * 140)
        print("💰 DETAILED PROFIT PROJECTIONS (Starting: $51.63)")
        print("=" * 140)
        
        timeframes = ["1_week", "1_month", "3_months", "6_months", "1_year"]
        timeframe_labels = ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]
        
        for config_name, result in results.items():
            config = result['config']
            print(f"\n🏆 {config['name']} ({config['win_rate']:.1f}% WR, {len(config['pairs'])} pairs):")
            
            print(f"{'Period':<12} {'Trades':<8} {'Simple Profit':<15} {'Balance':<12} {'Return':<10} {'Compound Profit':<15} {'Compound Balance':<15}")
            print("-" * 140)
            
            for tf, label in zip(timeframes, timeframe_labels):
                proj = result['projections'][tf]
                print(f"{label:<12} {proj['total_trades']:<7.0f} "
                      f"${proj['simple_profit']:<14.2f} ${proj['simple_balance']:<11.2f} "
                      f"{proj['simple_return_pct']:<9.1f}% ${proj['compounded_profit']:<14.2f} "
                      f"${proj['compounded_balance']:<14.2f}")
        
        # Best performer analysis
        print(f"\n🚀 PROFIT PROJECTION WINNER ANALYSIS:")
        
        # Compare 3-month projections (realistic target)
        three_month_comparison = []
        for config_name, result in results.items():
            config = result['config']
            proj = result['projections']['3_months']
            three_month_comparison.append({
                'name': config['name'],
                'win_rate': config['win_rate'],
                'pairs': len(config['pairs']),
                'total_trades': proj['total_trades'],
                'profit': proj['compounded_profit'],
                'return_pct': proj['compounded_return_pct'],
                'balance': proj['compounded_balance']
            })
        
        # Sort by profit
        three_month_comparison.sort(key=lambda x: x['profit'], reverse=True)
        
        print(f"\n📊 3-MONTH PROJECTIONS (MOST REALISTIC):")
        print(f"{'Rank':<4} {'Configuration':<25} {'Trades':<8} {'Profit':<12} {'Return':<10} {'Final Balance':<15}")
        print("-" * 120)
        
        for i, comp in enumerate(three_month_comparison, 1):
            status = "🏆" if i == 1 else "📊"
            print(f"{i:<4} {status} {comp['name']:<22} {comp['total_trades']:<7.0f} "
                  f"${comp['profit']:<11.2f} {comp['return_pct']:<9.1f}% ${comp['balance']:<14.2f}")

    def display_trade_volume_analysis(self, results: Dict):
        """Display trade volume analysis"""
        
        print(f"\n" + "=" * 100)
        print("📈 TRADE VOLUME ANALYSIS")
        print("=" * 100)
        
        print(f"\n🔢 ANNUAL TRADE VOLUME PROJECTIONS:")
        print(f"{'Configuration':<25} {'Daily':<8} {'Weekly':<8} {'Monthly':<10} {'Yearly':<10} {'Win Rate':<10}")
        print("-" * 100)
        
        for config_name, result in results.items():
            config = result['config']
            daily = config['daily_trades']
            weekly = daily * 7
            monthly = daily * 30
            yearly = daily * 365
            
            print(f"{config['name']:<25} {daily:<7.1f} {weekly:<7.0f} {monthly:<9.0f} {yearly:<9.0f} {config['win_rate']:<9.1f}%")
        
        # Volume vs Quality analysis
        print(f"\n📊 VOLUME vs QUALITY TRADE-OFF:")
        
        for config_name, result in results.items():
            config = result['config']
            yearly_trades = config['daily_trades'] * 365
            yearly_profit = result['projections']['1_year']['compounded_profit']
            
            print(f"\n{config['name']}:")
            print(f"   📈 Annual Trades: {yearly_trades:.0f}")
            print(f"   🎯 Win Rate: {config['win_rate']:.1f}%")
            print(f"   💰 Annual Profit: ${yearly_profit:.2f}")
            print(f"   ⚡ Profit per Trade: ${result['real_profit_per_trade']:.2f}")
            print(f"   🎲 Pairs: {len(config['pairs'])}")
            print(f"   💎 Quality Score: {config['quality_score']:.0%}")

def main():
    """Run detailed trade volume analysis"""
    
    analyzer = DetailedTradeVolumeAnalysis()
    results = analyzer.run_detailed_analysis()
    
    return results

if __name__ == "__main__":
    main() 