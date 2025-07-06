#!/usr/bin/env python3
"""
AI-Enhanced 70%+ Win Rate Backtest
Comprehensive Testing & Analysis
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ai_enhanced_70_wr_bot import AIEnhanced70WRBot
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_comprehensive_ai_backtest():
    """Run comprehensive AI backtest with multiple configurations"""
    print("🤖 AI-ENHANCED 70%+ WIN RATE COMPREHENSIVE BACKTEST")
    print("=" * 70)
    
    # Test configurations
    test_configs = [
        # Conservative Tests
        {"name": "Conservative 7-Day", "days": 7, "balance": 50},
        {"name": "Conservative 14-Day", "days": 14, "balance": 50},
        {"name": "Conservative 30-Day", "days": 30, "balance": 50},
        
        # Larger Balance Tests
        {"name": "Medium 14-Day", "days": 14, "balance": 100},
        {"name": "Large 30-Day", "days": 30, "balance": 200},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n🧠 Running {config['name']} Test...")
        print("-" * 50)
        
        try:
            # Create bot with specified balance
            bot = AIEnhanced70WRBot(config['balance'])
            
            # Run backtest
            results = bot.run_backtest('ETH', config['days'])
            
            # Display results
            print(f"💰 RESULTS FOR {config['name']}:")
            print(f"   Initial Balance: ${results['initial_balance']:.2f}")
            print(f"   Final Balance: ${results['final_balance']:.2f}")
            print(f"   Total Return: {results['total_return_pct']:.2f}%")
            print(f"   Total Trades: {results['total_trades']}")
            print(f"   🎯 WIN RATE: {results['win_rate']:.1%}")
            print(f"   Max Drawdown: {results['max_drawdown_pct']:.2f}%")
            print(f"   AI Avg Confidence: {results['ai_avg_confidence']:.2f}")
            
            # Win rate status
            if results['win_rate'] >= 0.70:
                status = "🎯 TARGET ACHIEVED!"
                print(f"   Status: {status}")
            elif results['win_rate'] >= 0.60:
                status = "🔄 Close to Target"
                print(f"   Status: {status}")
            else:
                status = "❌ Below Target"
                print(f"   Status: {status}")
            
            # AI Performance Breakdown
            if results['profitable_by_confidence']:
                print(f"\n🤖 AI CONFIDENCE ANALYSIS:")
                for conf_range, stats in results['profitable_by_confidence'].items():
                    print(f"   {conf_range}: {stats['trades']} trades, {stats['win_rate']:.1%} WR")
            
            # Market Regime Performance
            if results['regime_performance']:
                print(f"\n📊 MARKET REGIME PERFORMANCE:")
                for regime, stats in results['regime_performance'].items():
                    print(f"   {regime}: {stats['trades']} trades, {stats['win_rate']:.1%} WR")
            
            # Add to results
            all_results.append({
                'config': config,
                'results': results,
                'status': status
            })
            
        except Exception as e:
            logger.error(f"Error in {config['name']}: {str(e)}")
            continue
    
    # Overall Analysis
    print(f"\n🎯 COMPREHENSIVE AI ANALYSIS")
    print("=" * 70)
    
    if all_results:
        # Find best performing configuration
        best_wr = max(all_results, key=lambda x: x['results']['win_rate'])
        best_return = max(all_results, key=lambda x: x['results']['total_return_pct'])
        
        print(f"🏆 BEST WIN RATE: {best_wr['config']['name']}")
        print(f"   Win Rate: {best_wr['results']['win_rate']:.1%}")
        print(f"   Return: {best_wr['results']['total_return_pct']:.2f}%")
        print(f"   AI Confidence: {best_wr['results']['ai_avg_confidence']:.2f}")
        
        print(f"\n💰 BEST RETURN: {best_return['config']['name']}")
        print(f"   Return: {best_return['results']['total_return_pct']:.2f}%")
        print(f"   Win Rate: {best_return['results']['win_rate']:.1%}")
        
        # Calculate overall statistics
        total_trades = sum(r['results']['total_trades'] for r in all_results)
        total_wins = sum(r['results']['winning_trades'] for r in all_results)
        overall_wr = total_wins / total_trades if total_trades > 0 else 0
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Trades Across All Tests: {total_trades}")
        print(f"   Overall Win Rate: {overall_wr:.1%}")
        print(f"   Tests Achieving 70%+ WR: {len([r for r in all_results if r['results']['win_rate'] >= 0.70])}/{len(all_results)}")
        
        # AI Performance Summary
        all_confidences = []
        for r in all_results:
            if r['results']['ai_avg_confidence'] > 0:
                all_confidences.append(r['results']['ai_avg_confidence'])
        
        if all_confidences:
            avg_ai_confidence = np.mean(all_confidences)
            print(f"   Average AI Confidence: {avg_ai_confidence:.2f}")
        
        # Save results
        results_data = {
            'test_date': datetime.now().isoformat(),
            'target_win_rate': 0.70,
            'overall_win_rate': overall_wr,
            'total_trades': total_trades,
            'tests_achieving_target': len([r for r in all_results if r['results']['win_rate'] >= 0.70]),
            'total_tests': len(all_results),
            'detailed_results': all_results
        }
        
        with open('ai_70wr_comprehensive_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n✅ Comprehensive results saved to 'ai_70wr_comprehensive_results.json'")
        
        # Recommendations
        print(f"\n💡 AI ENHANCEMENT RECOMMENDATIONS:")
        
        if overall_wr >= 0.70:
            print("   🎯 AI system successfully achieving 70%+ win rate target!")
            print("   ✅ Ready for live paper trading validation")
            print("   ✅ Consider gradual position size scaling")
        elif overall_wr >= 0.60:
            print("   🔄 AI system shows strong potential (60%+ WR)")
            print("   📈 Recommend increasing AI confidence threshold")
            print("   📊 Focus on highest-confidence setups only")
        else:
            print("   ⚠️  AI system needs further optimization")
            print("   🔧 Consider additional feature engineering")
            print("   📚 Expand training data with more market regimes")
        
        # Feature importance insights
        print(f"\n🧠 AI MODEL INSIGHTS:")
        print("   • Ensemble approach with Random Forest, Gradient Boosting, Neural Network")
        print("   • 80% model consensus required for trade execution")
        print("   • Conservative 1% stop loss, 1.5% take profit")
        print("   • Maximum 2-hour holding time for high frequency")
        print("   • Risk-adjusted position sizing based on AI confidence")
        
    else:
        print("❌ No successful backtest results to analyze")
    
    return all_results

def analyze_ai_features():
    """Analyze AI feature importance and model performance"""
    print(f"\n🔬 AI FEATURE ANALYSIS")
    print("-" * 50)
    
    # Create a sample bot to analyze features
    bot = AIEnhanced70WRBot(50.0)
    
    # Train the AI system
    print("Training AI models for feature analysis...")
    training_results = bot.train_ai_system('ETH')
    
    print(f"AI Model Training Results:")
    for model_name, accuracy in training_results.items():
        print(f"   {model_name.upper()}: {accuracy:.1%} accuracy")
    
    # Feature importance analysis
    if hasattr(bot.ai_recognizer, 'feature_importance'):
        print(f"\n📊 FEATURE IMPORTANCE ANALYSIS:")
        
        feature_names = [
            'Price vs SMA5', 'Price vs SMA10', 'Price vs SMA20',
            'SMA5 vs SMA10', 'SMA10 vs SMA20', 'MACD-like',
            'RSI', 'ROC 5-period', 'ROC 10-period',
            'Volatility 5-period', 'Volatility 20-period', 'ATR',
            'Volume Ratio', 'Volume Trend', 'High-Low Ratio',
            'Upper Shadow', 'Lower Shadow', 'Distance from High',
            'Distance from Low', 'Price Position', 'Trend Strength',
            'Trend Consistency'
        ]
        
        for model_name, importance in bot.ai_recognizer.feature_importance.items():
            print(f"\n   {model_name.upper()} Top Features:")
            
            # Sort features by importance
            feature_importance = list(zip(feature_names[:len(importance)], importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, imp) in enumerate(feature_importance[:5]):
                print(f"      {i+1}. {feature}: {imp:.3f}")

if __name__ == "__main__":
    # Run comprehensive backtest
    results = run_comprehensive_ai_backtest()
    
    # Analyze AI features
    analyze_ai_features()
    
    print(f"\n🎯 AI-ENHANCED 70%+ WIN RATE ANALYSIS COMPLETE!")
    print("=" * 70) 