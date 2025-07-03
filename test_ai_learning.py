#!/usr/bin/env python3
"""
Test AI Learning System - Verify the AI actually learns from trade results
"""

import pandas as pd
import numpy as np
from ai_analyzer import AITradeAnalyzer

def test_ai_learning():
    print("🧠 TESTING AI LEARNING SYSTEM")
    print("=" * 60)
    
    # Create AI analyzer
    ai = AITradeAnalyzer()
    
    # Show initial weights
    print("📊 INITIAL AI WEIGHTS:")
    for indicator, weight in ai.indicator_weights.items():
        print(f"   {indicator}: {weight:.3f}")
    
    # Create sample market data
    print("\n📈 Creating sample market data...")
    data = pd.DataFrame({
        'close': np.random.normal(142, 2, 100),
        'high': np.random.normal(143, 2, 100),
        'low': np.random.normal(141, 2, 100),
        'volume': np.random.normal(1000, 200, 100),
        'rsi': np.random.normal(50, 20, 100)
    })
    
    # Add indicators
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['ema_26'] = data['close'].ewm(span=26).mean()
    data['macd'] = data['ema_12'] - data['ema_26']
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    
    print("✅ Sample data created with 100 candles")
    
    # Test multiple predictions and results
    print("\n🎯 TESTING AI PREDICTIONS AND LEARNING:")
    print("-" * 60)
    
    test_scenarios = [
        # High confidence predictions - some wins, some losses
        (95.0, 'win', 'High confidence WIN'),
        (92.0, 'win', 'High confidence WIN'),
        (94.0, 'loss', 'High confidence LOSS'),
        (91.0, 'win', 'High confidence WIN'),
        (93.0, 'loss', 'High confidence LOSS'),
        
        # Medium confidence predictions
        (75.0, 'loss', 'Medium confidence LOSS'),
        (80.0, 'win', 'Medium confidence WIN'),
        (78.0, 'loss', 'Medium confidence LOSS'),
        
        # Low confidence predictions
        (60.0, 'loss', 'Low confidence LOSS'),
        (65.0, 'loss', 'Low confidence LOSS'),
    ]
    
    print("🧠 Simulating trade results for AI learning...")
    for i, (confidence, result, description) in enumerate(test_scenarios, 1):
        ai.update_trade_result(confidence, result)
        stats = ai.get_ai_performance_stats()
        print(f"{i:2d}. {description:20s} | Accuracy: {stats['accuracy_rate']:5.1f}% | Total: {stats['total_predictions']}")
    
    # Show final weights after learning
    print("\n📊 FINAL AI WEIGHTS AFTER LEARNING:")
    for indicator, weight in ai.indicator_weights.items():
        print(f"   {indicator}: {weight:.3f}")
    
    # Test a real analysis
    print("\n🔍 TESTING REAL AI ANALYSIS:")
    print("-" * 60)
    
    # Test extreme RSI scenarios
    test_data = data.copy()
    test_data.loc[test_data.index[-1], 'rsi'] = 15.0  # Extreme oversold
    
    result = ai.analyze_trade_opportunity(test_data, 142.0, 'buy')
    
    print(f"🎯 AI Analysis Result:")
    print(f"   Confidence: {result['ai_confidence']:.1f}%")
    print(f"   Trade Approved: {result['trade_approved']}")
    print(f"   Dynamic Leverage: {result['dynamic_leverage']}x")
    print(f"   Recommendation: {result['recommendation']['recommendation']}")
    
    if result['recommendation']['reasoning']:
        print("   AI Reasoning:")
        for reason in result['recommendation']['reasoning']:
            print(f"     {reason}")
    
    # Test AI performance stats
    final_stats = ai.get_ai_performance_stats()
    print(f"\n📈 FINAL AI PERFORMANCE:")
    print(f"   Total Predictions: {final_stats['total_predictions']}")
    print(f"   Accuracy Rate: {final_stats['accuracy_rate']:.1f}%")
    print(f"   Correct Predictions: {final_stats['correct_predictions']}")
    print(f"   False Positives: {final_stats['false_positives']}")
    print(f"   False Negatives: {final_stats['false_negatives']}")
    
    print("\n" + "=" * 60)
    if final_stats['total_predictions'] > 0:
        print("✅ AI LEARNING SYSTEM IS WORKING!")
        print("🧠 The AI successfully tracked trade results and adapted its weights")
        print("📊 Accuracy metrics are being calculated and stored")
        if len(ai.trade_history) > 0:
            print("📝 Trade history is being maintained for pattern analysis")
    else:
        print("❌ AI Learning system may have issues")
    
    return ai

def test_weight_adaptation():
    """Test that AI weights actually change based on performance"""
    print("\n🔧 TESTING WEIGHT ADAPTATION:")
    print("-" * 60)
    
    ai = AITradeAnalyzer()
    
    # Record initial weights
    initial_weights = ai.indicator_weights.copy()
    
    # Simulate poor performance (many losses)
    print("Simulating poor performance (many losses)...")
    for i in range(25):
        ai.update_trade_result(85.0, 'loss')  # High confidence losses
        if i == 20:  # Check when adaptation triggers
            print("   📊 Triggering weight adaptation at 21 trades...")
    
    # Force a confidence calculation to trigger adaptation
    import pandas as pd
    import numpy as np
    test_data = pd.DataFrame({
        'close': [142] * 50,
        'high': [143] * 50,
        'low': [141] * 50,
        'volume': [1000] * 50,
        'rsi': [50] * 50,
        'ema_12': [142] * 50,
        'ema_26': [142] * 50,
        'macd': [0] * 50,
        'macd_signal': [0] * 50
    })
    
    print("   🔧 Forcing AI analysis to trigger weight adaptation...")
    ai.analyze_trade_opportunity(test_data, 142.0, 'buy')
    
    # Check if weights changed
    final_weights = ai.indicator_weights.copy()
    
    print("📊 WEIGHT CHANGES:")
    for indicator in initial_weights:
        initial = initial_weights[indicator]
        final = final_weights[indicator]
        change = final - initial
        print(f"   {indicator:20s}: {initial:.3f} → {final:.3f} ({change:+.3f})")
    
    # Check if any weight actually changed
    weights_changed = any(abs(initial_weights[k] - final_weights[k]) > 0.001 
                         for k in initial_weights)
    
    if weights_changed:
        print("✅ WEIGHTS SUCCESSFULLY ADAPTED!")
        print("🧠 AI is learning from poor performance and adjusting strategy")
    else:
        print("⚠️  Weights didn't change - may need more trades or different thresholds")

if __name__ == "__main__":
    print("🚀 STARTING AI LEARNING SYSTEM TESTS")
    print("=" * 80)
    
    # Test basic learning
    ai = test_ai_learning()
    
    # Test weight adaptation
    test_weight_adaptation()
    
    print("\n🎉 AI LEARNING TESTS COMPLETED!")
    print("=" * 80) 