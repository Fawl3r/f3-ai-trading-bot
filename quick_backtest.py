#!/usr/bin/env python3
"""
Quick Backtest - Test one mode to verify trading logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest_risk_manager import BacktestRiskManager
from ai_analyzer import AITradeAnalyzer
from indicators import TechnicalIndicators

def quick_test():
    print("🧪 QUICK BACKTEST TEST")
    print("=" * 40)
    
    # Initialize
    risk_manager = BacktestRiskManager()
    ai_analyzer = AITradeAnalyzer()
    indicators = TechnicalIndicators()
    
    # Select RISK MODE (75% threshold)
    risk_manager.select_risk_mode("2", 200.0)
    
    # Generate simple test data (volatile market)
    print("\n📊 Generating test data...")
    data = []
    price = 140.0
    
    # Create 1000 candles with known patterns
    for i in range(1000):
        # Create RSI oscillation pattern
        if i % 100 < 30:  # Oversold period
            price_change = np.random.uniform(-1, 0.5)  # More likely to go down
        elif i % 100 > 70:  # Overbought period  
            price_change = np.random.uniform(-0.5, 1)  # More likely to go up
        else:
            price_change = np.random.uniform(-0.5, 0.5)
        
        price += price_change
        price = max(120, min(160, price))  # Keep in range
        
        data.append({
            'timestamp': datetime.now() - timedelta(minutes=1000-i),
            'open': price + np.random.uniform(-0.2, 0.2),
            'high': price + np.random.uniform(0, 0.5),
            'low': price - np.random.uniform(0, 0.5),
            'close': price,
            'volume': np.random.uniform(500, 1500)
        })
    
    df = pd.DataFrame(data)
    df = indicators.calculate_all_indicators(df)
    
    print(f"✅ Generated {len(df)} candles")
    print(f"📊 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"📈 RSI range: {df['rsi'].min():.1f} - {df['rsi'].max():.1f}")
    
    # Count potential signals
    params = risk_manager.get_trading_params(200.0)
    oversold_count = len(df[df['rsi'] < params['rsi_oversold']])
    overbought_count = len(df[df['rsi'] > params['rsi_overbought']])
    
    print(f"🔍 Potential signals:")
    print(f"   📉 Oversold (RSI < {params['rsi_oversold']}): {oversold_count} candles")
    print(f"   📈 Overbought (RSI > {params['rsi_overbought']}): {overbought_count} candles")
    
    # Test AI analysis on some signals
    print(f"\n🧠 Testing AI analysis...")
    ai_approvals = 0
    
    for i in range(50, min(100, len(df))):
        current = df.iloc[i]
        if current['rsi'] < params['rsi_oversold']:
            recent_data = df.iloc[max(0, i-50):i+1]
            ai_result = ai_analyzer.analyze_trade_opportunity(recent_data, current['close'], 'buy')
            
            print(f"   RSI {current['rsi']:.1f} | AI: {ai_result['ai_confidence']:.1f}% | Threshold: 75%")
            
            if ai_result['ai_confidence'] >= 75.0:
                ai_approvals += 1
                print(f"   ✅ APPROVED! {ai_approvals} total approvals")
                
            if ai_approvals >= 3:  # Stop after finding a few
                break
    
    print(f"\n📊 Test Results:")
    print(f"   🎯 AI Approvals Found: {ai_approvals}")
    
    if ai_approvals > 0:
        print("   ✅ Trading logic is working!")
        print("   🚀 Ready for full backtest")
    else:
        print("   ⚠️  No AI approvals - may need to adjust thresholds")
    
    print("=" * 40)

if __name__ == "__main__":
    quick_test() 