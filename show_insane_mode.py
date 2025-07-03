#!/usr/bin/env python3
"""
Display All 4 Trading Risk Modes Including Complete Insane Mode
"""

def show_all_modes():
    print("🔥🧠💀 COMPLETE TRADING BOT RISK MODES")
    print("💰 Starting Balance: $200.00")
    print("=" * 80)
    
    print("\n1. 🛡️ SAFE MODE")
    print("   • Conservative trading with capital preservation")
    print("   • $4.00 per trade (2% of account) | 5 trades/day")
    print("   • High confidence required | 1.5% stop loss | 2% take profit")
    print("   • 5x leverage | Max 1% account risk per trade")
    print("   • Best for: New traders, capital preservation")
    
    print("\n2. ⚡ RISK MODE")
    print("   • Balanced trading with moderate risk/reward")
    print("   • $10.00 per trade (5% of account) | 10 trades/day")
    print("   • Moderate confidence | 2% stop loss | 3% take profit")
    print("   • 10x leverage | Max 2% account risk per trade")
    print("   • Best for: Experienced traders, balanced approach")
    
    print("\n3. 🚀💥 SUPER RISKY MODE")
    print("   • AGGRESSIVE trading for MAXIMUM PROFITS")
    print("   • $20.00 per trade (10% of account) | 20 trades/day")
    print("   • Low confidence needed | 3% stop loss | 5% take profit")
    print("   • 20x leverage | Max 5% account risk per trade")
    print("   • WARNING: HIGH RISK - Can lose money fast!")
    print("   • Best for: Expert traders, high risk tolerance")
    
    print("\n4. 🔥🧠💀 INSANE MODE - COMPLETE DETAILS:")
    print("   • AI-POWERED EXTREME LEVERAGE - Only HIGH-PROBABILITY trades!")
    print("   • $30.00 per trade (15% of account) | 8 trades/day")
    print("   • 90% AI confidence required | 2% stop loss | 8% take profit")
    print("   • 30x-50x DYNAMIC leverage | Max 3% account risk per trade")
    print("   • 🧠 AI filters multiple indicators for precision")
    print("   • ⚡ Leverage scales 30x-50x based on AI confidence")
    print("   • 🎯 Quality over quantity: Max 8 precision trades/day")
    print("   • 🤖 AI analyzes RSI, volume, trend, momentum, volatility")
    print("   • 💥 EXTREME OUTCOMES: Can double account OR lose 25% fast")
    print("   • 🚨 EXTREME RISK: Can make OR lose massive amounts!")
    print("   • 💀 WARNING: Most dangerous mode - Expert traders only")
    print("   • ⚠️ ONLY trades with 90%+ win probability!")
    print("   • 🔥 Requires 3 confirmations to activate")
    print("   • Best for: AI-assisted high-stakes trading")
    
    print("\n" + "=" * 80)
    print("🎯 INSANE MODE AI FEATURES:")
    print("• RSI Extreme Analysis (Buy <20, Sell >80)")
    print("• Volume Surge Confirmation") 
    print("• Multi-timeframe Trend Alignment")
    print("• Momentum Divergence Detection")
    print("• Volatility Squeeze Patterns")
    print("• Support/Resistance Level Analysis")
    print("• Dynamic Leverage: 90% confidence = 30x, 98% confidence = 50x")
    print("• AI Learning: Tracks accuracy and improves over time")
    print("=" * 80)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "4":
        print("\n🔥🧠💀 INSANE MODE SELECTED!")
        print("This mode uses AI to analyze market conditions and only")
        print("executes trades with 90%+ confidence using 30x-50x leverage.")
        print("It's designed for maximum profit potential with AI protection.")
    else:
        print(f"\nYou selected mode {choice}")
    
    return choice

if __name__ == "__main__":
    show_all_modes() 