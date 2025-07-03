#!/usr/bin/env python3
"""
Simple Startup Script for Insane Mode Bot
Ensures all 4 risk modes are available including Insane Mode with dynamic leverage
"""

print("🔥🧠💀 STARTING INSANE MODE BOT")
print("🚀 All 4 Risk Modes Available:")
print("   1. 🛡️  SAFE MODE - 5x leverage")
print("   2. ⚡ RISK MODE - 10x leverage") 
print("   3. 🚀💥 SUPER RISKY MODE - 20x leverage")
print("   4. 🔥🧠💀 INSANE MODE - 30x-50x DYNAMIC leverage with AI")
print()

try:
    # Import and run the insane mode bot
    from insane_mode_bot import InsaneModeBot
    
    print("✅ Starting Insane Mode Bot with $200...")
    bot = InsaneModeBot(initial_balance=200.0)
    bot.start()
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Creating simplified version...")
    
    # Fallback - create a simple demonstration
    from risk_manager import DynamicRiskManager
    
    print("🧪 TESTING RISK MANAGER - ALL 4 MODES")
    rm = DynamicRiskManager()
    
    while True:
        choice = rm.display_risk_options(200.0)
        if rm.select_risk_mode(choice, 200.0):
            params = rm.get_trading_params(200.0)
            print(f"\n✅ SELECTED: {rm.current_profile.name}")
            print(f"💰 Position Size: ${params['position_size_usd']:.2f}")
            print(f"⚖️  Leverage: {params['leverage']}x")
            if rm.current_profile.name == "INSANE MODE 🔥🧠💀":
                print("🧠 AI Analysis: ENABLED")
                print("⚡ Dynamic Leverage: 30x-50x based on AI confidence")
                print("🎯 Only 90%+ confidence trades will execute")
            break
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("🔧 Please ensure all files are present")

if __name__ == "__main__":
    pass 