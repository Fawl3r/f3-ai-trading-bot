#!/usr/bin/env python3
"""
Test script to show all 4 risk modes including Insane Mode
"""

from risk_manager import DynamicRiskManager

def test_all_modes():
    print("🧪 TESTING ALL 4 RISK MODES")
    print("=" * 50)
    
    rm = DynamicRiskManager()
    
    print("📋 Available Risk Modes:")
    for i, (key, profile) in enumerate(rm.profiles.items(), 1):
        print(f"{i}. {profile.name}")
        print(f"   Leverage: {profile.leverage}x")
        print(f"   Position Size: {profile.position_size_pct}% of account")
        print(f"   Max Daily Trades: {profile.max_daily_trades}")
        
        if key == "insane":
            print("   🧠 AI-POWERED with DYNAMIC leverage!")
            print("   ⚡ 30x-50x leverage based on AI confidence")
            print("   🎯 90% confidence required")
        print()
    
    print("=" * 50)
    print("🎯 Now test the selection menu:")
    print()
    
    # Show the actual selection menu
    choice = rm.display_risk_options(200.0)
    
    print(f"\nYou can select 1-4, including option 4 for INSANE MODE!")

if __name__ == "__main__":
    test_all_modes() 