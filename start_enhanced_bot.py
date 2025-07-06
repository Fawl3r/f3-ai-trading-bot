#!/usr/bin/env python3
"""
Enhanced Bot Launcher
Start the trading bot with detailed live updates
"""

import subprocess
import sys

def main():
    print("🚀" * 60)
    print("🤖 ENHANCED HYPERLIQUID TRADING BOT")
    print("📊 Real-time Progress Updates & Detailed Scanning")
    print("🚀" * 60)
    
    print("\n🎯 This enhanced version shows:")
    print("   📊 Which crypto it's currently scanning")
    print("   ⏰ Countdown timer to next scan")
    print("   💰 Live prices for each crypto")
    print("   🔍 Confidence levels for each scan")
    print("   📈 Detailed trade analysis")
    
    input("\n🚀 Press Enter to start enhanced bot...")
    
    try:
        subprocess.run([sys.executable, 'start_hyperliquid_trading.py'])
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped safely")

if __name__ == "__main__":
    main() 