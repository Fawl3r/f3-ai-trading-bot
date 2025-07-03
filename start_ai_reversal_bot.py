#!/usr/bin/env python3
"""
AI Reversal Trading Bot Launcher
"""

import os
import sys
import subprocess

def main():
    print("🤖 AI REVERSAL TRADING BOT LAUNCHER")
    print("📈 HIGH-PROFIT PERPETUAL FUTURES TRADING")
    print("💰 MIMICS YOUR MANUAL TRADING WITH AI PRECISION")
    print("="*70)
    
    print("\n🔍 Checking dependencies...")
    
    required = ['websocket-client', 'pandas', 'numpy', 'scikit-learn', 'plyer']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("⚠️ Missing:", ", ".join(missing))
        install = input("\n🔧 Install now? (y/n): ").lower()
        if install == 'y':
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        else:
            return
    
    print("✅ Dependencies ready!")
    
    print("\n📋 AI BOT FEATURES:")
    print("   🎯 Range high/low detection like your manual strategy")
    print("   💰 $150-$500 positions with 10x leverage")
    print("   🎯 $200-$500 profit targets (100-333% returns)")
    print("   🤖 70%+ AI confidence required")
    print("   🔔 Instant trade notifications")
    print("   🛡️ Smart reversal detection")
    
    start = input("\n🚀 Start AI Reversal Bot? (y/n): ").lower()
    if start == 'y':
        print("\n🚀 Starting AI Reversal Bot...")
        subprocess.run([sys.executable, 'ai_reversal_trading_bot.py'])
    else:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main() 