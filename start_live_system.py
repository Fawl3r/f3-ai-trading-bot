#!/usr/bin/env python3
"""
Master Live Trading System Launcher
Starts both the Hyperliquid trading bot and real dashboard
"""

import subprocess
import sys
import time
import os
from threading import Thread

def start_dashboard():
    """Start the Hyperliquid dashboard"""
    print("🚀 Starting Hyperliquid Live Dashboard...")
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', 
        'hyperliquid_dashboard.py', 
        '--server.port=8502',
        '--server.headless=false'
    ])

def start_trading_bot():
    """Start the Hyperliquid trading bot"""
    print("🤖 Starting Hyperliquid Trading Bot...")
    subprocess.run([sys.executable, 'start_hyperliquid_trading.py'])

def main():
    print("🏆" * 60)
    print("🚀 HYPERLIQUID LIVE TRADING SYSTEM LAUNCHER")
    print("📊 Real Dashboard + 🤖 AI Trading Bot")
    print("🏆" * 60)
    
    print("\n🎯 This will start:")
    print("   📊 Live Dashboard: http://localhost:8502")
    print("   🤖 Trading Bot: AI Opportunity Hunter")
    print("   💰 Network: Configured in your .env file")
    
    choice = input("\n🚀 Start complete system? (yes/no): ").lower()
    
    if choice != 'yes':
        print("❌ Launch cancelled")
        return
    
    print("\n🎬 STARTING LIVE SYSTEM...")
    print("=" * 50)
    
    try:
        # Start dashboard in background thread
        dashboard_thread = Thread(target=start_dashboard, daemon=True)
        dashboard_thread.start()
        
        print("✅ Dashboard starting... (will open in browser)")
        time.sleep(3)
        
        # Start trading bot in main thread
        start_trading_bot()
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n👋 Live trading system shutdown complete")

if __name__ == "__main__":
    main() 