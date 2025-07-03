#!/usr/bin/env python3
"""
Ultimate 75% Bot Launcher
Start live trading or simulation with integrated dashboard
"""

import os
import sys
import subprocess
import time

def print_banner():
    print("=" * 80)
    print("🎯 ULTIMATE 75% WINRATE BOT LAUNCHER")
    print("=" * 80)
    print("🏆 83.6% Win Rate Strategy")
    print("💎 Ultra Micro Profit Targeting (0.07%)")
    print("🛡️ Zero Stop Loss Design")
    print("⏱️ Time-Based Exit Optimization")
    print("📊 Advanced Live Dashboard Integration")
    print("=" * 80)

def start_live_trading():
    """Start live trading bot with dashboard"""
    print("\n🔴 Starting LIVE Trading Ultimate 75% Bot...")
    print("⚠️  WARNING: This will use REAL money!")
    print("📊 Advanced Dashboard will launch automatically!")
    
    confirm = input("\nType 'CONFIRM' to proceed with live trading: ")
    if confirm.upper() != 'CONFIRM':
        print("❌ Live trading cancelled")
        return
    
    try:
        # Check if file exists
        if not os.path.exists('live_ultimate_75_bot.py'):
            print("❌ live_ultimate_75_bot.py not found!")
            print("Please ensure the file is in the current directory.")
            return
        
        # Run the live bot (dashboard launches automatically)
        subprocess.run([sys.executable, 'live_ultimate_75_bot.py'])
        
    except KeyboardInterrupt:
        print("\n⏹️ Live trading stopped by user")
    except Exception as e:
        print(f"❌ Error starting live trading: {e}")

def start_live_simulation():
    """Start live simulation bot with dashboard"""
    print("\n🚀 Starting Live Simulation Ultimate 75% Bot...")
    print("📊 Advanced Dashboard will launch automatically!")
    print("💡 Uses real market data with simulated trading")
    
    try:
        # Check if file exists
        if not os.path.exists('live_simulation_ultimate_75.py'):
            print("❌ live_simulation_ultimate_75.py not found!")
            print("Please ensure the file is in the current directory.")
            return
        
        # Run the simulation (dashboard launches automatically)
        subprocess.run([sys.executable, 'live_simulation_ultimate_75.py'])
        
    except KeyboardInterrupt:
        print("\n⏹️ Live simulation stopped by user")
    except Exception as e:
        print(f"❌ Error starting live simulation: {e}")

def start_dashboard_only():
    """Start dashboard only"""
    print("\n📊 Starting Advanced Dashboard Only...")
    print("🎛️ Full control panel with customizable refresh rates!")
    print("⏱️ Choose from Ultra Fast (0.5s) to Manual (10s)")
    print("📈 Configurable charts, debug info, and compact mode")
    
    try:
        # Check if dashboard launcher exists
        if not os.path.exists('dashboard_launcher.py'):
            print("❌ dashboard_launcher.py not found!")
            return
        
        # Run dashboard
        subprocess.run([sys.executable, 'dashboard_launcher.py'])
        
    except KeyboardInterrupt:
        print("\n⏹️ Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

def view_documentation():
    """Display documentation"""
    print("\n📚 ULTIMATE 75% BOT DOCUMENTATION")
    print("=" * 50)
    
    docs = [
        "📊 Advanced Dashboard Features:",
        "  • Real-time performance metrics",
        "  • Live position monitoring", 
        "  • Performance charts & analytics",
        "  • Market data integration",
        "  • Trade log & exit analysis",
        "  • 🎛️ Customizable refresh rates (0.5s - 10s)",
        "  • 📈 Configurable chart settings",
        "  • 🎨 Compact mode & debug options",
        "",
        "⏱️ Dashboard Controls:",
        "  • Ultra Fast (0.5s) - Maximum responsiveness",
        "  • Fast (1s) - High frequency updates", 
        "  • Normal (2s) - Balanced performance ✅",
        "  • Slow (5s) - Reduced bandwidth usage",
        "  • Manual (10s) - Minimal auto-refresh",
        "  • Chart points: 10-200 history",
        "  • Chart height: 300-600 pixels",
        "  • Grid lines toggle",
        "",
        "🎯 Trading Strategy:",
        "  • 83.6% win rate target",
        "  • Ultra micro profit targets (0.07%)",
        "  • Zero stop loss design",
        "  • Time-based exit optimization",
        "  • 90%+ confidence entry requirements",
        "",
        "🚀 Quick Start:",
        "  1. Choose Live Simulation to test",
        "  2. Dashboard opens automatically", 
        "  3. Configure refresh rate in sidebar",
        "  4. Monitor real-time performance",
        "  5. Switch to Live Trading when ready",
        "",
        "📝 Files:",
        "  • live_simulation_ultimate_75.py - Simulation bot",
        "  • live_ultimate_75_bot.py - Live trading bot",
        "  • dashboard_launcher.py - Advanced dashboard",
        "  • DASHBOARD_CONTROLS_GUIDE.md - Controls documentation",
        "  • LIVE_ULTIMATE_75_README.md - Full documentation"
    ]
    
    for line in docs:
        print(line)

def main():
    """Main launcher menu"""
    print_banner()
    
    while True:
        print("\n🎮 SELECT OPERATION:")
        print("1. 🔴 Live Trading (Real Money)")
        print("2. 📊 Live Simulation (Safe Testing)")
        print("3. 📈 Dashboard Only")
        print("4. 📚 Documentation")
        print("5. ❌ Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            start_live_trading()
        elif choice == "2":
            start_live_simulation()
        elif choice == "3":
            start_dashboard_only()
        elif choice == "4":
            view_documentation()
        elif choice == "5":
            print("\n👋 Thanks for using Ultimate 75% Bot!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main() 