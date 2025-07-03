#!/usr/bin/env python3
"""
Lucrative $40 Profit Bot Launcher
Simple launcher with dependency checking
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['pandas', 'numpy', 'websockets', 'plyer']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        
        install = input("\n🔧 Install missing packages? (y/n): ").lower().strip()
        if install == 'y':
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages, check=True)
                print("✅ Dependencies installed successfully!")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install dependencies. Please install manually.")
                return False
        else:
            print("❌ Cannot run bot without required dependencies.")
            return False
    
    return True

def main():
    """Main launcher function"""
    print("💰 LUCRATIVE $40 PROFIT BOT LAUNCHER")
    print("=" * 50)
    print("🎯 HIGH-PERFORMANCE BOT FOR CONSISTENT $40 PROFITS")
    print("🏆 TARGET: 70%+ WIN RATE")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if bot file exists
    if not os.path.exists('lucrative_40_profit_bot.py'):
        print("❌ Bot file 'lucrative_40_profit_bot.py' not found!")
        print("   Make sure the bot file is in the current directory.")
        return
    
    print("\n🚀 Starting Lucrative $40 Profit Bot...")
    print("💡 This bot targets consistent $40 profits per trade")
    print("📊 Uses advanced AI analysis with real OKX market data")
    print("🔔 Includes sound and desktop notifications")
    print("\n⚠️ IMPORTANT:")
    print("   • This is for educational/simulation purposes")
    print("   • Always test thoroughly before live trading")
    print("   • Only trade what you can afford to lose")
    print("\n🎯 Press Ctrl+C to stop the bot anytime")
    print("=" * 50)
    
    try:
        # Import and run the bot
        from lucrative_40_profit_bot import main as bot_main
        bot_main()
    except KeyboardInterrupt:
        print("\n👋 Bot launcher stopped.")
    except ImportError as e:
        print(f"❌ Error importing bot: {e}")
        print("   Make sure 'lucrative_40_profit_bot.py' is valid.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 