#!/usr/bin/env python3
"""
Ultimate Win Rate Bot Launcher
Easy startup script for the ultimate trading bot
"""

import os
import sys
from datetime import datetime

def main():
    """Launch the ultimate win rate bot"""
    
    print("🏆" * 80)
    print("🚀 ULTIMATE WIN RATE BOT LAUNCHER")
    print("🎯 Advanced Trading with 70%+ Win Rate Target")
    print("🏆" * 80)
    
    print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Initializing ultimate trading system...")
    
    try:
        # Import and run the ultimate bot
        from ultimate_winrate_bot import UltimateWinRateBot
        
        print("✅ Ultimate Win Rate Bot imported successfully")
        print("🚀 Starting ultimate win rate test...\n")
        
        # Run the ultimate bot
        bot = UltimateWinRateBot()
        results = bot.test_ultimate_winrate()
        
        print("\n🏆 ULTIMATE BOT EXECUTION COMPLETE!")
        print("📊 Results summary:")
        
        # Calculate overall statistics
        total_trades = sum(r['total_trades'] for r in results.values())
        total_wins = sum(r['winning_trades'] for r in results.values())
        overall_winrate = (total_wins / max(total_trades, 1)) * 100
        
        print(f"   • Total Trades: {total_trades}")
        print(f"   • Total Wins: {total_wins:.1f}")
        print(f"   • Overall Win Rate: {overall_winrate:.1f}%")
        
        if overall_winrate >= 70:
            print("🎉 LEGENDARY ACHIEVEMENT: 70%+ win rate!")
        elif overall_winrate >= 60:
            print("🏆 ULTIMATE SUCCESS: 60%+ win rate!")
        elif overall_winrate >= 50:
            print("✅ EXCELLENT: 50%+ win rate!")
        else:
            print("📈 Good progress toward ultimate win rates")
        
        print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except ImportError as e:
        print(f"❌ Error importing ultimate bot: {e}")
        print("💡 Make sure all required files are present:")
        print("   • ultimate_winrate_bot.py")
        print("   • final_optimized_ai_bot.py")
        print("   • indicators.py")
        
    except Exception as e:
        print(f"❌ Error running ultimate bot: {e}")
        print("💡 Check the error details above and try again")
    
    input("\n🔥 Press Enter to exit...")

if __name__ == "__main__":
    main() 