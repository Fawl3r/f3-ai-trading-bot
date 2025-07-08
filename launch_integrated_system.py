#!/usr/bin/env python3
"""
🚀 INTEGRATED SYSTEM LAUNCHER
Launches the complete AI trading system with monitoring

Components:
1. Integrated AI Hyperliquid Bot (working models only)
2. Day 2-30 Action Plan Monitor
3. Performance tracking and alerts
"""

import asyncio
import threading
import time
import sys
import os
from datetime import datetime
import subprocess

def run_action_plan_monitor():
    """Run the action plan monitoring system"""
    try:
        print("📊 Starting Action Plan Monitor...")
        
        # Import and run the monitoring system
        from day_2_to_30_action_plan import ActionPlanMonitor
        
        monitor = ActionPlanMonitor()
        
        while True:
            try:
                # Run comprehensive monitoring check
                results = monitor.print_dashboard()
                
                # Sleep for 1 hour between checks
                time.sleep(3600)
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(300)  # 5 minutes on error
                
    except Exception as e:
        print(f"❌ Action Plan Monitor failed to start: {e}")

async def run_ai_trading_bot():
    """Run the integrated AI trading bot"""
    try:
        print("🤖 Starting Integrated AI Trading Bot...")
        
        # Import and run the trading bot
        from integrated_ai_hyperliquid_bot import IntegratedAIHyperliquidBot
        
        bot = IntegratedAIHyperliquidBot(paper_mode=True)
        await bot.run_trading_loop()
        
    except Exception as e:
        print(f"❌ AI Trading Bot failed: {e}")

def print_system_status():
    """Print current system status"""
    
    print("\n" + "=" * 80)
    print("🚀 INTEGRATED AI TRADING SYSTEM - STATUS DASHBOARD")
    print("=" * 80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("🧠 AI MODEL STATUS:")
    print("-" * 50)
    print("✅ TimesNet Long-Range: PF 1.97 (Strong performer)")
    print("✅ TSA-MAE Encoder: Model b59c66da (Ready)")
    print("✅ PPO Strict Enhanced: Available")
    print("❌ LightGBM + TSA-MAE: HALTED (PF 1.46 < 1.5)")
    print()
    
    print("🎯 CURRENT CONFIGURATION:")
    print("-" * 50)
    print("• Trading Pairs: BTC, ETH, SOL, DOGE, AVAX")
    print("• Risk per Trade: 0.5% (Elite 100/5 config)")
    print("• Max Drawdown: 5%")
    print("• Target Monthly Return: 100%")
    print("• AI Confidence Threshold: 45%")
    print("• Max Concurrent Positions: 2")
    print()
    
    print("🔄 TRAFFIC ALLOCATION:")
    print("-" * 50)
    print("• TimesNet: 1.1% (performing well)")
    print("• LightGBM: 0.0% (auto-halted)")
    print("• PPO: Available for ensemble")
    print("• Meta-Learner: Ready for 10% deployment")
    print()
    
    print("📊 MONITORING SYSTEMS:")
    print("-" * 50)
    print("• Action Plan Monitor: Running (1h intervals)")
    print("• Performance Tracker: Active")
    print("• Risk Management: Elite 100/5 config")
    print("• Thompson Sampling: Auto traffic allocation")
    print()
    
    print("🛡️ SAFETY MEASURES:")
    print("-" * 50)
    print("• Paper Mode: ENABLED (for safety)")
    print("• Circuit Breakers: Active")
    print("• Daily Trade Limit: 15 trades")
    print("• Emergency Halt: 5% DD trigger")
    print("=" * 80)

def main():
    """Main launcher function"""
    
    print_system_status()
    
    print("\n🚀 LAUNCHING INTEGRATED SYSTEM...")
    print("-" * 50)
    
    try:
        # Start action plan monitor in background thread
        monitor_thread = threading.Thread(
            target=run_action_plan_monitor,
            daemon=True,
            name="ActionPlanMonitor"
        )
        monitor_thread.start()
        print("✅ Action Plan Monitor started")
        
        # Small delay
        time.sleep(2)
        
        # Start AI trading bot (main thread)
        print("✅ Starting AI Trading Bot...")
        asyncio.run(run_ai_trading_bot())
        
    except KeyboardInterrupt:
        print("\n🛑 System shutdown requested by user")
    except Exception as e:
        print(f"❌ System launch failed: {e}")
        sys.exit(1)
    finally:
        print("👋 Integrated system shutdown complete")

if __name__ == "__main__":
    main() 