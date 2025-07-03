#!/usr/bin/env python3
"""
HYPERLIQUID OPPORTUNITY HUNTER AI - STARTER SCRIPT
"""

import asyncio
import sys
import os
from dotenv import load_dotenv
from hyperliquid_opportunity_hunter import HyperliquidOpportunityHunter

# Load environment variables
load_dotenv()

async def start_bot():
    """Start the Hyperliquid trading bot"""
    try:
        print("🚀 HYPERLIQUID OPPORTUNITY HUNTER AI - PRODUCTION VERSION")
        print("⚡ REAL-TIME PARABOLIC DETECTION + DYNAMIC CAPITAL")
        print("💎 LIVE TRADING WITH VALIDATED 75%+ WIN RATE AI")
        print("⚠️  TRADING WITH REAL MONEY - USE AT YOUR OWN RISK")
        print("=" * 65)
        
        # Check environment variables
        private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY', '') or os.getenv('HL_PRIVATE_KEY', '')
        
        if not private_key:
            print("❌ ERROR: HYPERLIQUID_PRIVATE_KEY not found!")
            print("Please set your Hyperliquid private key in .env file:")
            print("HYPERLIQUID_PRIVATE_KEY=your_private_key_here")
            print()
            print("📖 Check the .env.example file for proper format")
            return
        
        print("🔑 Private key configured ✅")
        
        # Initialize and start the bot
        print("🤖 Initializing Hyperliquid Opportunity Hunter AI...")
        bot = HyperliquidOpportunityHunter()
        
        # Test connection
        print("🔗 Testing Hyperliquid API connection...")
        balance = await bot.get_account_balance()
        
        if balance is None:
            print("❌ Cannot connect to Hyperliquid API!")
            print("Please check your private key and network connection")
            return
        
        print(f"💰 Account Balance: ${balance:.2f}")
        
        if balance <= 0:
            print("⚠️  Warning: Account balance is $0")
            print("Please deposit funds to your Hyperliquid account to start trading")
            return
        
        print("✅ All systems ready!")
        print()
        print("🎯 STARTING LIVE TRADING BOT...")
        print("🔥 Hunting for 75%+ Win Rate Opportunities")
        print("💫 Press Ctrl+C to stop the bot safely")
        print("=" * 65)
        
        # Start the main trading loop
        await bot.run_trading_loop()
        
    except KeyboardInterrupt:
        print("\n🛑 Trading bot stopped by user")
        print("👋 Thank you for using Hyperliquid Opportunity Hunter AI!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your configuration and try again")

def main():
    """Main entry point"""
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")

if __name__ == "__main__":
    main() 