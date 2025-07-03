#!/usr/bin/env python3
"""
📊 EXTENDED 15 BOT STATUS CHECKER
Quick status check for the Extended 15 Production Bot
"""

import json
import os
from datetime import datetime
from hyperliquid.info import Info
from hyperliquid.utils import constants

def print_status():
    """Print current bot status"""
    
    print("📊 EXTENDED 15 BOT STATUS CHECK")
    print("=" * 60)
    print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return
    
    # Check connection
    try:
        info = Info(constants.MAINNET_API_URL if config['is_mainnet'] else constants.TESTNET_API_URL)
        user_state = info.user_state(config['wallet_address'])
        
        if user_state and 'marginSummary' in user_state:
            balance = float(user_state['marginSummary'].get('accountValue', 0))
            print(f"✅ Connected to Hyperliquid")
            print(f"💰 Account Balance: ${balance:.2f}")
            
            # Check positions
            positions = []
            if 'assetPositions' in user_state:
                for pos in user_state['assetPositions']:
                    if float(pos['position']['szi']) != 0:
                        positions.append({
                            'symbol': pos['position']['coin'],
                            'size': float(pos['position']['szi']),
                            'pnl': float(pos['position']['unrealizedPnl'])
                        })
            
            print(f"📊 Active Positions: {len(positions)}")
            if positions:
                total_pnl = sum(pos['pnl'] for pos in positions)
                print(f"💰 Total Unrealized PnL: ${total_pnl:.2f}")
                print("🎲 Position Details:")
                for pos in positions:
                    pnl_emoji = "🟢" if pos['pnl'] > 0 else "🔴"
                    print(f"   {pnl_emoji} {pos['symbol']}: ${pos['pnl']:.2f}")
            
        else:
            print("❌ Failed to get account data")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    # Check log file
    print("\n📋 LOG FILE STATUS:")
    try:
        if os.path.exists('extended_15_bot.log'):
            with open('extended_15_bot.log', 'r') as f:
                lines = f.readlines()
            
            if lines:
                print(f"✅ Log file exists ({len(lines)} lines)")
                print("📝 Last 5 entries:")
                for line in lines[-5:]:
                    print(f"   {line.strip()}")
            else:
                print("❌ Log file is empty")
        else:
            print("❌ Log file not found")
    except Exception as e:
        print(f"❌ Log error: {e}")
    
    print("\n" + "=" * 60)
    print("🚀 EXTENDED 15 BOT STATUS COMPLETE")

if __name__ == "__main__":
    print_status() 