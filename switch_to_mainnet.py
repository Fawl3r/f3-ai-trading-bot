#!/usr/bin/env python3
"""
Switch to Mainnet Mode
Updates configuration and restarts live trading system
"""

import os
import sys
import subprocess
from pathlib import Path

def update_env_file():
    """Update .env file for mainnet mode"""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("❌ .env file not found!")
        return False
    
    # Read current .env
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update testnet setting
    updated_lines = []
    testnet_found = False
    
    for line in lines:
        if line.startswith('HYPERLIQUID_TESTNET'):
            updated_lines.append('HYPERLIQUID_TESTNET=false\n')
            testnet_found = True
            print("✅ Updated HYPERLIQUID_TESTNET=false")
        else:
            updated_lines.append(line)
    
    # Add if not found
    if not testnet_found:
        updated_lines.append('HYPERLIQUID_TESTNET=false\n')
        print("✅ Added HYPERLIQUID_TESTNET=false")
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    return True

def main():
    print("🚨" * 30)
    print("🎯 SWITCHING TO HYPERLIQUID MAINNET")
    print("💰 REAL MONEY TRADING MODE")
    print("🚨" * 30)
    
    print("\n⚠️  WARNING: This will trade with REAL MONEY!")
    print(f"💰 Your balance: ~$51.63")
    print("🎯 Minimum trade size: $11+")
    print("🔥 Trades will be REAL and IRREVERSIBLE!")
    
    confirm = input("\n🚨 Confirm switch to MAINNET? (type 'MAINNET'): ")
    
    if confirm != 'MAINNET':
        print("❌ Switch cancelled - staying in testnet")
        return
    
    print("\n🔧 Updating configuration...")
    
    if not update_env_file():
        print("❌ Failed to update .env file")
        return
    
    print("✅ Configuration updated!")
    print("\n🚀 Starting MAINNET live trading system...")
    print("=" * 50)
    
    try:
        # Start the live system
        subprocess.run([sys.executable, 'start_live_system.py'])
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        print("💡 Try running manually: python start_live_system.py")

if __name__ == "__main__":
    main() 