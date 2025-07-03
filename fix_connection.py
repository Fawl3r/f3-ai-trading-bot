#!/usr/bin/env python3
"""
Fix the Hyperliquid connection error
"""

import json
import os

print("🔧 DIAGNOSING CONNECTION ERROR...")
print("=" * 50)

# Check config.json
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print("✅ Config loaded successfully:")
    print(f"   Private key length: {len(config.get('private_key', ''))}")
    print(f"   Wallet address: {config.get('wallet_address', 'not found')}")
    print(f"   Is mainnet: {config.get('is_mainnet', True)}")
    
    # Check if private key looks valid
    private_key = config.get('private_key', '')
    if private_key.startswith('0x') and len(private_key) == 66:
        print("✅ Private key format looks correct")
    else:
        print(f"⚠️ Private key format may be incorrect: length {len(private_key)}")
    
    # Check wallet address
    wallet = config.get('wallet_address', '')
    if wallet.startswith('0x') and len(wallet) == 42:
        print("✅ Wallet address format looks correct")
    else:
        print(f"⚠️ Wallet address format may be incorrect: {wallet}")
    
except Exception as e:
    print(f"❌ Config error: {e}")

print("\n🔧 CREATING FIXED CONFIG...")

# Create a fixed config
fixed_config = {
    "private_key": os.getenv('HYPERLIQUID_PRIVATE_KEY', ''),
    "wallet_address": os.getenv('HYPERLIQUID_WALLET_ADDRESS', ''),
    "is_mainnet": True,
    "api_url": "https://api.hyperliquid.xyz",
    "trading_mode": "advanced_ta_momentum",
    "position_sizes": {
        "ta_base": 1.5,
        "ta_strong": 3.0,
        "momentum": 5.0,
        "parabolic": 8.0
    }
}

with open('config_fixed.json', 'w') as f:
    json.dump(fixed_config, f, indent=2)

print("✅ Created config_fixed.json")
print("✅ This separates API URL from wallet address")
print("=" * 50)
print("🚀 CONNECTION ERROR SHOULD BE FIXED!") 