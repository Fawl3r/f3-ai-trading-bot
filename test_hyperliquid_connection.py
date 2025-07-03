#!/usr/bin/env python3
"""
Test Hyperliquid connection and basic functionality
"""

import os
from dotenv import load_dotenv
from hyperliquid.info import Info
from hyperliquid.utils import constants
import eth_account

# Load environment variables
load_dotenv()

def test_connection():
    """Test basic Hyperliquid connection"""
    print("🧪 Testing Hyperliquid Connection...")
    print("=" * 50)
    
    try:
        # Test basic market data connection (no auth needed)
        print("📊 Testing market data connection...")
        info = Info(constants.TESTNET_API_URL, skip_ws=True)
        
        # Get all market prices
        all_mids = info.all_mids()
        print(f"✅ Connected! Found {len(all_mids)} trading pairs")
        
        # Show some sample prices
        sample_pairs = ['BTC', 'ETH', 'SOL']
        print("\n💰 Sample prices:")
        for pair in sample_pairs:
            if pair in all_mids:
                price = all_mids[pair]
                print(f"   {pair}: ${float(price):,.2f}")
        
        # Test private key authentication if provided
        private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY', '') or os.getenv('HL_PRIVATE_KEY', '')
        
        if private_key:
            print("\n🔑 Testing private key authentication...")
            try:
                account = eth_account.Account.from_key(private_key)
                account_address = account.address
                print(f"✅ Private key valid! Address: {account_address[:10]}...")
                
                # Test getting user state
                user_state = info.user_state(account_address)
                if user_state:
                    if 'marginSummary' in user_state:
                        balance = float(user_state['marginSummary']['accountValue'])
                        print(f"💰 Account balance: ${balance:.2f}")
                    else:
                        print("💰 Account balance: $0.00 (new account)")
                else:
                    print("⚠️  Could not retrieve account info (account might be new)")
                    
            except Exception as e:
                print(f"❌ Private key authentication failed: {e}")
        else:
            print("\n⚠️  No private key found in environment")
            print("   Set HYPERLIQUID_PRIVATE_KEY in .env to test authentication")
        
        print("\n✅ Hyperliquid connection test completed successfully!")
        print("🚀 Ready to start trading!")
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection")
        print("2. Verify Hyperliquid is not under maintenance")
        print("3. Try again in a few minutes")

if __name__ == "__main__":
    test_connection() 