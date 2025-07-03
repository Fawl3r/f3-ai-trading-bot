#!/usr/bin/env python3
"""
🔧 HYPERLIQUID LIVE CONNECTION TEST
Test script to verify if the F3 AI Trading Bot can connect to live Hyperliquid trading
"""
import json
import asyncio
import websockets
import aiohttp
from eth_account import Account

class HyperliquidConnectionTest:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        try:
            with open('f3_config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return None
    
    async def test_credentials(self):
        """Test if credentials are properly configured"""
        print("🔧 TESTING HYPERLIQUID CREDENTIALS...")
        print("=" * 50)
        
        if not self.config:
            print("❌ No config file found!")
            return False
            
        private_key = self.config['hyperliquid']['private_key']
        wallet_address = self.config['hyperliquid']['wallet_address']
        is_mainnet = self.config['hyperliquid']['is_mainnet']
        
        # Check if credentials are placeholder text
        if private_key == "YOUR_HYPERLIQUID_PRIVATE_KEY_HERE":
            print("❌ PRIVATE KEY NOT CONFIGURED")
            print("   Replace 'YOUR_HYPERLIQUID_PRIVATE_KEY_HERE' with your actual private key")
            return False
            
        if wallet_address == "YOUR_WALLET_ADDRESS_HERE":
            print("❌ WALLET ADDRESS NOT CONFIGURED") 
            print("   Replace 'YOUR_WALLET_ADDRESS_HERE' with your actual wallet address")
            return False
            
        if not private_key or not wallet_address:
            print("❌ CREDENTIALS ARE EMPTY")
            return False
            
        # Test if private key is valid
        try:
            account = Account.from_key(private_key)
            derived_address = account.address
            print(f"✅ Private key is valid")
            print(f"📍 Derived address: {derived_address}")
            
            if derived_address.lower() != wallet_address.lower():
                print(f"⚠️  WARNING: Derived address doesn't match configured address!")
                print(f"   Configured: {wallet_address}")
                print(f"   Derived:    {derived_address}")
                
        except Exception as e:
            print(f"❌ Invalid private key: {e}")
            return False
            
        print(f"🌐 Network: {'MAINNET' if is_mainnet else 'TESTNET'}")
        return True
    
    async def test_api_connection(self):
        """Test connection to Hyperliquid API"""
        print("\n🌐 TESTING API CONNECTION...")
        print("=" * 50)
        
        base_url = "https://api.hyperliquid.xyz"
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test basic API endpoint
                async with session.get(f"{base_url}/info", timeout=10) as response:
                    if response.status == 200:
                        print("✅ API connection successful")
                        data = await response.json()
                        print(f"📊 API responding normally")
                        return True
                    else:
                        print(f"❌ API connection failed: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"❌ API connection error: {e}")
            return False
    
    async def test_websocket_connection(self):
        """Test WebSocket connection"""
        print("\n📡 TESTING WEBSOCKET CONNECTION...")
        print("=" * 50)
        
        try:
            uri = "wss://api.hyperliquid.xyz/ws"
            async with websockets.connect(uri, timeout=10) as websocket:
                print("✅ WebSocket connection successful")
                
                # Send ping
                ping_msg = {"method": "ping"}
                await websocket.send(json.dumps(ping_msg))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                print(f"📡 WebSocket response received")
                return True
                
        except Exception as e:
            print(f"❌ WebSocket connection error: {e}")
            return False
    
    async def test_account_info(self):
        """Test if we can get account information"""
        print("\n💰 TESTING ACCOUNT ACCESS...")
        print("=" * 50)
        
        if not self.config:
            return False
            
        wallet_address = self.config['hyperliquid']['wallet_address']
        
        if wallet_address == "YOUR_WALLET_ADDRESS_HERE":
            print("❌ Cannot test account - wallet address not configured")
            return False
            
        try:
            base_url = "https://api.hyperliquid.xyz"
            async with aiohttp.ClientSession() as session:
                # Get user state
                payload = {
                    "type": "clearinghouseState",
                    "user": wallet_address
                }
                
                async with session.post(f"{base_url}/info", json=payload, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'marginSummary' in data:
                            balance = float(data['marginSummary']['accountValue'])
                            print(f"✅ Account access successful")
                            print(f"💰 Account Balance: ${balance:,.2f}")
                            
                            if balance > 0:
                                print(f"🎯 Ready for live trading!")
                                return True
                            else:
                                print(f"⚠️  Account balance is zero - deposit funds to trade")
                                return False
                        else:
                            print(f"❌ Unexpected account data structure")
                            return False
                    else:
                        print(f"❌ Account access failed: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"❌ Account access error: {e}")
            return False
    
    async def run_full_test(self):
        """Run complete connection test"""
        print("🚀 F3 AI TRADING BOT - HYPERLIQUID CONNECTION TEST")
        print("=" * 60)
        print()
        
        results = []
        
        # Test credentials
        results.append(await self.test_credentials())
        
        # Test API connection
        results.append(await self.test_api_connection())
        
        # Test WebSocket
        results.append(await self.test_websocket_connection())
        
        # Test account access
        results.append(await self.test_account_info())
        
        # Summary
        print("\n🎯 CONNECTION TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(results)
        total = len(results)
        
        print(f"✅ Tests Passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - READY FOR LIVE TRADING!")
            print("🚀 You can now run the bot with real trading enabled")
        else:
            print("❌ SOME TESTS FAILED - LIVE TRADING NOT READY")
            print("🔧 Fix the issues above before attempting live trading")
            
        return passed == total

async def main():
    tester = HyperliquidConnectionTest()
    await tester.run_full_test()

if __name__ == "__main__":
    asyncio.run(main()) 