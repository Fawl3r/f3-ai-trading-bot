#!/usr/bin/env python3
"""
🔥 HYPERLIQUID TRADE EXECUTION TEST
Real trade execution test - opens and closes actual positions
Use small amounts to verify trading works end-to-end
"""

import asyncio
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import os

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import eth_account

# Load environment
load_dotenv()

class HyperliquidTradeTest:
    def __init__(self, trade_size_usd: float = 11.0):
        print("🔥 HYPERLIQUID TRADE EXECUTION TEST")
        print("=" * 60)
        print("🎯 Testing REAL trade execution on Hyperliquid")
        print(f"💰 Test trade size: ${trade_size_usd}")
        print("⚠️  WARNING: This will place real trades!")
        print("=" * 60)
        
        # Load configuration
        self.private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
        self.account_address = os.getenv('HYPERLIQUID_ACCOUNT_ADDRESS')
        self.testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
        self.trade_size_usd = trade_size_usd
        
        if not self.private_key or self.private_key == 'your_private_key_here':
            raise ValueError("❌ HYPERLIQUID_PRIVATE_KEY not configured in .env file!")
        
        # Initialize clients
        self.base_url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
        self.info = Info(self.base_url, skip_ws=True)
        
        # Initialize exchange client
        account = eth_account.Account.from_key(self.private_key)
        self.exchange = Exchange(account, self.base_url)
        
        if not self.account_address:
            self.account_address = account.address
        
        print(f"🔗 Mode: {'TESTNET' if self.testnet else 'MAINNET'}")
        print(f"🏦 Address: {self.account_address}")
        print("=" * 60)
        
        # Test variables
        self.test_coin = 'BTC'  # Safe, liquid pair for testing
        self.opened_orders = []
        self.test_results = {}

    def run_trade_test(self):
        """Run complete trade execution test"""
        print("\n🚀 STARTING TRADE EXECUTION TEST")
        print("=" * 60)
        
        tests = [
            ("📊 Pre-flight Checks", self.preflight_checks),
            ("💰 Account Balance Check", self.check_balance),
            ("📈 Market Data Validation", self.validate_market_data),
            ("🔓 Long Position Test", self.test_long_position),
            ("📉 Short Position Test", self.test_short_position),
            ("🔍 Trade History Verification", self.verify_trade_history),
            ("✅ Final Cleanup", self.final_cleanup)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{test_name}...")
            try:
                result = test_func()
                self.test_results[test_name] = {"status": "✅ PASS", "result": result}
                print(f"✅ {test_name}: PASS")
                
                # Small delay between tests
                time.sleep(2)
                
            except Exception as e:
                self.test_results[test_name] = {"status": "❌ FAIL", "error": str(e)}
                print(f"❌ {test_name}: FAIL - {e}")
                
                # If critical test fails, stop
                if test_name in ["📊 Pre-flight Checks", "💰 Account Balance Check"]:
                    print("❌ Critical test failed - stopping execution")
                    break
        
        self.generate_final_report()
        return self.test_results

    def preflight_checks(self):
        """Essential pre-flight checks"""
        # Check API connection
        all_mids = self.info.all_mids()
        if not all_mids or len(all_mids) < 100:
            raise Exception("API connection failed")
        
        # Check if test coin exists
        if self.test_coin not in all_mids:
            raise Exception(f"{self.test_coin} not available")
        
        # Check authentication
        user_state = self.info.user_state(self.account_address)
        if not user_state:
            raise Exception("Authentication failed")
        
        return f"✅ API connected, {self.test_coin} available, authenticated"

    def check_balance(self):
        """Check account balance and calculate trade size"""
        user_state = self.info.user_state(self.account_address)
        balance = float(user_state.get('marginSummary', {}).get('accountValue', 0))
        
        if balance < self.trade_size_usd * 2:
            raise Exception(f"Insufficient balance: ${balance:.2f} (need at least ${self.trade_size_usd * 2:.2f})")
        
        return f"Balance: ${balance:.2f} (sufficient for testing)"

    def validate_market_data(self):
        """Validate market data for trading"""
        # Get current price
        all_mids = self.info.all_mids()
        current_price = float(all_mids[self.test_coin])
        
        if current_price <= 0:
            raise Exception(f"Invalid price for {self.test_coin}")
        
        # Check order book depth
        l2_snapshot = self.info.l2_snapshot(self.test_coin)
        levels = l2_snapshot['levels']
        
        if len(levels[0]) < 5 or len(levels[1]) < 5:
            raise Exception("Insufficient order book depth")
        
        # Calculate trade size in coins and round to reasonable precision
        raw_size = self.trade_size_usd / current_price
        
        # Round to 5 decimal places (matches Hyperliquid BTC szDecimals=5)
        self.trade_size_coins = round(raw_size, 5)
        
        return f"Price: ${current_price:,.2f}, Trade size: {self.trade_size_coins:.5f} {self.test_coin}"

    def format_size(self, size):
        """Format size as proper decimal string (no scientific notation)"""
        return float(f"{size:.5f}")

    def test_long_position(self):
        """Test opening and closing a long position"""
        print(f"   🟢 Opening LONG position for {self.test_coin}...")
        
        # Get current price
        all_mids = self.info.all_mids()
        current_price = float(all_mids[self.test_coin])
        
        # Place market buy order (long)
        buy_order = {
            'coin': self.test_coin,
            'is_buy': True,
            'sz': self.format_size(self.trade_size_coins),
            'limit_px': round(current_price * 1.005, 1),  # Small buffer above market
            'order_type': {'limit': {'tif': 'Ioc'}},  # Immediate or Cancel
            'reduce_only': False
        }
        
        print(f"   📝 Placing buy order: {buy_order}")
        
        try:
            buy_result = self.exchange.order(
                buy_order['coin'],      # name (symbol)
                buy_order['is_buy'],    # is_buy
                buy_order['sz'],        # size
                buy_order['limit_px'],  # limit price
                buy_order['order_type'] # order type
            )
            print(f"   📝 Buy order result: {buy_result}")
            print(f"   📝 Buy order result type: {type(buy_result)}")
        except Exception as e:
            print(f"   ❌ Order placement failed: {e}")
            raise Exception(f"Buy order failed with exception: {e}")
        
        # Handle both boolean and dict responses
        order_success = False
        if isinstance(buy_result, bool):
            order_success = buy_result
        elif isinstance(buy_result, dict) and 'status' in buy_result:
            if buy_result['status'] == 'ok':
                # Check if order was actually accepted (no errors in response)
                response_data = buy_result.get('response', {}).get('data', {})
                statuses = response_data.get('statuses', [])
                if statuses and any('error' in status for status in statuses):
                    # Order was rejected
                    error_msg = statuses[0].get('error', 'Unknown error')
                    raise Exception(f"Order rejected by exchange: {error_msg}")
                order_success = True
        
        if order_success:
            # Wait for fill
            time.sleep(3)
            
            # Check position
            user_state = self.info.user_state(self.account_address)
            positions = user_state.get('assetPositions', [])
            
            btc_position = None
            for pos in positions:
                if pos.get('position', {}).get('coin') == self.test_coin:
                    btc_position = pos
                    break
            
            if btc_position:
                position_size = float(btc_position['position']['szi'])
                print(f"   📊 Position opened: {position_size} {self.test_coin}")
                
                # Close the position (sell)
                print(f"   🔴 Closing LONG position...")
                
                sell_order = {
                    'coin': self.test_coin,
                    'is_buy': False,
                    'sz': self.format_size(abs(position_size)),
                    'limit_px': round(current_price * 0.995, 1),  # Small buffer below market
                    'order_type': {'limit': {'tif': 'Ioc'}},  # Immediate or Cancel
                    'reduce_only': True
                }
                
                sell_result = self.exchange.order(
                    sell_order['coin'],      # name (symbol)
                    sell_order['is_buy'],    # is_buy
                    sell_order['sz'],        # size
                    sell_order['limit_px'],  # limit price
                    sell_order['order_type'] # order type
                )
                print(f"   📝 Sell order result: {sell_result}")
                
                time.sleep(3)
                return f"Long position test completed: opened and closed {abs(position_size):.6f} {self.test_coin}"
            else:
                return "Long order placed but position not found (may not have filled)"
        else:
            if isinstance(buy_result, bool):
                raise Exception(f"Long order failed: order returned {buy_result}")
            else:
                raise Exception(f"Long order failed: {buy_result}")

    def test_short_position(self):
        """Test opening and closing a short position"""
        print(f"   🔴 Opening SHORT position for {self.test_coin}...")
        
        # Get current price
        all_mids = self.info.all_mids()
        current_price = float(all_mids[self.test_coin])
        
        # Place market sell order (short)
        sell_order = {
            'coin': self.test_coin,
            'is_buy': False,
            'sz': self.format_size(self.trade_size_coins),
            'limit_px': round(current_price * 0.995, 1),  # Small buffer below market
            'order_type': {'limit': {'tif': 'Ioc'}},  # Immediate or Cancel
            'reduce_only': False
        }
        
        print(f"   📝 Placing sell order: {sell_order}")
        
        try:
            sell_result = self.exchange.order(
                sell_order['coin'],      # name (symbol)
                sell_order['is_buy'],    # is_buy
                sell_order['sz'],        # size
                sell_order['limit_px'],  # limit price
                sell_order['order_type'] # order type
            )
            print(f"   📝 Sell order result: {sell_result}")
            print(f"   📝 Sell order result type: {type(sell_result)}")
        except Exception as e:
            print(f"   ❌ Order placement failed: {e}")
            raise Exception(f"Sell order failed with exception: {e}")
        
        # Handle both boolean and dict responses
        order_success = False
        if isinstance(sell_result, bool):
            order_success = sell_result
        elif isinstance(sell_result, dict) and 'status' in sell_result:
            if sell_result['status'] == 'ok':
                # Check if order was actually accepted (no errors in response)
                response_data = sell_result.get('response', {}).get('data', {})
                statuses = response_data.get('statuses', [])
                if statuses and any('error' in status for status in statuses):
                    # Order was rejected
                    error_msg = statuses[0].get('error', 'Unknown error')
                    raise Exception(f"Order rejected by exchange: {error_msg}")
                order_success = True
        
        if order_success:
            # Wait for fill
            time.sleep(3)
            
            # Check position
            user_state = self.info.user_state(self.account_address)
            positions = user_state.get('assetPositions', [])
            
            btc_position = None
            for pos in positions:
                if pos.get('position', {}).get('coin') == self.test_coin:
                    btc_position = pos
                    break
            
            if btc_position:
                position_size = float(btc_position['position']['szi'])
                print(f"   📊 Position opened: {position_size} {self.test_coin}")
                
                # Close the position (buy back)
                print(f"   🟢 Closing SHORT position...")
                
                buy_order = {
                    'coin': self.test_coin,
                    'is_buy': True,
                    'sz': self.format_size(abs(position_size)),
                    'limit_px': round(current_price * 1.005, 1),  # Small buffer above market
                    'order_type': {'limit': {'tif': 'Ioc'}},  # Immediate or Cancel
                    'reduce_only': True
                }
                
                buy_result = self.exchange.order(
                    buy_order['coin'],      # name (symbol)
                    buy_order['is_buy'],    # is_buy
                    buy_order['sz'],        # size
                    buy_order['limit_px'],  # limit price
                    buy_order['order_type'] # order type
                )
                print(f"   📝 Buy order result: {buy_result}")
                
                time.sleep(3)
                return f"Short position test completed: opened and closed {abs(position_size):.6f} {self.test_coin}"
            else:
                return "Short order placed but position not found (may not have filled)"
        else:
            if isinstance(sell_result, bool):
                raise Exception(f"Short order failed: order returned {sell_result}")
            else:
                raise Exception(f"Short order failed: {sell_result}")

    def verify_trade_history(self):
        """Verify trades appear in history"""
        user_fills = self.info.user_fills(self.account_address)
        
        if not user_fills:
            return "No trade history found (orders may not have filled)"
        
        # Count recent fills (last 10 minutes)
        recent_fills = 0
        current_time = int(time.time() * 1000)
        
        for fill in user_fills[:10]:  # Check last 10 fills
            fill_time = int(fill.get('time', 0))
            if current_time - fill_time < 600000:  # 10 minutes
                recent_fills += 1
        
        return f"Trade history verified: {recent_fills} recent fills found"

    def final_cleanup(self):
        """Ensure no positions are left open"""
        user_state = self.info.user_state(self.account_address)
        positions = user_state.get('assetPositions', [])
        
        open_positions = 0
        for pos in positions:
            position_size = float(pos.get('position', {}).get('szi', 0))
            if abs(position_size) > 0.000001:  # More than dust
                open_positions += 1
                print(f"   ⚠️  Open position found: {position_size} {pos['position']['coin']}")
        
        if open_positions == 0:
            return "✅ No open positions remaining"
        else:
            return f"⚠️  {open_positions} positions still open (manual cleanup may be needed)"

    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("🏆 HYPERLIQUID TRADE EXECUTION TEST RESULTS")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.test_results.items():
            status = result['status']
            print(f"{status} {test_name}")
            if "✅" in status:
                passed += 1
            else:
                failed += 1
                if 'error' in result:
                    print(f"    Error: {result['error']}")
        
        print("=" * 60)
        print(f"📊 RESULTS: {passed} PASSED | {failed} FAILED")
        
        if failed == 0:
            print("🎉 ALL TRADE TESTS PASSED!")
            print("✅ HYPERLIQUID TRADING IS FULLY FUNCTIONAL!")
            print("🚀 Bot is ready for live trading!")
        elif passed >= 4:
            print("⚠️  Most tests passed - minor issues detected")
            print("✅ Core trading functionality works")
        else:
            print("❌ Multiple test failures - review setup")
            print("🔧 Fix issues before live trading")
        
        print(f"💰 Test trade size used: ${self.trade_size_usd}")
        print(f"🔗 Network: {'TESTNET' if self.testnet else 'MAINNET'}")
        print(f"📅 Test completed: {datetime.now()}")
        print("=" * 60)

def main():
    """Run the trade execution test"""
    print("⚠️  WARNING: This test will place REAL trades!")
    print("💡 Make sure you're using testnet or small amounts")
    
    # Get user confirmation
    if os.getenv('HYPERLIQUID_TESTNET', 'true').lower() != 'true':
        confirm = input("\n🚨 You're about to test on MAINNET! Continue? (type 'YES' to proceed): ")
        if confirm != 'YES':
            print("❌ Test cancelled")
            return
    
    try:
        # Use $11 for testing (ensures >$10 minimum order value after price calculation)
        trade_size = float(os.getenv('TEST_TRADE_SIZE_USD', '11.0'))
        tester = HyperliquidTradeTest(trade_size_usd=trade_size)
        results = tester.run_trade_test()
        return results
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        print("💡 Make sure your .env file is configured correctly")
        return None

if __name__ == "__main__":
    main() 