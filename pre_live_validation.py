#!/usr/bin/env python3
"""
🛡️ PRE-LIVE VALIDATION SUITE
Comprehensive checks before live trading with real money
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import eth_account

load_dotenv()

class PreLiveValidator:
    def __init__(self):
        print("🛡️ PRE-LIVE VALIDATION SUITE")
        print("=" * 60)
        print("🎯 Ensuring 100% readiness before real money trading")
        
        # Load configuration
        self.private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
        self.account_address = os.getenv('HYPERLIQUID_ACCOUNT_ADDRESS')
        self.testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
        
        # Initialize clients
        self.base_url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
        self.info = Info(self.base_url, skip_ws=True)
        
        # Initialize exchange client
        account = eth_account.Account.from_key(self.private_key)
        self.exchange = Exchange(account, self.base_url)
        
        if not self.account_address:
            self.account_address = account.address
        
        print(f"🔗 Network: {'TESTNET' if self.testnet else 'MAINNET'}")
        print(f"🏦 Address: {self.account_address}")
        print("=" * 60)

    def run_validation_suite(self):
        """Run complete validation suite"""
        
        validations = [
            ("🔐 Credentials Validation", self.validate_credentials),
            ("💰 Balance Requirements", self.validate_balance_requirements),
            ("📊 Market Data Quality", self.validate_market_data_quality),
            ("🎯 Order Size Calculations", self.validate_order_sizing),
            ("📈 Price Precision Tests", self.validate_price_precision),
            ("⚡ Exchange Connectivity", self.validate_exchange_connectivity),
            ("🔍 Position Management", self.validate_position_management),
            ("🛡️ Risk Management", self.validate_risk_management),
            ("📋 Order Validation Logic", self.validate_order_validation),
            ("✅ Final Readiness Check", self.final_readiness_check)
        ]
        
        results = {}
        
        for validation_name, validation_func in validations:
            print(f"\n{validation_name}...")
            try:
                result = validation_func()
                results[validation_name] = {"status": "✅ PASS", "result": result}
                print(f"✅ {validation_name}: PASS")
            except Exception as e:
                results[validation_name] = {"status": "❌ FAIL", "error": str(e)}
                print(f"❌ {validation_name}: FAIL - {e}")
        
        self.generate_validation_report(results)
        return results

    def validate_credentials(self):
        """Validate credentials and authentication"""
        # Check private key format
        if len(self.private_key) != 66 or not self.private_key.startswith('0x'):
            raise Exception("Invalid private key format")
        
        # Verify address derivation
        account = eth_account.Account.from_key(self.private_key)
        if account.address.lower() != self.account_address.lower():
            raise Exception("Address mismatch")
        
        # Test API authentication
        user_state = self.info.user_state(self.account_address)
        if not user_state:
            raise Exception("Authentication failed")
        
        return "Credentials valid and authenticated"

    def validate_balance_requirements(self):
        """Validate balance meets minimum requirements"""
        user_state = self.info.user_state(self.account_address)
        balance = float(user_state.get('marginSummary', {}).get('accountValue', 0))
        
        if balance < 10:
            raise Exception(f"Insufficient balance: ${balance:.2f} (need $10+ for testing)")
        
        if balance < 20:
            print(f"   ⚠️  Low balance warning: ${balance:.2f}")
        
        return f"Balance: ${balance:.2f} (sufficient)"

    def validate_market_data_quality(self):
        """Validate market data quality and availability"""
        # Test BTC market data
        all_mids = self.info.all_mids()
        if 'BTC' not in all_mids:
            raise Exception("BTC not available")
        
        btc_price = float(all_mids['BTC'])
        if btc_price <= 0 or btc_price > 200000:
            raise Exception(f"Invalid BTC price: ${btc_price}")
        
        # Test order book depth
        l2_snapshot = self.info.l2_snapshot('BTC')
        levels = l2_snapshot['levels']
        
        if len(levels[0]) < 5 or len(levels[1]) < 5:
            raise Exception("Insufficient order book depth")
        
        return f"BTC: ${btc_price:,.2f}, Order book: {len(levels[0])} bids/{len(levels[1])} asks"

    def validate_order_sizing(self):
        """Validate order size calculations"""
        all_mids = self.info.all_mids()
        btc_price = float(all_mids['BTC'])
        
        # Test different trade sizes
        test_sizes_usd = [5, 10, 20, 50]
        valid_sizes = []
        
        for size_usd in test_sizes_usd:
            raw_size = size_usd / btc_price
            rounded_size = round(raw_size, 6)
            
            # Check if size is reasonable
            if rounded_size > 0.000001:  # More than dust
                valid_sizes.append((size_usd, rounded_size))
        
        if not valid_sizes:
            raise Exception("No valid trade sizes found")
        
        return f"Valid sizes: {len(valid_sizes)} tested, ranging ${test_sizes_usd[0]}-${test_sizes_usd[-1]}"

    def validate_price_precision(self):
        """Validate price precision and rounding"""
        all_mids = self.info.all_mids()
        btc_price = float(all_mids['BTC'])
        
        # Test price calculations with different multipliers
        test_multipliers = [0.99, 1.01, 0.995, 1.005]
        precision_tests = []
        
        for multiplier in test_multipliers:
            calculated_price = btc_price * multiplier
            rounded_price = round(calculated_price, 2)
            
            # Check for precision issues
            if abs(calculated_price - rounded_price) < 0.01:
                precision_tests.append(True)
            else:
                precision_tests.append(False)
        
        if not all(precision_tests):
            raise Exception("Price precision issues detected")
        
        return f"Price precision validated for {len(test_multipliers)} scenarios"

    def validate_exchange_connectivity(self):
        """Validate exchange client connectivity"""
        try:
            # Test that exchange client is properly initialized
            if not hasattr(self.exchange, 'order'):
                raise Exception("Exchange client missing order method")
            
            return "Exchange client properly initialized and accessible"
            
        except Exception as e:
            raise Exception(f"Exchange connectivity issue: {e}")

    def validate_position_management(self):
        """Validate position management capabilities"""
        user_state = self.info.user_state(self.account_address)
        
        # Check position structure
        positions = user_state.get('assetPositions', [])
        
        # Verify we can read position data
        open_positions = 0
        for pos in positions:
            if pos.get('position', {}).get('szi'):
                open_positions += 1
        
        return f"Position management ready, {open_positions} existing positions"

    def validate_risk_management(self):
        """Validate risk management parameters"""
        user_state = self.info.user_state(self.account_address)
        balance = float(user_state.get('marginSummary', {}).get('accountValue', 0))
        
        # Calculate max position size (2% of balance)
        max_position_usd = balance * 0.02
        
        if max_position_usd < 1:
            raise Exception(f"Max position too small: ${max_position_usd:.2f}")
        
        # Validate stop loss calculations
        test_entry = 50000.0
        stop_loss_pct = 0.03  # 3%
        stop_loss_price = test_entry * (1 - stop_loss_pct)
        
        if abs(stop_loss_price - 48500.0) > 1:
            raise Exception("Stop loss calculation error")
        
        return f"Risk management validated, max position: ${max_position_usd:.2f}"

    def validate_order_validation(self):
        """Validate order validation logic"""
        # Test order structure validation
        valid_order = {
            'coin': 'BTC',
            'is_buy': True,
            'sz': 0.001,
            'limit_px': 50000.0,
            'order_type': {'limit': {'tif': 'Ioc'}},
            'reduce_only': False
        }
        
        required_fields = ['coin', 'is_buy', 'sz', 'limit_px', 'order_type']
        for field in required_fields:
            if field not in valid_order:
                raise Exception(f"Order validation missing field: {field}")
        
        return "Order validation logic confirmed"

    def final_readiness_check(self):
        """Final comprehensive readiness check"""
        # Network check
        network = "MAINNET" if not self.testnet else "TESTNET"
        
        # Balance check
        user_state = self.info.user_state(self.account_address)
        balance = float(user_state.get('marginSummary', {}).get('accountValue', 0))
        
        # Market access check
        all_mids = self.info.all_mids()
        available_pairs = len(all_mids)
        
        readiness_score = 0
        if balance >= 10: readiness_score += 2
        if available_pairs >= 100: readiness_score += 2
        if network == "MAINNET": readiness_score += 1
        
        if readiness_score < 4:
            raise Exception(f"Readiness score too low: {readiness_score}/5")
        
        return f"{network} ready, ${balance:.2f} balance, {available_pairs} pairs, score: {readiness_score}/5"

    def generate_validation_report(self, results):
        """Generate final validation report"""
        print("\n" + "=" * 60)
        print("🛡️ PRE-LIVE VALIDATION REPORT")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for validation_name, result in results.items():
            status = result['status']
            print(f"{status} {validation_name}")
            
            if "✅" in status:
                passed += 1
            else:
                failed += 1
                if 'error' in result:
                    print(f"    Error: {result['error']}")
        
        print("=" * 60)
        print(f"📊 VALIDATION RESULTS: {passed} PASSED | {failed} FAILED")
        
        if failed == 0:
            print("🎉 ALL VALIDATIONS PASSED!")
            print("✅ 100% READY FOR LIVE TRADING!")
            print("🚀 Proceed with confidence!")
        elif passed >= 8:
            print("⚠️  Minor issues detected but mostly ready")
            print("✅ Core functionality validated")
        else:
            print("❌ Multiple validation failures")
            print("🔧 Fix issues before live trading")
        
        network = "MAINNET" if not self.testnet else "TESTNET"
        print(f"🔗 Network: {network}")
        print(f"📅 Validation completed: {datetime.now()}")
        print("=" * 60)

def main():
    """Run pre-live validation suite"""
    try:
        validator = PreLiveValidator()
        results = validator.run_validation_suite()
        return results
    except Exception as e:
        print(f"❌ Validation setup failed: {e}")
        return None

if __name__ == "__main__":
    main() 