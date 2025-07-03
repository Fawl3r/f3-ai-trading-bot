#!/usr/bin/env python3
"""
Test order book methods in Hyperliquid SDK
"""

from hyperliquid.info import Info
from hyperliquid.utils import constants

def test_order_book_methods():
    info = Info(constants.TESTNET_API_URL, skip_ws=True)
    
    print("🔍 Testing order book methods...")
    
    # Check available methods
    methods = [method for method in dir(info) if 'book' in method.lower() or 'order' in method.lower()]
    print(f"📊 Found potential order book methods: {methods}")
    
    # Test common method names
    test_methods = ['l2_book', 'order_book', 'book', 'l2_snapshot', 'orderbook']
    
    for method_name in test_methods:
        if hasattr(info, method_name):
            print(f"✅ Found method: {method_name}")
            try:
                method = getattr(info, method_name)
                if callable(method):
                    result = method("BTC")
                    print(f"✅ {method_name}('BTC') works: {type(result)}")
                    if result:
                        print(f"   Sample data: {str(result)[:100]}...")
                    break
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
        else:
            print(f"❌ Method {method_name} not found")
    
    # Try alternative approaches
    print("\n🔍 Trying alternative approaches...")
    try:
        # Try getting meta data first
        meta = info.meta()
        print(f"✅ Meta data available: {len(meta.get('universe', []))} assets")
        
        # Try getting all mids (prices)
        all_mids = info.all_mids()
        print(f"✅ All mids available: {len(all_mids)} pairs")
        
        # This is sufficient for trading - order book might not be essential
        return True
        
    except Exception as e:
        print(f"❌ Alternative approach failed: {e}")
        return False

if __name__ == "__main__":
    test_order_book_methods() 