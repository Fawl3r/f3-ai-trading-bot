#!/usr/bin/env python3
"""
Test script to fetch live SOL-USD data from OKX without authentication
"""

import requests
import json
import websocket
import threading
import time
from datetime import datetime

def test_public_api():
    """Test public REST API endpoint"""
    print("🧪 Testing OKX Public REST API...")
    
    url = "https://www.okx.com/api/v5/market/candles"
    params = {
        'instId': 'SOL-USD-SWAP',
        'bar': '1m',
        'limit': '5'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Response: {json.dumps(data, indent=2)}")
            
            if data.get('code') == '0' and data.get('data'):
                print("✅ Public API working!")
                latest_candle = data['data'][0]  # OKX returns newest first
                price = float(latest_candle[4])  # Close price
                print(f"💰 Current SOL price: ${price:.4f}")
                return True
            else:
                print(f"❌ API error: {data}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_public_websocket():
    """Test public WebSocket endpoint"""
    print("\n🧪 Testing OKX Public WebSocket...")
    
    ws_url = "wss://ws.okx.com:8443/ws/v5/public"
    
    def on_message(ws, message):
        try:
            data = json.loads(message)
            print(f"📨 WebSocket message: {data}")
            
            if 'data' in data:
                for candle_data in data['data']:
                    price = float(candle_data[4])  # Close price
                    print(f"💰 Live SOL price: ${price:.4f}")
                    
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
    def on_error(ws, error):
        print(f"❌ WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")
    
    def on_open(ws):
        print("✅ WebSocket connected!")
        
        # Subscribe to SOL-USD-SWAP 1-minute candles
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "channel": "candle1m",
                "instId": "SOL-USD-SWAP"
            }]
        }
        ws.send(json.dumps(subscribe_msg))
        print(f"📡 Subscribed to SOL-USD-SWAP candles")
    
    # Create WebSocket connection
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    
    # Run for 30 seconds
    def stop_ws():
        time.sleep(30)
        ws.close()
        print("🛑 Stopping WebSocket test")
    
    stop_thread = threading.Thread(target=stop_ws)
    stop_thread.start()
    
    ws.run_forever()

if __name__ == "__main__":
    print("🚀 Testing OKX Live Data Access")
    print("=" * 50)
    
    # Test REST API
    if test_public_api():
        print("\n🎉 REST API test successful!")
    else:
        print("\n❌ REST API test failed!")
    
    # Test WebSocket
    print("\n" + "=" * 50)
    test_public_websocket() 