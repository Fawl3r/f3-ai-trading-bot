#!/usr/bin/env python3
"""
Live Data Simulator for OKX Trading Bot
Fetches REAL market data from OKX but executes trades in simulation mode
"""

import asyncio
import json
import threading
import time
import websocket
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional
import pandas as pd
import requests
import random
import numpy as np

from okx_client import OKXClient
from simulation_trader import SimulationTradingEngine
from config import INSTRUMENT_ID

class LiveDataSimulator:
    """Fetches real market data from OKX for simulation trading"""
    
    def __init__(self, symbol: str = "SOL-USD-SWAP"):
        self.symbol = symbol
        self.okx_client = OKXClient()
        
        # WebSocket connection
        self.ws = None
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        
        # Data storage
        self.latest_candle = None
        self.price_history = []
        self.callback_function = None
        
        # Connection state
        self.is_connected = False
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Threading
        self.data_thread = None
        self.heartbeat_thread = None
        
        print(f"🌐 Live Data Simulator initialized for {symbol}")
        print("📡 Will fetch REAL market data from OKX")
    
    def set_callback(self, callback: Callable):
        """Set callback function for data updates"""
        self.callback_function = callback
    
    def start_live_feed(self):
        """Start live data feed from OKX"""
        if self.is_running:
            return
        
        self.is_running = True
        print("🚀 Starting live OKX data feed...")
        
        # Load initial historical data
        self._load_historical_data()
        
        # Start WebSocket connection
        self._start_websocket()
        
        # Start heartbeat
        self._start_heartbeat()
        
        print("✅ Live data feed started")
    
    def stop_live_feed(self):
        """Stop live data feed"""
        self.is_running = False
        
        if self.ws:
            self.ws.close()
        
        print("🛑 Live data feed stopped")
    
    def _load_historical_data(self):
        """Load initial historical data from OKX"""
        try:
            print("📚 Loading historical data from OKX...")
            
            # Use public API endpoint directly for historical data (no auth required)
            url = f"{self.okx_client.base_url}/api/v5/market/candles"
            params = {
                'instId': INSTRUMENT_ID,
                'bar': '1m',
                'limit': '100'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == '0' and data.get('data'):
                    candles_data = []
                    
                    # OKX returns newest first, reverse to get chronological order
                    for candle in reversed(data['data']):
                        candle_data = {
                            'timestamp': int(candle[0]),
                            'datetime': datetime.fromtimestamp(int(candle[0]) / 1000),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5])
                        }
                        candles_data.append(candle_data)
                        
                        # Call callback for each historical candle
                        if self.callback_function:
                            self.callback_function(candle_data)
                    
                    self.price_history = candles_data
                    self.latest_candle = candles_data[-1] if candles_data else None
                    
                    print(f"✅ Loaded {len(candles_data)} historical candles")
                    if self.latest_candle:
                        print(f"📊 Latest price: ${self.latest_candle['close']:.4f}")
                    
                else:
                    print(f"❌ Failed to load historical data: {data}")
                    
            else:
                print(f"❌ HTTP error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            # Generate some sample data as fallback
            self._generate_fallback_data()
    
    def _generate_fallback_data(self):
        """Generate fallback sample data when OKX API is not accessible"""
        print("🔄 Generating fallback sample data...")
        
        # Create 100 sample candles with realistic SOL price movement
        base_price = 150.0  # Starting SOL price
        now = datetime.now()
        candles_data = []
        
        for i in range(100):
            timestamp = now - timedelta(minutes=100-i)
            
            # Generate realistic price movement
            price_change = random.uniform(-0.02, 0.02)  # ±2% per minute
            if i > 0:
                previous_close = candles_data[-1]['close']
                new_close = previous_close * (1 + price_change)
            else:
                new_close = base_price
            
            # Generate OHLC
            spread = new_close * 0.001  # 0.1% spread
            open_price = new_close + random.uniform(-spread, spread)
            high = max(open_price, new_close) + random.uniform(0, spread)
            low = min(open_price, new_close) - random.uniform(0, spread)
            volume = random.uniform(10000, 50000)
            
            candle_data = {
                'timestamp': int(timestamp.timestamp() * 1000),
                'datetime': timestamp,
                'open': round(open_price, 4),
                'high': round(high, 4),
                'low': round(low, 4),
                'close': round(new_close, 4),
                'volume': round(volume, 2)
            }
            
            candles_data.append(candle_data)
            
            # Call callback for each candle
            if self.callback_function:
                self.callback_function(candle_data)
        
        self.price_history = candles_data
        self.latest_candle = candles_data[-1] if candles_data else None
        
        print(f"✅ Generated {len(candles_data)} fallback candles")
        if self.latest_candle:
            print(f"📊 Fallback price: ${self.latest_candle['close']:.4f}")
    
    def _start_websocket(self):
        """Start WebSocket connection to OKX"""
        def run_websocket():
            while self.is_running:
                try:
                    print("🔌 Connecting to OKX WebSocket...")
                    
                    self.ws = websocket.WebSocketApp(
                        self.ws_url,
                        on_message=self._on_websocket_message,
                        on_error=self._on_websocket_error,
                        on_close=self._on_websocket_close,
                        on_open=self._on_websocket_open
                    )
                    
                    self.ws.run_forever()
                    
                except Exception as e:
                    print(f"❌ WebSocket error: {e}")
                    if self.is_running:
                        self._handle_reconnect()
        
        self.data_thread = threading.Thread(target=run_websocket)
        self.data_thread.daemon = True
        self.data_thread.start()
    
    def _on_websocket_open(self, ws):
        """Handle WebSocket connection open"""
        print("✅ Connected to OKX WebSocket")
        self.is_connected = True
        self.reconnect_attempts = 0
        
        # Subscribe to candlestick data
        subscribe_msg = {
            "op": "subscribe",
            "args": [
                {
                    "channel": "candle1m",
                    "instId": INSTRUMENT_ID
                }
            ]
        }
        
        ws.send(json.dumps(subscribe_msg))
        print(f"📡 Subscribed to {INSTRUMENT_ID} 1-minute candles")
    
    def _on_websocket_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle subscription confirmation
            if data.get('event') == 'subscribe':
                print(f"✅ Subscription confirmed: {data.get('arg', {}).get('channel')}")
                return
            
            # Handle candle data
            if 'data' in data:
                for candle_data in data['data']:
                    self._process_candle_data(candle_data)
                    
        except Exception as e:
            print(f"❌ Error processing WebSocket message: {e}")
    
    def _process_candle_data(self, raw_candle):
        """Process incoming candle data"""
        try:
            # Convert OKX candle format to our format
            candle = {
                'timestamp': int(raw_candle[0]),
                'datetime': datetime.fromtimestamp(int(raw_candle[0]) / 1000),
                'open': float(raw_candle[1]),
                'high': float(raw_candle[2]),
                'low': float(raw_candle[3]),
                'close': float(raw_candle[4]),
                'volume': float(raw_candle[5])
            }
            
            # Update latest candle
            self.latest_candle = candle
            
            # Add to history
            self.price_history.append(candle)
            
            # Keep only last 200 candles
            if len(self.price_history) > 200:
                self.price_history = self.price_history[-200:]
            
            # Print live update
            print(f"📈 Live Update: {candle['datetime'].strftime('%H:%M:%S')} | "
                  f"${candle['close']:.4f} | Vol: {candle['volume']:.0f}")
            
            # Call callback function
            if self.callback_function:
                self.callback_function(candle)
                
        except Exception as e:
            print(f"❌ Error processing candle data: {e}")
    
    def _on_websocket_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"❌ WebSocket error: {error}")
        self.is_connected = False
    
    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        print(f"🔌 WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        
        if self.is_running:
            self._handle_reconnect()
    
    def _handle_reconnect(self):
        """Handle WebSocket reconnection"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            wait_time = min(30, 5 * self.reconnect_attempts)
            
            print(f"🔄 Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} in {wait_time}s...")
            time.sleep(wait_time)
        else:
            print("❌ Max reconnection attempts reached")
            self.is_running = False
    
    def _start_heartbeat(self):
        """Start heartbeat to keep connection alive"""
        def heartbeat_loop():
            while self.is_running:
                try:
                    if self.is_connected and self.ws:
                        ping_msg = {"op": "ping"}
                        self.ws.send(json.dumps(ping_msg))
                    
                    time.sleep(25)  # Send ping every 25 seconds
                    
                except Exception as e:
                    print(f"❌ Heartbeat error: {e}")
                    time.sleep(30)
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
    
    def get_current_price(self) -> float:
        """Get current market price"""
        if self.latest_candle:
            return self.latest_candle['close']
        return 0.0
    
    def get_market_data(self) -> Dict:
        """Get current market data"""
        if not self.latest_candle:
            return {}
        
        return {
            'current_price': self.latest_candle['close'],
            'high_24h': self._calculate_24h_high(),
            'low_24h': self._calculate_24h_low(),
            'volume_24h': self._calculate_24h_volume(),
            'price_change_24h': self._calculate_24h_change(),
            'timestamp': self.latest_candle['datetime'].isoformat(),
            'is_live': True
        }
    
    def _calculate_24h_high(self) -> float:
        """Calculate 24h high from available data"""
        if not self.price_history:
            return 0.0
        return max(candle['high'] for candle in self.price_history)
    
    def _calculate_24h_low(self) -> float:
        """Calculate 24h low from available data"""
        if not self.price_history:
            return 0.0
        return min(candle['low'] for candle in self.price_history)
    
    def _calculate_24h_volume(self) -> float:
        """Calculate 24h volume from available data"""
        if not self.price_history:
            return 0.0
        return sum(candle['volume'] for candle in self.price_history)
    
    def _calculate_24h_change(self) -> float:
        """Calculate 24h price change from available data"""
        if len(self.price_history) < 2:
            return 0.0
        
        first_price = self.price_history[0]['close']
        current_price = self.latest_candle['close']
        
        return ((current_price - first_price) / first_price) * 100
    
    def get_price_history(self, limit: int = 100) -> List[Dict]:
        """Get recent price history"""
        return self.price_history[-limit:] if self.price_history else []

class LiveSimulationTradingEngine(SimulationTradingEngine):
    """Enhanced simulation trading engine with live OKX data"""
    
    def __init__(self, initial_balance: float = 10000.0):
        super().__init__(initial_balance)
        
        # Replace simulation data generator with live data
        self.live_data = LiveDataSimulator()
        self.data_generator = None  # Disable simulation data generator
        
        print("🌐 Live Simulation Trading Engine initialized")
        print("📡 Using REAL market data from OKX")
    
    def start_simulation(self):
        """Start the live simulation trading engine"""
        if self.is_running:
            return
        
        self.is_running = True
        print("🚀 Starting live simulation trading engine...")
        print(f"💰 Initial balance: ${self.balance:,.2f}")
        print(f"📊 Position size: ${self.position_size_usd}")
        print(f"⚖️  Leverage: {self.leverage}x")
        print("📡 Market data: LIVE from OKX")
        
        # Start live data feed
        self.live_data.set_callback(self._on_new_candle)
        self.live_data.start_live_feed()
        
        # Start trading logic
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.live_data.stop_live_feed()
        if self.trading_thread:
            self.trading_thread.join()
        print("🛑 Live simulation trading engine stopped")
    
    def get_market_data(self) -> Dict:
        """Get current market data from live feed"""
        return self.live_data.get_market_data()

def main():
    """Test the live data simulator"""
    def on_candle_update(candle):
        print(f"📊 {candle['datetime'].strftime('%H:%M:%S')} | "
              f"O: ${candle['open']:.4f} | H: ${candle['high']:.4f} | "
              f"L: ${candle['low']:.4f} | C: ${candle['close']:.4f} | "
              f"V: {candle['volume']:.0f}")
    
    # Create live data simulator
    live_data = LiveDataSimulator()
    live_data.set_callback(on_candle_update)
    
    try:
        # Start live data feed
        live_data.start_live_feed()
        
        # Let it run for a while
        print("📡 Live data feed running... Press Ctrl+C to stop")
        while True:
            time.sleep(60)
            
            # Print market summary every minute
            market_data = live_data.get_market_data()
            if market_data:
                print(f"\n📊 Market Summary:")
                print(f"Current Price: ${market_data['current_price']:.4f}")
                print(f"24h Change: {market_data['price_change_24h']:.2f}%")
                print(f"24h High: ${market_data['high_24h']:.4f}")
                print(f"24h Low: ${market_data['low_24h']:.4f}")
                print("-" * 50)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping live data feed...")
    finally:
        live_data.stop_live_feed()

if __name__ == "__main__":
    main() 