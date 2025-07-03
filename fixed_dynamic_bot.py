#!/usr/bin/env python3
"""
Fixed Dynamic Live Simulation Trading Bot with $200 Starting Balance
Fixes WebSocket connection issues and uses dynamic risk management
"""

import requests
import json
import websocket
import threading
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from indicators import TechnicalIndicators
from simulation_trader import SimulationTradingEngine
from risk_manager import DynamicRiskManager

class FixedLiveDataFeed:
    """Fixed live data feed with proper WebSocket handling"""
    
    def __init__(self, symbol: str = "SOL-USD-SWAP"):
        self.symbol = symbol
        self.base_url = "https://www.okx.com"
        
        # Data storage
        self.candle_buffer = []
        self.latest_price = None
        self.callback_functions = []
        
        # Connection state
        self.is_running = False
        self.ws = None
        self.reconnect_count = 0
        self.max_reconnects = 5
        
        print(f"🌐 Fixed Live OKX Data Feed initialized for {symbol}")
    
    def add_callback(self, callback):
        """Add callback function for data updates"""
        self.callback_functions.append(callback)
    
    def start(self):
        """Start live data feed"""
        if self.is_running:
            return
        
        self.is_running = True
        print("🚀 Starting fixed live OKX data feed...")
        
        # Load initial historical data
        if self._load_historical_data():
            # Start WebSocket for live updates
            self._start_websocket()
            print("✅ Fixed live data feed started successfully!")
        else:
            print("❌ Failed to load initial data, using sample data...")
            self._use_sample_data()
    
    def stop(self):
        """Stop live data feed"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        print("🛑 Fixed live data feed stopped")
    
    def _load_historical_data(self):
        """Load initial historical data from OKX public API"""
        try:
            print("📚 Loading historical data from OKX...")
            
            url = f"{self.base_url}/api/v5/market/candles"
            params = {
                'instId': self.symbol,
                'bar': '1m',
                'limit': '200'  # Get 200 minutes of data
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == '0' and data.get('data'):
                    # Process candles (OKX returns newest first, so reverse)
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
                        
                        self.candle_buffer.append(candle_data)
                        
                        # Notify callbacks
                        for callback in self.callback_functions:
                            callback(candle_data)
                    
                    self.latest_price = self.candle_buffer[-1]['close'] if self.candle_buffer else None
                    
                    print(f"✅ Loaded {len(self.candle_buffer)} historical candles")
                    print(f"💰 Current {self.symbol} price: ${self.latest_price:.4f}")
                    
                    return True
                    
            print(f"❌ Failed to load data: HTTP {response.status_code}")
            return False
            
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return False
    
    def _use_sample_data(self):
        """Use sample data if API fails"""
        print("🧪 Using sample data for testing...")
        
        # Generate sample price around $142
        base_price = 142.0
        
        for i in range(50):
            price = base_price + np.random.normal(0, 2)  # Random walk
            candle_data = {
                'timestamp': int(time.time() * 1000) + i * 60000,
                'datetime': datetime.now() + timedelta(minutes=i),
                'open': price,
                'high': price + abs(np.random.normal(0, 0.5)),
                'low': price - abs(np.random.normal(0, 0.5)),
                'close': price,
                'volume': np.random.uniform(1000, 5000)
            }
            
            self.candle_buffer.append(candle_data)
            
            # Notify callbacks
            for callback in self.callback_functions:
                callback(candle_data)
        
        self.latest_price = price
        print(f"✅ Generated {len(self.candle_buffer)} sample candles")
        print(f"💰 Sample {self.symbol} price: ${self.latest_price:.4f}")
        
        # Start periodic price updates
        self._start_sample_updates()
    
    def _start_sample_updates(self):
        """Start periodic sample price updates"""
        def update_sample_price():
            while self.is_running:
                try:
                    time.sleep(10)  # Update every 10 seconds for testing
                    
                    if self.latest_price:
                        # Small random price movement
                        change = np.random.normal(0, 0.5)
                        new_price = max(self.latest_price + change, 50)  # Don't go below $50
                        
                        candle_data = {
                            'timestamp': int(time.time() * 1000),
                            'datetime': datetime.now(),
                            'open': self.latest_price,
                            'high': max(self.latest_price, new_price),
                            'low': min(self.latest_price, new_price),
                            'close': new_price,
                            'volume': np.random.uniform(1000, 5000)
                        }
                        
                        self.candle_buffer.append(candle_data)
                        if len(self.candle_buffer) > 500:
                            self.candle_buffer.pop(0)
                        
                        self.latest_price = new_price
                        
                        print(f"💹 Sample update: ${self.latest_price:.4f}")
                        
                        # Notify callbacks
                        for callback in self.callback_functions:
                            callback(candle_data)
                            
                except Exception as e:
                    print(f"❌ Error in sample updates: {e}")
        
        sample_thread = threading.Thread(target=update_sample_price, daemon=True)
        sample_thread.start()
    
    def _start_websocket(self):
        """Start WebSocket for live price updates with better error handling"""
        def run_websocket():
            try:
                ws_url = "wss://ws.okx.com:8443/ws/v5/public"
                
                def on_message(ws, message):
                    try:
                        data = json.loads(message)
                        
                        # Handle subscription confirmation
                        if data.get('event') == 'subscribe':
                            print("✅ WebSocket subscription confirmed")
                            self.reconnect_count = 0  # Reset on successful connection
                            return
                        
                        # Handle candle data
                        if 'data' in data and data.get('arg', {}).get('channel', '').startswith('candle'):
                            for candle_raw in data['data']:
                                candle_data = {
                                    'timestamp': int(candle_raw[0]),
                                    'datetime': datetime.fromtimestamp(int(candle_raw[0]) / 1000),
                                    'open': float(candle_raw[1]),
                                    'high': float(candle_raw[2]),
                                    'low': float(candle_raw[3]),
                                    'close': float(candle_raw[4]),
                                    'volume': float(candle_raw[5])
                                }
                                
                                # Update buffer
                                self.candle_buffer.append(candle_data)
                                if len(self.candle_buffer) > 500:
                                    self.candle_buffer.pop(0)
                                
                                self.latest_price = candle_data['close']
                                
                                print(f"💹 Live update: ${self.latest_price:.4f}")
                                
                                # Notify callbacks
                                for callback in self.callback_functions:
                                    callback(candle_data)
                                
                    except Exception as e:
                        print(f"❌ Error processing WebSocket message: {e}")
                
                def on_error(ws, error):
                    print(f"❌ WebSocket error: {error}")
                
                def on_close(ws, close_status_code, close_msg):
                    print(f"🔌 WebSocket closed: {close_status_code}")
                    
                    if self.is_running and self.reconnect_count < self.max_reconnects:
                        self.reconnect_count += 1
                        print(f"🔄 Attempting to reconnect... ({self.reconnect_count}/{self.max_reconnects})")
                        time.sleep(5)
                        self._start_websocket()
                    else:
                        print("❌ Max reconnect attempts reached, switching to sample data...")
                        self._use_sample_data()
                
                def on_open(ws):
                    print("✅ WebSocket connected to OKX")
                    
                    # Subscribe to candle data (FIXED INDENTATION)
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": [{
                            "channel": "candle1m",
                            "instId": self.symbol
                        }]
                    }
                    
                    try:
                        ws.send(json.dumps(subscribe_msg))
                        print(f"📡 Subscribed to {self.symbol} live candles")
                    except Exception as e:
                        print(f"❌ Error sending subscription: {e}")
                
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open
                )
                
                # Add ping interval to keep connection alive
                self.ws.run_forever(ping_interval=60, ping_timeout=10)
                
            except Exception as e:
                print(f"❌ WebSocket thread error: {e}")
                if self.is_running:
                    print("🔄 Falling back to sample data...")
                    self._use_sample_data()
        
        # Run WebSocket in separate thread
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
    
    def get_latest_price(self) -> float:
        """Get latest price"""
        return self.latest_price if self.latest_price else 142.0  # Default price
    
    def get_candle_data(self, limit: int = 100) -> List[Dict]:
        """Get recent candle data"""
        return self.candle_buffer[-limit:] if len(self.candle_buffer) >= limit else self.candle_buffer

class FixedDynamicBot:
    """Fixed dynamic trading bot with $200 starting balance"""
    
    def __init__(self, initial_balance: float = 200.0):
        print("🎯 FIXED DYNAMIC SIMULATION BOT - Starting with $200!")
        print("💡 Risk management adjusts automatically based on balance")
        print("🔧 Fixed WebSocket issues for stable connection")
        
        # Initialize risk manager
        self.risk_manager = DynamicRiskManager()
        self.initial_balance = initial_balance
        
        # Get user's risk preference
        self._select_risk_mode()
        
        # Initialize components
        self.data_feed = FixedLiveDataFeed()
        self.trader = SimulationTradingEngine(initial_balance)
        self.indicators = TechnicalIndicators()
        
        self.is_running = False
        self.data_buffer = pd.DataFrame()
        
        # Performance tracking
        self.start_time = None
        self.last_status_update = datetime.now()
        self.daily_trades = 0
        self.last_trade_date = datetime.now().date()
        self.last_signal_time = None
        self.last_balance_check = initial_balance
        
        print(f"🤖 Fixed Dynamic Bot initialized with ${initial_balance:,.2f}")
        print(f"🎯 Risk Mode: {self.risk_manager.current_profile.name}")
    
    def _select_risk_mode(self):
        """Let user select risk mode"""
        while True:
            choice = self.risk_manager.display_risk_options(self.initial_balance)
            if self.risk_manager.select_risk_mode(choice, self.initial_balance):
                break
        
        print(f"\n🎉 Ready to trade with {self.risk_manager.current_profile.name}!")
    
    def start(self):
        """Start the fixed dynamic trading bot"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        print("\n🚀 Starting Fixed Dynamic Live Simulation Trading Bot")
        print("=" * 80)
        print(f"💰 Starting Balance: ${self.initial_balance:,.2f}")
        print(f"🎯 Risk Mode: {self.risk_manager.current_profile.name}")
        
        # Show initial trading parameters
        params = self.risk_manager.get_trading_params(self.initial_balance)
        print(f"💼 Position Size: ${params['position_size_usd']:.2f} ({params['position_size_pct']}% of account)")
        print(f"⚖️  Leverage: {params['leverage']}x | RSI: Buy<{params['rsi_oversold']} Sell>{params['rsi_overbought']}")
        print("=" * 80)
        
        # Setup data callback
        self.data_feed.add_callback(self._on_new_candle)
        
        # Start data feed
        self.data_feed.start()
        
        # Start monitoring
        self._start_monitoring()
        
        print("🎯 Bot running with DYNAMIC risk management!")
        print("📈 Position sizes adjust automatically as balance changes!")
        print("🔧 Fixed WebSocket connection for stability!")
        print("Press Ctrl+C to stop...")
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping bot...")
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        self.data_feed.stop()
        print("✅ Fixed dynamic bot stopped successfully")
    
    def _on_new_candle(self, candle: Dict):
        """Process new candle data"""
        try:
            # Add to dataframe
            new_row = pd.DataFrame([candle])
            self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)
            
            # Keep last 500 candles
            if len(self.data_buffer) > 500:
                self.data_buffer = self.data_buffer.tail(500).reset_index(drop=True)
            
            # Calculate indicators if we have enough data
            if len(self.data_buffer) >= 20:  # Reduced requirement for faster testing
                self.data_buffer = self.indicators.calculate_all_indicators(self.data_buffer)
                
                # Check for trading signals
                self._check_dynamic_signals(candle)
            
        except Exception as e:
            print(f"❌ Error processing candle: {e}")
    
    def _reset_daily_counters(self):
        """Reset daily trade counters"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
            print(f"📅 New trading day! Daily trades reset to 0")
    
    def _check_signal_cooldown(self) -> bool:
        """Check signal cooldown"""
        if not self.last_signal_time:
            return True
        
        current_balance = self.trader.get_statistics()['balance']
        params = self.risk_manager.get_trading_params(current_balance)
        cooldown = params['signal_cooldown']
        time_since_last = (datetime.now() - self.last_signal_time).total_seconds()
        
        if time_since_last < cooldown:
            remaining = cooldown - time_since_last
            return False
        
        return True
    
    def _check_balance_changes(self):
        """Check for balance changes and adjust risk"""
        current_balance = self.trader.get_statistics()['balance']
        
        balance_change_pct = abs((current_balance - self.last_balance_check) / self.last_balance_check) * 100
        
        if balance_change_pct > 15:  # 15% change threshold
            print(f"\n💡 BALANCE CHANGE: ${self.last_balance_check:.2f} → ${current_balance:.2f}")
            self.risk_manager.adjust_for_balance_changes(current_balance, self.initial_balance)
            self.last_balance_check = current_balance
            
            params = self.risk_manager.get_trading_params(current_balance)
            print(f"🔄 New Position Size: ${params['position_size_usd']:.2f} ({params['position_size_pct']}%)")
    
    def _check_dynamic_signals(self, current_candle: Dict):
        """Check for trading signals with dynamic risk"""
        try:
            if len(self.data_buffer) < 20:
                return
            
            self._reset_daily_counters()
            
            if not self._check_signal_cooldown():
                return
            
            # Get current balance and parameters
            current_balance = self.trader.get_statistics()['balance']
            params = self.risk_manager.get_trading_params(current_balance)
            
            # Get indicators
            latest = self.data_buffer.iloc[-1]
            rsi = latest.get('rsi', 50)
            price = current_candle['close']
            
            rsi_oversold = params['rsi_oversold']
            rsi_overbought = params['rsi_overbought']
            
            # BUY SIGNAL
            if rsi < rsi_oversold and not self.trader.has_position():
                confidence = ((rsi_oversold - rsi) / rsi_oversold) * 100
                confidence = min(confidence, 100)
                
                should_execute, reason = self.risk_manager.should_execute_trade(
                    confidence, self.daily_trades, current_balance, self.initial_balance
                )
                
                if should_execute:
                    position_size = params['position_size_usd']
                    
                    self.trader.execute_trade(
                        side='buy',
                        price=price,
                        size=position_size,
                        reason=f"RSI oversold: {rsi:.1f}",
                        confidence=confidence
                    )
                    
                    self.daily_trades += 1
                    self.last_signal_time = datetime.now()
                    
                    print(f"🟢 BUY! ${price:.4f} | RSI: {rsi:.1f} | Size: ${position_size:.2f} | Conf: {confidence:.1f}%")
                    print(f"📊 Balance: ${current_balance:.2f} | Trades: {self.daily_trades}/{params['max_daily_trades']}")
                else:
                    print(f"🚫 BUY BLOCKED: {reason}")
            
            # SELL SIGNAL
            elif rsi > rsi_overbought and self.trader.has_position():
                confidence = ((rsi - rsi_overbought) / (100 - rsi_overbought)) * 100
                confidence = min(confidence, 100)
                
                should_execute, reason = self.risk_manager.should_execute_trade(
                    confidence, self.daily_trades, current_balance, self.initial_balance
                )
                
                if should_execute:
                    pnl_before = self.trader.get_statistics()['total_pnl']
                    
                    self.trader.close_position(
                        price=price,
                        reason=f"RSI overbought: {rsi:.1f}"
                    )
                    
                    pnl_after = self.trader.get_statistics()['total_pnl']
                    trade_pnl = pnl_after - pnl_before
                    
                    self.daily_trades += 1
                    self.last_signal_time = datetime.now()
                    
                    print(f"🔴 SELL! ${price:.4f} | RSI: {rsi:.1f} | P&L: ${trade_pnl:.2f}")
                    
                    self._check_balance_changes()
                else:
                    print(f"🚫 SELL BLOCKED: {reason}")
            
        except Exception as e:
            print(f"❌ Error checking signals: {e}")
    
    def _start_monitoring(self):
        """Start performance monitoring"""
        def monitoring_loop():
            while self.is_running:
                try:
                    time.sleep(60)  # Update every minute
                    
                    if datetime.now() - self.last_status_update > timedelta(seconds=60):
                        self._print_status()
                        self.last_status_update = datetime.now()
                        
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def _print_status(self):
        """Print comprehensive status"""
        try:
            stats = self.trader.get_statistics()
            current_balance = stats['balance']
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            current_price = self.data_feed.get_latest_price()
            
            risk_metrics = self.risk_manager.get_dynamic_risk_metrics(current_balance, self.initial_balance)
            
            balance_change = current_balance - self.initial_balance
            balance_change_pct = (balance_change / self.initial_balance) * 100
            
            print("\n" + "=" * 80)
            print("📊 FIXED DYNAMIC BOT STATUS")
            print("=" * 80)
            print(f"🎯 Risk Mode: {risk_metrics['mode']}")
            print(f"⏰ Uptime: {str(uptime).split('.')[0]}")
            print(f"💰 Balance: ${current_balance:,.2f} (Started: ${self.initial_balance:,.2f})")
            print(f"📈 Performance: ${balance_change:+.2f} ({balance_change_pct:+.1f}%)")
            print(f"📊 Drawdown: {risk_metrics['drawdown_pct']:.1f}% / {risk_metrics['max_drawdown_allowed']:.1f}%")
            print(f"🎯 Win Rate: {stats['win_rate']:.1f}% | Trades: {stats['total_trades']}")
            print(f"💹 SOL Price: ${current_price:.4f} | Buffer: {len(self.data_buffer)} candles")
            print(f"💼 Position Size: ${risk_metrics['position_size']:.2f} ({risk_metrics['position_size_pct']:.1f}%)")
            print(f"📅 Daily: {self.daily_trades}/{risk_metrics['max_daily_trades']} | Leverage: {risk_metrics['leverage']}x")
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ Status error: {e}")

def main():
    """Main entry point for fixed dynamic bot"""
    print("🚀 FIXED DYNAMIC OKX Trading Bot")
    print("💰 Starting with $200 - Risk adjusts automatically!")
    print("🔧 Fixed WebSocket connection issues")
    print("📡 Uses REAL market data from OKX")
    
    bot = FixedDynamicBot(initial_balance=200.0)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
        bot.stop()
    except Exception as e:
        print(f"❌ Bot error: {e}")
        bot.stop()

if __name__ == "__main__":
    main() 