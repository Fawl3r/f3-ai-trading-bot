#!/usr/bin/env python3
"""
Dynamic Live Simulation Trading Bot with $200 Starting Balance
Uses dynamic risk management that adjusts based on current account balance
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

class DynamicLiveDataFeed:
    """Live data feed for dynamic simulation"""
    
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
        
        print(f"🌐 Dynamic Live OKX Data Feed initialized for {symbol}")
    
    def add_callback(self, callback):
        """Add callback function for data updates"""
        self.callback_functions.append(callback)
    
    def start(self):
        """Start live data feed"""
        if self.is_running:
            return
        
        self.is_running = True
        print("🚀 Starting dynamic live OKX data feed...")
        
        # Load initial historical data
        self._load_historical_data()
        
        # Start WebSocket for live updates
        self._start_websocket()
        
        print("✅ Dynamic live data feed started successfully!")
    
    def stop(self):
        """Stop live data feed"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        print("🛑 Dynamic live data feed stopped")
    
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
    
    def _start_websocket(self):
        """Start WebSocket for live price updates"""
        def run_websocket():
            try:
                ws_url = "wss://ws.okx.com:8443/ws/v5/public"
                
                def on_message(ws, message):
                    try:
                        data = json.loads(message)
                        
                        # Handle subscription confirmation
                        if data.get('event') == 'subscribe':
                            print("✅ WebSocket subscription confirmed")
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
                                if len(self.candle_buffer) > 500:  # Keep last 500 candles
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
                    if self.is_running:
                        print("🔄 Attempting to reconnect...")
                        time.sleep(5)
                        self._start_websocket()
                
                def on_open(ws):
                    print("✅ WebSocket connected to OKX")
                    
                    # Subscribe to candle data
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": [{
                            "channel": "candle1m",
                            "instId": self.symbol
                        }]
                    }
                    ws.send(json.dumps(subscribe_msg))
                    print(f"📡 Subscribed to {self.symbol} live candles")
                
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open
                )
                
                self.ws.run_forever()
                
            except Exception as e:
                print(f"❌ WebSocket thread error: {e}")
        
        # Run WebSocket in separate thread
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
    
    def get_latest_price(self) -> float:
        """Get latest price"""
        return self.latest_price
    
    def get_candle_data(self, limit: int = 100) -> List[Dict]:
        """Get recent candle data"""
        return self.candle_buffer[-limit:] if len(self.candle_buffer) >= limit else self.candle_buffer

class DynamicSimulationBot:
    """Dynamic trading bot with $200 starting balance and risk management"""
    
    def __init__(self, initial_balance: float = 200.0):
        print("🎯 DYNAMIC SIMULATION BOT - Starting with $200!")
        print("💡 Risk management will adjust based on your balance")
        
        # Initialize risk manager with $200
        self.risk_manager = DynamicRiskManager()
        self.initial_balance = initial_balance
        
        # Get user's risk preference
        self._select_risk_mode()
        
        # Initialize components
        self.data_feed = DynamicLiveDataFeed()
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
        
        print(f"🤖 Dynamic Simulation Bot initialized with ${initial_balance:,.2f}")
        print(f"🎯 Risk Mode: {self.risk_manager.current_profile.name}")
    
    def _select_risk_mode(self):
        """Let user select risk mode with current balance"""
        while True:
            choice = self.risk_manager.display_risk_options(self.initial_balance)
            if self.risk_manager.select_risk_mode(choice, self.initial_balance):
                break
        
        print(f"\n🎉 Ready to trade with {self.risk_manager.current_profile.name}!")
        print(f"💰 Starting balance: ${self.initial_balance}")
    
    def start(self):
        """Start the dynamic trading bot"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        print("\n🚀 Starting Dynamic Live Simulation Trading Bot")
        print("=" * 70)
        print(f"💰 Starting Balance: ${self.initial_balance:,.2f}")
        print(f"🎯 Risk Mode: {self.risk_manager.current_profile.name}")
        
        # Show initial trading parameters
        params = self.risk_manager.get_trading_params(self.initial_balance)
        print(f"💼 Initial Position Size: ${params['position_size_usd']:.2f} ({params['position_size_pct']}% of account)")
        print(f"⚖️  Leverage: {params['leverage']}x")
        print(f"🎯 RSI Thresholds: Buy<{params['rsi_oversold']} Sell>{params['rsi_overbought']}")
        print("=" * 70)
        
        # Setup data callback
        self.data_feed.add_callback(self._on_new_candle)
        
        # Start data feed
        self.data_feed.start()
        
        # Start status monitoring
        self._start_monitoring()
        
        print("🎯 Bot is now running with LIVE OKX data and DYNAMIC risk management!")
        print("📈 Position sizes will adjust automatically as your balance changes!")
        print("Press Ctrl+C to stop...")
        
        try:
            # Keep running
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping bot...")
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        self.data_feed.stop()
        print("✅ Dynamic bot stopped successfully")
    
    def _on_new_candle(self, candle: Dict):
        """Process new candle data with dynamic risk management"""
        try:
            # Add to dataframe
            new_row = pd.DataFrame([candle])
            self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)
            
            # Keep last 500 candles
            if len(self.data_buffer) > 500:
                self.data_buffer = self.data_buffer.tail(500).reset_index(drop=True)
            
            # Calculate indicators if we have enough data
            if len(self.data_buffer) >= 50:
                self.data_buffer = self.indicators.calculate_all_indicators(self.data_buffer)
                
                # Check for trading signals with dynamic risk
                self._check_dynamic_trading_signals(candle)
            
        except Exception as e:
            print(f"❌ Error processing candle: {e}")
    
    def _reset_daily_counters(self):
        """Reset daily trade counters"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
            print(f"📅 New trading day started! Daily trades reset to 0")
    
    def _check_signal_cooldown(self) -> bool:
        """Check if signal cooldown has passed"""
        if not self.last_signal_time:
            return True
        
        current_balance = self.trader.get_statistics()['balance']
        params = self.risk_manager.get_trading_params(current_balance)
        cooldown = params['signal_cooldown']
        time_since_last = (datetime.now() - self.last_signal_time).total_seconds()
        
        if time_since_last < cooldown:
            remaining = cooldown - time_since_last
            print(f"⏳ Signal cooldown: {remaining:.0f}s remaining")
            return False
        
        return True
    
    def _check_balance_changes(self):
        """Check for significant balance changes and adjust risk accordingly"""
        current_balance = self.trader.get_statistics()['balance']
        
        # Check if balance changed significantly (>10%)
        balance_change_pct = abs((current_balance - self.last_balance_check) / self.last_balance_check) * 100
        
        if balance_change_pct > 10:
            print(f"\n💡 BALANCE CHANGE DETECTED: ${self.last_balance_check:.2f} → ${current_balance:.2f}")
            self.risk_manager.adjust_for_balance_changes(current_balance, self.initial_balance)
            self.last_balance_check = current_balance
            
            # Show new trading parameters
            params = self.risk_manager.get_trading_params(current_balance)
            print(f"🔄 Updated Position Size: ${params['position_size_usd']:.2f} ({params['position_size_pct']}% of account)")
    
    def _check_dynamic_trading_signals(self, current_candle: Dict):
        """Dynamic trading signal detection with balance-based risk management"""
        try:
            if len(self.data_buffer) < 50:
                return
            
            # Reset daily counters if needed
            self._reset_daily_counters()
            
            # Check signal cooldown
            if not self._check_signal_cooldown():
                return
            
            # Get current balance and dynamic parameters
            current_balance = self.trader.get_statistics()['balance']
            params = self.risk_manager.get_trading_params(current_balance)
            
            # Check for significant balance changes
            self._check_balance_changes()
            
            # Get latest indicators
            latest = self.data_buffer.iloc[-1]
            rsi = latest.get('rsi', 50)
            price = current_candle['close']
            
            # Get dynamic RSI thresholds
            rsi_oversold = params['rsi_oversold']
            rsi_overbought = params['rsi_overbought']
            
            print(f"📊 Balance: ${current_balance:.2f} | RSI: {rsi:.1f} | Pos Size: ${params['position_size_usd']:.2f}")
            
            # BUY SIGNAL LOGIC
            if rsi < rsi_oversold and not self.trader.has_position():
                # Calculate confidence based on how oversold
                confidence = ((rsi_oversold - rsi) / rsi_oversold) * 100
                confidence = min(confidence, 100)  # Cap at 100%
                
                # Check if trade should be executed with current balance
                should_execute, reason = self.risk_manager.should_execute_trade(
                    confidence, self.daily_trades, current_balance, self.initial_balance
                )
                
                if should_execute:
                    position_size = params['position_size_usd']
                    
                    self.trader.execute_trade(
                        side='buy',
                        price=price,
                        size=position_size,
                        reason=f"RSI oversold: {rsi:.1f} | Balance: ${current_balance:.2f}",
                        confidence=confidence
                    )
                    
                    self.daily_trades += 1
                    self.last_signal_time = datetime.now()
                    
                    print(f"🟢 BUY EXECUTED! ${price:.4f} | RSI: {rsi:.1f} | Confidence: {confidence:.1f}%")
                    print(f"💰 Position Size: ${position_size:.2f} ({params['position_size_pct']}% of ${current_balance:.2f})")
                    print(f"📊 Daily Trades: {self.daily_trades}/{params['max_daily_trades']}")
                else:
                    print(f"🚫 BUY BLOCKED: {reason}")
            
            # SELL SIGNAL LOGIC
            elif rsi > rsi_overbought and self.trader.has_position():
                # Calculate confidence based on how overbought
                confidence = ((rsi - rsi_overbought) / (100 - rsi_overbought)) * 100
                confidence = min(confidence, 100)  # Cap at 100%
                
                # Check if trade should be executed with current balance
                should_execute, reason = self.risk_manager.should_execute_trade(
                    confidence, self.daily_trades, current_balance, self.initial_balance
                )
                
                if should_execute:
                    pnl_before = self.trader.get_statistics()['total_pnl']
                    
                    self.trader.close_position(
                        price=price,
                        reason=f"RSI overbought: {rsi:.1f} | Balance: ${current_balance:.2f}"
                    )
                    
                    pnl_after = self.trader.get_statistics()['total_pnl']
                    trade_pnl = pnl_after - pnl_before
                    
                    self.daily_trades += 1
                    self.last_signal_time = datetime.now()
                    
                    print(f"🔴 SELL EXECUTED! ${price:.4f} | RSI: {rsi:.1f} | P&L: ${trade_pnl:.2f}")
                    print(f"📊 Daily Trades: {self.daily_trades}/{params['max_daily_trades']}")
                    
                    # Check balance changes after trade
                    self._check_balance_changes()
                else:
                    print(f"🚫 SELL BLOCKED: {reason}")
            
            # NEUTRAL MARKET
            else:
                # Occasionally show status
                if self.daily_trades == 0 and int(time.time()) % 60 == 0:  # Every minute
                    print(f"😴 Waiting... RSI: {rsi:.1f} | Need: <{rsi_oversold} or >{rsi_overbought} | Balance: ${current_balance:.2f}")
            
        except Exception as e:
            print(f"❌ Error checking dynamic signals: {e}")
    
    def _start_monitoring(self):
        """Start dynamic performance monitoring"""
        def monitoring_loop():
            while self.is_running:
                try:
                    time.sleep(45)  # Update every 45 seconds
                    
                    if datetime.now() - self.last_status_update > timedelta(seconds=45):
                        self._print_dynamic_status()
                        self.last_status_update = datetime.now()
                        
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def _print_dynamic_status(self):
        """Print dynamic bot status with balance-based risk metrics"""
        try:
            stats = self.trader.get_statistics()
            current_balance = stats['balance']
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            current_price = self.data_feed.get_latest_price()
            
            # Get dynamic risk metrics
            risk_metrics = self.risk_manager.get_dynamic_risk_metrics(current_balance, self.initial_balance)
            
            # Calculate performance
            balance_change = current_balance - self.initial_balance
            balance_change_pct = (balance_change / self.initial_balance) * 100
            
            print("\n" + "=" * 80)
            print("📊 DYNAMIC LIVE SIMULATION BOT STATUS")
            print("=" * 80)
            print(f"🎯 Risk Mode: {risk_metrics['mode']}")
            print(f"⏰ Uptime: {str(uptime).split('.')[0]}")
            print(f"💰 Balance: ${current_balance:,.2f} (Started: ${self.initial_balance:,.2f})")
            print(f"📈 Performance: ${balance_change:+.2f} ({balance_change_pct:+.1f}%)")
            print(f"📊 Drawdown: {risk_metrics['drawdown_pct']:.1f}% (Max: {risk_metrics['max_drawdown_allowed']:.1f}%)")
            print(f"🎯 Win Rate: {stats['win_rate']:.1f}%")
            print(f"🔄 Total Trades: {stats['total_trades']}")
            print(f"📊 Active Positions: {1 if self.trader.has_position() else 0}")
            print(f"💹 Current SOL Price: ${current_price:.4f}")
            print(f"💼 Current Position Size: ${risk_metrics['position_size']:.2f} ({risk_metrics['position_size_pct']:.1f}% of account)")
            print(f"🎯 Max Risk per Trade: ${risk_metrics['max_risk_per_trade']:.2f}")
            print(f"📅 Daily Trades: {self.daily_trades}/{risk_metrics['max_daily_trades']}")
            print(f"⚖️  Leverage: {risk_metrics['leverage']}x")
            print("=" * 80)
            
            # Show warning if approaching limits
            if risk_metrics['drawdown_pct'] > risk_metrics['max_drawdown_allowed'] * 0.8:
                print("⚠️  WARNING: Approaching maximum drawdown limit!")
            
            if current_balance < self.initial_balance * 0.2:
                print("🚨 CRITICAL: Balance below 20% of initial amount!")
            
        except Exception as e:
            print(f"❌ Error printing dynamic status: {e}")

def main():
    """Main entry point"""
    print("🚀 DYNAMIC OKX Live Simulation Trading Bot")
    print("💰 Starting with $200 - Risk adjusts automatically!")
    print("📡 Using REAL market data from OKX")
    print("🎯 Choose your risk level for customized trading!")
    
    bot = DynamicSimulationBot(initial_balance=200.0)
    
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