#!/usr/bin/env python3
"""
Enhanced Live Simulation Trading Bot with Risk Management
3 Risk Modes: Safe, Risk, Super Risky
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
from risk_manager import RiskManager

class EnhancedLiveOKXDataFeed:
    """Enhanced data feed with risk-aware processing"""
    
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
        
        print(f"🌐 Enhanced Live OKX Data Feed initialized for {symbol}")
    
    def add_callback(self, callback):
        """Add callback function for data updates"""
        self.callback_functions.append(callback)
    
    def start(self):
        """Start live data feed"""
        if self.is_running:
            return
        
        self.is_running = True
        print("🚀 Starting enhanced live OKX data feed...")
        
        # Load initial historical data
        self._load_historical_data()
        
        # Start WebSocket for live updates
        self._start_websocket()
        
        print("✅ Enhanced live data feed started successfully!")
    
    def stop(self):
        """Stop live data feed"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        print("🛑 Enhanced live data feed stopped")
    
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
                    
                    # Subscribe to candle data (correct channel name)
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": [{
                            "channel": "candle1m",  # Use lowercase 'm' for correct channel name
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

class EnhancedLiveSimulationBot:
    """Enhanced trading bot with 3-tier risk management"""
    
    def __init__(self, initial_balance: float = 10000.0):
        # Initialize risk manager
        self.risk_manager = RiskManager()
        
        # Get user's risk preference
        self._select_risk_mode()
        
        # Initialize components
        self.data_feed = EnhancedLiveOKXDataFeed()
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
        
        # Get trading parameters from risk profile
        self.trading_params = self.risk_manager.get_trading_params()
        
        print(f"🤖 Enhanced Live Simulation Bot initialized with ${initial_balance:,.2f}")
        print(f"🎯 Risk Mode: {self.risk_manager.current_profile.name}")
    
    def _select_risk_mode(self):
        """Let user select risk mode"""
        while True:
            choice = self.risk_manager.display_risk_options()
            if self.risk_manager.select_risk_mode(choice):
                break
        
        print(f"\n🎉 Ready to trade with {self.risk_manager.current_profile.name}!")
    
    def start(self):
        """Start the enhanced trading bot"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        print("\n🚀 Starting Enhanced Live Simulation Trading Bot")
        print("=" * 60)
        print(f"🎯 Risk Mode: {self.risk_manager.current_profile.name}")
        print(f"💰 Position Size: ${self.trading_params['position_size_usd']} per trade")
        print(f"⚖️  Leverage: {self.trading_params['leverage']}x")
        print("=" * 60)
        
        # Setup data callback
        self.data_feed.add_callback(self._on_new_candle)
        
        # Start data feed
        self.data_feed.start()
        
        # Start status monitoring
        self._start_monitoring()
        
        print("🎯 Bot is now running with LIVE OKX data and risk management!")
        print("📱 Monitor the console for real-time updates")
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
        print("✅ Enhanced bot stopped successfully")
    
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
            if len(self.data_buffer) >= 50:
                self.data_buffer = self.indicators.calculate_all_indicators(self.data_buffer)
                
                # Check for trading signals
                self._check_enhanced_trading_signals(candle)
            
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
        
        cooldown = self.trading_params['signal_cooldown']
        time_since_last = (datetime.now() - self.last_signal_time).total_seconds()
        
        if time_since_last < cooldown:
            remaining = cooldown - time_since_last
            print(f"⏳ Signal cooldown: {remaining:.0f}s remaining")
            return False
        
        return True
    
    def _check_enhanced_trading_signals(self, current_candle: Dict):
        """Enhanced trading signal detection with risk management"""
        try:
            if len(self.data_buffer) < 50:
                return
            
            # Reset daily counters if needed
            self._reset_daily_counters()
            
            # Check signal cooldown
            if not self._check_signal_cooldown():
                return
            
            # Get latest indicators
            latest = self.data_buffer.iloc[-1]
            rsi = latest.get('rsi', 50)
            price = current_candle['close']
            
            # Get risk-adjusted RSI thresholds
            rsi_oversold = self.trading_params['rsi_oversold']
            rsi_overbought = self.trading_params['rsi_overbought']
            
            print(f"📊 RSI: {rsi:.1f} | Thresholds: Buy<{rsi_oversold} Sell>{rsi_overbought}")
            
            # BUY SIGNAL LOGIC
            if rsi < rsi_oversold and not self.trader.has_position():
                # Calculate confidence based on how oversold
                confidence = ((rsi_oversold - rsi) / rsi_oversold) * 100
                confidence = min(confidence, 100)  # Cap at 100%
                
                # Check if trade should be executed
                should_execute, reason = self.risk_manager.should_execute_trade(
                    confidence, self.daily_trades
                )
                
                if should_execute:
                    position_size = self.trading_params['position_size_usd']
                    
                    self.trader.execute_trade(
                        side='buy',
                        price=price,
                        size=position_size,
                        reason=f"RSI oversold: {rsi:.1f} | Risk: {self.risk_manager.current_profile.name}",
                        confidence=confidence
                    )
                    
                    self.daily_trades += 1
                    self.last_signal_time = datetime.now()
                    
                    print(f"🟢 BUY EXECUTED! ${price:.4f} | RSI: {rsi:.1f} | Confidence: {confidence:.1f}%")
                    print(f"💰 Position Size: ${position_size} | Daily Trades: {self.daily_trades}")
                else:
                    print(f"🚫 BUY BLOCKED: {reason}")
            
            # SELL SIGNAL LOGIC
            elif rsi > rsi_overbought and self.trader.has_position():
                # Calculate confidence based on how overbought
                confidence = ((rsi - rsi_overbought) / (100 - rsi_overbought)) * 100
                confidence = min(confidence, 100)  # Cap at 100%
                
                # Check if trade should be executed
                should_execute, reason = self.risk_manager.should_execute_trade(
                    confidence, self.daily_trades
                )
                
                if should_execute:
                    self.trader.close_position(
                        price=price,
                        reason=f"RSI overbought: {rsi:.1f} | Risk: {self.risk_manager.current_profile.name}"
                    )
                    
                    self.daily_trades += 1
                    self.last_signal_time = datetime.now()
                    
                    print(f"🔴 SELL EXECUTED! ${price:.4f} | RSI: {rsi:.1f} | Confidence: {confidence:.1f}%")
                    print(f"📊 Daily Trades: {self.daily_trades}")
                else:
                    print(f"🚫 SELL BLOCKED: {reason}")
            
            # NEUTRAL MARKET
            else:
                if self.daily_trades == 0:  # Only show once at start
                    print(f"😴 Waiting for signals... RSI: {rsi:.1f} (Need: <{rsi_oversold} or >{rsi_overbought})")
            
        except Exception as e:
            print(f"❌ Error checking enhanced signals: {e}")
    
    def _start_monitoring(self):
        """Start enhanced performance monitoring"""
        def monitoring_loop():
            while self.is_running:
                try:
                    time.sleep(30)  # Update every 30 seconds
                    
                    if datetime.now() - self.last_status_update > timedelta(seconds=30):
                        self._print_enhanced_status()
                        self.last_status_update = datetime.now()
                        
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def _print_enhanced_status(self):
        """Print enhanced bot status with risk metrics"""
        try:
            stats = self.trader.get_statistics()
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            current_price = self.data_feed.get_latest_price()
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            print("\n" + "=" * 70)
            print("📊 ENHANCED LIVE SIMULATION BOT STATUS")
            print("=" * 70)
            print(f"🎯 Risk Mode: {risk_metrics['mode']}")
            print(f"⏰ Uptime: {str(uptime).split('.')[0]}")
            print(f"💰 Balance: ${stats['balance']:,.2f}")
            print(f"📈 Total P&L: ${stats['total_pnl']:,.2f}")
            print(f"🎯 Win Rate: {stats['win_rate']:.1f}%")
            print(f"🔄 Total Trades: {stats['total_trades']}")
            print(f"📊 Active Positions: {1 if self.trader.has_position() else 0}")
            print(f"💹 Current SOL Price: ${current_price:.4f}")
            print(f"📡 Data Buffer: {len(self.data_buffer)} candles")
            print(f"📅 Daily Trades: {self.daily_trades}/{risk_metrics['max_daily_trades']}")
            print(f"⚖️  Leverage: {risk_metrics['leverage']}x")
            print(f"💼 Position Size: ${risk_metrics['position_size']}")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Error printing enhanced status: {e}")

def main():
    """Main entry point"""
    print("🚀 Enhanced OKX Live Simulation Trading Bot")
    print("📡 Using REAL market data from OKX")
    print("💰 Trading in SIMULATION mode with RISK MANAGEMENT")
    print("🎯 Choose your risk level for customized trading!")
    
    bot = EnhancedLiveSimulationBot(initial_balance=10000)
    
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