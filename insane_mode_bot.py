#!/usr/bin/env python3
"""
Insane Mode Trading Bot with AI Analysis
Includes all 4 risk modes: Safe, Risk, Super Risky, and INSANE
Uses AI-powered analysis for extreme leverage trading (30x-50x)
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
from ai_analyzer import AITradeAnalyzer

class InsaneModeDataFeed:
    """Enhanced data feed for all risk modes including Insane Mode"""
    
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
        self.max_reconnects = 3
        
        print(f"🌐 Insane Mode Data Feed initialized for {symbol}")
    
    def add_callback(self, callback):
        """Add callback function for data updates"""
        self.callback_functions.append(callback)
    
    def start(self):
        """Start live data feed"""
        if self.is_running:
            return
        
        self.is_running = True
        print("🚀 Starting Insane Mode data feed...")
        
        # Load initial historical data
        if self._load_historical_data():
            # Start WebSocket for live updates
            self._start_websocket()
            print("✅ Insane Mode data feed started successfully!")
        else:
            print("❌ API failed, using enhanced sample data...")
            self._use_enhanced_sample_data()
    
    def stop(self):
        """Stop live data feed"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        print("🛑 Insane Mode data feed stopped")
    
    def _load_historical_data(self):
        """Load initial historical data from OKX public API"""
        try:
            print("📚 Loading historical data from OKX...")
            
            url = f"{self.base_url}/api/v5/market/candles"
            params = {
                'instId': self.symbol,
                'bar': '1m',
                'limit': '200'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == '0' and data.get('data'):
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
    
    def _use_enhanced_sample_data(self):
        """Use enhanced sample data with realistic price movements for testing"""
        print("🧪 Using enhanced sample data for INSANE MODE testing...")
        
        # Start with realistic SOL price
        base_price = 142.0
        current_price = base_price
        
        # Generate realistic market data
        for i in range(100):
            # Create realistic price movements
            trend = np.sin(i * 0.1) * 0.5  # Gentle trending
            noise = np.random.normal(0, 0.3)  # Random noise
            momentum = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]) * 0.5  # Occasional momentum
            
            price_change = trend + noise + momentum
            current_price = max(current_price + price_change, 50)  # Don't go below $50
            
            # Create realistic OHLC
            open_price = current_price - price_change
            high_price = max(open_price, current_price) + abs(np.random.normal(0, 0.2))
            low_price = min(open_price, current_price) - abs(np.random.normal(0, 0.2))
            
            candle_data = {
                'timestamp': int(time.time() * 1000) + i * 60000,
                'datetime': datetime.now() + timedelta(minutes=i),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': current_price,
                'volume': np.random.uniform(800, 2000) * (1 + abs(price_change))  # Higher volume on big moves
            }
            
            self.candle_buffer.append(candle_data)
            
            for callback in self.callback_functions:
                callback(candle_data)
        
        self.latest_price = current_price
        print(f"✅ Generated {len(self.candle_buffer)} enhanced sample candles")
        print(f"💰 Sample {self.symbol} price: ${self.latest_price:.4f}")
        
        # Start realistic price updates
        self._start_enhanced_updates()
    
    def _start_enhanced_updates(self):
        """Start enhanced sample price updates with realistic patterns"""
        def update_enhanced_price():
            update_count = 0
            trend_direction = 1
            trend_strength = 0
            
            while self.is_running:
                try:
                    time.sleep(5)  # Update every 5 seconds for faster testing
                    update_count += 1
                    
                    if self.latest_price:
                        # Create realistic market patterns
                        if update_count % 20 == 0:  # Change trend occasionally
                            trend_direction = np.random.choice([-1, 1])
                            trend_strength = np.random.uniform(0.1, 0.5)
                        
                        # Create price movement
                        trend_component = trend_direction * trend_strength * 0.1
                        random_component = np.random.normal(0, 0.2)
                        
                        # Occasional strong moves for RSI extremes
                        if np.random.random() < 0.05:  # 5% chance
                            strong_move = np.random.choice([-1, 1]) * np.random.uniform(1, 3)
                            random_component += strong_move
                        
                        price_change = trend_component + random_component
                        new_price = max(self.latest_price + price_change, 50)
                        
                        # Create realistic OHLC
                        open_price = self.latest_price
                        high_price = max(open_price, new_price) + abs(np.random.normal(0, 0.1))
                        low_price = min(open_price, new_price) - abs(np.random.normal(0, 0.1))
                        
                        candle_data = {
                            'timestamp': int(time.time() * 1000),
                            'datetime': datetime.now(),
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': new_price,
                            'volume': np.random.uniform(1000, 3000) * (1 + abs(price_change))
                        }
                        
                        self.candle_buffer.append(candle_data)
                        if len(self.candle_buffer) > 500:
                            self.candle_buffer.pop(0)
                        
                        self.latest_price = new_price
                        
                        print(f"💹 Enhanced update: ${self.latest_price:.4f} (Δ{price_change:+.2f})")
                        
                        for callback in self.callback_functions:
                            callback(candle_data)
                            
                except Exception as e:
                    print(f"❌ Error in enhanced updates: {e}")
        
        update_thread = threading.Thread(target=update_enhanced_price, daemon=True)
        update_thread.start()
    
    def _start_websocket(self):
        """Start WebSocket for live price updates"""
        def run_websocket():
            try:
                ws_url = "wss://ws.okx.com:8443/ws/v5/public"
                
                def on_message(ws, message):
                    try:
                        data = json.loads(message)
                        
                        if data.get('event') == 'subscribe':
                            print("✅ WebSocket subscription confirmed")
                            self.reconnect_count = 0
                            return
                        
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
                                
                                self.candle_buffer.append(candle_data)
                                if len(self.candle_buffer) > 500:
                                    self.candle_buffer.pop(0)
                                
                                self.latest_price = candle_data['close']
                                
                                print(f"💹 Live update: ${self.latest_price:.4f}")
                                
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
                        print(f"🔄 Reconnecting... ({self.reconnect_count}/{self.max_reconnects})")
                        time.sleep(5)
                        self._start_websocket()
                    else:
                        print("❌ Max reconnects reached, using sample data...")
                        self._use_enhanced_sample_data()
                
                def on_open(ws):
                    print("✅ WebSocket connected to OKX")
                    
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
                
                self.ws.run_forever(ping_interval=60, ping_timeout=10)
                
            except Exception as e:
                print(f"❌ WebSocket thread error: {e}")
                if self.is_running:
                    print("🔄 Falling back to sample data...")
                    self._use_enhanced_sample_data()
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
    
    def get_latest_price(self) -> float:
        """Get latest price"""
        return self.latest_price if self.latest_price else 142.0
    
    def get_candle_data(self, limit: int = 100) -> List[Dict]:
        """Get recent candle data"""
        return self.candle_buffer[-limit:] if len(self.candle_buffer) >= limit else self.candle_buffer

class InsaneModeBot:
    """Trading bot with all 4 risk modes including AI-powered Insane Mode"""
    
    def __init__(self, initial_balance: float = 200.0):
        print("🔥🧠💀 INSANE MODE TRADING BOT INITIALIZED!")
        print("💰 Starting with $200 - Risk adjusts dynamically!")
        print("🤖 AI-powered analysis for extreme leverage trading!")
        
        # Initialize components
        self.risk_manager = DynamicRiskManager()
        self.ai_analyzer = AITradeAnalyzer()
        self.initial_balance = initial_balance
        
        # Get user's risk preference (now includes Insane Mode)
        self._select_risk_mode()
        
        # Initialize trading components
        self.data_feed = InsaneModeDataFeed()
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
        
        # Insane Mode specific
        self.is_insane_mode = self.risk_manager.current_profile.name == "INSANE MODE 🔥🧠💀"
        self.ai_trade_count = 0
        self.ai_success_count = 0
        self.pending_ai_trades = {}  # Track trades for learning feedback
        
        print(f"🤖 Insane Mode Bot initialized with ${initial_balance:,.2f}")
        print(f"🎯 Risk Mode: {self.risk_manager.current_profile.name}")
        if self.is_insane_mode:
            print("🧠 AI Analysis System: ACTIVATED")
            print("⚡ Dynamic Leverage: 30x-50x based on AI confidence")
    
    def _select_risk_mode(self):
        """Let user select risk mode including new Insane Mode"""
        while True:
            choice = self.risk_manager.display_risk_options(self.initial_balance)
            if self.risk_manager.select_risk_mode(choice, self.initial_balance):
                break
        
        print(f"\n🎉 Ready to trade with {self.risk_manager.current_profile.name}!")
    
    def start(self):
        """Start the Insane Mode trading bot"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        print("\n🚀 Starting INSANE MODE Trading Bot")
        print("=" * 80)
        print(f"💰 Starting Balance: ${self.initial_balance:,.2f}")
        print(f"🎯 Risk Mode: {self.risk_manager.current_profile.name}")
        
        # Show initial trading parameters
        params = self.risk_manager.get_trading_params(self.initial_balance)
        print(f"💼 Position Size: ${params['position_size_usd']:.2f} ({params['position_size_pct']}% of account)")
        print(f"⚖️  Leverage: {params['leverage']}x")
        print(f"🎯 RSI Thresholds: Buy<{params['rsi_oversold']} Sell>{params['rsi_overbought']}")
        
        if self.is_insane_mode:
            print("🧠 AI Analysis: ACTIVE - Only 90%+ confidence trades executed")
            print("⚡ Dynamic Leverage: 30x-50x based on AI assessment")
            print("🎯 Quality Focus: Max 8 trades/day for precision")
        
        print("=" * 80)
        
        # Setup data callback
        self.data_feed.add_callback(self._on_new_candle)
        
        # Start data feed
        self.data_feed.start()
        
        # Start monitoring
        self._start_monitoring()
        
        mode_name = "INSANE MODE" if self.is_insane_mode else "STANDARD"
        print(f"🎯 {mode_name} Bot running with dynamic risk management!")
        print("📈 Position sizes adjust automatically as balance changes!")
        if self.is_insane_mode:
            print("🧠 AI filtering ensures only high-probability trades!")
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
        
        if self.is_insane_mode and self.ai_trade_count > 0:
            ai_accuracy = (self.ai_success_count / self.ai_trade_count) * 100
            print(f"🧠 AI Performance: {ai_accuracy:.1f}% accuracy ({self.ai_success_count}/{self.ai_trade_count})")
        
        print("✅ Insane Mode bot stopped successfully")
    
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
            min_data_required = 50 if not self.is_insane_mode else 80  # More data for AI analysis
            if len(self.data_buffer) >= min_data_required:
                self.data_buffer = self.indicators.calculate_all_indicators(self.data_buffer)
                
                # Check for trading signals
                if self.is_insane_mode:
                    self._check_insane_mode_signals(candle)
                else:
                    self._check_standard_signals(candle)
            
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
        
        return time_since_last >= cooldown
    
    def _check_standard_signals(self, current_candle: Dict):
        """Check for standard trading signals (Safe, Risk, Super Risky modes)"""
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
                    self.trader.close_position(
                        price=price,
                        reason=f"RSI overbought: {rsi:.1f}"
                    )
                    
                    self.daily_trades += 1
                    self.last_signal_time = datetime.now()
                    
                    print(f"🔴 SELL! ${price:.4f} | RSI: {rsi:.1f}")
                else:
                    print(f"🚫 SELL BLOCKED: {reason}")
            
        except Exception as e:
            print(f"❌ Error checking standard signals: {e}")
    
    def _check_insane_mode_signals(self, current_candle: Dict):
        """Check for INSANE MODE signals with AI analysis"""
        try:
            if len(self.data_buffer) < 80:  # Need more data for AI
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
            
            rsi_oversold = params['rsi_oversold']  # 20.0 for insane mode
            rsi_overbought = params['rsi_overbought']  # 80.0 for insane mode
            
            print(f"🧠 AI Scanning: RSI {rsi:.1f} | Need: <{rsi_oversold} or >{rsi_overbought}")
            
            # BUY SIGNAL - AI ANALYSIS
            if rsi < rsi_oversold and not self.trader.has_position():
                print("🔍 AI analyzing BUY opportunity...")
                
                # Perform AI analysis
                ai_result = self.ai_analyzer.analyze_trade_opportunity(
                    self.data_buffer, price, 'buy'
                )
                
                if ai_result['trade_approved']:
                    # Use dynamic leverage from AI
                    dynamic_leverage = ai_result['dynamic_leverage']
                    position_size = params['position_size_usd']
                    
                    # Apply dynamic leverage to position calculation
                    leveraged_position = position_size * (dynamic_leverage / params['leverage'])
                    
                    should_execute, reason = self.risk_manager.should_execute_trade(
                        ai_result['ai_confidence'], self.daily_trades, current_balance, self.initial_balance
                    )
                    
                    if should_execute:
                        self.trader.execute_trade(
                            side='buy',
                            price=price,
                            size=leveraged_position,
                            reason=f"AI BUY: {ai_result['ai_confidence']:.1f}% confidence | {dynamic_leverage}x leverage",
                            confidence=ai_result['ai_confidence']
                        )
                        
                        self.daily_trades += 1
                        self.ai_trade_count += 1
                        self.last_signal_time = datetime.now()
                        
                        # Track this trade for learning
                        trade_id = f"buy_{datetime.now().timestamp()}"
                        self.pending_ai_trades[trade_id] = {
                            'confidence': ai_result['ai_confidence'],
                            'entry_price': price,
                            'timestamp': datetime.now()
                        }
                        
                        print(f"🔥🧠 AI BUY EXECUTED! ${price:.4f}")
                        print(f"🎯 AI Confidence: {ai_result['ai_confidence']:.1f}%")
                        print(f"⚡ Dynamic Leverage: {dynamic_leverage}x")
                        print(f"💰 Position Size: ${leveraged_position:.2f}")
                        
                        # Show AI reasoning
                        for reason in ai_result['recommendation']['reasoning']:
                            print(f"   {reason}")
                    else:
                        print(f"🚫 AI BUY BLOCKED: {reason}")
                else:
                    print(f"❌ AI REJECTED BUY:")
                    for reason in ai_result['recommendation']['reasoning']:
                        print(f"   {reason}")
            
            # SELL SIGNAL - AI ANALYSIS
            elif rsi > rsi_overbought and self.trader.has_position():
                print("🔍 AI analyzing SELL opportunity...")
                
                # Perform AI analysis
                ai_result = self.ai_analyzer.analyze_trade_opportunity(
                    self.data_buffer, price, 'sell'
                )
                
                if ai_result['trade_approved']:
                    should_execute, reason = self.risk_manager.should_execute_trade(
                        ai_result['ai_confidence'], self.daily_trades, current_balance, self.initial_balance
                    )
                    
                    if should_execute:
                        pnl_before = self.trader.get_statistics()['total_pnl']
                        
                        self.trader.close_position(
                            price=price,
                            reason=f"AI SELL: {ai_result['ai_confidence']:.1f}% confidence"
                        )
                        
                        pnl_after = self.trader.get_statistics()['total_pnl']
                        trade_pnl = pnl_after - pnl_before
                        
                        self.daily_trades += 1
                        self.last_signal_time = datetime.now()
                        
                        # Update AI learning with completed trade
                        if self.pending_ai_trades:
                            # Get the oldest pending trade (FIFO)
                            trade_id = list(self.pending_ai_trades.keys())[0]
                            trade_info = self.pending_ai_trades.pop(trade_id)
                            
                            # Feed result back to AI
                            if trade_pnl > 0:
                                self.ai_success_count += 1
                                self.ai_analyzer.update_trade_result(trade_info['confidence'], 'win')
                                print(f"🧠 AI Learning: WIN recorded (confidence: {trade_info['confidence']:.1f}%)")
                            else:
                                self.ai_analyzer.update_trade_result(trade_info['confidence'], 'loss')
                                print(f"🧠 AI Learning: LOSS recorded (confidence: {trade_info['confidence']:.1f}%)")
                        else:
                            # Fallback if no pending trades
                            if trade_pnl > 0:
                                self.ai_success_count += 1
                        
                        print(f"🔥🧠 AI SELL EXECUTED! ${price:.4f}")
                        print(f"🎯 AI Confidence: {ai_result['ai_confidence']:.1f}%")
                        print(f"💰 Trade P&L: ${trade_pnl:.2f}")
                        
                        # Show AI reasoning
                        for reason in ai_result['recommendation']['reasoning']:
                            print(f"   {reason}")
                    else:
                        print(f"🚫 AI SELL BLOCKED: {reason}")
                else:
                    print(f"❌ AI REJECTED SELL:")
                    for reason in ai_result['recommendation']['reasoning']:
                        print(f"   {reason}")
            
        except Exception as e:
            print(f"❌ Error checking INSANE MODE signals: {e}")
    
    def _start_monitoring(self):
        """Start performance monitoring"""
        def monitoring_loop():
            while self.is_running:
                try:
                    sleep_time = 90 if self.is_insane_mode else 60  # Less frequent for AI mode
                    time.sleep(sleep_time)
                    
                    if datetime.now() - self.last_status_update > timedelta(seconds=sleep_time):
                        self._print_status()
                        self.last_status_update = datetime.now()
                        
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def _print_status(self):
        """Print comprehensive status including AI metrics"""
        try:
            stats = self.trader.get_statistics()
            current_balance = stats['balance']
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            current_price = self.data_feed.get_latest_price()
            
            risk_metrics = self.risk_manager.get_dynamic_risk_metrics(current_balance, self.initial_balance)
            
            balance_change = current_balance - self.initial_balance
            balance_change_pct = (balance_change / self.initial_balance) * 100
            
            print("\n" + "=" * 80)
            if self.is_insane_mode:
                print("🔥🧠💀 INSANE MODE BOT STATUS")
            else:
                print("📊 TRADING BOT STATUS")
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
            
            if self.is_insane_mode:
                ai_accuracy = (self.ai_success_count / self.ai_trade_count * 100) if self.ai_trade_count > 0 else 0
                print(f"🧠 AI Performance: {ai_accuracy:.1f}% accuracy ({self.ai_success_count}/{self.ai_trade_count})")
                ai_stats = self.ai_analyzer.get_ai_performance_stats()
                print(f"🤖 AI Total Predictions: {ai_stats['total_predictions']} | Accuracy: {ai_stats['accuracy_rate']:.1f}%")
            
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ Status error: {e}")

def main():
    """Main entry point for Insane Mode bot"""
    print("🔥🧠💀 INSANE MODE OKX Trading Bot")
    print("💰 Starting with $200 - All 4 risk modes available!")
    print("🤖 AI-powered analysis for extreme leverage trading!")
    print("📡 Uses REAL market data from OKX")
    print("⚡ Dynamic leverage 30x-50x for Insane Mode!")
    
    bot = InsaneModeBot(initial_balance=200.0)
    
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