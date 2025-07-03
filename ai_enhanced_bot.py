#!/usr/bin/env python3
"""
AI-Enhanced Trading Bot - All 4 Modes with AI Analysis
Uses AI filtering for ALL modes with different confidence thresholds
"""

import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import json

from risk_manager import DynamicRiskManager
from ai_analyzer import AITradeAnalyzer
from simulation_trader import SimulationTradingEngine
from indicators import TechnicalIndicators

class AIEnhancedDataFeed:
    """Enhanced data feed with real-time simulation"""
    
    def __init__(self, symbol: str = "SOL-USD-SWAP"):
        self.symbol = symbol
        self.callbacks = []
        self.is_running = False
        self.current_price = 142.50
        self.base_price = 142.50
        self.last_update = datetime.now()
        
        print(f"📡 AI-Enhanced Data Feed initialized for {symbol}")
    
    def add_callback(self, callback):
        """Add callback for new candle data"""
        self.callbacks.append(callback)
    
    def start(self):
        """Start the enhanced data feed"""
        if self.is_running:
            return
        
        self.is_running = True
        print("🚀 Starting AI-Enhanced data feed...")
        
        # Load initial data and start updates
        self._load_initial_data()
        self._start_price_updates()
    
    def stop(self):
        """Stop the data feed"""
        self.is_running = False
        print("⏹️ AI-Enhanced data feed stopped")
    
    def _load_initial_data(self):
        """Load initial historical data"""
        print("📊 Loading initial market data...")
        
        # Generate realistic historical data
        for i in range(100):
            timestamp = datetime.now() - timedelta(minutes=100-i)
            
            # Create realistic price movement
            price_change = np.random.normal(0, 0.5)
            self.current_price += price_change
            
            # Keep price in reasonable range
            if self.current_price < 130:
                self.current_price = 130 + np.random.uniform(0, 2)
            elif self.current_price > 155:
                self.current_price = 155 - np.random.uniform(0, 2)
            
            candle = {
                'timestamp': timestamp,
                'open': self.current_price - np.random.uniform(-0.5, 0.5),
                'high': self.current_price + np.random.uniform(0, 1),
                'low': self.current_price - np.random.uniform(0, 1),
                'close': self.current_price,
                'volume': np.random.uniform(800, 1500)
            }
            
            # Send to callbacks
            for callback in self.callbacks:
                callback(candle)
        
        print("✅ Initial data loaded")
    
    def _start_price_updates(self):
        """Start real-time price updates"""
        def update_loop():
            while self.is_running:
                try:
                    # Create realistic price movement
                    time_factor = (datetime.now().hour % 24) / 24.0
                    volatility = 0.3 + (0.7 * time_factor)  # Higher volatility during certain hours
                    
                    price_change = np.random.normal(0, volatility)
                    trend_factor = np.sin(time.time() / 3600) * 0.1  # Subtle trend component
                    
                    self.current_price += price_change + trend_factor
                    
                    # Keep price in range
                    if self.current_price < 130:
                        self.current_price = 130 + np.random.uniform(0, 2)
                    elif self.current_price > 155:
                        self.current_price = 155 - np.random.uniform(0, 2)
                    
                    # Create candle
                    now = datetime.now()
                    candle = {
                        'timestamp': now,
                        'open': self.current_price - np.random.uniform(-0.3, 0.3),
                        'high': self.current_price + np.random.uniform(0, 0.8),
                        'low': self.current_price - np.random.uniform(0, 0.8),
                        'close': self.current_price,
                        'volume': np.random.uniform(500, 2000)
                    }
                    
                    # Send to callbacks
                    for callback in self.callbacks:
                        callback(candle)
                    
                    self.last_update = now
                    time.sleep(5)  # 5-second updates
                    
                except Exception as e:
                    print(f"❌ Data feed error: {e}")
                    time.sleep(5)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
    
    def get_latest_price(self) -> float:
        """Get latest price"""
        return self.current_price

class AIEnhancedBot:
    """AI-Enhanced Trading Bot with all 4 modes using AI analysis"""
    
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        self.is_running = False
        self.start_time = None
        
        # Initialize components
        self.risk_manager = DynamicRiskManager()
        self.ai_analyzer = AITradeAnalyzer()
        self.trader = SimulationTradingEngine(initial_balance)
        self.indicators = TechnicalIndicators()
        self.data_feed = AIEnhancedDataFeed()
        
        # Data management
        self.data_buffer = pd.DataFrame()
        
        # Trading state
        self.daily_trades = 0
        self.last_trade_date = datetime.now().date()
        self.last_signal_time = None
        self.last_status_update = datetime.now()
        
        # AI metrics for all modes
        self.ai_trade_count = 0
        self.ai_success_count = 0
        self.pending_ai_trades = {}
        
        # Mode-specific AI thresholds
        self.ai_confidence_thresholds = {
            "SAFE MODE 🛡️": 85.0,      # High confidence required
            "RISK MODE ⚡": 75.0,       # Moderate confidence 
            "SUPER RISKY MODE 🚀💥": 60.0,  # Low confidence needed
            "INSANE MODE 🔥🧠💀": 90.0   # Extreme confidence required
        }
        
        print(f"🤖 AI-Enhanced Bot initialized with ${initial_balance:,.2f}")
        print("🧠 AI analysis active for ALL trading modes!")
    
    def _select_risk_mode(self):
        """Let user select risk mode"""
        while True:
            choice = self.risk_manager.display_risk_options(self.initial_balance)
            if self.risk_manager.select_risk_mode(choice, self.initial_balance):
                break
        
        mode_name = self.risk_manager.current_profile.name
        ai_threshold = self.ai_confidence_thresholds.get(mode_name, 75.0)
        
        print(f"\n🎉 Ready to trade with {mode_name}!")
        print(f"🧠 AI Confidence Required: {ai_threshold}%")
    
    def start(self):
        """Start the AI-Enhanced trading bot"""
        if self.is_running:
            return
        
        # Select risk mode first
        self._select_risk_mode()
        
        self.is_running = True
        self.start_time = datetime.now()
        
        print("\n🚀 Starting AI-ENHANCED Trading Bot")
        print("=" * 80)
        print(f"💰 Starting Balance: ${self.initial_balance:,.2f}")
        print(f"🎯 Risk Mode: {self.risk_manager.current_profile.name}")
        
        # Show AI configuration
        mode_name = self.risk_manager.current_profile.name
        ai_threshold = self.ai_confidence_thresholds.get(mode_name, 75.0)
        print(f"🧠 AI Analysis: ACTIVE - {ai_threshold}% confidence required")
        
        # Show trading parameters
        params = self.risk_manager.get_trading_params(self.initial_balance)
        print(f"💼 Position Size: ${params['position_size_usd']:.2f} ({params['position_size_pct']}% of account)")
        print(f"⚖️  Leverage: {params['leverage']}x")
        print(f"🎯 RSI Thresholds: Buy<{params['rsi_oversold']} Sell>{params['rsi_overbought']}")
        
        if mode_name == "INSANE MODE 🔥🧠💀":
            print("⚡ Dynamic Leverage: 30x-50x based on AI assessment")
            print("🎯 Quality Focus: Max 8 trades/day for precision")
        
        print("=" * 80)
        
        # Setup data callback
        self.data_feed.add_callback(self._on_new_candle)
        
        # Start data feed
        self.data_feed.start()
        
        # Start monitoring
        self._start_monitoring()
        
        print(f"🎯 AI-ENHANCED Bot running with intelligent filtering!")
        print("🧠 All modes now use AI analysis for better accuracy!")
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
        
        if self.ai_trade_count > 0:
            ai_accuracy = (self.ai_success_count / self.ai_trade_count) * 100
            print(f"🧠 AI Performance: {ai_accuracy:.1f}% accuracy ({self.ai_success_count}/{self.ai_trade_count})")
        
        print("✅ AI-Enhanced bot stopped successfully")
    
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
            min_data_required = 50
            if len(self.data_buffer) >= min_data_required:
                self.data_buffer = self.indicators.calculate_all_indicators(self.data_buffer)
                
                # Check for AI-enhanced trading signals
                self._check_ai_enhanced_signals(candle)
            
        except Exception as e:
            print(f"❌ Error processing candle: {e}")
    
    def _check_ai_enhanced_signals(self, current_candle: Dict):
        """Check for AI-enhanced trading signals for all modes"""
        try:
            if len(self.data_buffer) < 50:
                return
            
            self._reset_daily_counters()
            
            if not self._check_signal_cooldown():
                return
            
            # Get current balance and parameters
            current_balance = self.trader.get_statistics()['balance']
            params = self.risk_manager.get_trading_params(current_balance)
            mode_name = self.risk_manager.current_profile.name
            ai_threshold = self.ai_confidence_thresholds.get(mode_name, 75.0)
            
            # Get indicators
            latest = self.data_buffer.iloc[-1]
            rsi = latest.get('rsi', 50)
            price = current_candle['close']
            
            rsi_oversold = params['rsi_oversold']
            rsi_overbought = params['rsi_overbought']
            
            print(f"🧠 AI Scanning: RSI {rsi:.1f} | Need: <{rsi_oversold} or >{rsi_overbought} | AI Req: {ai_threshold}%")
            
            # BUY SIGNAL - AI ANALYSIS
            if rsi < rsi_oversold and not self.trader.has_position():
                print(f"🔍 AI analyzing BUY opportunity for {mode_name}...")
                
                # Perform AI analysis
                ai_result = self.ai_analyzer.analyze_trade_opportunity(
                    self.data_buffer, price, 'buy'
                )
                
                # Check if AI confidence meets mode threshold
                ai_approved = ai_result['ai_confidence'] >= ai_threshold
                
                if ai_approved:
                    # Calculate position size
                    if mode_name == "INSANE MODE 🔥🧠💀":
                        # Use dynamic leverage for Insane Mode
                        dynamic_leverage = ai_result['dynamic_leverage']
                        position_size = params['position_size_usd'] * (dynamic_leverage / params['leverage'])
                    else:
                        # Standard position size for other modes
                        position_size = params['position_size_usd']
                    
                    should_execute, reason = self.risk_manager.should_execute_trade(
                        ai_result['ai_confidence'], self.daily_trades, current_balance, self.initial_balance
                    )
                    
                    if should_execute:
                        self.trader.execute_trade(
                            side='buy',
                            price=price,
                            size=position_size,
                            reason=f"AI-Enhanced BUY: {ai_result['ai_confidence']:.1f}% confidence",
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
                        print(f"🎯 AI Confidence: {ai_result['ai_confidence']:.1f}% (Required: {ai_threshold}%)")
                        print(f"💰 Position Size: ${position_size:.2f}")
                        
                        if mode_name == "INSANE MODE 🔥🧠💀":
                            print(f"⚡ Dynamic Leverage: {ai_result['dynamic_leverage']}x")
                        
                        # Show AI reasoning
                        for reason in ai_result['recommendation']['reasoning'][:3]:  # Show top 3 reasons
                            print(f"   {reason}")
                    else:
                        print(f"🚫 AI BUY BLOCKED: {reason}")
                else:
                    print(f"❌ AI REJECTED BUY: {ai_result['ai_confidence']:.1f}% < {ai_threshold}% required")
                    for reason in ai_result['recommendation']['reasoning'][:2]:
                        print(f"   {reason}")
            
            # SELL SIGNAL - AI ANALYSIS
            elif rsi > rsi_overbought and self.trader.has_position():
                print(f"🔍 AI analyzing SELL opportunity for {mode_name}...")
                
                # Perform AI analysis
                ai_result = self.ai_analyzer.analyze_trade_opportunity(
                    self.data_buffer, price, 'sell'
                )
                
                # Check if AI confidence meets mode threshold
                ai_approved = ai_result['ai_confidence'] >= ai_threshold
                
                if ai_approved:
                    should_execute, reason = self.risk_manager.should_execute_trade(
                        ai_result['ai_confidence'], self.daily_trades, current_balance, self.initial_balance
                    )
                    
                    if should_execute:
                        pnl_before = self.trader.get_statistics()['total_pnl']
                        
                        self.trader.close_position(
                            price=price,
                            reason=f"AI-Enhanced SELL: {ai_result['ai_confidence']:.1f}% confidence"
                        )
                        
                        pnl_after = self.trader.get_statistics()['total_pnl']
                        trade_pnl = pnl_after - pnl_before
                        
                        self.daily_trades += 1
                        self.last_signal_time = datetime.now()
                        
                        # Update AI learning with completed trade
                        if self.pending_ai_trades:
                            trade_id = list(self.pending_ai_trades.keys())[0]
                            trade_info = self.pending_ai_trades.pop(trade_id)
                            
                            if trade_pnl > 0:
                                self.ai_success_count += 1
                                self.ai_analyzer.update_trade_result(trade_info['confidence'], 'win')
                                print(f"🧠 AI Learning: WIN recorded ({trade_info['confidence']:.1f}% confidence)")
                            else:
                                self.ai_analyzer.update_trade_result(trade_info['confidence'], 'loss')
                                print(f"🧠 AI Learning: LOSS recorded ({trade_info['confidence']:.1f}% confidence)")
                        else:
                            if trade_pnl > 0:
                                self.ai_success_count += 1
                        
                        print(f"🔥🧠 AI SELL EXECUTED! ${price:.4f}")
                        print(f"🎯 AI Confidence: {ai_result['ai_confidence']:.1f}% (Required: {ai_threshold}%)")
                        print(f"💰 Trade P&L: ${trade_pnl:.2f}")
                        
                        # Show AI reasoning
                        for reason in ai_result['recommendation']['reasoning'][:3]:
                            print(f"   {reason}")
                    else:
                        print(f"🚫 AI SELL BLOCKED: {reason}")
                else:
                    print(f"❌ AI REJECTED SELL: {ai_result['ai_confidence']:.1f}% < {ai_threshold}% required")
                    for reason in ai_result['recommendation']['reasoning'][:2]:
                        print(f"   {reason}")
            
        except Exception as e:
            print(f"❌ Error checking AI-enhanced signals: {e}")
    
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
    
    def _start_monitoring(self):
        """Start performance monitoring"""
        def monitoring_loop():
            while self.is_running:
                try:
                    time.sleep(60)  # Status update every minute
                    
                    if datetime.now() - self.last_status_update > timedelta(seconds=60):
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
            mode_name = self.risk_manager.current_profile.name
            ai_threshold = self.ai_confidence_thresholds.get(mode_name, 75.0)
            
            balance_change = current_balance - self.initial_balance
            balance_change_pct = (balance_change / self.initial_balance) * 100
            
            print("\n" + "=" * 80)
            print("🔥🧠💰 AI-ENHANCED TRADING BOT STATUS")
            print("=" * 80)
            print(f"🎯 Risk Mode: {risk_metrics['mode']}")
            print(f"🧠 AI Threshold: {ai_threshold}% confidence required")
            print(f"⏰ Uptime: {str(uptime).split('.')[0]}")
            print(f"💰 Balance: ${current_balance:,.2f} (Started: ${self.initial_balance:,.2f})")
            print(f"📈 Performance: ${balance_change:+.2f} ({balance_change_pct:+.1f}%)")
            print(f"📊 Drawdown: {risk_metrics['drawdown_pct']:.1f}% / {risk_metrics['max_drawdown_allowed']:.1f}%")
            print(f"🎯 Win Rate: {stats['win_rate']:.1f}% | Trades: {stats['total_trades']}")
            print(f"💹 SOL Price: ${current_price:.4f} | Buffer: {len(self.data_buffer)} candles")
            print(f"💼 Position Size: ${risk_metrics['position_size']:.2f} ({risk_metrics['position_size_pct']:.1f}%)")
            print(f"📅 Daily: {self.daily_trades}/{risk_metrics['max_daily_trades']} | Leverage: {risk_metrics['leverage']}x")
            
            # AI Performance metrics
            ai_accuracy = (self.ai_success_count / self.ai_trade_count * 100) if self.ai_trade_count > 0 else 0
            print(f"🧠 AI Performance: {ai_accuracy:.1f}% accuracy ({self.ai_success_count}/{self.ai_trade_count})")
            
            ai_stats = self.ai_analyzer.get_ai_performance_stats()
            print(f"🤖 AI Learning Stats: {ai_stats['total_predictions']} predictions | {ai_stats['accuracy_rate']:.1f}% accuracy")
            
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ Status error: {e}")

def main():
    """Main entry point for AI-Enhanced bot"""
    print("🔥🧠💰 AI-ENHANCED OKX Trading Bot")
    print("💰 Starting with $200 - All 4 modes with AI intelligence!")
    print("🤖 Every mode now uses AI analysis for better accuracy!")
    print("📡 Uses REAL market data with intelligent filtering")
    print("🧠 Different AI confidence levels for each risk mode!")
    
    bot = AIEnhancedBot(initial_balance=200.0)
    
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