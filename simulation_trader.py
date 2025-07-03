#!/usr/bin/env python3
"""
Simulation Trading Engine for OKX Trading Bot
Generates realistic trading activity for simulation mode
"""

import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random
from simulation_data_generator import SimulationDataGenerator
from strategy import TradingSignal, AdvancedTradingStrategy
from indicators import TechnicalIndicators

class SimulationPosition:
    """Simulated trading position"""
    
    def __init__(self, side: str, size: float, entry_price: float, 
                 stop_loss: float = None, take_profit: float = None):
        self.side = side  # 'long' or 'short'
        self.size = size
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = datetime.now()
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.id = f"{side}_{int(self.entry_time.timestamp())}"
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
            self.unrealized_pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
            self.unrealized_pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
    
    def should_close(self, current_price: float) -> str:
        """Check if position should be closed"""
        if self.stop_loss and (
            (self.side == 'long' and current_price <= self.stop_loss) or
            (self.side == 'short' and current_price >= self.stop_loss)
        ):
            return 'stop_loss'
        
        if self.take_profit and (
            (self.side == 'long' and current_price >= self.take_profit) or
            (self.side == 'short' and current_price <= self.take_profit)
        ):
            return 'take_profit'
        
        return None

class SimulationTradingEngine:
    """Simulation trading engine that generates realistic trading activity"""
    
    def __init__(self, initial_balance: float = 10000.0):
        # Trading state
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: List[SimulationPosition] = []
        self.closed_trades = []
        
        # Trading parameters
        self.position_size_usd = 100.0
        self.leverage = 10
        self.max_positions = 3
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        
        # Market data
        self.data_generator = SimulationDataGenerator()
        self.strategy = AdvancedTradingStrategy()
        self.price_history = []
        self.current_price = 150.0
        
        # Threading
        self.is_running = False
        self.trading_thread = None
        self.callback_function = None
        
        # Trading logic
        self.last_signal_time = None
        self.signal_cooldown = 30  # seconds between signals
        
    def set_callback(self, callback):
        """Set callback for trade updates"""
        self.callback_function = callback
    
    def start_simulation(self):
        """Start the simulation trading engine"""
        if self.is_running:
            return
        
        self.is_running = True
        print("🎮 Starting simulation trading engine...")
        print(f"💰 Initial balance: ${self.balance:,.2f}")
        print(f"📊 Position size: ${self.position_size_usd}")
        print(f"⚖️  Leverage: {self.leverage}x")
        
        # Start data generator
        self.data_generator.set_callback(self._on_new_candle)
        self.data_generator.start_simulation()
        
        # Start trading logic
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.data_generator.stop_simulation()
        if self.trading_thread:
            self.trading_thread.join()
        print("🛑 Simulation trading engine stopped")
    
    def _on_new_candle(self, candle: dict):
        """Process new market data"""
        self.current_price = candle['close']
        self.price_history.append(candle)
        
        # Keep only last 200 candles for indicators
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-200:]
        
        # Update existing positions
        self._update_positions()
        
        # Check for position closures
        self._check_position_closures()
    
    def _trading_loop(self):
        """Main trading logic loop"""
        while self.is_running:
            try:
                if len(self.price_history) >= 50:  # Need enough data for indicators
                    self._evaluate_trading_signals()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(10)
    
    def _evaluate_trading_signals(self):
        """Evaluate potential trading signals"""
        # Check cooldown
        if self.last_signal_time:
            time_since_last = (datetime.now() - self.last_signal_time).total_seconds()
            if time_since_last < self.signal_cooldown:
                return
        
        # Don't open new positions if we have too many
        if len(self.positions) >= self.max_positions:
            return
        
        # Convert price history to DataFrame
        df = pd.DataFrame(self.price_history)
        
        # Add technical indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        # Generate signal using strategy
        signal = self.strategy.generate_signal(df)
        
        if signal and signal.confidence > 0.6:  # 60% minimum confidence
            self._execute_signal(signal)
    
    def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal"""
        try:
            # Check if we already have a position in this direction
            existing_positions = [p for p in self.positions if p.side == signal.signal_type]
            if existing_positions:
                return  # Don't add to existing position
            
            # Calculate position size
            position_size = self.position_size_usd / self.current_price
            
            # Create position
            position = SimulationPosition(
                side=signal.signal_type,
                size=position_size,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            self.positions.append(position)
            self.last_signal_time = datetime.now()
            
            print(f"📈 POSITION OPENED: {signal.signal_type.upper()} "
                  f"${position_size:.4f} @ ${signal.entry_price:.4f} "
                  f"(confidence: {signal.confidence:.1%})")
            
            # Log trade for metrics
            self._log_trade_open(position, signal)
            
        except Exception as e:
            print(f"Error executing signal: {e}")
    
    def _update_positions(self):
        """Update all open positions"""
        for position in self.positions:
            position.update_pnl(self.current_price)
    
    def _check_position_closures(self):
        """Check if any positions should be closed"""
        positions_to_close = []
        
        for position in self.positions:
            close_reason = position.should_close(self.current_price)
            if close_reason:
                positions_to_close.append((position, close_reason))
        
        # Close positions
        for position, reason in positions_to_close:
            self._close_position(position, reason)
    
    def _close_position(self, position: SimulationPosition, reason: str):
        """Close a position"""
        try:
            # Calculate final P&L
            position.update_pnl(self.current_price)
            final_pnl = position.unrealized_pnl
            
            # Update statistics
            self.total_trades += 1
            self.total_pnl += final_pnl
            self.balance += final_pnl
            
            if final_pnl > 0:
                self.winning_trades += 1
            
            # Update drawdown
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            
            current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Remove from active positions
            self.positions.remove(position)
            
            # Add to closed trades
            trade_record = {
                'id': position.id,
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'exit_price': self.current_price,
                'entry_time': position.entry_time,
                'exit_time': datetime.now(),
                'pnl': final_pnl,
                'pnl_pct': position.unrealized_pnl_pct,
                'close_reason': reason
            }
            self.closed_trades.append(trade_record)
            
            print(f"💰 POSITION CLOSED: {position.side.upper()} "
                  f"${position.size:.4f} @ ${self.current_price:.4f} "
                  f"P&L: ${final_pnl:.2f} ({reason})")
            
            # Log trade closure
            self._log_trade_close(trade_record)
            
        except Exception as e:
            print(f"Error closing position: {e}")
    
    def _log_trade_open(self, position: SimulationPosition, signal: TradingSignal):
        """Log trade opening to callback"""
        if self.callback_function:
            trade_data = {
                'event': 'trade_open',
                'position_id': position.id,
                'symbol': 'SOL-USD-SWAP',
                'side': position.side,
                'size': position.size,
                'price': position.entry_price,
                'timestamp': position.entry_time.isoformat(),
                'confidence': signal.confidence,
                'reason': signal.reason
            }
            self.callback_function(trade_data)
    
    def _log_trade_close(self, trade_record: dict):
        """Log trade closure to callback"""
        if self.callback_function:
            trade_data = {
                'event': 'trade_close',
                'position_id': trade_record['id'],
                'symbol': 'SOL-USD-SWAP',
                'side': trade_record['side'],
                'size': trade_record['size'],
                'price': trade_record['exit_price'],
                'pnl': trade_record['pnl'],
                'timestamp': trade_record['exit_time'].isoformat(),
                'close_reason': trade_record['close_reason']
            }
            self.callback_function(trade_data)
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        # Calculate unrealized P&L from open positions
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions)
        total_equity = self.balance + unrealized_pnl
        
        # Calculate win rate
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        
        # Calculate profit factor
        winning_pnl = sum(trade['pnl'] for trade in self.closed_trades if trade['pnl'] > 0)
        losing_pnl = abs(sum(trade['pnl'] for trade in self.closed_trades if trade['pnl'] < 0))
        profit_factor = winning_pnl / max(losing_pnl, 1)
        
        return {
            'total_pnl': self.total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': total_equity,
            'balance': self.balance,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown * 100,
            'active_positions': len(self.positions),
            'return_pct': ((total_equity - self.initial_balance) / self.initial_balance) * 100
        }
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent trades"""
        recent_trades = self.closed_trades[-limit:] if self.closed_trades else []
        return [
            {
                'timestamp': trade['exit_time'].isoformat(),
                'symbol': 'SOL-USD-SWAP',
                'side': trade['side'],
                'size': trade['size'],
                'price': trade['exit_price'],
                'pnl': trade['pnl'],
                'status': 'closed'
            }
            for trade in reversed(recent_trades)
        ]
    
    def get_active_positions(self) -> List[Dict]:
        """Get active positions"""
        return [
            {
                'id': pos.id,
                'side': pos.side,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'current_price': self.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'entry_time': pos.entry_time.isoformat()
            }
            for pos in self.positions
        ]
    
    def get_statistics(self) -> Dict:
        """Get trading statistics (alias for get_performance_metrics)"""
        return self.get_performance_metrics()
    
    def has_position(self) -> bool:
        """Check if there are any active positions"""
        return len(self.positions) > 0
    
    def execute_trade(self, side: str, price: float, size: float, reason: str = "", confidence: float = 0.0):
        """Execute a simulated trade"""
        try:
            # Calculate actual position size
            actual_size = size / price
            
            # Create position
            position = SimulationPosition(
                side=side,
                size=actual_size,
                entry_price=price
            )
            
            self.positions.append(position)
            
            # Log the trade
            print(f"🔄 Executed {side.upper()} trade: {actual_size:.4f} SOL @ ${price:.4f}")
            print(f"📝 Reason: {reason}")
            print(f"🎯 Confidence: {confidence:.1f}%")
            
            # Update balance (subtract fees)
            fee = size * 0.001  # 0.1% fee
            self.balance -= fee
            
        except Exception as e:
            print(f"❌ Error executing trade: {e}")
    
    def close_position(self, price: float, reason: str = ""):
        """Close all positions"""
        if not self.positions:
            return
        
        for position in self.positions[:]:  # Copy list to avoid modification during iteration
            # Calculate P&L
            if position.side == 'long':
                pnl = (price - position.entry_price) * position.size
            else:  # short
                pnl = (position.entry_price - price) * position.size
            
            # Update statistics
            self.total_pnl += pnl
            self.balance += pnl
            
            if pnl > 0:
                self.winning_trades += 1
            
            # Log the closure
            print(f"🔚 Closed {position.side.upper()} position: P&L ${pnl:.2f}")
            print(f"📝 Reason: {reason}")
            
            # Remove position
            self.positions.remove(position)
    
    def inject_trading_event(self, event_type: str):
        """Inject specific trading events for testing"""
        if event_type == "market_pump":
            self.data_generator.inject_market_event("pump", 0.05)
        elif event_type == "market_dump":
            self.data_generator.inject_market_event("dump", 0.05)
        elif event_type == "high_volatility":
            self.data_generator.inject_market_event("high_volatility")
        elif event_type == "close_all_positions":
            for position in self.positions.copy():
                self._close_position(position, "manual_close")

def main():
    """Test the simulation trading engine"""
    def on_trade_update(trade_data):
        if trade_data['event'] == 'trade_open':
            print(f"🟢 Trade opened: {trade_data['side']} @ ${trade_data['price']:.4f}")
        elif trade_data['event'] == 'trade_close':
            pnl_sign = "+" if trade_data['pnl'] >= 0 else ""
            print(f"🔴 Trade closed: {pnl_sign}${trade_data['pnl']:.2f}")
    
    # Create trading engine
    engine = SimulationTradingEngine()
    engine.set_callback(on_trade_update)
    
    try:
        # Start simulation
        engine.start_simulation()
        
        # Let it run for a while
        time.sleep(30)
        
        # Inject some market events
        print("\n🚀 Injecting market pump...")
        engine.inject_trading_event("market_pump")
        time.sleep(10)
        
        print("\n📉 Injecting market dump...")
        engine.inject_trading_event("market_dump")
        time.sleep(10)
        
        # Show performance
        metrics = engine.get_performance_metrics()
        print(f"\n📊 Performance Summary:")
        print(f"Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Active Positions: {metrics['active_positions']}")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping simulation...")
    finally:
        engine.stop_simulation()

if __name__ == "__main__":
    main() 