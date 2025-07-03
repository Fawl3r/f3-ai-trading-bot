import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from okx_client import OKXClient
from strategy import TradingSignal, AdvancedTradingStrategy
from config import (
    INSTRUMENT_ID, POSITION_SIZE_USD, LEVERAGE, STOP_LOSS_PCT, 
    TAKE_PROFIT_PCT, MAX_DRAWDOWN_PCT, MAX_DAILY_TRADES
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Position:
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
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
            self.unrealized_pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
            self.unrealized_pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100

class TradingEngine:
    def __init__(self):
        self.okx_client = OKXClient()
        self.strategy = AdvancedTradingStrategy()
        self.current_position: Optional[Position] = None
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Risk management
        self.daily_loss_limit = POSITION_SIZE_USD * 0.1  # 10% of position size
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        
        # Trading state
        self.is_trading_enabled = True
        self.last_heartbeat = datetime.now()
        
    def initialize(self):
        """Initialize trading engine and set leverage"""
        try:
            # Set leverage
            logger.info(f"Setting leverage to {LEVERAGE}x for {INSTRUMENT_ID}")
            response = self.okx_client.set_leverage(INSTRUMENT_ID, LEVERAGE)
            if response.get('code') == '0':
                logger.info("Leverage set successfully")
            else:
                logger.warning(f"Leverage setting warning: {response}")
            
            # Get initial account info
            balance = self.okx_client.get_account_balance()
            logger.info(f"Account balance: {balance}")
            
            # Check for existing positions
            positions = self.okx_client.get_positions(INSTRUMENT_ID)
            if positions.get('data'):
                self._sync_existing_position(positions['data'][0])
            
            logger.info("Trading engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            return False
    
    def _sync_existing_position(self, position_data: Dict):
        """Sync with existing position from OKX"""
        try:
            if float(position_data.get('pos', 0)) != 0:
                side = 'long' if float(position_data['pos']) > 0 else 'short'
                size = abs(float(position_data['pos']))
                entry_price = float(position_data['avgPx'])
                
                self.current_position = Position(side, size, entry_price)
                logger.info(f"Synced existing {side} position: {size} @ {entry_price}")
        except Exception as e:
            logger.error(f"Error syncing existing position: {e}")
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on current price and leverage"""
        notional_value = POSITION_SIZE_USD * LEVERAGE
        return round(notional_value / price, 4)
    
    def _should_trade(self) -> bool:
        """Check if trading should be enabled based on risk controls"""
        if not self.is_trading_enabled:
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            logger.warning("Daily loss limit reached - stopping trading")
            self.is_trading_enabled = False
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning("Max consecutive losses reached - taking a break")
            time.sleep(300)  # 5-minute cooling off period
            self.consecutive_losses = 0
            return False
        
        # Check max daily trades
        stats = self.strategy.get_strategy_stats()
        if stats['trades_today'] >= MAX_DAILY_TRADES:
            logger.warning("Max daily trades reached")
            return False
        
        return True
    
    def execute_signal(self, signal: TradingSignal, current_price: float) -> bool:
        """Execute trading signal"""
        if not self._should_trade():
            return False
        
        try:
            if signal.signal_type == 'close':
                return self._close_position()
            
            elif signal.signal_type in ['buy', 'sell']:
                # Close existing position if opposite direction
                if self.current_position:
                    if ((signal.signal_type == 'buy' and self.current_position.side == 'short') or
                        (signal.signal_type == 'sell' and self.current_position.side == 'long')):
                        self._close_position()
                
                # Open new position
                return self._open_position(signal, current_price)
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    def _open_position(self, signal: TradingSignal, current_price: float) -> bool:
        """Open new position"""
        try:
            size = self._calculate_position_size(current_price)
            side_okx = 'buy' if signal.signal_type == 'buy' else 'sell'
            
            # Calculate stop loss and take profit prices
            if signal.signal_type == 'buy':
                stop_loss_price = current_price * (1 - STOP_LOSS_PCT / 100)
                take_profit_price = current_price * (1 + TAKE_PROFIT_PCT / 100)
            else:
                stop_loss_price = current_price * (1 + STOP_LOSS_PCT / 100)
                take_profit_price = current_price * (1 - TAKE_PROFIT_PCT / 100)
            
            # Place market order
            logger.info(f"Placing {signal.signal_type} order: {size} @ {current_price}")
            logger.info(f"Signal confidence: {signal.confidence:.2%}")
            logger.info(f"Reasons: {signal.reason}")
            
            response = self.okx_client.place_order(
                inst_id=INSTRUMENT_ID,
                side=side_okx,
                ord_type='market',
                sz=str(size)
            )
            
            if response.get('code') == '0':
                # Create position object
                position_side = 'long' if signal.signal_type == 'buy' else 'short'
                self.current_position = Position(
                    side=position_side,
                    size=size,
                    entry_price=current_price,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price
                )
                
                self.total_trades += 1
                logger.info(f"Position opened successfully: {position_side} {size} @ {current_price}")
                
                # Place stop loss and take profit orders
                self._place_exit_orders()
                
                return True
            else:
                logger.error(f"Failed to place order: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False
    
    def _close_position(self) -> bool:
        """Close current position"""
        if not self.current_position:
            return True
        
        try:
            logger.info(f"Closing {self.current_position.side} position")
            
            response = self.okx_client.close_position(INSTRUMENT_ID)
            
            if response.get('code') == '0':
                # Update P&L tracking
                self._update_pnl_stats(self.current_position.unrealized_pnl)
                
                logger.info(f"Position closed. P&L: ${self.current_position.unrealized_pnl:.2f} "
                           f"({self.current_position.unrealized_pnl_pct:.2f}%)")
                
                self.current_position = None
                return True
            else:
                logger.error(f"Failed to close position: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def _place_exit_orders(self):
        """Place stop loss and take profit orders"""
        if not self.current_position:
            return
        
        try:
            # Note: OKX may require separate stop-loss and take-profit orders
            # This is a simplified implementation
            pass
        except Exception as e:
            logger.error(f"Error placing exit orders: {e}")
    
    def _update_pnl_stats(self, pnl: float):
        """Update P&L statistics"""
        self.total_pnl += pnl
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Update drawdown tracking
        if self.total_pnl > self.peak_equity:
            self.peak_equity = self.total_pnl
        
        current_drawdown = (self.peak_equity - self.total_pnl) / max(self.peak_equity, 1) * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def update_position(self, current_price: float, df) -> bool:
        """Update current position and check exit conditions"""
        if not self.current_position:
            return False
        
        # Update P&L
        self.current_position.update_pnl(current_price)
        
        # Check stop loss
        if self.current_position.stop_loss:
            if ((self.current_position.side == 'long' and current_price <= self.current_position.stop_loss) or
                (self.current_position.side == 'short' and current_price >= self.current_position.stop_loss)):
                logger.info("Stop loss triggered")
                return self._close_position()
        
        # Check take profit
        if self.current_position.take_profit:
            if ((self.current_position.side == 'long' and current_price >= self.current_position.take_profit) or
                (self.current_position.side == 'short' and current_price <= self.current_position.take_profit)):
                logger.info("Take profit triggered")
                return self._close_position()
        
        # Check strategy exit conditions
        if self.strategy.should_exit_position(
            df, self.current_position.side, 
            self.current_position.entry_price, 
            self.current_position.unrealized_pnl_pct
        ):
            logger.info("Strategy exit condition met")
            return self._close_position()
        
        # Emergency exit on large drawdown
        if self.current_position.unrealized_pnl_pct <= -MAX_DRAWDOWN_PCT:
            logger.warning("Emergency exit - max drawdown reached")
            return self._close_position()
        
        return False
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        
        position_info = {}
        if self.current_position:
            position_info = {
                'side': self.current_position.side,
                'size': self.current_position.size,
                'entry_price': self.current_position.entry_price,
                'unrealized_pnl': self.current_position.unrealized_pnl,
                'unrealized_pnl_pct': self.current_position.unrealized_pnl_pct
            }
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'is_trading_enabled': self.is_trading_enabled,
            'current_position': position_info
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        self.is_trading_enabled = True
        logger.info("Daily statistics reset")
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        logger.warning("EMERGENCY STOP ACTIVATED")
        self.is_trading_enabled = False
        
        if self.current_position:
            self._close_position()
    
    def heartbeat(self):
        """Send heartbeat to confirm system is alive"""
        self.last_heartbeat = datetime.now()
        stats = self.get_performance_stats()
        
        logger.info(f"Heartbeat - Trades: {stats['total_trades']}, "
                   f"Win Rate: {stats['win_rate']:.1f}%, "
                   f"Total P&L: ${stats['total_pnl']:.2f}, "
                   f"Daily P&L: ${stats['daily_pnl']:.2f}")
        
        if self.current_position:
            logger.info(f"Position: {self.current_position.side} "
                       f"P&L: ${self.current_position.unrealized_pnl:.2f} "
                       f"({self.current_position.unrealized_pnl_pct:.2f}%)")
        
        return stats 