#!/usr/bin/env python3
"""
Smart Pyramid + Equity Reset System
Automatically scales winning positions and resets after blockbuster trades
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Enhanced position with pyramiding state"""
    symbol: str
    side: str
    entry_price: float
    current_size: float
    scale_steps: int = 0
    initial_stop_distance: float = 0.0
    blockbuster_flag: bool = False
    entry_time: datetime = None
    
    def get_current_pnl_r(self, current_price: float) -> float:
        """Calculate current P&L in R multiples"""
        if self.initial_stop_distance == 0:
            return 0.0
        
        if self.side == 'long':
            pnl_points = current_price - self.entry_price
        else:
            pnl_points = self.entry_price - current_price
        
        return pnl_points / self.initial_stop_distance

class SmartPyramidSystem:
    """Smart pyramiding with equity reset"""
    
    def __init__(self, config: Dict = None):
        # Default configuration
        self.config = {
            'base_risk_pct': 0.01,          # 1% base risk
            'profit_step_r': 1.0,           # Add unit every 1R profit
            'scale_factor': [1.0, 0.8, 0.6], # Size multipliers
            'max_steps': 3,                  # Max pyramid levels
            'trail_dist_atr': [1.2, 1.1, 1.0], # Trailing stop distances
            'reset_threshold_r': 5.0,        # Blockbuster threshold
            'cooldown_trades': 3,            # Trades after big loss
            'max_total_risk': 0.03,          # 3% max risk per symbol
            'enabled': True
        }
        
        if config:
            self.config.update(config)
        
        self.positions = {}
        self.trade_history = []
        self.blockbuster_pending = False
        self.cooldown_counter = 0
        self.last_trade_result = None
        
    def calculate_atr(self, candles: list, period: int = 14) -> float:
        """Calculate ATR for position sizing"""
        if len(candles) < period + 1:
            return 0.0
        
        high = np.array([c['high'] for c in candles[-period-1:]])
        low = np.array([c['low'] for c in candles[-period-1:]])
        close = np.array([c['close'] for c in candles[-period-1:]])
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        
        return np.mean(tr)
    
    def calculate_position_size(self, balance: float, atr: float, current_price: float, 
                              scale_step: int = 0) -> float:
        """Calculate position size based on risk and scale factor"""
        # Apply cooldown if needed
        if self.cooldown_counter > 0:
            risk_pct = self.config['base_risk_pct'] * 0.5  # Half size during cooldown
        else:
            risk_pct = self.config['base_risk_pct']
        
        # Apply scale factor
        if scale_step < len(self.config['scale_factor']):
            risk_pct *= self.config['scale_factor'][scale_step]
        
        # Calculate position size
        risk_amount = balance * risk_pct
        stop_distance = atr * self.config['trail_dist_atr'][0]
        position_value = risk_amount / (stop_distance / current_price)
        
        return position_value
    
    def should_add_position(self, position: Position, current_price: float) -> bool:
        """Check if we should pyramid into position"""
        if not self.config['enabled']:
            return False
        
        if position.scale_steps >= self.config['max_steps']:
            return False
        
        current_r = position.get_current_pnl_r(current_price)
        required_r = self.config['profit_step_r'] * (position.scale_steps + 1)
        
        return current_r >= required_r
    
    def get_trailing_stop(self, position: Position, current_price: float, atr: float) -> float:
        """Calculate trailing stop based on pyramid level"""
        # Get appropriate trail distance
        trail_idx = min(position.scale_steps, len(self.config['trail_dist_atr']) - 1)
        trail_distance = atr * self.config['trail_dist_atr'][trail_idx]
        
        if position.side == 'long':
            return current_price - trail_distance
        else:
            return current_price + trail_distance
    
    def enter_position(self, symbol: str, side: str, signal: dict, 
                      balance: float, current_price: float, candles: list) -> Optional[Dict]:
        """Enter new position with smart sizing"""
        # Check if we already have a position
        if symbol in self.positions:
            return None
        
        # Calculate ATR
        atr = self.calculate_atr(candles)
        if atr == 0:
            return None
        
        # Calculate position size
        position_size = self.calculate_position_size(balance, atr, current_price, 0)
        
        # Check max risk
        position_risk = (position_size / balance) * (atr / current_price)
        if position_risk > self.config['max_total_risk']:
            position_size = balance * self.config['max_total_risk'] / (atr / current_price)
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=current_price,
            current_size=position_size,
            scale_steps=0,
            initial_stop_distance=atr * self.config['trail_dist_atr'][0],
            entry_time=datetime.now()
        )
        
        self.positions[symbol] = position
        
        # Calculate stop price
        stop_price = self.get_trailing_stop(position, current_price, atr)
        
        logger.info(f"📈 New {side} position: {symbol} @ {current_price:.2f}, "
                   f"size: ${position_size:.2f}, stop: {stop_price:.2f}")
        
        return {
            'action': 'enter',
            'symbol': symbol,
            'side': side,
            'size': position_size,
            'price': current_price,
            'stop': stop_price,
            'scale_step': 0
        }
    
    def manage_position(self, symbol: str, current_price: float, 
                       balance: float, candles: list) -> Optional[Dict]:
        """Manage existing position with pyramiding"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        atr = self.calculate_atr(candles)
        
        # Check if we should add to position
        if self.should_add_position(position, current_price):
            # Calculate add-on size
            add_size = self.calculate_position_size(
                balance, atr, current_price, position.scale_steps + 1
            )
            
            # Check total risk
            total_risk = ((position.current_size + add_size) / balance) * (atr / current_price)
            if total_risk <= self.config['max_total_risk']:
                position.current_size += add_size
                position.scale_steps += 1
                
                # Update trailing stop
                new_stop = self.get_trailing_stop(position, current_price, atr)
                
                logger.info(f"🔺 Pyramid add #{position.scale_steps}: {symbol} @ {current_price:.2f}, "
                           f"add size: ${add_size:.2f}, new stop: {new_stop:.2f}")
                
                return {
                    'action': 'pyramid',
                    'symbol': symbol,
                    'side': position.side,
                    'add_size': add_size,
                    'price': current_price,
                    'new_stop': new_stop,
                    'scale_step': position.scale_steps
                }
        
        # Just update trailing stop
        new_stop = self.get_trailing_stop(position, current_price, atr)
        
        return {
            'action': 'trail_stop',
            'symbol': symbol,
            'new_stop': new_stop
        }
    
    def close_position(self, symbol: str, exit_price: float, 
                      exit_reason: str = 'signal') -> Optional[Dict]:
        """Close position and handle equity reset"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate final P&L
        if position.side == 'long':
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        
        pnl_r = position.get_current_pnl_r(exit_price)
        pnl_amount = position.current_size * pnl_pct
        
        # Check if blockbuster trade
        if pnl_r >= self.config['reset_threshold_r']:
            position.blockbuster_flag = True
            self.blockbuster_pending = True
            logger.info(f"🎯 BLOCKBUSTER TRADE! {pnl_r:.1f}R gain!")
        
        # Handle big loss
        if pnl_r <= -2.0:  # Big loss threshold
            self.cooldown_counter = self.config['cooldown_trades']
            logger.info(f"❄️ Entering cooldown mode for {self.cooldown_counter} trades")
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'position_size': position.current_size,
            'scale_steps': position.scale_steps,
            'pnl_pct': pnl_pct,
            'pnl_r': pnl_r,
            'pnl_amount': pnl_amount,
            'blockbuster': position.blockbuster_flag,
            'timestamp': datetime.now()
        }
        
        self.trade_history.append(trade_record)
        self.last_trade_result = pnl_r
        
        # Remove position
        del self.positions[symbol]
        
        # Decrement cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        
        logger.info(f"📊 Closed {position.side} {symbol}: {pnl_pct:.2%} ({pnl_r:.1f}R)")
        
        return trade_record
    
    def get_next_trade_config(self) -> Dict:
        """Get configuration for next trade (handles equity reset)"""
        config = self.config.copy()
        
        # Apply equity reset if blockbuster pending
        if self.blockbuster_pending:
            logger.info("🔄 Equity reset activated - returning to base size")
            self.blockbuster_pending = False
            # Config already has base settings
        
        # Apply cooldown if active
        if self.cooldown_counter > 0:
            config['base_risk_pct'] *= 0.5
            logger.info(f"❄️ Cooldown active: {self.cooldown_counter} trades remaining")
        
        return config
    
    def get_statistics(self) -> Dict:
        """Calculate pyramiding statistics"""
        if not self.trade_history:
            return {}
        
        df = pd.DataFrame(self.trade_history)
        
        # Basic stats
        total_trades = len(df)
        winning_trades = len(df[df['pnl_r'] > 0])
        win_rate = winning_trades / total_trades
        
        # Pyramiding stats
        pyramided_trades = df[df['scale_steps'] > 0]
        pyramid_success_rate = len(pyramided_trades[pyramided_trades['pnl_r'] > 0]) / len(pyramided_trades) if len(pyramided_trades) > 0 else 0
        
        # Blockbuster stats
        blockbusters = df[df['blockbuster'] == True]
        blockbuster_count = len(blockbusters)
        blockbuster_contribution = blockbusters['pnl_amount'].sum() / df['pnl_amount'].sum() if df['pnl_amount'].sum() > 0 else 0
        
        # R-multiple stats
        avg_win_r = df[df['pnl_r'] > 0]['pnl_r'].mean() if winning_trades > 0 else 0
        avg_loss_r = df[df['pnl_r'] <= 0]['pnl_r'].mean() if winning_trades < total_trades else 0
        expectancy_r = (win_rate * avg_win_r) + ((1 - win_rate) * avg_loss_r)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win_r': avg_win_r,
            'avg_loss_r': avg_loss_r,
            'expectancy_r': expectancy_r,
            'pyramid_trades': len(pyramided_trades),
            'pyramid_success_rate': pyramid_success_rate,
            'avg_pyramid_steps': pyramided_trades['scale_steps'].mean() if len(pyramided_trades) > 0 else 0,
            'blockbuster_count': blockbuster_count,
            'blockbuster_contribution': blockbuster_contribution,
            'max_win_r': df['pnl_r'].max(),
            'max_loss_r': df['pnl_r'].min()
        }

class SmartPyramidBacktester:
    """Backtester for smart pyramiding system"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.pyramid_system = SmartPyramidSystem()
        
    def generate_trending_data(self, days: int = 30) -> list:
        """Generate data with trending periods for pyramiding"""
        candles = []
        base_price = 100
        current_price = base_price
        
        start_time = datetime.now() - timedelta(days=days)
        
        for i in range(days * 288):  # 5-minute candles
            # Create trending markets
            cycle = (i % 1000) / 1000
            
            if cycle < 0.3:  # Strong uptrend
                trend = 0.0004
                volatility = 0.002
            elif cycle < 0.5:  # Consolidation
                trend = 0
                volatility = 0.001
            elif cycle < 0.8:  # Strong downtrend
                trend = -0.0004
                volatility = 0.002
            else:  # Volatile
                trend = 0
                volatility = 0.004
            
            # Add some big moves for pyramiding opportunities
            if np.random.random() < 0.01:  # 1% chance of big move
                trend *= 5
            
            change = np.random.normal(trend, volatility)
            new_price = current_price * (1 + change)
            
            # OHLC
            high = new_price * (1 + abs(np.random.normal(0, volatility/2)))
            low = new_price * (1 - abs(np.random.normal(0, volatility/2)))
            
            candle = {
                'timestamp': int((start_time + timedelta(minutes=i*5)).timestamp() * 1000),
                'open': current_price,
                'high': max(high, current_price, new_price),
                'low': min(low, current_price, new_price),
                'close': new_price,
                'volume': max(100, 1000 + np.random.normal(0, 200))
            }
            
            candles.append(candle)
            current_price = new_price
        
        return candles
    
    def run_backtest(self, days: int = 30) -> dict:
        """Run pyramiding backtest"""
        logger.info(f"🚀 Smart Pyramid Backtest ({days} days)")
        
        # Generate trending data
        candles = self.generate_trending_data(days)
        
        # Initialize
        balance = self.initial_balance
        equity_curve = [balance]
        
        # Simple trend-following signals for testing
        for i in range(100, len(candles) - 50):
            current_candle = candles[i]
            current_price = current_candle['close']
            
            # Update existing positions
            for symbol in list(self.pyramid_system.positions.keys()):
                position = self.pyramid_system.positions[symbol]
                
                # Check stop loss
                atr = self.pyramid_system.calculate_atr(candles[:i+1])
                stop_price = self.pyramid_system.get_trailing_stop(position, current_price, atr)
                
                if (position.side == 'long' and current_price <= stop_price) or \
                   (position.side == 'short' and current_price >= stop_price):
                    # Close position
                    trade = self.pyramid_system.close_position(symbol, current_price, 'stop')
                    if trade:
                        balance += trade['pnl_amount']
                        equity_curve.append(balance)
                else:
                    # Manage position (potential pyramid)
                    action = self.pyramid_system.manage_position(symbol, current_price, balance, candles[:i+1])
                    if action and action['action'] == 'pyramid':
                        # Deduct cost of pyramid addition
                        balance -= action['add_size']
            
            # Generate new signals (simple MA crossover for demo)
            if 'TEST' not in self.pyramid_system.positions and i % 50 == 0:  # Periodic signals
                closes = [c['close'] for c in candles[i-50:i+1]]
                sma_10 = np.mean(closes[-10:])
                sma_30 = np.mean(closes[-30:])
                
                signal = None
                if sma_10 > sma_30 * 1.01:  # Long signal
                    signal = {'direction': 'long', 'strength': 0.7}
                elif sma_10 < sma_30 * 0.99:  # Short signal
                    signal = {'direction': 'short', 'strength': 0.7}
                
                if signal and signal['direction'] != 'hold':
                    # Enter position
                    entry = self.pyramid_system.enter_position(
                        'TEST', signal['direction'], signal, 
                        balance, current_price, candles[:i+1]
                    )
                    if entry:
                        balance -= entry['size']
        
        # Close any remaining positions
        for symbol in list(self.pyramid_system.positions.keys()):
            trade = self.pyramid_system.close_position(symbol, candles[-1]['close'], 'end')
            if trade:
                balance += trade['pnl_amount']
                equity_curve.append(balance)
        
        # Get statistics
        stats = self.pyramid_system.get_statistics()
        
        # Calculate additional metrics
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        max_dd = 0
        if len(equity_array) > 0:
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (running_max - equity_array) / running_max
            max_dd = np.max(drawdown)
        
        sharpe = 0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.sqrt(252) * (np.mean(returns) / np.std(returns))
        
        results = {
            **stats,
            'final_balance': balance,
            'total_return': (balance - self.initial_balance) / self.initial_balance,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe
        }
        
        return results

def main():
    """Test smart pyramiding system"""
    print("🚀 SMART PYRAMID + EQUITY RESET SYSTEM")
    print("=" * 50)
    print("Features: Auto-scaling, Blockbuster reset, Cooldown mode\n")
    
    # Test different configurations
    configs = [
        {
            "name": "Conservative",
            "config": {
                "base_risk_pct": 0.005,  # 0.5%
                "profit_step_r": 1.5,    # Add every 1.5R
                "scale_factor": [1.0, 0.5, 0.3],
                "reset_threshold_r": 7.0
            }
        },
        {
            "name": "Standard",
            "config": {}  # Use defaults
        },
        {
            "name": "Aggressive",
            "config": {
                "base_risk_pct": 0.02,   # 2%
                "profit_step_r": 0.75,   # Add every 0.75R
                "scale_factor": [1.0, 1.0, 0.8],
                "max_steps": 4,
                "reset_threshold_r": 4.0
            }
        }
    ]
    
    all_results = []
    
    for test_config in configs:
        print(f"\n🧪 Testing {test_config['name']} Configuration...")
        print("-" * 40)
        
        backtester = SmartPyramidBacktester()
        backtester.pyramid_system.config.update(test_config['config'])
        
        results = backtester.run_backtest(30)
        
        print(f"📊 RESULTS:")
        print(f"   Total Trades: {results.get('total_trades', 0)}")
        print(f"   Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"   Expectancy: {results.get('expectancy_r', 0):.2f}R")
        print(f"   📈 Pyramid Trades: {results.get('pyramid_trades', 0)}")
        print(f"   Pyramid Success: {results.get('pyramid_success_rate', 0):.1%}")
        print(f"   🎯 Blockbusters: {results.get('blockbuster_count', 0)}")
        print(f"   Blockbuster Impact: {results.get('blockbuster_contribution', 0):.1%} of profits")
        print(f"   Max Win: {results.get('max_win_r', 0):.1f}R")
        print(f"   Total Return: {results.get('total_return', 0):.1%}")
        
        all_results.append({
            'config': test_config['name'],
            'results': results
        })
    
    # Summary
    print(f"\n🎯 CONFIGURATION COMPARISON")
    print("=" * 50)
    
    best_return = max(all_results, key=lambda x: x['results'].get('total_return', 0))
    best_expectancy = max(all_results, key=lambda x: x['results'].get('expectancy_r', 0))
    
    print(f"Best Return: {best_return['config']} ({best_return['results']['total_return']:.1%})")
    print(f"Best Expectancy: {best_expectancy['config']} ({best_expectancy['results']['expectancy_r']:.2f}R)")
    
    # Save results
    with open('smart_pyramid_results.json', 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to 'smart_pyramid_results.json'")

if __name__ == "__main__":
    main() 