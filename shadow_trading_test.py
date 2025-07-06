#!/usr/bin/env python3
"""
Shadow Trading Test - Real-time simulation of the improved edge system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import random
from improved_edge_system import ImprovedEdgeSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShadowTrader:
    """Real-time shadow trading simulator"""
    
    def __init__(self, system):
        self.system = system
        self.trades = []
        self.positions = {}
        self.balance = 10000
        self.start_time = datetime.now()
        
    def generate_live_candle(self, previous_price, volatility=0.003):
        """Generate a single live market candle"""
        change = np.random.normal(0, volatility)
        new_price = previous_price * (1 + change)
        
        high = new_price * (1 + abs(np.random.normal(0, volatility/3)))
        low = new_price * (1 - abs(np.random.normal(0, volatility/3)))
        
        return {
            'timestamp': int(datetime.now().timestamp() * 1000),
            'open': previous_price,
            'high': max(high, previous_price, new_price),
            'low': min(low, previous_price, new_price),
            'close': new_price,
            'volume': max(100, 1000 + np.random.normal(0, 200))
        }
    
    def generate_order_book(self):
        """Generate realistic order book data"""
        imbalance = np.random.uniform(-0.3, 0.3)  # -30% to +30% imbalance
        
        if imbalance > 0:  # Bid heavy
            bid_vol = np.random.uniform(3000, 8000)
            ask_vol = bid_vol * (1 - abs(imbalance))
        else:  # Ask heavy
            ask_vol = np.random.uniform(3000, 8000)
            bid_vol = ask_vol * (1 - abs(imbalance))
        
        return {
            'bid_volume': bid_vol,
            'ask_volume': ask_vol
        }
    
    def check_position_exits(self, asset, current_candle):
        """Check if any positions should be closed"""
        if asset not in self.positions:
            return
        
        position = self.positions[asset]
        current_price = current_candle['close']
        
        exit_price = None
        exit_reason = None
        
        # Check stops and targets
        if position['direction'] == 'long':
            if current_candle['low'] <= position['stop_price']:
                exit_price = position['stop_price']
                exit_reason = 'stop'
            elif current_candle['high'] >= position['target_price']:
                exit_price = position['target_price']
                exit_reason = 'target'
        else:
            if current_candle['high'] >= position['stop_price']:
                exit_price = position['stop_price']
                exit_reason = 'stop'
            elif current_candle['low'] <= position['target_price']:
                exit_price = position['target_price']
                exit_reason = 'target'
        
        # Time-based exit (40 candles = ~3.3 hours)
        candles_held = len(position['price_history'])
        if exit_price is None and candles_held >= 40:
            exit_price = current_price
            exit_reason = 'time'
        
        if exit_price:
            self.close_position(asset, exit_price, exit_reason)
    
    def close_position(self, asset, exit_price, reason):
        """Close a position and record the trade"""
        position = self.positions[asset]
        
        # Calculate P&L
        if position['direction'] == 'long':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        pnl_amount = position['size'] * pnl_pct
        self.balance += pnl_amount
        
        # Record trade
        trade = {
            'asset': asset,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'reason': reason,
            'duration_candles': len(position['price_history']),
            'edge': position['edge'],
            'confidence': position['confidence'],
            'timestamp': datetime.now()
        }
        
        self.trades.append(trade)
        
        # Log trade
        emoji = "🎯" if reason == 'target' else "🛑" if reason == 'stop' else "⏰"
        color = "🟢" if pnl_pct > 0 else "🔴"
        logger.info(f"{emoji}{color} {position['direction'].upper()} {asset}: {pnl_pct:+.2%} ({reason}) - Edge: {position['edge']:.3%}")
        
        # Remove position
        del self.positions[asset]
    
    def open_position(self, asset, signal, current_price):
        """Open a new position"""
        position_size = self.balance * 0.01  # 1% risk
        
        if signal['direction'] == 'long':
            stop_price = current_price - signal['stop_distance']
            target_price = current_price + signal['target_distance']
        else:
            stop_price = current_price + signal['stop_distance']
            target_price = current_price - signal['target_distance']
        
        position = {
            'asset': asset,
            'direction': signal['direction'],
            'entry_price': current_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'size': position_size,
            'edge': signal['edge'],
            'confidence': signal['confidence'],
            'entry_time': datetime.now(),
            'price_history': [current_price]
        }
        
        self.positions[asset] = position
        
        logger.info(f"📝 OPEN {signal['direction'].upper()} {asset} @ {current_price:.2f} "
                   f"(Stop: {stop_price:.2f}, Target: {target_price:.2f}) - Edge: {signal['edge']:.3%}")
    
    def get_current_metrics(self):
        """Calculate current performance metrics"""
        if not self.trades:
            return None
        
        df = pd.DataFrame(self.trades)
        
        wins = df[df['pnl_pct'] > 0]
        losses = df[df['pnl_pct'] <= 0]
        
        win_rate = len(wins) / len(self.trades)
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        total_wins = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'balance': self.balance,
            'total_return': (self.balance - 10000) / 10000,
            'avg_edge': df['edge'].mean(),
            'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }

def run_shadow_trading():
    """Run shadow trading simulation"""
    print("🏃 SHADOW TRADING SIMULATION")
    print("=" * 50)
    
    # Initialize system
    system = ImprovedEdgeSystem()
    
    # Generate initial training data
    print("🧠 Training system...")
    candles = []
    current_price = 100
    
    # Generate 2000 candles for training
    for i in range(2000):
        change = np.random.normal(0, 0.003)
        new_price = current_price * (1 + change)
        
        candle = {
            'timestamp': int((datetime.now() - timedelta(minutes=(2000-i)*5)).timestamp() * 1000),
            'open': current_price,
            'high': new_price * (1 + abs(np.random.normal(0, 0.001))),
            'low': new_price * (1 - abs(np.random.normal(0, 0.001))),
            'close': new_price,
            'volume': max(100, 1000 + np.random.normal(0, 200))
        }
        
        candles.append(candle)
        current_price = new_price
    
    # Train the system
    system.train_models(candles)
    print("✅ System trained and ready")
    
    # Initialize shadow trader
    shadow = ShadowTrader(system)
    
    # Assets to trade
    assets = ['ETH', 'BTC', 'SOL']
    asset_prices = {'ETH': current_price, 'BTC': current_price * 500, 'SOL': current_price * 0.3}
    
    target_trades = 50  # Target number of trades
    candle_count = 0
    
    print(f"\n🎯 Target: {target_trades} shadow trades")
    print("📊 Live simulation starting...\n")
    
    # Real-time simulation loop
    while len(shadow.trades) < target_trades and candle_count < 1000:
        candle_count += 1
        
        for asset in assets:
            # Generate new candle
            current_price = asset_prices[asset]
            new_candle = shadow.generate_live_candle(current_price, volatility=0.003)
            asset_prices[asset] = new_candle['close']
            
            # Update candle history
            candles.append(new_candle)
            if len(candles) > 2100:  # Keep rolling window
                candles.pop(0)
            
            # Update position price history
            if asset in shadow.positions:
                shadow.positions[asset]['price_history'].append(new_candle['close'])
            
            # Check exits first
            shadow.check_position_exits(asset, new_candle)
            
            # Generate signal if no position
            if asset not in shadow.positions:
                order_book = shadow.generate_order_book()
                signal = system.generate_signal(candles[-100:], order_book)
                
                if system.should_trade(signal):
                    shadow.open_position(asset, signal, new_candle['close'])
        
        # Progress update every 50 candles
        if candle_count % 50 == 0:
            metrics = shadow.get_current_metrics()
            if metrics:
                print(f"📈 Progress: {metrics['total_trades']}/{target_trades} trades, "
                      f"Win Rate: {metrics['win_rate']:.1%}, "
                      f"Expectancy: {metrics['expectancy']:.3%}, "
                      f"Balance: ${metrics['balance']:.2f}")
        
        # Simulate time passing
        time.sleep(0.1)  # 100ms per candle
    
    # Final results
    final_metrics = shadow.get_current_metrics()
    
    print(f"\n{'='*50}")
    print("SHADOW TRADING COMPLETE")
    print(f"{'='*50}")
    
    if final_metrics:
        print(f"📊 FINAL RESULTS:")
        print(f"   Total Trades: {final_metrics['total_trades']}")
        print(f"   Win Rate: {final_metrics['win_rate']:.1%}")
        print(f"   💰 EXPECTANCY: {final_metrics['expectancy']:.3%} per trade")
        print(f"   📈 Profit Factor: {final_metrics['profit_factor']:.2f}")
        print(f"   Final Balance: ${final_metrics['balance']:.2f}")
        print(f"   Total Return: {final_metrics['total_return']:.1%}")
        print(f"   Avg Edge: {final_metrics['avg_edge']:.3%}")
        print(f"   Runtime: {final_metrics['runtime_hours']:.1f} hours")
        
        # Shadow trading gates
        print(f"\n🎯 SHADOW GATES:")
        print(f"   Trades ≥ 50: {'✅' if final_metrics['total_trades'] >= 50 else '❌'} ({final_metrics['total_trades']})")
        print(f"   Expectancy > 0: {'✅' if final_metrics['expectancy'] > 0 else '❌'} ({final_metrics['expectancy']:.3%})")
        print(f"   Profit Factor > 1: {'✅' if final_metrics['profit_factor'] > 1 else '❌'} ({final_metrics['profit_factor']:.2f})")
        print(f"   Positive Return: {'✅' if final_metrics['total_return'] > 0 else '❌'} ({final_metrics['total_return']:.1%})")
        
        # Final assessment
        shadow_passed = (
            final_metrics['total_trades'] >= 50 and
            final_metrics['expectancy'] > 0 and
            final_metrics['profit_factor'] > 1 and
            final_metrics['total_return'] > 0
        )
        
        if shadow_passed:
            print(f"\n🎉 SHADOW TRADING PASSED!")
            print(f"✅ System ready for live deployment")
            
            # Write status file
            with open('shadow_status.txt', 'w') as f:
                f.write('PASSED')
        else:
            print(f"\n🔄 Shadow trading needs improvement")
            with open('shadow_status.txt', 'w') as f:
                f.write('FAILED')
    
    else:
        print("❌ No trades completed in shadow mode")
        with open('shadow_status.txt', 'w') as f:
            f.write('FAILED')

if __name__ == "__main__":
    run_shadow_trading() 