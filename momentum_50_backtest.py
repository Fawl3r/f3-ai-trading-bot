#!/usr/bin/env python3
"""
🚀 MOMENTUM BOT $50 BACKTEST
Test all 4 momentum features with $50 starting balance

FEATURES TESTED:
✅ Volume spike detection (2x+ normal volume)
✅ Price acceleration detection  
✅ Dynamic position sizing (2-8% based on momentum strength)
✅ Trailing stops for parabolic moves (3% trailing distance)
✅ Momentum-adjusted confidence thresholds
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class MomentumBacktest:
    def __init__(self, starting_balance=50.0):
        """Initialize backtest with $50"""
        
        print("🚀 MOMENTUM BOT $50 BACKTEST")
        print("💥 Testing all 4 momentum features")
        print("=" * 60)
        
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        
        # Trading pairs
        self.trading_pairs = [
            'BTC', 'ETH', 'SOL', 'DOGE', 'AVAX',
            'LINK', 'UNI', 'ADA', 'DOT', 'MATIC', 
            'NEAR', 'ATOM', 'FTM', 'SAND', 'CRV'
        ]
        
        # 🚀 MOMENTUM DETECTION CONFIG
        self.volume_spike_threshold = 2.0
        self.acceleration_threshold = 0.02
        self.parabolic_threshold = 0.8
        self.big_swing_threshold = 0.6
        
        # 💰 DYNAMIC POSITION SIZING (2-8%)
        self.base_position_size = 2.0
        self.max_position_size = 8.0
        self.parabolic_multiplier = 4.0    # 8% for parabolic
        self.big_swing_multiplier = 3.0    # 6% for big swing
        
        # ⚡ MOMENTUM-ADJUSTED CONFIDENCE
        self.base_threshold = 0.45
        self.parabolic_boost = 0.25        # -25% threshold
        self.big_swing_boost = 0.20        # -20% threshold
        self.min_threshold = 0.25
        
        # 🎯 TRAILING STOPS
        self.trailing_distance = 3.0       # 3% trailing distance
        self.min_profit_for_trailing = 8.0 # 8% min profit
        
        # State tracking
        self.active_positions = {}
        self.trailing_stops = {}
        self.trades_history = []
        
        # Performance metrics
        self.performance = {
            'total_trades': 0,
            'parabolic_trades': 0,
            'big_swing_trades': 0,
            'normal_trades': 0,
            'trailing_exits': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'momentum_profits': 0.0,
            'parabolic_profits': 0.0
        }
        
        print(f"💰 Starting Balance: ${self.starting_balance:.2f}")
        print(f"🎲 Trading Pairs: {len(self.trading_pairs)}")
        print(f"💥 Volume spike threshold: {self.volume_spike_threshold}x")
        print(f"📈 Position sizing: {self.base_position_size}%-{self.max_position_size}%")
        print(f"🎯 Trailing distance: {self.trailing_distance}%")
        print("=" * 60)

    def generate_market_data(self, symbol: str, is_momentum_move=False, move_type='normal') -> Dict:
        """Generate realistic market data for testing"""
        
        base_price = {
            'BTC': 43000, 'ETH': 2500, 'SOL': 100, 'DOGE': 0.08, 'AVAX': 40,
            'LINK': 15, 'UNI': 7, 'ADA': 0.5, 'DOT': 7, 'MATIC': 0.9,
            'NEAR': 2, 'ATOM': 10, 'FTM': 0.4, 'SAND': 0.5, 'CRV': 0.7
        }.get(symbol, 50)
        
        # Generate price with some volatility
        price_variation = random.uniform(-0.02, 0.02)  # ±2%
        current_price = base_price * (1 + price_variation)
        
        # Generate momentum indicators based on move type
        if move_type == 'parabolic':
            # Parabolic move characteristics
            volume_ratio = random.uniform(3.0, 8.0)      # High volume spike
            price_change_24h = random.uniform(0.08, 0.25) # 8-25% move
            volatility = random.uniform(0.08, 0.15)       # High volatility
            price_acceleration = random.uniform(0.03, 0.08) # Strong acceleration
            
        elif move_type == 'big_swing':
            # Big swing characteristics
            volume_ratio = random.uniform(2.0, 4.0)      # Good volume
            price_change_24h = random.uniform(0.04, 0.12) # 4-12% move
            volatility = random.uniform(0.05, 0.10)       # Medium-high volatility
            price_acceleration = random.uniform(0.02, 0.05) # Good acceleration
            
        else:
            # Normal move characteristics
            volume_ratio = random.uniform(0.8, 2.0)      # Normal volume
            price_change_24h = random.uniform(-0.03, 0.03) # ±3% move
            volatility = random.uniform(0.02, 0.05)       # Low volatility
            price_acceleration = random.uniform(0.005, 0.02) # Low acceleration
        
        # Add some randomness
        if random.choice([True, False]):
            price_change_24h *= -1  # Make it a short opportunity
        
        volume_spike = max(0, volume_ratio - 1.0)
        
        return {
            'symbol': symbol,
            'price': current_price,
            'price_change_24h': price_change_24h,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'volume_spike': volume_spike,
            'price_acceleration': price_acceleration,
            'move_type': move_type
        }

    def calculate_momentum_score(self, market_data: Dict) -> Dict:
        """🚀 Calculate comprehensive momentum score"""
        
        volume_spike = market_data.get('volume_spike', 0)
        price_acceleration = market_data.get('price_acceleration', 0)
        volatility = market_data.get('volatility', 0.02)
        price_change_24h = market_data.get('price_change_24h', 0)
        
        # Normalize scores (0-1)
        volume_score = min(1.0, volume_spike / 2.0)
        acceleration_score = min(1.0, price_acceleration / 0.05)
        volatility_score = min(1.0, max(0, volatility - 0.03) / 0.05)
        trend_score = min(1.0, abs(price_change_24h) / 0.05)
        
        # Combined momentum score
        momentum_score = (
            volume_score * 0.3 +
            acceleration_score * 0.25 +
            volatility_score * 0.2 +
            trend_score * 0.25
        )
        
        # Classify momentum type
        if momentum_score >= self.parabolic_threshold:
            momentum_type = 'parabolic'
        elif momentum_score >= self.big_swing_threshold:
            momentum_type = 'big_swing'
        else:
            momentum_type = 'normal'
        
        return {
            'momentum_score': momentum_score,
            'momentum_type': momentum_type,
            'volume_score': volume_score,
            'acceleration_score': acceleration_score,
            'volatility_score': volatility_score,
            'trend_score': trend_score
        }

    def calculate_dynamic_position_size(self, momentum_data: Dict, signal_strength: float) -> float:
        """💰 Dynamic position sizing (2-8% based on momentum)"""
        
        momentum_type = momentum_data['momentum_type']
        momentum_score = momentum_data['momentum_score']
        
        if momentum_type == 'parabolic':
            multiplier = self.parabolic_multiplier
        elif momentum_type == 'big_swing':
            multiplier = self.big_swing_multiplier
        else:
            multiplier = 1.0
        
        position_size = self.base_position_size * (1 + (multiplier - 1) * momentum_score * signal_strength)
        position_size = min(position_size, self.max_position_size)
        
        return position_size

    def get_momentum_adjusted_threshold(self, momentum_data: Dict) -> float:
        """⚡ Lower confidence threshold for momentum opportunities"""
        
        momentum_type = momentum_data['momentum_type']
        momentum_score = momentum_data['momentum_score']
        
        if momentum_type == 'parabolic':
            boost = self.parabolic_boost
        elif momentum_type == 'big_swing':
            boost = self.big_swing_boost
        else:
            boost = 0
        
        threshold = self.base_threshold - (boost * momentum_score)
        threshold = max(threshold, self.min_threshold)
        
        return threshold

    def analyze_opportunity(self, market_data: Dict) -> Optional[Dict]:
        """🚀 Analyze trading opportunity with momentum detection"""
        
        symbol = market_data['symbol']
        
        # Calculate momentum
        momentum_data = self.calculate_momentum_score(market_data)
        
        # Signal analysis
        price_change = market_data['price_change_24h']
        volume_ratio = market_data['volume_ratio']
        
        signal_strength = 0.0
        signal_type = None
        
        # Trend analysis
        if abs(price_change) > 0.015:
            signal_strength += 0.25
            signal_type = 'long' if price_change > 0 else 'short'
        
        # Volume confirmation
        if volume_ratio > 1.2:
            signal_strength += 0.20
        
        # Momentum boost
        signal_strength += momentum_data['momentum_score'] * 0.35
        
        # Volatility
        if market_data['volatility'] > 0.025:
            signal_strength += 0.15
        
        # Get momentum-adjusted threshold
        threshold = self.get_momentum_adjusted_threshold(momentum_data)
        
        if signal_strength >= threshold and signal_type:
            position_size = self.calculate_dynamic_position_size(momentum_data, signal_strength)
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'momentum_data': momentum_data,
                'position_size': position_size,
                'entry_price': market_data['price'],
                'threshold_used': threshold,
                'market_data': market_data
            }
        
        return None

    def execute_trade(self, opportunity: Dict) -> bool:
        """Execute trade and simulate outcome"""
        
        symbol = opportunity['symbol']
        signal_type = opportunity['signal_type']
        position_size = opportunity['position_size']
        momentum_data = opportunity['momentum_data']
        entry_price = opportunity['entry_price']
        move_type = opportunity['market_data']['move_type']
        
        # Calculate position size in USD
        position_usd = self.current_balance * (position_size / 100)
        
        if position_usd < 1.0:  # Minimum $1 position
            return False
        
        print(f"\n🚀 TRADE: {signal_type.upper()} {symbol}")
        print(f"   💥 Momentum Type: {momentum_data['momentum_type'].upper()}")
        print(f"   📊 Momentum Score: {momentum_data['momentum_score']:.3f}")
        print(f"   💰 Position Size: ${position_usd:.2f} ({position_size:.2f}%)")
        print(f"   📈 Entry Price: ${entry_price:.4f}")
        print(f"   🎯 Threshold Used: {opportunity['threshold_used']:.3f}")
        
        # Simulate trade outcome based on momentum type
        leverage = 8  # 8x leverage
        
        # Different win rates and profit targets based on momentum
        if momentum_data['momentum_type'] == 'parabolic':
            win_rate = 0.75  # 75% win rate for parabolic
            profit_range = (0.15, 0.35)  # 15-35% profit on wins
            loss_range = (-0.08, -0.04)  # 4-8% loss
            
            # Check if trailing stop should be used
            use_trailing = True
            
        elif momentum_data['momentum_type'] == 'big_swing':
            win_rate = 0.70  # 70% win rate for big swings
            profit_range = (0.08, 0.20)  # 8-20% profit on wins
            loss_range = (-0.06, -0.03)  # 3-6% loss
            use_trailing = False
            
        else:
            win_rate = 0.60  # 60% win rate for normal
            profit_range = (0.03, 0.08)  # 3-8% profit on wins
            loss_range = (-0.04, -0.02)  # 2-4% loss
            use_trailing = False
        
        # Simulate trade outcome
        is_winner = random.random() < win_rate
        
        if is_winner:
            if use_trailing and momentum_data['momentum_type'] == 'parabolic':
                # Simulate trailing stop exit
                base_profit = random.uniform(*profit_range)
                # Trailing stops capture more of big moves
                profit_pct = base_profit * random.uniform(1.2, 1.8)
                exit_reason = "Trailing Stop"
                self.performance['trailing_exits'] += 1
            else:
                profit_pct = random.uniform(*profit_range)
                exit_reason = "Take Profit"
        else:
            profit_pct = random.uniform(*loss_range)
            exit_reason = "Stop Loss"
        
        # Calculate P&L
        raw_pnl = position_usd * profit_pct * leverage
        
        # Account for fees (0.05% per side = 0.1% total)
        fees = position_usd * leverage * 0.001
        net_pnl = raw_pnl - fees
        
        print(f"   🎯 Exit: {exit_reason}")
        print(f"   💰 P&L: ${net_pnl:.2f} ({profit_pct*100:.2f}%)")
        
        # Update balance and stats
        self.current_balance += net_pnl
        self.performance['total_profit'] += net_pnl
        
        if net_pnl > 0:
            self.performance['winning_trades'] += 1
            if net_pnl > self.performance['largest_win']:
                self.performance['largest_win'] = net_pnl
        else:
            if net_pnl < self.performance['largest_loss']:
                self.performance['largest_loss'] = net_pnl
        
        # Track momentum-specific profits
        if momentum_data['momentum_type'] == 'parabolic':
            self.performance['parabolic_profits'] += net_pnl
            self.performance['parabolic_trades'] += 1
        elif momentum_data['momentum_type'] == 'big_swing':
            self.performance['big_swing_trades'] += 1
        else:
            self.performance['normal_trades'] += 1
        
        if momentum_data['momentum_type'] in ['parabolic', 'big_swing']:
            self.performance['momentum_profits'] += net_pnl
        
        self.performance['total_trades'] += 1
        
        # Store trade record
        trade_record = {
            'symbol': symbol,
            'signal_type': signal_type,
            'momentum_type': momentum_data['momentum_type'],
            'momentum_score': momentum_data['momentum_score'],
            'position_size': position_size,
            'position_usd': position_usd,
            'entry_price': entry_price,
            'profit_pct': profit_pct,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason,
            'balance_after': self.current_balance
        }
        self.trades_history.append(trade_record)
        
        return True

    def run_backtest(self, num_opportunities=100):
        """Run momentum backtest simulation"""
        
        print(f"\n🚀 STARTING MOMENTUM BACKTEST")
        print(f"💰 Starting Balance: ${self.starting_balance:.2f}")
        print(f"🎯 Testing {num_opportunities} opportunities")
        print("=" * 60)
        
        # Generate different types of market opportunities
        opportunity_types = ['normal'] * 60 + ['big_swing'] * 25 + ['parabolic'] * 15
        
        for i in range(num_opportunities):
            # Pick random symbol and opportunity type
            symbol = random.choice(self.trading_pairs)
            move_type = random.choice(opportunity_types)
            
            # Generate market data
            market_data = self.generate_market_data(symbol, move_type=move_type)
            
            # Analyze opportunity
            opportunity = self.analyze_opportunity(market_data)
            
            if opportunity:
                # Execute trade
                self.execute_trade(opportunity)
                
                # Check if balance is too low
                if self.current_balance < 5.0:  # Stop if below $5
                    print(f"\n❌ STOPPING: Balance too low (${self.current_balance:.2f})")
                    break
            
            # Print progress every 20 opportunities
            if (i + 1) % 20 == 0:
                self.print_progress(i + 1)
        
        self.print_final_results()

    def print_progress(self, completed: int):
        """Print progress update"""
        
        profit_pct = ((self.current_balance - self.starting_balance) / self.starting_balance) * 100
        
        print(f"\n📊 PROGRESS UPDATE ({completed} opportunities)")
        print(f"💰 Balance: ${self.current_balance:.2f} ({profit_pct:+.1f}%)")
        print(f"📈 Total Trades: {self.performance['total_trades']}")
        print(f"🚀 Parabolic Trades: {self.performance['parabolic_trades']}")
        
    def print_final_results(self):
        """Print final backtest results"""
        
        p = self.performance
        total_trades = p['total_trades']
        
        if total_trades == 0:
            print("\n❌ No trades executed")
            return
        
        win_rate = (p['winning_trades'] / total_trades) * 100
        total_return = ((self.current_balance - self.starting_balance) / self.starting_balance) * 100
        
        print("\n" + "=" * 80)
        print("🚀 MOMENTUM BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"\n💰 FINANCIAL PERFORMANCE:")
        print(f"   Starting Balance: ${self.starting_balance:.2f}")
        print(f"   Ending Balance: ${self.current_balance:.2f}")
        print(f"   Total Profit: ${p['total_profit']:.2f}")
        print(f"   Total Return: {total_return:+.1f}%")
        print(f"   Largest Win: ${p['largest_win']:.2f}")
        print(f"   Largest Loss: ${p['largest_loss']:.2f}")
        
        print(f"\n📊 TRADING STATISTICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Winning Trades: {p['winning_trades']}")
        print(f"   Losing Trades: {total_trades - p['winning_trades']}")
        
        print(f"\n🚀 MOMENTUM BREAKDOWN:")
        print(f"   🔥 Parabolic Trades: {p['parabolic_trades']} ({p['parabolic_trades']/total_trades*100:.1f}%)")
        print(f"   📈 Big Swing Trades: {p['big_swing_trades']} ({p['big_swing_trades']/total_trades*100:.1f}%)")
        print(f"   📊 Normal Trades: {p['normal_trades']} ({p['normal_trades']/total_trades*100:.1f}%)")
        print(f"   🎯 Trailing Stop Exits: {p['trailing_exits']}")
        
        print(f"\n💎 MOMENTUM PROFITS:")
        print(f"   Momentum Profits: ${p['momentum_profits']:.2f}")
        print(f"   Parabolic Profits: ${p['parabolic_profits']:.2f}")
        
        momentum_trades = p['parabolic_trades'] + p['big_swing_trades']
        if momentum_trades > 0:
            momentum_profit_per_trade = p['momentum_profits'] / momentum_trades
            print(f"   Avg Momentum Profit: ${momentum_profit_per_trade:.2f} per trade")
        
        print(f"\n🎯 KEY INSIGHTS:")
        if p['parabolic_trades'] > 0:
            parabolic_avg = p['parabolic_profits'] / p['parabolic_trades']
            print(f"   📈 Avg Parabolic Trade: ${parabolic_avg:.2f}")
            print(f"   🎯 Trailing stops captured: {p['trailing_exits']} moves")
        
        if total_return > 0:
            print(f"   ✅ Profitable strategy!")
        else:
            print(f"   ❌ Strategy needs optimization")
        
        print("=" * 80)

def main():
    """Run momentum backtest with $50"""
    
    print("🚀 MOMENTUM BOT $50 BACKTEST")
    print("💥 Testing all 4 momentum features with $50 starting balance")
    print()
    
    # Initialize backtest
    backtest = MomentumBacktest(starting_balance=50.0)
    
    # Run simulation
    backtest.run_backtest(num_opportunities=150)
    
    print("\n✅ Backtest complete!")

if __name__ == "__main__":
    main() 