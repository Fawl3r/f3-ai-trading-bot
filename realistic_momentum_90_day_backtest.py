#!/usr/bin/env python3
"""
🎯 REALISTIC MOMENTUM BOT - 90 DAY BACKTEST
Comprehensive testing with realistic market simulation

✅ 90 days of realistic market data
✅ Proper momentum detection
✅ Conservative position sizing
✅ Detailed performance tracking
"""

import asyncio
import json
import logging
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticMarketSimulator:
    """📊 Realistic market data simulator based on crypto patterns"""
    
    def __init__(self):
        # Base prices for major cryptos (realistic levels)
        self.base_prices = {
            'BTC': 65000,
            'ETH': 2500, 
            'SOL': 150,
            'AVAX': 35,
            'LINK': 15
        }
        
        # Realistic market parameters
        self.daily_volatility = {
            'BTC': 0.04,    # 4% daily volatility
            'ETH': 0.05,    # 5% daily volatility
            'SOL': 0.08,    # 8% daily volatility
            'AVAX': 0.09,   # 9% daily volatility
            'LINK': 0.07    # 7% daily volatility
        }
        
        # Volume patterns (normalized)
        self.base_volume = {
            'BTC': 1000000,
            'ETH': 800000,
            'SOL': 500000,
            'AVAX': 200000,
            'LINK': 150000
        }
        
        self.market_data = {}
        self.generate_90_day_data()
    
    def generate_90_day_data(self):
        """Generate 90 days of realistic market data"""
        
        print("📊 Generating 90 days of realistic market data...")
        
        start_date = datetime.now() - timedelta(days=90)
        
        for symbol in self.base_prices.keys():
            prices = []
            volumes = []
            timestamps = []
            
            current_price = self.base_prices[symbol]
            
            # Generate data for each day
            for day in range(90):
                for hour in range(24):  # Hourly data
                    timestamp = start_date + timedelta(days=day, hours=hour)
                    
                    # Realistic price movement with trends
                    daily_vol = self.daily_volatility[symbol]
                    
                    # Add weekly and monthly trends
                    week_trend = 0.001 * np.sin(day / 7 * 2 * np.pi)
                    month_trend = 0.002 * np.sin(day / 30 * 2 * np.pi)
                    
                    # Random walk with drift
                    price_change = np.random.normal(week_trend + month_trend, daily_vol / 24)
                    current_price = current_price * (1 + price_change)
                    
                    # Prevent unrealistic prices
                    min_price = self.base_prices[symbol] * 0.3
                    max_price = self.base_prices[symbol] * 3.0
                    current_price = max(min_price, min(max_price, current_price))
                    
                    # Volume with realistic patterns
                    base_vol = self.base_volume[symbol]
                    volume_multiplier = np.random.lognormal(0, 0.5)
                    volume = base_vol * volume_multiplier
                    
                    # Higher volume during high volatility
                    if abs(price_change) > daily_vol / 12:
                        volume *= 1.5 + abs(price_change) * 10
                    
                    prices.append(current_price)
                    volumes.append(volume)
                    timestamps.append(timestamp)
            
            self.market_data[symbol] = {
                'prices': prices,
                'volumes': volumes,
                'timestamps': timestamps
            }
        
        print("✅ Market data generation complete")
    
    def get_market_data(self, symbol, timestamp):
        """Get market data for specific symbol and time"""
        if symbol not in self.market_data:
            return None
        
        data = self.market_data[symbol]
        timestamps = data['timestamps']
        prices = data['prices']
        volumes = data['volumes']
        
        # Find closest timestamp
        closest_idx = 0
        min_diff = float('inf')
        
        for i, ts in enumerate(timestamps):
            diff = abs((ts - timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        # Get 24 hours of data for analysis
        start_idx = max(0, closest_idx - 24)
        end_idx = min(len(prices), closest_idx + 1)
        
        if end_idx - start_idx < 6:
            return None
        
        period_prices = prices[start_idx:end_idx]
        period_volumes = volumes[start_idx:end_idx]
        
        current_price = prices[closest_idx]
        price_24h_ago = period_prices[0]
        price_change_24h = (current_price - price_24h_ago) / price_24h_ago
        
        # Volume analysis
        avg_volume = sum(period_volumes) / len(period_volumes)
        current_volume = volumes[closest_idx]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volatility
        volatility = np.std(period_prices) / np.mean(period_prices) if len(period_prices) > 1 else 0
        
        # Trend strength
        recent_prices = period_prices[-6:] if len(period_prices) >= 6 else period_prices
        if len(recent_prices) >= 2:
            trend_strength = abs((recent_prices[-1] - recent_prices[0]) / recent_prices[0])
        else:
            trend_strength = 0
        
        return {
            'symbol': symbol,
            'price': current_price,
            'price_change_24h': price_change_24h,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'timestamp': timestamp
        }

class RealisticMomentumBacktest:
    """🧪 Comprehensive 90-day backtest"""
    
    def __init__(self, starting_balance=50.0):
        print("🎯 REALISTIC MOMENTUM BOT - 90 DAY BACKTEST")
        print("💎 Testing with realistic market simulation")
        print("=" * 60)
        
        self.starting_balance = starting_balance
        self.balance = starting_balance
        
        # Bot settings
        self.trading_pairs = ['BTC', 'ETH', 'SOL', 'AVAX', 'LINK']
        self.min_position_size = 0.5
        self.max_position_size = 2.0
        self.base_position_size = 1.0
        self.min_profit_target = 3.0
        self.max_profit_target = 8.0
        self.stop_loss_pct = 2.0
        self.daily_loss_limit = 5.0
        self.max_open_positions = 3
        self.max_daily_trades = 10
        
        # Momentum thresholds
        self.volume_spike_min = 1.5
        self.volatility_min = 0.02
        self.trend_strength_min = 0.01
        
        # Market simulator
        self.market_sim = RealisticMarketSimulator()
        
        # Trading state
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_date = None
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_fees': 0.0,
            'max_drawdown': 0.0,
            'peak_balance': starting_balance,
            'daily_stats': [],
            'trade_history': [],
            'win_rate_by_pair': {},
            'profit_by_pair': {},
            'momentum_type_stats': {
                'high_momentum': {'trades': 0, 'wins': 0, 'profit': 0},
                'medium_momentum': {'trades': 0, 'wins': 0, 'profit': 0},
                'low_momentum': {'trades': 0, 'wins': 0, 'profit': 0}
            }
        }
        
        print(f"💰 Starting balance: ${starting_balance:.2f}")
        print(f"🎲 Trading pairs: {len(self.trading_pairs)}")
        print(f"📊 Position size: {self.min_position_size}%-{self.max_position_size}%")
        print("=" * 60)
    
    def reset_daily_limits(self, current_date):
        """Reset daily limits for new trading day"""
        if self.current_date != current_date:
            if self.current_date:
                self.performance['daily_stats'].append({
                    'date': self.current_date,
                    'balance': self.balance,
                    'daily_trades': self.daily_trades,
                    'daily_pnl': self.daily_pnl,
                    'active_positions': len(self.active_positions)
                })
            
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.current_date = current_date
    
    def can_trade(self, timestamp):
        """Check if we can make new trades"""
        current_date = timestamp.date()
        self.reset_daily_limits(current_date)
        
        # Check trading hours (8 AM to 10 PM UTC)
        hour = timestamp.hour
        if not (8 <= hour <= 22):
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            return False
        
        # Check max daily trades
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        # Check max open positions
        if len(self.active_positions) >= self.max_open_positions:
            return False
        
        return True
    
    def analyze_momentum(self, market_data):
        """📈 Realistic momentum analysis"""
        
        symbol = market_data['symbol']
        price_change = market_data['price_change_24h']
        volume_ratio = market_data['volume_ratio']
        volatility = market_data['volatility']
        trend_strength = market_data['trend_strength']
        
        # Realistic momentum scoring
        momentum_score = 0
        signals = []
        
        # Volume spike
        if volume_ratio >= self.volume_spike_min:
            momentum_score += 0.3
            signals.append(f"Volume: {volume_ratio:.1f}x")
        
        # Volatility
        if volatility >= self.volatility_min:
            momentum_score += 0.25
            signals.append(f"Vol: {volatility*100:.1f}%")
        
        # Trend strength
        if trend_strength >= self.trend_strength_min:
            momentum_score += 0.25
            signals.append(f"Trend: {trend_strength*100:.1f}%")
        
        # Price momentum
        if abs(price_change) >= 0.02:
            momentum_score += 0.2
            signals.append(f"Price: {price_change*100:.1f}%")
        
        # Determine signal
        signal_type = None
        if momentum_score >= 0.6:
            if price_change > 0:
                signal_type = 'long'
            else:
                signal_type = 'short'
        
        # Classify momentum type
        if momentum_score >= 0.8:
            momentum_type = 'high_momentum'
        elif momentum_score >= 0.6:
            momentum_type = 'medium_momentum'
        else:
            momentum_type = 'low_momentum'
        
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'momentum_score': momentum_score,
            'momentum_type': momentum_type,
            'confidence': min(momentum_score, 0.8),
            'signals': signals,
            'price_change': price_change
        }
    
    def calculate_position_size(self, momentum_data):
        """💰 Conservative position sizing"""
        
        confidence = momentum_data['confidence']
        momentum_score = momentum_data['momentum_score']
        
        base_size = self.base_position_size
        confidence_multiplier = 1 + (confidence * 0.5)
        momentum_multiplier = 1 + (momentum_score * 0.3)
        
        position_size = base_size * confidence_multiplier * momentum_multiplier
        position_size = max(position_size, self.min_position_size)
        position_size = min(position_size, self.max_position_size)
        
        return position_size
    
    def calculate_targets(self, momentum_data):
        """🎯 Realistic profit targets"""
        
        momentum_score = momentum_data['momentum_score']
        
        if momentum_score >= 0.8:
            profit_target = self.max_profit_target
        elif momentum_score >= 0.7:
            profit_target = 6.0
        else:
            profit_target = self.min_profit_target
        
        return {
            'profit_target': profit_target,
            'stop_loss': self.stop_loss_pct
        }
    
    def execute_trade(self, market_data, momentum_data, targets, timestamp):
        """📝 Execute a trade"""
        
        symbol = momentum_data['symbol']
        signal_type = momentum_data['signal_type']
        
        position_size_pct = self.calculate_position_size(momentum_data)
        position_value = self.balance * (position_size_pct / 100)
        
        entry_price = market_data['price']
        profit_target = targets['profit_target']
        stop_loss = targets['stop_loss']
        
        if signal_type == 'long':
            take_profit_price = entry_price * (1 + profit_target / 100)
            stop_loss_price = entry_price * (1 - stop_loss / 100)
        else:
            take_profit_price = entry_price * (1 - profit_target / 100)
            stop_loss_price = entry_price * (1 + stop_loss / 100)
        
        entry_fee = position_value * 0.0002
        
        trade = {
            'id': f"{symbol}_{int(timestamp.timestamp())}",
            'symbol': symbol,
            'signal_type': signal_type,
            'entry_time': timestamp,
            'entry_price': entry_price,
            'position_size_pct': position_size_pct,
            'position_value': position_value,
            'take_profit_price': take_profit_price,
            'stop_loss_price': stop_loss_price,
            'momentum_type': momentum_data['momentum_type'],
            'momentum_score': momentum_data['momentum_score'],
            'entry_fee': entry_fee,
            'status': 'open'
        }
        
        self.active_positions[trade['id']] = trade
        self.balance -= entry_fee
        self.performance['total_fees'] += entry_fee
        self.daily_trades += 1
    
    def check_exits(self, market_data, timestamp):
        """🔍 Check for trade exits"""
        
        symbol = market_data['symbol']
        current_price = market_data['price']
        
        positions_to_close = []
        
        for trade_id, trade in self.active_positions.items():
            if trade['symbol'] != symbol:
                continue
            
            should_exit = False
            exit_reason = ""
            
            if trade['signal_type'] == 'long':
                if current_price >= trade['take_profit_price']:
                    should_exit = True
                    exit_reason = "take_profit"
                elif current_price <= trade['stop_loss_price']:
                    should_exit = True
                    exit_reason = "stop_loss"
            else:
                if current_price <= trade['take_profit_price']:
                    should_exit = True
                    exit_reason = "take_profit"
                elif current_price >= trade['stop_loss_price']:
                    should_exit = True
                    exit_reason = "stop_loss"
            
            # Time-based exit (max 24 hours)
            time_open = (timestamp - trade['entry_time']).total_seconds() / 3600
            if time_open > 24:
                should_exit = True
                exit_reason = "time_limit"
            
            if should_exit:
                positions_to_close.append((trade_id, exit_reason, current_price))
        
        # Close positions
        for trade_id, exit_reason, exit_price in positions_to_close:
            self.close_trade(trade_id, exit_reason, exit_price, timestamp)
    
    def close_trade(self, trade_id, exit_reason, exit_price, timestamp):
        """💼 Close a trade"""
        
        trade = self.active_positions[trade_id]
        
        entry_price = trade['entry_price']
        position_value = trade['position_value']
        
        if trade['signal_type'] == 'long':
            price_change = (exit_price - entry_price) / entry_price
        else:
            price_change = (entry_price - exit_price) / entry_price
        
        exit_fee = position_value * 0.0002
        pnl_dollar = position_value * price_change - trade['entry_fee'] - exit_fee
        pnl_percent = (pnl_dollar / self.balance) * 100
        
        self.balance += position_value + pnl_dollar
        
        # Track performance
        self.performance['total_trades'] += 1
        self.performance['total_fees'] += exit_fee
        self.performance['total_profit'] += pnl_dollar
        
        is_winner = pnl_dollar > 0
        if is_winner:
            self.performance['winning_trades'] += 1
        else:
            self.performance['losing_trades'] += 1
        
        self.daily_pnl += pnl_percent
        
        # Track by pair
        symbol = trade['symbol']
        if symbol not in self.performance['win_rate_by_pair']:
            self.performance['win_rate_by_pair'][symbol] = {'wins': 0, 'total': 0}
            self.performance['profit_by_pair'][symbol] = 0
        
        self.performance['win_rate_by_pair'][symbol]['total'] += 1
        if is_winner:
            self.performance['win_rate_by_pair'][symbol]['wins'] += 1
        self.performance['profit_by_pair'][symbol] += pnl_dollar
        
        # Track by momentum type
        momentum_type = trade['momentum_type']
        self.performance['momentum_type_stats'][momentum_type]['trades'] += 1
        if is_winner:
            self.performance['momentum_type_stats'][momentum_type]['wins'] += 1
        self.performance['momentum_type_stats'][momentum_type]['profit'] += pnl_dollar
        
        # Update peak and drawdown
        if self.balance > self.performance['peak_balance']:
            self.performance['peak_balance'] = self.balance
        
        drawdown = (self.performance['peak_balance'] - self.balance) / self.performance['peak_balance'] * 100
        self.performance['max_drawdown'] = max(self.performance['max_drawdown'], drawdown)
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'signal_type': trade['signal_type'],
            'entry_time': trade['entry_time'],
            'exit_time': timestamp,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size_pct': trade['position_size_pct'],
            'position_value': position_value,
            'momentum_type': momentum_type,
            'momentum_score': trade['momentum_score'],
            'exit_reason': exit_reason,
            'pnl_dollar': pnl_dollar,
            'pnl_percent': price_change * 100,
            'fees': trade['entry_fee'] + exit_fee,
            'is_winner': is_winner
        }
        
        self.performance['trade_history'].append(trade_record)
        del self.active_positions[trade_id]
    
    async def run_backtest(self):
        """🚀 Run the complete 90-day backtest"""
        
        print("\n🚀 STARTING 90-DAY BACKTEST")
        print("=" * 60)
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=90)
        
        current_time = start_time
        hour_count = 0
        total_hours = 90 * 24
        
        while current_time < end_time:
            # Progress indicator
            if hour_count % 168 == 0:
                week = hour_count // 168 + 1
                progress = (hour_count / total_hours) * 100
                print(f"📅 Week {week}/13 - Progress: {progress:.1f}% - Balance: ${self.balance:.2f}")
            
            if self.can_trade(current_time):
                for symbol in self.trading_pairs:
                    if any(pos['symbol'] == symbol for pos in self.active_positions.values()):
                        continue
                    
                    market_data = self.market_sim.get_market_data(symbol, current_time)
                    if not market_data:
                        continue
                    
                    self.check_exits(market_data, current_time)
                    
                    momentum_data = self.analyze_momentum(market_data)
                    
                    if momentum_data['signal_type']:
                        targets = self.calculate_targets(momentum_data)
                        self.execute_trade(market_data, momentum_data, targets, current_time)
                        break
            
            else:
                for symbol in self.trading_pairs:
                    market_data = self.market_sim.get_market_data(symbol, current_time)
                    if market_data:
                        self.check_exits(market_data, current_time)
            
            current_time += timedelta(hours=1)
            hour_count += 1
        
        # Close remaining positions
        final_time = end_time
        for trade_id in list(self.active_positions.keys()):
            trade = self.active_positions[trade_id]
            symbol = trade['symbol']
            market_data = self.market_sim.get_market_data(symbol, final_time)
            if market_data:
                self.close_trade(trade_id, "backtest_end", market_data['price'], final_time)
        
        print("\n✅ BACKTEST COMPLETE")
        self.print_results()
    
    def print_results(self):
        """📊 Print comprehensive backtest results"""
        
        print("\n" + "=" * 80)
        print("🎯 REALISTIC MOMENTUM BOT - 90 DAY BACKTEST RESULTS")
        print("=" * 80)
        
        # Overall Performance
        total_return = (self.balance / self.starting_balance - 1) * 100
        total_trades = self.performance['total_trades']
        win_rate = (self.performance['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n💰 OVERALL PERFORMANCE:")
        print(f"   Starting Balance: ${self.starting_balance:.2f}")
        print(f"   Final Balance: ${self.balance:.2f}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Total Profit: ${self.performance['total_profit']:.2f}")
        print(f"   Total Fees: ${self.performance['total_fees']:.2f}")
        print(f"   Max Drawdown: {self.performance['max_drawdown']:.2f}%")
        
        print(f"\n📊 TRADING STATISTICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Winning Trades: {self.performance['winning_trades']}")
        print(f"   Losing Trades: {self.performance['losing_trades']}")
        print(f"   Win Rate: {win_rate:.1f}%")
        
        if total_trades > 0:
            avg_profit_per_trade = self.performance['total_profit'] / total_trades
            print(f"   Average Profit/Trade: ${avg_profit_per_trade:.2f}")
        
        # Performance by trading pair
        print(f"\n🎲 PERFORMANCE BY PAIR:")
        for symbol in self.trading_pairs:
            if symbol in self.performance['win_rate_by_pair']:
                pair_stats = self.performance['win_rate_by_pair'][symbol]
                pair_profit = self.performance['profit_by_pair'][symbol]
                pair_win_rate = (pair_stats['wins'] / pair_stats['total'] * 100) if pair_stats['total'] > 0 else 0
                print(f"   {symbol}: {pair_stats['total']} trades, {pair_win_rate:.1f}% win rate, ${pair_profit:.2f} profit")
        
        # Performance by momentum type
        print(f"\n📈 PERFORMANCE BY MOMENTUM TYPE:")
        for momentum_type, stats in self.performance['momentum_type_stats'].items():
            if stats['trades'] > 0:
                win_rate = (stats['wins'] / stats['trades']) * 100
                avg_profit = stats['profit'] / stats['trades']
                print(f"   {momentum_type.title()}: {stats['trades']} trades, {win_rate:.1f}% win rate, ${avg_profit:.2f} avg profit")
        
        # Recent trades sample
        if self.performance['trade_history']:
            print(f"\n📝 RECENT TRADES (Last 10):")
            recent_trades = self.performance['trade_history'][-10:]
            for trade in recent_trades:
                result = "🟢 WIN" if trade['is_winner'] else "🔴 LOSS"
                print(f"   {trade['symbol']} {trade['signal_type']} - {result} - ${trade['pnl_dollar']:.2f} ({trade['pnl_percent']:.1f}%) - {trade['exit_reason']}")
        
        # Daily performance summary
        if len(self.performance['daily_stats']) > 0:
            profitable_days = sum(1 for day in self.performance['daily_stats'] if day['daily_pnl'] > 0)
            total_days = len(self.performance['daily_stats'])
            
            print(f"\n📅 DAILY PERFORMANCE:")
            print(f"   Total Trading Days: {total_days}")
            print(f"   Profitable Days: {profitable_days} ({profitable_days/total_days*100:.1f}%)")
            
            if self.performance['daily_stats']:
                best_day = max(self.performance['daily_stats'], key=lambda x: x['daily_pnl'])
                worst_day = min(self.performance['daily_stats'], key=lambda x: x['daily_pnl'])
                
                print(f"   Best Day: +{best_day['daily_pnl']:.2f}% on {best_day['date']}")
                print(f"   Worst Day: {worst_day['daily_pnl']:.2f}% on {worst_day['date']}")
        
        # Risk Assessment
        print(f"\n🛡️ RISK ASSESSMENT:")
        if total_return > 0:
            risk_rating = "🟢 Conservative" if total_return < 50 else "🟡 Moderate" if total_return < 100 else "🔴 Aggressive"
        else:
            risk_rating = "🔴 High Risk"
        
        print(f"   Risk Rating: {risk_rating}")
        print(f"   Max Drawdown: {self.performance['max_drawdown']:.2f}%")
        print(f"   Consistency: {win_rate:.1f}% win rate")
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")
        if total_return > 20 and win_rate > 60 and self.performance['max_drawdown'] < 20:
            assessment = "🟢 EXCELLENT - Strong performance with good risk management"
        elif total_return > 10 and win_rate > 50:
            assessment = "🟡 GOOD - Positive returns with room for improvement"
        elif total_return > 0:
            assessment = "🟠 ACCEPTABLE - Profitable but needs optimization"
        else:
            assessment = "🔴 POOR - Needs significant improvements"
        
        print(f"   {assessment}")
        print(f"   Realistic annual projection: {total_return * 4:.1f}% (4x quarterly)")
        
        print("\n" + "=" * 80)

async def main():
    """🚀 Run the 90-day backtest"""
    
    backtest = RealisticMomentumBacktest(starting_balance=50.0)
    await backtest.run_backtest()

if __name__ == "__main__":
    asyncio.run(main()) 