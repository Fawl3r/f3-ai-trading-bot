#!/usr/bin/env python3
"""
🎯 REALISTIC ENHANCED BACKTEST WITH MARKET EVENTS
Advanced backtesting with unexpected events and adaptive position sizing

✅ Market crash events and black swans
✅ Adaptive position sizing based on performance
✅ Realistic market conditions and slippage
✅ AI learning from unexpected events
"""

import asyncio
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketEventSimulator:
    """🌪️ Simulates unexpected market events for realistic testing"""
    
    def __init__(self):
        self.events_history = []
        
        # Market event types and probabilities
        self.event_types = {
            'flash_crash': {'probability': 0.02, 'impact': -0.15, 'duration': 2},      # 2% chance, -15% drop, 2 hours
            'pump_dump': {'probability': 0.03, 'impact': 0.25, 'duration': 4},        # 3% chance, +25% pump, 4 hours  
            'regulatory_fud': {'probability': 0.01, 'impact': -0.08, 'duration': 24}, # 1% chance, -8% drop, 24 hours
            'whale_manipulation': {'probability': 0.04, 'impact': 0.12, 'duration': 6}, # 4% chance, +12%, 6 hours
            'exchange_hack': {'probability': 0.005, 'impact': -0.25, 'duration': 48}, # 0.5% chance, -25%, 48 hours
            'major_adoption': {'probability': 0.02, 'impact': 0.18, 'duration': 12},  # 2% chance, +18%, 12 hours
            'liquidity_crunch': {'probability': 0.03, 'impact': -0.05, 'duration': 8}, # 3% chance, -5%, 8 hours
            'weekend_gap': {'probability': 0.15, 'impact': 0.03, 'duration': 1}       # 15% chance, ±3%, 1 hour
        }
        
        # Active events tracker
        self.active_events = {}
    
    def check_for_events(self, timestamp, symbol):
        """Check if any market events should trigger"""
        
        events_triggered = []
        
        # Weekend effects (higher volatility)
        is_weekend = timestamp.weekday() >= 5
        weekend_multiplier = 1.5 if is_weekend else 1.0
        
        for event_type, config in self.event_types.items():
            # Adjust probability for weekends
            probability = config['probability'] * weekend_multiplier
            
            # Special conditions
            if event_type == 'weekend_gap' and not is_weekend:
                continue
                
            if random.random() < probability / 24:  # Hourly probability
                event = {
                    'type': event_type,
                    'symbol': symbol,
                    'start_time': timestamp,
                    'end_time': timestamp + timedelta(hours=config['duration']),
                    'impact': config['impact'] * (0.8 + random.random() * 0.4),  # ±20% variation
                    'severity': random.choice(['mild', 'moderate', 'severe'])
                }
                
                # Adjust impact based on severity
                if event['severity'] == 'mild':
                    event['impact'] *= 0.5
                elif event['severity'] == 'severe':
                    event['impact'] *= 1.5
                
                event_id = f"{event_type}_{symbol}_{int(timestamp.timestamp())}"
                self.active_events[event_id] = event
                events_triggered.append(event)
                
                logger.warning(f"🌪️ MARKET EVENT: {event_type} - {symbol} - Impact: {event['impact']*100:.1f}%")
        
        return events_triggered
    
    def get_event_impact(self, timestamp, symbol):
        """Get current cumulative impact from all active events"""
        
        total_impact = 0
        events_to_remove = []
        
        for event_id, event in self.active_events.items():
            if timestamp > event['end_time']:
                events_to_remove.append(event_id)
                continue
                
            if event['symbol'] == symbol or event['type'] in ['regulatory_fud', 'exchange_hack']:
                # Time decay for event impact
                time_elapsed = (timestamp - event['start_time']).total_seconds() / 3600
                time_total = (event['end_time'] - event['start_time']).total_seconds() / 3600
                decay_factor = 1 - (time_elapsed / time_total) * 0.5  # 50% decay over time
                
                total_impact += event['impact'] * decay_factor
        
        # Remove expired events
        for event_id in events_to_remove:
            del self.active_events[event_id]
        
        return total_impact

class AdaptivePositionSizer:
    """📊 Adaptive position sizing based on performance and market conditions"""
    
    def __init__(self, base_size=1.0, min_size=0.25, max_size=3.0):
        self.base_size = base_size
        self.min_size = min_size
        self.max_size = max_size
        
        # Performance tracking for adaptation
        self.recent_trades = []
        self.win_streak = 0
        self.loss_streak = 0
        self.current_drawdown = 0
        self.peak_balance = 50.0
        
        # Kelly Criterion parameters
        self.kelly_lookback = 20  # Last 20 trades for Kelly calculation
        
    def update_performance(self, trade_result, current_balance):
        """Update performance metrics for adaptive sizing"""
        
        self.recent_trades.append(trade_result)
        if len(self.recent_trades) > 50:  # Keep last 50 trades
            self.recent_trades.pop(0)
        
        # Update streaks
        if trade_result['is_winner']:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
        
        # Update drawdown
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
    
    def calculate_kelly_fraction(self):
        """Calculate Kelly Criterion fraction for optimal position sizing"""
        
        if len(self.recent_trades) < 10:
            return 0.5  # Conservative default
        
        # Get recent trades for calculation
        recent = self.recent_trades[-self.kelly_lookback:] if len(self.recent_trades) >= self.kelly_lookback else self.recent_trades
        
        wins = [t for t in recent if t['is_winner']]
        losses = [t for t in recent if not t['is_winner']]
        
        if not wins or not losses:
            return 0.5  # Conservative if all wins or all losses
        
        # Calculate win rate and average returns
        win_rate = len(wins) / len(recent)
        avg_win = sum(t['return_pct'] for t in wins) / len(wins)
        avg_loss = abs(sum(t['return_pct'] for t in losses) / len(losses))
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = loss_rate
        if avg_loss == 0:
            return 0.5
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Cap Kelly fraction for safety
        return max(0.1, min(kelly_fraction, 0.8))
    
    def calculate_adaptive_size(self, momentum_score, volatility, balance):
        """Calculate adaptive position size based on multiple factors"""
        
        # Base size from momentum
        base_multiplier = 1 + (momentum_score * 0.5)
        
        # Kelly Criterion adjustment
        kelly_fraction = self.calculate_kelly_fraction()
        kelly_multiplier = 0.5 + (kelly_fraction * 1.0)  # Scale Kelly to reasonable range
        
        # Win streak bonus (controlled)
        streak_multiplier = 1.0
        if self.win_streak >= 3:
            streak_multiplier = 1.2  # 20% bonus for win streak
        elif self.win_streak >= 5:
            streak_multiplier = 1.4  # 40% bonus for long win streak
        elif self.loss_streak >= 2:
            streak_multiplier = 0.7  # 30% reduction for loss streak
        
        # Drawdown protection
        drawdown_multiplier = 1.0
        if self.current_drawdown > 0.1:  # 10% drawdown
            drawdown_multiplier = 0.6
        elif self.current_drawdown > 0.05:  # 5% drawdown
            drawdown_multiplier = 0.8
        
        # Volatility adjustment
        volatility_multiplier = 1.0
        if volatility > 0.08:  # High volatility
            volatility_multiplier = 0.7
        elif volatility > 0.05:
            volatility_multiplier = 0.85
        
        # Combine all factors
        adaptive_size = (self.base_size * 
                        base_multiplier * 
                        kelly_multiplier * 
                        streak_multiplier * 
                        drawdown_multiplier * 
                        volatility_multiplier)
        
        # Apply bounds
        adaptive_size = max(self.min_size, min(adaptive_size, self.max_size))
        
        return adaptive_size

class RealisticEnhancedBacktest:
    """🧪 Enhanced backtest with events and adaptive sizing"""
    
    def __init__(self, starting_balance=50.0):
        print("🎯 REALISTIC ENHANCED BACKTEST WITH MARKET EVENTS")
        print("💥 AI Training with Unexpected Events & Adaptive Sizing")
        print("=" * 65)
        
        self.starting_balance = starting_balance
        self.balance = starting_balance
        
        # Trading pairs
        self.trading_pairs = ['BTC', 'ETH', 'SOL', 'AVAX', 'LINK']
        
        # Base settings (will be adapted)
        self.stop_loss_pct = 2.0
        self.daily_loss_limit = 8.0  # Increased for realistic markets
        self.max_open_positions = 3
        self.max_daily_trades = 12
        
        # Enhanced components
        self.event_simulator = MarketEventSimulator()
        self.position_sizer = AdaptivePositionSizer()
        
        # Market simulation with realistic parameters
        self.base_prices = {'BTC': 65000, 'ETH': 2500, 'SOL': 150, 'AVAX': 35, 'LINK': 15}
        self.daily_volatility = {'BTC': 0.05, 'ETH': 0.06, 'SOL': 0.08, 'AVAX': 0.09, 'LINK': 0.07}
        
        # Trading state
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_date = None
        
        # Enhanced performance tracking
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_fees': 0.0,
            'max_drawdown': 0.0,
            'peak_balance': starting_balance,
            'trade_history': [],
            'event_history': [],
            'adaptive_sizing_history': [],
            'daily_stats': [],
            'monthly_stats': []
        }
        
        # Generate realistic market data with events
        self.generate_market_data_with_events()
        
        print(f"💰 Starting balance: ${starting_balance:.2f}")
        print(f"🎲 Trading pairs: {len(self.trading_pairs)}")
        print(f"⚡ Adaptive sizing: 0.25%-3.0%")
        print(f"🌪️ Market events: ENABLED")
        print(f"🧠 AI learning: ENABLED")
        print("=" * 65)
    
    def generate_market_data_with_events(self):
        """Generate 90 days of realistic data with unexpected events"""
        
        print("📊 Generating 90 days of realistic market data with events...")
        
        self.market_data = {}
        start_date = datetime.now() - timedelta(days=90)
        
        for symbol in self.trading_pairs:
            prices = []
            volumes = []
            timestamps = []
            events_log = []
            
            current_price = self.base_prices[symbol]
            daily_vol = self.daily_volatility[symbol]
            
            for day in range(90):
                for hour in range(24):
                    timestamp = start_date + timedelta(days=day, hours=hour)
                    
                    # Check for market events
                    events = self.event_simulator.check_for_events(timestamp, symbol)
                    if events:
                        events_log.extend(events)
                    
                    # Get event impact
                    event_impact = self.event_simulator.get_event_impact(timestamp, symbol)
                    
                    # Base price movement
                    trend = 0.0001 * np.sin(day / 15 * 2 * np.pi)  # 15-day cycles
                    noise = np.random.normal(0, daily_vol / 24)
                    
                    # Apply event impact
                    total_change = trend + noise + event_impact
                    
                    # Add realistic slippage during high volatility
                    if abs(total_change) > daily_vol / 12:
                        slippage = abs(total_change) * 0.1  # 10% slippage on volatile moves
                        total_change += random.choice([-1, 1]) * slippage
                    
                    current_price = current_price * (1 + total_change)
                    
                    # Realistic price bounds
                    min_price = self.base_prices[symbol] * 0.3
                    max_price = self.base_prices[symbol] * 3.0
                    current_price = max(min_price, min(max_price, current_price))
                    
                    # Volume with event correlation
                    base_volume = 100000
                    volume_multiplier = 1 + abs(event_impact) * 5  # Higher volume during events
                    volume = base_volume * volume_multiplier * np.random.lognormal(0, 0.4)
                    
                    prices.append(current_price)
                    volumes.append(volume)
                    timestamps.append(timestamp)
            
            self.market_data[symbol] = {
                'prices': prices,
                'volumes': volumes,
                'timestamps': timestamps,
                'events': events_log
            }
        
        print("✅ Market data with events generated")
    
    def get_market_data(self, symbol, timestamp):
        """Get market data with realistic conditions"""
        
        if symbol not in self.market_data:
            return None
        
        data = self.market_data[symbol]
        timestamps = data['timestamps']
        prices = data['prices']
        volumes = data['volumes']
        
        # Find closest timestamp
        closest_idx = min(range(len(timestamps)), 
                         key=lambda i: abs((timestamps[i] - timestamp).total_seconds()))
        
        # Need enough historical data
        start_idx = max(0, closest_idx - 24)
        end_idx = min(len(prices), closest_idx + 1)
        
        if end_idx - start_idx < 6:
            return None
        
        period_prices = prices[start_idx:end_idx]
        period_volumes = volumes[start_idx:end_idx]
        
        current_price = prices[closest_idx]
        price_24h_ago = period_prices[0]
        price_change_24h = (current_price - price_24h_ago) / price_24h_ago
        
        # Enhanced market analysis
        avg_volume = sum(period_volumes) / len(period_volumes)
        current_volume = volumes[closest_idx]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        volatility = np.std(period_prices) / np.mean(period_prices) if len(period_prices) > 1 else 0
        
        # Trend analysis with more sophistication
        if len(period_prices) >= 6:
            recent_trend = (period_prices[-1] - period_prices[-6]) / period_prices[-6]
        else:
            recent_trend = 0
        
        # Get current event impact
        event_impact = self.event_simulator.get_event_impact(timestamp, symbol)
        
        return {
            'symbol': symbol,
            'price': current_price,
            'price_change_24h': price_change_24h,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'trend_strength': abs(recent_trend),
            'event_impact': event_impact,
            'timestamp': timestamp
        }
    
    def analyze_momentum_with_events(self, market_data):
        """Enhanced momentum analysis considering market events"""
        
        symbol = market_data['symbol']
        price_change = market_data['price_change_24h']
        volume_ratio = market_data['volume_ratio']
        volatility = market_data['volatility']
        trend_strength = market_data['trend_strength']
        event_impact = market_data['event_impact']
        
        # Base momentum scoring
        momentum_score = 0
        signals = []
        
        # Volume analysis (enhanced thresholds during events)
        volume_threshold = 1.8 if abs(event_impact) > 0.05 else 1.5
        if volume_ratio >= volume_threshold:
            momentum_score += 0.25
            signals.append(f"Volume: {volume_ratio:.1f}x")
        
        # Volatility analysis
        volatility_threshold = 0.03 if abs(event_impact) > 0.05 else 0.025
        if volatility >= volatility_threshold:
            momentum_score += 0.25
            signals.append(f"Vol: {volatility*100:.1f}%")
        
        # Trend strength
        if trend_strength >= 0.02:
            momentum_score += 0.25
            signals.append(f"Trend: {trend_strength*100:.1f}%")
        
        # Price momentum with event consideration
        price_threshold = 0.03 if abs(event_impact) > 0.05 else 0.025
        if abs(price_change) >= price_threshold:
            momentum_score += 0.25
            signals.append(f"Price: {price_change*100:.1f}%")
        
        # Event impact adjustment
        if abs(event_impact) > 0.1:  # Major event
            momentum_score *= 0.7  # Reduce confidence during major events
            signals.append(f"Event: {event_impact*100:.1f}%")
        elif abs(event_impact) > 0.05:  # Moderate event
            momentum_score *= 0.85
        
        # Signal generation with higher threshold
        signal_type = None
        threshold = 0.8  # Higher threshold for realistic trading
        
        if momentum_score >= threshold:
            # Consider event direction
            adjusted_price_change = price_change + event_impact
            if adjusted_price_change > 0:
                signal_type = 'long'
            else:
                signal_type = 'short'
        
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'momentum_score': momentum_score,
            'confidence': min(momentum_score, 0.9),
            'signals': signals,
            'price_change': price_change,
            'event_impact': event_impact,
            'volatility': volatility
        }
    
    async def run_enhanced_backtest(self):
        """🚀 Run enhanced backtest with events and adaptive sizing"""
        
        print("\n🚀 STARTING ENHANCED 90-DAY BACKTEST")
        print("🌪️ With market events and adaptive position sizing")
        print("=" * 65)
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=90)
        current_time = start_time
        hour_count = 0
        
        while current_time < end_time:
            # Progress updates
            if hour_count % 168 == 0:  # Weekly
                week = hour_count // 168 + 1
                progress = (hour_count / (90 * 24)) * 100
                print(f"📅 Week {week}/13 - Progress: {progress:.1f}% - Balance: ${self.balance:.2f}")
            
            # Daily reset logic
            current_date = current_time.date()
            if self.current_date != current_date:
                if self.current_date:
                    self.performance['daily_stats'].append({
                        'date': self.current_date,
                        'balance': self.balance,
                        'daily_pnl': self.daily_pnl,
                        'daily_trades': self.daily_trades
                    })
                self.daily_pnl = 0.0
                self.daily_trades = 0
                self.current_date = current_date
            
            # Trading logic
            if self.can_trade(current_time):
                for symbol in self.trading_pairs:
                    if any(pos['symbol'] == symbol for pos in self.active_positions.values()):
                        continue
                    
                    market_data = self.get_market_data(symbol, current_time)
                    if not market_data:
                        continue
                    
                    # Check exits first
                    self.check_exits_with_slippage(market_data, current_time)
                    
                    # Analyze momentum with events
                    momentum_data = self.analyze_momentum_with_events(market_data)
                    
                    if momentum_data['signal_type']:
                        # Calculate adaptive position size
                        adaptive_size = self.position_sizer.calculate_adaptive_size(
                            momentum_data['momentum_score'],
                            momentum_data['volatility'],
                            self.balance
                        )
                        
                        self.execute_trade_with_realistic_costs(
                            market_data, momentum_data, adaptive_size, current_time
                        )
                        break
            
            else:
                # Check exits even outside trading hours
                for symbol in self.trading_pairs:
                    market_data = self.get_market_data(symbol, current_time)
                    if market_data:
                        self.check_exits_with_slippage(market_data, current_time)
            
            current_time += timedelta(hours=1)
            hour_count += 1
        
        # Close remaining positions
        for trade_id in list(self.active_positions.keys()):
            trade = self.active_positions[trade_id]
            market_data = self.get_market_data(trade['symbol'], end_time)
            if market_data:
                self.close_trade_with_costs(trade_id, "backtest_end", market_data['price'], end_time)
        
        print("\n✅ ENHANCED BACKTEST COMPLETE")
        self.print_enhanced_results()
    
    def can_trade(self, timestamp):
        """Enhanced trading conditions check"""
        hour = timestamp.hour
        if not (6 <= hour <= 23):  # Extended hours for crypto
            return False
        
        if self.daily_pnl <= -self.daily_loss_limit:
            return False
        
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        if len(self.active_positions) >= self.max_open_positions:
            return False
        
        return True
    
    def execute_trade_with_realistic_costs(self, market_data, momentum_data, position_size_pct, timestamp):
        """Execute trade with realistic slippage and fees"""
        
        symbol = momentum_data['symbol']
        signal_type = momentum_data['signal_type']
        
        position_value = self.balance * (position_size_pct / 100)
        entry_price = market_data['price']
        
        # Realistic slippage based on volatility and position size
        volatility = market_data['volatility']
        slippage = min(0.005, volatility * position_size_pct * 0.001)  # Max 0.5% slippage
        
        if signal_type == 'long':
            entry_price *= (1 + slippage)  # Buy higher
        else:
            entry_price *= (1 - slippage)  # Sell lower
        
        # Dynamic profit targets based on volatility
        if volatility > 0.08:
            profit_target = 5.0  # Lower target in high volatility
        elif volatility > 0.05:
            profit_target = 6.0
        else:
            profit_target = 8.0
        
        # Calculate exit prices
        if signal_type == 'long':
            take_profit_price = entry_price * (1 + profit_target / 100)
            stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100)
        else:
            take_profit_price = entry_price * (1 - profit_target / 100)
            stop_loss_price = entry_price * (1 + self.stop_loss_pct / 100)
        
        # Realistic fees (higher during high volatility)
        base_fee = 0.001  # 0.1% base fee
        volatility_fee_multiplier = 1 + (volatility * 2)  # Higher fees during volatility
        entry_fee = position_value * base_fee * volatility_fee_multiplier
        
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
            'momentum_score': momentum_data['momentum_score'],
            'event_impact': momentum_data['event_impact'],
            'entry_fee': entry_fee,
            'slippage': slippage,
            'volatility': volatility
        }
        
        self.active_positions[trade['id']] = trade
        self.balance -= entry_fee
        self.performance['total_fees'] += entry_fee
        self.daily_trades += 1
        
        # Log adaptive sizing decision
        self.performance['adaptive_sizing_history'].append({
            'timestamp': timestamp,
            'symbol': symbol,
            'adaptive_size': position_size_pct,
            'momentum_score': momentum_data['momentum_score'],
            'volatility': volatility
        })
    
    def check_exits_with_slippage(self, market_data, timestamp):
        """Check exits with realistic slippage"""
        
        symbol = market_data['symbol']
        current_price = market_data['price']
        volatility = market_data['volatility']
        
        positions_to_close = []
        
        for trade_id, trade in self.active_positions.items():
            if trade['symbol'] != symbol:
                continue
            
            should_exit = False
            exit_reason = ""
            
            # Apply slippage to current price for exit
            exit_slippage = min(0.003, volatility * 0.5)  # Max 0.3% exit slippage
            
            if trade['signal_type'] == 'long':
                slipped_price = current_price * (1 - exit_slippage)  # Sell lower
                if slipped_price >= trade['take_profit_price']:
                    should_exit = True
                    exit_reason = "take_profit"
                elif slipped_price <= trade['stop_loss_price']:
                    should_exit = True
                    exit_reason = "stop_loss"
            else:
                slipped_price = current_price * (1 + exit_slippage)  # Cover higher
                if slipped_price <= trade['take_profit_price']:
                    should_exit = True
                    exit_reason = "take_profit"
                elif slipped_price >= trade['stop_loss_price']:
                    should_exit = True
                    exit_reason = "stop_loss"
            
            # Time-based exit
            time_open = (timestamp - trade['entry_time']).total_seconds() / 3600
            if time_open > 18:  # 18 hour max
                should_exit = True
                exit_reason = "time_limit"
                slipped_price = current_price * (1 - exit_slippage if trade['signal_type'] == 'long' else 1 + exit_slippage)
            
            if should_exit:
                positions_to_close.append((trade_id, exit_reason, slipped_price))
        
        for trade_id, exit_reason, exit_price in positions_to_close:
            self.close_trade_with_costs(trade_id, exit_reason, exit_price, timestamp)
    
    def close_trade_with_costs(self, trade_id, exit_reason, exit_price, timestamp):
        """Close trade with realistic costs and update AI learning"""
        
        trade = self.active_positions[trade_id]
        
        entry_price = trade['entry_price']
        position_value = trade['position_value']
        
        # Calculate returns
        if trade['signal_type'] == 'long':
            price_change_pct = (exit_price - entry_price) / entry_price
        else:
            price_change_pct = (entry_price - exit_price) / entry_price
        
        # Realistic exit fees
        volatility = trade['volatility']
        exit_fee = position_value * 0.001 * (1 + volatility * 2)
        
        # Calculate P&L
        gross_pnl = position_value * price_change_pct
        net_pnl = gross_pnl - trade['entry_fee'] - exit_fee
        
        # Update balance
        self.balance += net_pnl
        
        # Track performance
        self.performance['total_trades'] += 1
        self.performance['total_fees'] += exit_fee
        self.performance['total_profit'] += net_pnl
        
        is_winner = net_pnl > 0
        if is_winner:
            self.performance['winning_trades'] += 1
        else:
            self.performance['losing_trades'] += 1
        
        self.daily_pnl += (net_pnl / self.starting_balance) * 100
        
        # Update drawdown
        if self.balance > self.performance['peak_balance']:
            self.performance['peak_balance'] = self.balance
        
        drawdown = (self.performance['peak_balance'] - self.balance) / self.performance['peak_balance'] * 100
        self.performance['max_drawdown'] = max(self.performance['max_drawdown'], drawdown)
        
        # Create trade record for AI learning
        trade_result = {
            'symbol': trade['symbol'],
            'signal_type': trade['signal_type'],
            'entry_time': trade['entry_time'],
            'exit_time': timestamp,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size_pct': trade['position_size_pct'],
            'momentum_score': trade['momentum_score'],
            'event_impact': trade['event_impact'],
            'exit_reason': exit_reason,
            'net_pnl': net_pnl,
            'return_pct': price_change_pct * 100,
            'total_fees': trade['entry_fee'] + exit_fee,
            'is_winner': is_winner,
            'volatility': volatility
        }
        
        self.performance['trade_history'].append(trade_result)
        
        # Update adaptive position sizer with trade result
        self.position_sizer.update_performance(trade_result, self.balance)
        
        del self.active_positions[trade_id]
    
    def print_enhanced_results(self):
        """Print comprehensive enhanced results"""
        
        print("\n" + "=" * 80)
        print("🎯 REALISTIC ENHANCED BACKTEST RESULTS")
        print("🌪️ With Market Events & Adaptive Position Sizing")
        print("=" * 80)
        
        # Calculate metrics
        total_return = (self.balance / self.starting_balance - 1) * 100
        total_trades = self.performance['total_trades']
        win_rate = (self.performance['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n💰 ENHANCED PERFORMANCE:")
        print(f"   Starting Balance: ${self.starting_balance:.2f}")
        print(f"   Final Balance: ${self.balance:.2f}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Net Profit: ${self.balance - self.starting_balance:.2f}")
        print(f"   Total Fees: ${self.performance['total_fees']:.2f}")
        print(f"   Max Drawdown: {self.performance['max_drawdown']:.2f}%")
        
        print(f"\n📊 TRADING STATISTICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Avg Profit/Trade: ${(self.balance - self.starting_balance) / total_trades:.2f}" if total_trades > 0 else "   No trades")
        
        # Adaptive sizing analysis
        if self.performance['adaptive_sizing_history']:
            sizes = [entry['adaptive_size'] for entry in self.performance['adaptive_sizing_history']]
            print(f"\n⚡ ADAPTIVE SIZING ANALYSIS:")
            print(f"   Average Size: {np.mean(sizes):.2f}%")
            print(f"   Size Range: {min(sizes):.2f}% - {max(sizes):.2f}%")
            print(f"   Kelly Fraction: {self.position_sizer.calculate_kelly_fraction():.3f}")
        
        # Event impact analysis
        events_during_trading = [trade for trade in self.performance['trade_history'] if abs(trade['event_impact']) > 0.02]
        if events_during_trading:
            print(f"\n🌪️ MARKET EVENTS IMPACT:")
            print(f"   Trades during events: {len(events_during_trading)}")
            event_win_rate = sum(1 for t in events_during_trading if t['is_winner']) / len(events_during_trading) * 100
            print(f"   Event win rate: {event_win_rate:.1f}%")
        
        # Recent performance
        if self.performance['trade_history']:
            print(f"\n📝 RECENT TRADES (Last 5):")
            recent = self.performance['trade_history'][-5:]
            for trade in recent:
                result = "🟢" if trade['is_winner'] else "🔴"
                event_flag = "⚡" if abs(trade['event_impact']) > 0.02 else ""
                print(f"   {trade['symbol']} {trade['signal_type']} {event_flag} - {result} ${trade['net_pnl']:.2f} ({trade['return_pct']:.1f}%)")
        
        print(f"\n🎯 ENHANCED ASSESSMENT:")
        if total_return > 50 and win_rate > 60:
            assessment = "🟢 EXCELLENT - Strong performance with realistic conditions"
        elif total_return > 25 and win_rate > 55:
            assessment = "🟡 GOOD - Solid returns considering market events"
        elif total_return > 10:
            assessment = "🟠 ACCEPTABLE - Profitable despite challenges"
        else:
            assessment = "🔴 NEEDS IMPROVEMENT - Struggling with realistic conditions"
        
        print(f"   {assessment}")
        print(f"   Realistic Annual Projection: {total_return * 4:.1f}%")
        print(f"   AI Learning: Enhanced with event adaptation")
        
        print("\n" + "=" * 80)

async def main():
    """🚀 Run enhanced backtest"""
    backtest = RealisticEnhancedBacktest(starting_balance=50.0)
    await backtest.run_enhanced_backtest()

if __name__ == "__main__":
    asyncio.run(main()) 