#!/usr/bin/env python3
"""
🚀 IMPROVED LIVE SIMULATION 30-DAY BACKTEST
Enhanced Risk Management & Conservative Approach

Key Improvements:
- Lower leverage (3-8x instead of 16-20x)
- Wider stop losses (2.5% instead of 1.2%)
- Smaller position sizes (3-6% instead of 8-15%)
- Higher AI confidence threshold (75% instead of 65%)
- Better entry filtering
- Trend following bias
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ImprovedLiveSimulation30DayBacktest:
    """Improved 30-day live simulation with better risk management"""
    
    def __init__(self, initial_balance: float = 50.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
        # IMPROVED TRADING CONFIGURATION
        self.config = {
            "trading_pairs": ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE'],
            "position_size_base_pct": 3.0,  # 3% base (was 8%)
            "position_size_max_pct": 6.0,   # Max 6% (was 15%)
            "leverage_min": 3,              # Min 3x (was 8x)
            "leverage_max": 8,              # Max 8x (was 20x)
            "stop_loss_pct": 2.5,           # 2.5% stop (was 1.2%)
            "take_profit_pct": 5.0,         # 5% target (was 2.4%)
            "max_daily_trades": 8,          # Max 8 trades/day (was 15)
            "max_concurrent_trades": 2,     # Max 2 concurrent (was 3)
            "ai_confidence_threshold": 75.0, # 75% confidence (was 65%)
            "max_hold_hours": 24,           # Max 24 hours (was 8)
            "slippage_pct": 0.05,           # 0.05% slippage
            "fee_pct": 0.075,               # 0.075% trading fee
            "daily_loss_limit_pct": 3.0,    # 3% daily loss limit (was 5%)
            "drawdown_limit_pct": 12.0,     # 12% max drawdown (was 15%)
            "trend_filter": True,           # Enable trend filtering
            "volume_filter": True,          # Enable volume filtering
            "rsi_filter": True,             # Enable RSI filtering
        }
        
        # Performance tracking
        self.trades = []
        self.daily_performance = []
        self.positions = {}
        self.daily_trades_count = 0
        self.last_trade_day = None
        self.daily_loss = 0.0
        
        # Risk management
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.circuit_breaker_triggered = False
        
        print("🚀 IMPROVED LIVE SIMULATION 30-DAY BACKTEST")
        print(f"💰 Starting Balance: ${initial_balance:.2f}")
        print(f"🛡️ Enhanced Risk Management & Conservative Approach")
        print("=" * 70)
    
    def generate_realistic_market_data(self, days: int = 30) -> pd.DataFrame:
        """Generate realistic 30-day market data with better price action"""
        print("📊 Generating realistic 30-day market data...")
        
        # Generate 30 days of 15-minute candles (less noise)
        total_candles = days * 24 * 4  # 15-minute intervals
        
        # Base prices for different assets
        base_prices = {
            'BTC': 65000,
            'ETH': 2500,
            'SOL': 150,
            'AVAX': 35,
            'DOGE': 0.15
        }
        
        all_data = []
        
        for symbol in self.config['trading_pairs']:
            base_price = base_prices[symbol]
            
            # Generate more realistic price series
            returns = np.random.normal(0, 0.004, total_candles)  # 0.4% volatility per 15min
            
            # Add trending behavior
            trend_strength = np.random.uniform(-0.001, 0.001)
            trend = np.linspace(0, trend_strength * total_candles, total_candles)
            
            # Add mean reversion
            mean_reversion = np.sin(np.linspace(0, 2*np.pi, total_candles)) * 0.002
            
            # Combine components
            returns = returns + trend + mean_reversion
            
            # Generate prices
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Generate more realistic OHLC
            high_multiplier = 1 + np.abs(np.random.normal(0, 0.002, total_candles))
            low_multiplier = 1 - np.abs(np.random.normal(0, 0.002, total_candles))
            
            # Generate volumes with realistic patterns
            base_volume = np.random.lognormal(8, 0.3, total_candles)
            volume_trend = 1 + np.abs(returns) * 5  # Higher volume on big moves
            volumes = base_volume * volume_trend
            
            # Create timestamps
            start_time = datetime.now() - timedelta(days=days)
            timestamps = [start_time + timedelta(minutes=15*i) for i in range(total_candles)]
            
            # Create DataFrame
            symbol_data = pd.DataFrame({
                'timestamp': timestamps,
                'symbol': symbol,
                'open': prices,
                'high': prices * high_multiplier,
                'low': prices * low_multiplier,
                'close': prices,
                'volume': volumes
            })
            
            # Calculate technical indicators
            symbol_data = self.calculate_technical_indicators(symbol_data)
            
            all_data.append(symbol_data)
        
        # Combine all data
        market_data = pd.concat(all_data, ignore_index=True)
        market_data = market_data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Generated {len(market_data)} data points across {len(self.config['trading_pairs'])} pairs")
        return market_data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        data = data.copy()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        data['ma_10'] = data['close'].rolling(window=10).mean()
        data['ma_20'] = data['close'].rolling(window=20).mean()
        data['ma_50'] = data['close'].rolling(window=50).mean()
        
        # Trend indicators
        data['trend_short'] = np.where(data['close'] > data['ma_10'], 1, -1)
        data['trend_medium'] = np.where(data['close'] > data['ma_20'], 1, -1)
        data['trend_long'] = np.where(data['close'] > data['ma_50'], 1, -1)
        
        # Volume indicators
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Volatility
        data['volatility'] = data['close'].rolling(window=20).std()
        data['atr'] = data[['high', 'low', 'close']].apply(
            lambda x: max(x['high'] - x['low'], 
                         abs(x['high'] - x['close']), 
                         abs(x['low'] - x['close'])), axis=1
        ).rolling(window=14).mean()
        
        # Momentum
        data['momentum'] = data['close'].pct_change(periods=10) * 100
        data['momentum_ma'] = data['momentum'].rolling(window=5).mean()
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        data['bb_std'] = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
        
        return data
    
    def generate_improved_ai_signal(self, data: pd.DataFrame, current_idx: int) -> Dict:
        """Generate improved AI trading signal with better filtering"""
        if current_idx < 50:
            return {'signal': 'hold', 'confidence': 0, 'direction': None}
        
        # Get recent data window
        window_data = data.iloc[max(0, current_idx-50):current_idx+1]
        current_row = data.iloc[current_idx]
        
        # Initialize signal
        confidence = 0
        signal = 'hold'
        direction = None
        
        # Get indicators
        rsi = current_row['rsi']
        volume_ratio = current_row['volume_ratio']
        momentum = current_row['momentum']
        trend_short = current_row['trend_short']
        trend_medium = current_row['trend_medium']
        close_price = current_row['close']
        bb_upper = current_row['bb_upper']
        bb_lower = current_row['bb_lower']
        
        # IMPROVED ENTRY LOGIC
        
        # 1. RSI-based signals (more conservative)
        if rsi < 20:  # Very oversold
            confidence += 35
            signal = 'buy'
            direction = 'long'
        elif rsi < 30 and trend_medium > 0:  # Oversold in uptrend
            confidence += 25
            signal = 'buy'
            direction = 'long'
        elif rsi > 80:  # Very overbought
            confidence += 35
            signal = 'sell'
            direction = 'short'
        elif rsi > 70 and trend_medium < 0:  # Overbought in downtrend
            confidence += 25
            signal = 'sell'
            direction = 'short'
        
        # 2. Trend confirmation (critical)
        if signal == 'buy' and trend_short > 0 and trend_medium > 0:
            confidence += 30
        elif signal == 'sell' and trend_short < 0 and trend_medium < 0:
            confidence += 30
        elif signal == 'buy' and trend_short < 0:
            confidence -= 20  # Penalty for counter-trend
        elif signal == 'sell' and trend_short > 0:
            confidence -= 20  # Penalty for counter-trend
        
        # 3. Volume confirmation
        if volume_ratio > 1.5:
            confidence += 20
        elif volume_ratio > 2.0:
            confidence += 30
        elif volume_ratio < 0.8:
            confidence -= 15  # Penalty for low volume
        
        # 4. Momentum confirmation
        if signal == 'buy' and momentum > 0.5:
            confidence += 15
        elif signal == 'sell' and momentum < -0.5:
            confidence += 15
        elif signal == 'buy' and momentum < -1.0:
            confidence -= 10  # Penalty for negative momentum
        elif signal == 'sell' and momentum > 1.0:
            confidence -= 10  # Penalty for positive momentum
        
        # 5. Bollinger Bands confirmation
        if signal == 'buy' and close_price <= bb_lower:
            confidence += 20  # Oversold by BB
        elif signal == 'sell' and close_price >= bb_upper:
            confidence += 20  # Overbought by BB
        
        # 6. Market structure (additional filtering)
        recent_volatility = window_data['volatility'].tail(5).mean()
        avg_volatility = window_data['volatility'].mean()
        
        if recent_volatility > avg_volatility * 1.5:
            confidence -= 10  # Penalty for high volatility
        
        # 7. Time-based filtering (avoid choppy periods)
        recent_rsi = window_data['rsi'].tail(5)
        rsi_choppiness = recent_rsi.std()
        
        if rsi_choppiness > 15:
            confidence -= 15  # Penalty for choppy RSI
        
        # Apply filters
        if self.config['trend_filter'] and abs(trend_medium) < 1:
            confidence -= 20  # Penalty for sideways trend
        
        if self.config['volume_filter'] and volume_ratio < 1.2:
            confidence -= 10  # Penalty for low volume
        
        if self.config['rsi_filter'] and 35 < rsi < 65:
            confidence -= 15  # Penalty for neutral RSI
        
        # Cap confidence
        confidence = max(0, min(confidence, 95))
        
        return {
            'signal': signal,
            'confidence': confidence,
            'direction': direction,
            'entry_price': current_row['close'],
            'symbol': current_row['symbol'],
            'timestamp': current_row['timestamp'],
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'momentum': momentum,
            'trend_score': trend_short + trend_medium,
            'volatility': recent_volatility
        }
    
    def calculate_dynamic_position_size(self, signal: Dict) -> float:
        """Calculate dynamic position size with improved risk management"""
        confidence = signal['confidence']
        
        # Base position size (conservative)
        base_size = self.config['position_size_base_pct']
        
        # Confidence multiplier (less aggressive)
        confidence_multiplier = 1.0 + (confidence - 75) / 200  # Smaller multiplier
        
        # Trend multiplier
        trend_score = signal.get('trend_score', 0)
        trend_multiplier = 1.0 + abs(trend_score) * 0.1
        
        # Volatility adjustment (reduce size for high volatility)
        volatility = signal.get('volatility', 0.02)
        avg_volatility = 0.02
        volatility_multiplier = max(0.5, 1.0 - (volatility - avg_volatility) / avg_volatility)
        
        # Calculate position size
        position_size = base_size * confidence_multiplier * trend_multiplier * volatility_multiplier
        
        # Cap at maximum
        position_size = min(position_size, self.config['position_size_max_pct'])
        
        # Reduce size if balance is low
        if self.current_balance < self.initial_balance * 0.9:
            position_size *= 0.8
        
        if self.current_balance < self.initial_balance * 0.7:
            position_size *= 0.6
        
        # Reduce size after consecutive losses
        if self.consecutive_losses >= 2:
            position_size *= 0.7
        
        if self.consecutive_losses >= 3:
            position_size *= 0.5
        
        return position_size
    
    def calculate_smart_leverage(self, signal: Dict) -> int:
        """Calculate smart leverage based on multiple factors"""
        confidence = signal['confidence']
        volatility = signal.get('volatility', 0.02)
        trend_score = abs(signal.get('trend_score', 0))
        
        # Base leverage calculation
        if confidence >= 85 and trend_score >= 2:
            base_leverage = self.config['leverage_max']
        elif confidence >= 80:
            base_leverage = int(self.config['leverage_max'] * 0.8)
        elif confidence >= 75:
            base_leverage = int(self.config['leverage_max'] * 0.6)
        else:
            base_leverage = self.config['leverage_min']
        
        # Volatility adjustment
        avg_volatility = 0.02
        if volatility > avg_volatility * 1.5:
            base_leverage = max(self.config['leverage_min'], int(base_leverage * 0.6))
        elif volatility > avg_volatility * 1.2:
            base_leverage = max(self.config['leverage_min'], int(base_leverage * 0.8))
        
        # Consecutive losses adjustment
        if self.consecutive_losses >= 2:
            base_leverage = max(self.config['leverage_min'], int(base_leverage * 0.7))
        
        return base_leverage
    
    def execute_improved_trade(self, signal: Dict, current_time: datetime) -> bool:
        """Execute trade with improved risk management"""
        if signal['signal'] == 'hold':
            return False
        
        # Check daily limits
        current_day = current_time.date()
        if self.last_trade_day != current_day:
            self.daily_trades_count = 0
            self.daily_loss = 0.0
            self.last_trade_day = current_day
        
        if self.daily_trades_count >= self.config['max_daily_trades']:
            return False
        
        # Check concurrent positions
        if len(self.positions) >= self.config['max_concurrent_trades']:
            return False
        
        # Check daily loss limit
        if self.daily_loss >= self.current_balance * self.config['daily_loss_limit_pct'] / 100:
            return False
        
        # Check circuit breaker
        if self.circuit_breaker_triggered:
            return False
        
        # Additional filters for consecutive losses
        if self.consecutive_losses >= 3 and signal['confidence'] < 85:
            return False
        
        # Calculate position parameters
        position_size_pct = self.calculate_dynamic_position_size(signal)
        leverage = self.calculate_smart_leverage(signal)
        
        # Calculate position value
        position_value = self.current_balance * position_size_pct / 100
        leveraged_value = position_value * leverage
        
        # Apply slippage
        entry_price = signal['entry_price']
        slippage = entry_price * self.config['slippage_pct'] / 100
        if signal['direction'] == 'long':
            entry_price += slippage
        else:
            entry_price -= slippage
        
        # Calculate stops and targets
        if signal['direction'] == 'long':
            stop_loss = entry_price * (1 - self.config['stop_loss_pct'] / 100)
            take_profit = entry_price * (1 + self.config['take_profit_pct'] / 100)
        else:
            stop_loss = entry_price * (1 + self.config['stop_loss_pct'] / 100)
            take_profit = entry_price * (1 - self.config['take_profit_pct'] / 100)
        
        # Create position
        trade_id = f"{signal['symbol']}_{len(self.trades)}"
        position = {
            'trade_id': trade_id,
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'entry_price': entry_price,
            'entry_time': current_time,
            'position_size': position_value,
            'leverage': leverage,
            'leveraged_value': leveraged_value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': signal['confidence'],
            'max_hold_time': current_time + timedelta(hours=self.config['max_hold_hours']),
            'trend_score': signal.get('trend_score', 0)
        }
        
        self.positions[trade_id] = position
        self.daily_trades_count += 1
        
        print(f"📈 TRADE EXECUTED: {signal['symbol']} {signal['direction'].upper()} | "
              f"Size: ${position_value:.2f} | Leverage: {leverage}x | "
              f"Confidence: {signal['confidence']:.1f}% | Trend: {signal.get('trend_score', 0)}")
        
        return True
    
    def manage_improved_positions(self, current_data: pd.DataFrame, current_time: datetime) -> List[Dict]:
        """Manage positions with improved exit logic"""
        closed_trades = []
        positions_to_close = []
        
        for trade_id, position in self.positions.items():
            # Get current price for the symbol
            symbol_data = current_data[current_data['symbol'] == position['symbol']]
            if symbol_data.empty:
                continue
            
            current_price = symbol_data.iloc[-1]['close']
            current_rsi = symbol_data.iloc[-1]['rsi']
            
            # Check exit conditions
            exit_reason = None
            exit_price = current_price
            
            # Stop loss check
            if position['direction'] == 'long':
                if current_price <= position['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price >= position['take_profit']:
                    exit_reason = 'take_profit'
                # Early exit for overbought conditions
                elif current_rsi > 80 and current_price > position['entry_price'] * 1.02:
                    exit_reason = 'rsi_exit'
            else:  # short
                if current_price >= position['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price <= position['take_profit']:
                    exit_reason = 'take_profit'
                # Early exit for oversold conditions
                elif current_rsi < 20 and current_price < position['entry_price'] * 0.98:
                    exit_reason = 'rsi_exit'
            
            # Time exit check
            if current_time >= position['max_hold_time']:
                exit_reason = 'time_exit'
            
            # Close position if exit condition met
            if exit_reason:
                closed_trade = self.close_improved_position(position, exit_price, exit_reason, current_time)
                closed_trades.append(closed_trade)
                positions_to_close.append(trade_id)
        
        # Remove closed positions
        for trade_id in positions_to_close:
            del self.positions[trade_id]
        
        return closed_trades
    
    def close_improved_position(self, position: Dict, exit_price: float, exit_reason: str, exit_time: datetime) -> Dict:
        """Close position with improved P&L calculation"""
        # Apply slippage
        slippage = exit_price * self.config['slippage_pct'] / 100
        if position['direction'] == 'long':
            exit_price -= slippage
        else:
            exit_price += slippage
        
        # Calculate P&L
        if position['direction'] == 'long':
            price_change = (exit_price - position['entry_price']) / position['entry_price']
        else:
            price_change = (position['entry_price'] - exit_price) / position['entry_price']
        
        # Apply leverage
        leveraged_return = price_change * position['leverage']
        
        # Calculate gross P&L
        gross_pnl = position['position_size'] * leveraged_return
        
        # Apply trading fees
        entry_fee = position['position_size'] * self.config['fee_pct'] / 100
        exit_fee = position['position_size'] * self.config['fee_pct'] / 100
        total_fees = entry_fee + exit_fee
        
        # Net P&L
        net_pnl = gross_pnl - total_fees
        
        # Update balance
        self.current_balance += net_pnl
        
        # Track daily loss
        if net_pnl < 0:
            self.daily_loss += abs(net_pnl)
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        else:
            self.consecutive_losses = 0
        
        # Update peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Check circuit breaker
        if current_drawdown >= self.config['drawdown_limit_pct']:
            self.circuit_breaker_triggered = True
            print(f"🚨 CIRCUIT BREAKER TRIGGERED! Drawdown: {current_drawdown:.2f}%")
        
        # Create trade record
        trade_record = {
            'trade_id': position['trade_id'],
            'symbol': position['symbol'],
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'hold_time_minutes': (exit_time - position['entry_time']).total_seconds() / 60,
            'position_size': position['position_size'],
            'leverage': position['leverage'],
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'fees': total_fees,
            'return_pct': leveraged_return * 100,
            'exit_reason': exit_reason,
            'confidence': position['confidence'],
            'trend_score': position.get('trend_score', 0),
            'is_winner': net_pnl > 0
        }
        
        self.trades.append(trade_record)
        
        # Print trade result
        result_emoji = "✅" if net_pnl > 0 else "❌"
        print(f"{result_emoji} TRADE CLOSED: {position['symbol']} {position['direction'].upper()} | "
              f"P&L: ${net_pnl:.2f} ({leveraged_return*100:.2f}%) | "
              f"Reason: {exit_reason} | Balance: ${self.current_balance:.2f}")
        
        return trade_record
    
    def run_improved_backtest(self) -> Dict:
        """Run the improved 30-day backtest"""
        print("\n🚀 STARTING IMPROVED 30-DAY SIMULATION...")
        
        # Generate market data
        market_data = self.generate_realistic_market_data(days=30)
        
        # Group by timestamp for simultaneous processing
        timestamps = market_data['timestamp'].unique()
        
        print(f"📊 Processing {len(timestamps)} time intervals...")
        
        # Main backtest loop
        for i, timestamp in enumerate(timestamps):
            if i % 500 == 0:
                progress = (i / len(timestamps)) * 100
                print(f"⏳ Progress: {progress:.1f}% | Balance: ${self.current_balance:.2f} | "
                      f"Trades: {len(self.trades)} | Consecutive Losses: {self.consecutive_losses}")
            
            # Get current data slice
            current_data = market_data[market_data['timestamp'] == timestamp]
            
            # Manage existing positions
            closed_trades = self.manage_improved_positions(current_data, timestamp)
            
            # Skip if circuit breaker triggered
            if self.circuit_breaker_triggered:
                continue
            
            # Generate new signals for each symbol
            for symbol in self.config['trading_pairs']:
                symbol_data = market_data[market_data['symbol'] == symbol]
                symbol_data = symbol_data[symbol_data['timestamp'] <= timestamp].reset_index(drop=True)
                
                if len(symbol_data) < 50:
                    continue
                
                # Check if we already have a position for this symbol
                has_position = any(pos['symbol'] == symbol for pos in self.positions.values())
                if has_position:
                    continue
                
                # Generate improved AI signal
                current_idx = len(symbol_data) - 1
                signal = self.generate_improved_ai_signal(symbol_data, current_idx)
                
                # Execute trade if signal is strong enough
                if signal['confidence'] >= self.config['ai_confidence_threshold']:
                    self.execute_improved_trade(signal, timestamp)
        
        # Close any remaining positions
        final_timestamp = timestamps[-1]
        final_data = market_data[market_data['timestamp'] == final_timestamp]
        
        for trade_id, position in list(self.positions.items()):
            symbol_data = final_data[final_data['symbol'] == position['symbol']]
            if not symbol_data.empty:
                final_price = symbol_data.iloc[0]['close']
                closed_trade = self.close_improved_position(position, final_price, 'backtest_end', final_timestamp)
                self.trades.append(closed_trade)
        
        self.positions.clear()
        
        # Calculate final results
        results = self.calculate_comprehensive_results()
        
        print("\n🎉 IMPROVED 30-DAY BACKTEST COMPLETE!")
        self.print_detailed_results(results)
        
        return results
    
    def calculate_comprehensive_results(self) -> Dict:
        """Calculate comprehensive backtest results"""
        if not self.trades:
            return {
                'error': 'No trades executed',
                'final_balance': self.current_balance,
                'total_return': 0
            }
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['is_winner']])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t['net_pnl'] for t in self.trades)
        total_profit = sum(t['net_pnl'] for t in self.trades if t['net_pnl'] > 0)
        total_loss = sum(t['net_pnl'] for t in self.trades if t['net_pnl'] < 0)
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Return metrics
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Trade analysis
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        returns = [t['return_pct'] for t in self.trades]
        volatility = np.std(returns) if returns else 0
        sharpe_ratio = (np.mean(returns) / volatility) if volatility > 0 else 0
        
        # Time analysis
        avg_hold_time = np.mean([t['hold_time_minutes'] for t in self.trades])
        
        # Exit reason analysis
        exit_reasons = {}
        for trade in self.trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Symbol performance
        symbol_performance = {}
        for symbol in self.config['trading_pairs']:
            symbol_trades = [t for t in self.trades if t['symbol'] == symbol]
            if symbol_trades:
                symbol_pnl = sum(t['net_pnl'] for t in symbol_trades)
                symbol_wins = len([t for t in symbol_trades if t['is_winner']])
                symbol_wr = (symbol_wins / len(symbol_trades)) * 100
                symbol_performance[symbol] = {
                    'trades': len(symbol_trades),
                    'pnl': symbol_pnl,
                    'win_rate': symbol_wr,
                    'avg_confidence': np.mean([t['confidence'] for t in symbol_trades])
                }
        
        # Daily performance
        daily_pnl = {}
        for trade in self.trades:
            day = trade['exit_time'].date()
            daily_pnl[day] = daily_pnl.get(day, 0) + trade['net_pnl']
        
        profitable_days = len([pnl for pnl in daily_pnl.values() if pnl > 0])
        total_days = len(daily_pnl)
        daily_win_rate = (profitable_days / total_days) * 100 if total_days > 0 else 0
        
        # Confidence analysis
        high_conf_trades = [t for t in self.trades if t['confidence'] >= 80]
        high_conf_wr = (len([t for t in high_conf_trades if t['is_winner']]) / len(high_conf_trades)) * 100 if high_conf_trades else 0
        
        return {
            'final_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'max_consecutive_losses': self.max_consecutive_losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'avg_hold_time_minutes': avg_hold_time,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'exit_reasons': exit_reasons,
            'symbol_performance': symbol_performance,
            'daily_win_rate': daily_win_rate,
            'profitable_days': profitable_days,
            'total_days': total_days,
            'high_confidence_win_rate': high_conf_wr,
            'high_confidence_trades': len(high_conf_trades),
            'circuit_breaker_triggered': self.circuit_breaker_triggered
        }
    
    def print_detailed_results(self, results: Dict):
        """Print comprehensive results with analysis"""
        print("\n" + "="*80)
        print("📊 IMPROVED 30-DAY LIVE SIMULATION RESULTS")
        print("="*80)
        
        print(f"💰 FINANCIAL PERFORMANCE:")
        print(f"   Initial Balance:     ${results['initial_balance']:.2f}")
        print(f"   Final Balance:       ${results['final_balance']:.2f}")
        print(f"   Total Return:        {results['total_return']:.2f}%")
        print(f"   Total P&L:           ${results['total_pnl']:.2f}")
        print(f"   Max Drawdown:        {results['max_drawdown']:.2f}%")
        
        print(f"\n📈 TRADING STATISTICS:")
        print(f"   Total Trades:        {results['total_trades']}")
        print(f"   Winning Trades:      {results['winning_trades']}")
        print(f"   Losing Trades:       {results['losing_trades']}")
        print(f"   Win Rate:            {results['win_rate']:.2f}%")
        print(f"   Profit Factor:       {results['profit_factor']:.2f}")
        
        print(f"\n⚡ RISK METRICS:")
        print(f"   Volatility:          {results['volatility']:.2f}%")
        print(f"   Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print(f"   Max Consecutive Losses: {results['max_consecutive_losses']}")
        
        print(f"\n⏱️ TRADE ANALYSIS:")
        print(f"   Average Win:         ${results['avg_win']:.2f}")
        print(f"   Average Loss:        ${results['avg_loss']:.2f}")
        print(f"   Average Trade:       ${results['avg_trade']:.2f}")
        print(f"   Avg Hold Time:       {results['avg_hold_time_minutes']:.1f} minutes")
        
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"   High Confidence Trades: {results['high_confidence_trades']}")
        print(f"   High Confidence WR:   {results['high_confidence_win_rate']:.2f}%")
        
        print(f"\n🚪 EXIT REASONS:")
        for reason, count in results['exit_reasons'].items():
            pct = (count / results['total_trades']) * 100
            print(f"   {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        print(f"\n📅 DAILY PERFORMANCE:")
        print(f"   Profitable Days:     {results['profitable_days']}/{results['total_days']}")
        print(f"   Daily Win Rate:      {results['daily_win_rate']:.2f}%")
        
        print(f"\n🔄 SYMBOL PERFORMANCE:")
        for symbol, perf in results['symbol_performance'].items():
            print(f"   {symbol}: {perf['trades']} trades, ${perf['pnl']:.2f} P&L, "
                  f"{perf['win_rate']:.1f}% WR, {perf['avg_confidence']:.1f}% conf")
        
        if results['circuit_breaker_triggered']:
            print(f"\n🚨 CIRCUIT BREAKER: Triggered due to excessive drawdown")
        
        print("\n" + "="*80)
        
        # Performance grade
        if results['total_return'] > 100:
            grade = "A+ (Exceptional)"
        elif results['total_return'] > 50:
            grade = "A (Excellent)"
        elif results['total_return'] > 25:
            grade = "B (Good)"
        elif results['total_return'] > 10:
            grade = "C+ (Decent)"
        elif results['total_return'] > 0:
            grade = "C (Profitable)"
        else:
            grade = "F (Unprofitable)"
        
        print(f"🏆 PERFORMANCE GRADE: {grade}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if results['win_rate'] < 40:
            print("   - Consider tightening entry criteria")
        if results['profit_factor'] < 1.5:
            print("   - Review risk/reward ratios")
        if results['max_drawdown'] > 10:
            print("   - Implement stricter position sizing")
        if results['high_confidence_win_rate'] > results['win_rate'] + 10:
            print("   - Focus on high-confidence trades only")
        
        print("="*80)
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"improved_live_simulation_30_day_results_{timestamp}.json"
        
        # Prepare data for JSON serialization
        json_data = {
            'results': results,
            'trades': self.trades,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert datetime objects to strings
        for trade in json_data['trades']:
            trade['entry_time'] = trade['entry_time'].isoformat()
            trade['exit_time'] = trade['exit_time'].isoformat()
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")

def main():
    """Run the improved 30-day live simulation backtest"""
    # Initialize improved backtest
    backtest = ImprovedLiveSimulation30DayBacktest(initial_balance=50.0)
    
    # Run backtest
    results = backtest.run_improved_backtest()
    
    # Save results
    backtest.save_results(results)
    
    return results

if __name__ == "__main__":
    results = main() 