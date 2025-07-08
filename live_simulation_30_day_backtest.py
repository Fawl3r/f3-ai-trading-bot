#!/usr/bin/env python3
"""
🚀 LIVE SIMULATION 30-DAY BACKTEST
Comprehensive $50 Starting Balance Test

Features:
- 30-day realistic market simulation
- Advanced AI signal generation with TSA-MAE embeddings
- Dynamic position sizing and leverage
- Real-time risk management
- Comprehensive performance analytics
- Live trading simulation with slippage and fees
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LiveSimulation30DayBacktest:
    """Comprehensive 30-day live simulation backtest"""
    
    def __init__(self, initial_balance: float = 50.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
        # Trading configuration optimized for $50 start
        self.config = {
            "trading_pairs": ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE'],
            "position_size_base_pct": 8.0,  # 8% base position size
            "position_size_max_pct": 15.0,  # Max 15% per trade
            "leverage_min": 8,               # Minimum leverage
            "leverage_max": 20,              # Maximum leverage
            "stop_loss_pct": 1.2,            # 1.2% stop loss
            "take_profit_pct": 2.4,          # 2.4% take profit (2:1 R:R)
            "max_daily_trades": 15,          # Max trades per day
            "max_concurrent_trades": 3,      # Max concurrent positions
            "ai_confidence_threshold": 65.0, # AI confidence threshold
            "max_hold_hours": 8,             # Max hold time
            "slippage_pct": 0.05,            # 0.05% slippage
            "fee_pct": 0.075,                # 0.075% trading fee
            "daily_loss_limit_pct": 5.0,     # 5% daily loss limit
            "drawdown_limit_pct": 15.0,      # 15% max drawdown
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
        
        print("🚀 LIVE SIMULATION 30-DAY BACKTEST")
        print(f"💰 Starting Balance: ${initial_balance:.2f}")
        print(f"🎯 Target: 30-day comprehensive test")
        print("=" * 60)
    
    def generate_realistic_market_data(self, days: int = 30) -> pd.DataFrame:
        """Generate realistic 30-day market data"""
        print("📊 Generating realistic 30-day market data...")
        
        # Generate 30 days of 5-minute candles
        total_candles = days * 24 * 12  # 5-minute intervals
        
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
            
            # Generate price series with realistic volatility
            returns = np.random.normal(0, 0.002, total_candles)  # 0.2% volatility per 5min
            
            # Add some trend and momentum
            trend = np.sin(np.linspace(0, 4*np.pi, total_candles)) * 0.001
            momentum = np.cumsum(np.random.normal(0, 0.0005, total_candles))
            
            returns = returns + trend + momentum
            
            # Generate prices
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Generate volumes with correlation to price movement
            volumes = np.random.lognormal(8, 0.5, total_candles)
            volumes = volumes * (1 + np.abs(returns) * 10)  # Higher volume on big moves
            
            # Create timestamps
            start_time = datetime.now() - timedelta(days=days)
            timestamps = [start_time + timedelta(minutes=5*i) for i in range(total_candles)]
            
            # Create DataFrame
            symbol_data = pd.DataFrame({
                'timestamp': timestamps,
                'symbol': symbol,
                'open': prices,
                'high': prices * (1 + np.abs(np.random.normal(0, 0.001, total_candles))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.001, total_candles))),
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
        """Calculate technical indicators"""
        data = data.copy()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        data['ma_20'] = data['close'].rolling(window=20).mean()
        data['ma_50'] = data['close'].rolling(window=50).mean()
        
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
        
        return data
    
    def generate_ai_signal(self, data: pd.DataFrame, current_idx: int) -> Dict:
        """Generate AI trading signal with TSA-MAE embeddings simulation"""
        if current_idx < 50:
            return {'signal': 'hold', 'confidence': 0, 'direction': None}
        
        # Get recent data window
        window_data = data.iloc[max(0, current_idx-50):current_idx+1]
        current_row = data.iloc[current_idx]
        
        # Simulate AI analysis
        rsi = current_row['rsi']
        volume_ratio = current_row['volume_ratio']
        momentum = current_row['momentum']
        volatility = current_row['volatility']
        
        # AI confidence calculation
        confidence = 0
        signal = 'hold'
        direction = None
        
        # RSI signals
        if rsi < 25:  # Oversold
            confidence += 30
            signal = 'buy'
            direction = 'long'
        elif rsi > 75:  # Overbought
            confidence += 30
            signal = 'sell'
            direction = 'short'
        
        # Volume confirmation
        if volume_ratio > 1.5:
            confidence += 25
        elif volume_ratio > 2.0:
            confidence += 35
        
        # Momentum confirmation
        if abs(momentum) > 0.8:
            confidence += 20
        elif abs(momentum) > 1.5:
            confidence += 30
        
        # Volatility adjustment
        if volatility > window_data['volatility'].quantile(0.7):
            confidence += 15
        
        # MA trend confirmation
        if current_row['close'] > current_row['ma_20'] and signal == 'buy':
            confidence += 15
        elif current_row['close'] < current_row['ma_20'] and signal == 'sell':
            confidence += 15
        
        # Simulate TSA-MAE embedding boost
        if confidence > 50:
            tsa_mae_boost = np.random.uniform(5, 15)
            confidence += tsa_mae_boost
        
        return {
            'signal': signal,
            'confidence': min(confidence, 95),  # Cap at 95%
            'direction': direction,
            'entry_price': current_row['close'],
            'symbol': current_row['symbol'],
            'timestamp': current_row['timestamp'],
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'momentum': momentum
        }
    
    def calculate_position_size(self, signal: Dict) -> float:
        """Calculate dynamic position size based on confidence and balance"""
        confidence = signal['confidence']
        
        # Base position size
        base_size = self.config['position_size_base_pct']
        
        # Confidence multiplier
        confidence_multiplier = 1.0 + (confidence - 50) / 100
        
        # Calculate position size
        position_size = base_size * confidence_multiplier
        
        # Cap at maximum
        position_size = min(position_size, self.config['position_size_max_pct'])
        
        # Reduce size if balance is low
        if self.current_balance < self.initial_balance * 0.8:
            position_size *= 0.7
        
        return position_size
    
    def calculate_leverage(self, signal: Dict) -> int:
        """Calculate dynamic leverage based on volatility and confidence"""
        confidence = signal['confidence']
        
        # Base leverage calculation
        if confidence >= 80:
            leverage = self.config['leverage_max']
        elif confidence >= 70:
            leverage = int(self.config['leverage_max'] * 0.8)
        else:
            leverage = self.config['leverage_min']
        
        # Adjust for volatility (lower leverage for high volatility)
        if 'volatility' in signal:
            volatility = signal.get('volatility', 0.02)
            if volatility > 0.04:  # High volatility
                leverage = max(self.config['leverage_min'], int(leverage * 0.7))
        
        return leverage
    
    def execute_trade(self, signal: Dict, current_time: datetime) -> bool:
        """Execute trade with realistic slippage and fees"""
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
        
        # Calculate position parameters
        position_size_pct = self.calculate_position_size(signal)
        leverage = self.calculate_leverage(signal)
        
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
            'max_hold_time': current_time + timedelta(hours=self.config['max_hold_hours'])
        }
        
        self.positions[trade_id] = position
        self.daily_trades_count += 1
        
        print(f"📈 TRADE EXECUTED: {signal['symbol']} {signal['direction'].upper()} | "
              f"Size: ${position_value:.2f} | Leverage: {leverage}x | "
              f"Confidence: {signal['confidence']:.1f}%")
        
        return True
    
    def manage_positions(self, current_data: pd.DataFrame, current_time: datetime) -> List[Dict]:
        """Manage open positions and check exit conditions"""
        closed_trades = []
        positions_to_close = []
        
        for trade_id, position in self.positions.items():
            # Get current price for the symbol
            symbol_data = current_data[current_data['symbol'] == position['symbol']]
            if symbol_data.empty:
                continue
            
            current_price = symbol_data.iloc[-1]['close']
            
            # Check exit conditions
            exit_reason = None
            exit_price = current_price
            
            # Stop loss check
            if position['direction'] == 'long':
                if current_price <= position['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price >= position['take_profit']:
                    exit_reason = 'take_profit'
            else:  # short
                if current_price >= position['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price <= position['take_profit']:
                    exit_reason = 'take_profit'
            
            # Time exit check
            if current_time >= position['max_hold_time']:
                exit_reason = 'time_exit'
            
            # Close position if exit condition met
            if exit_reason:
                closed_trade = self.close_position(position, exit_price, exit_reason, current_time)
                closed_trades.append(closed_trade)
                positions_to_close.append(trade_id)
        
        # Remove closed positions
        for trade_id in positions_to_close:
            del self.positions[trade_id]
        
        return closed_trades
    
    def close_position(self, position: Dict, exit_price: float, exit_reason: str, exit_time: datetime) -> Dict:
        """Close position and calculate P&L"""
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
            'is_winner': net_pnl > 0
        }
        
        self.trades.append(trade_record)
        
        # Print trade result
        result_emoji = "✅" if net_pnl > 0 else "❌"
        print(f"{result_emoji} TRADE CLOSED: {position['symbol']} {position['direction'].upper()} | "
              f"P&L: ${net_pnl:.2f} ({leveraged_return*100:.2f}%) | "
              f"Reason: {exit_reason} | Balance: ${self.current_balance:.2f}")
        
        return trade_record
    
    def run_backtest(self) -> Dict:
        """Run the complete 30-day backtest"""
        print("\n🚀 STARTING 30-DAY LIVE SIMULATION...")
        
        # Generate market data
        market_data = self.generate_realistic_market_data(days=30)
        
        # Group by timestamp for simultaneous processing
        timestamps = market_data['timestamp'].unique()
        
        print(f"📊 Processing {len(timestamps)} time intervals...")
        
        # Main backtest loop
        for i, timestamp in enumerate(timestamps):
            if i % 1000 == 0:
                progress = (i / len(timestamps)) * 100
                print(f"⏳ Progress: {progress:.1f}% | Balance: ${self.current_balance:.2f}")
            
            # Get current data slice
            current_data = market_data[market_data['timestamp'] == timestamp]
            
            # Manage existing positions
            closed_trades = self.manage_positions(current_data, timestamp)
            
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
                
                # Generate AI signal
                current_idx = len(symbol_data) - 1
                signal = self.generate_ai_signal(symbol_data, current_idx)
                
                # Execute trade if signal is strong enough
                if signal['confidence'] >= self.config['ai_confidence_threshold']:
                    self.execute_trade(signal, timestamp)
        
        # Close any remaining positions
        final_timestamp = timestamps[-1]
        final_data = market_data[market_data['timestamp'] == final_timestamp]
        
        for trade_id, position in list(self.positions.items()):
            symbol_data = final_data[final_data['symbol'] == position['symbol']]
            if not symbol_data.empty:
                final_price = symbol_data.iloc[0]['close']
                closed_trade = self.close_position(position, final_price, 'backtest_end', final_timestamp)
                self.trades.append(closed_trade)
        
        self.positions.clear()
        
        # Calculate final results
        results = self.calculate_results()
        
        print("\n🎉 30-DAY BACKTEST COMPLETE!")
        self.print_results(results)
        
        return results
    
    def calculate_results(self) -> Dict:
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
                    'win_rate': symbol_wr
                }
        
        # Daily performance
        daily_pnl = {}
        for trade in self.trades:
            day = trade['exit_time'].date()
            daily_pnl[day] = daily_pnl.get(day, 0) + trade['net_pnl']
        
        profitable_days = len([pnl for pnl in daily_pnl.values() if pnl > 0])
        total_days = len(daily_pnl)
        daily_win_rate = (profitable_days / total_days) * 100 if total_days > 0 else 0
        
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
            'exit_reasons': exit_reasons,
            'symbol_performance': symbol_performance,
            'daily_win_rate': daily_win_rate,
            'profitable_days': profitable_days,
            'total_days': total_days,
            'circuit_breaker_triggered': self.circuit_breaker_triggered
        }
    
    def print_results(self, results: Dict):
        """Print comprehensive results"""
        print("\n" + "="*80)
        print("📊 30-DAY LIVE SIMULATION RESULTS")
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
        
        print(f"\n⏱️ TRADE ANALYSIS:")
        print(f"   Average Win:         ${results['avg_win']:.2f}")
        print(f"   Average Loss:        ${results['avg_loss']:.2f}")
        print(f"   Average Trade:       ${results['avg_trade']:.2f}")
        print(f"   Avg Hold Time:       {results['avg_hold_time_minutes']:.1f} minutes")
        print(f"   Max Consecutive Losses: {results['max_consecutive_losses']}")
        
        print(f"\n🚪 EXIT REASONS:")
        for reason, count in results['exit_reasons'].items():
            pct = (count / results['total_trades']) * 100
            print(f"   {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        print(f"\n📅 DAILY PERFORMANCE:")
        print(f"   Profitable Days:     {results['profitable_days']}/{results['total_days']}")
        print(f"   Daily Win Rate:      {results['daily_win_rate']:.2f}%")
        
        print(f"\n🔄 SYMBOL PERFORMANCE:")
        for symbol, perf in results['symbol_performance'].items():
            print(f"   {symbol}: {perf['trades']} trades, ${perf['pnl']:.2f} P&L, {perf['win_rate']:.1f}% WR")
        
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
        elif results['total_return'] > 0:
            grade = "C (Profitable)"
        else:
            grade = "F (Unprofitable)"
        
        print(f"🏆 PERFORMANCE GRADE: {grade}")
        print("="*80)
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"live_simulation_30_day_results_{timestamp}.json"
        
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
    """Run the 30-day live simulation backtest"""
    # Initialize backtest
    backtest = LiveSimulation30DayBacktest(initial_balance=50.0)
    
    # Run backtest
    results = backtest.run_backtest()
    
    # Save results
    backtest.save_results(results)
    
    return results

if __name__ == "__main__":
    results = main() 