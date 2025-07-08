#!/usr/bin/env python3
"""
🚀 BALANCED LIVE SIMULATION 30-DAY BACKTEST
Optimal Balance Between Risk & Opportunity

Key Features:
- Moderate leverage (5-12x)
- Balanced stop losses (1.8%)
- Reasonable position sizes (4-8%)
- Medium AI confidence threshold (70%)
- Smart entry filtering
- Trend-aware trading
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BalancedLiveSimulation30DayBacktest:
    """Balanced 30-day live simulation with optimal risk/reward"""
    
    def __init__(self, initial_balance: float = 50.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
        # BALANCED TRADING CONFIGURATION
        self.config = {
            "trading_pairs": ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE'],
            "position_size_base_pct": 4.0,  # 4% base position
            "position_size_max_pct": 8.0,   # Max 8% position
            "leverage_min": 5,              # Min 5x leverage
            "leverage_max": 12,             # Max 12x leverage
            "stop_loss_pct": 1.8,           # 1.8% stop loss
            "take_profit_pct": 3.6,         # 3.6% take profit (2:1 R:R)
            "max_daily_trades": 12,         # Max 12 trades/day
            "max_concurrent_trades": 3,     # Max 3 concurrent
            "ai_confidence_threshold": 70.0, # 70% confidence threshold
            "max_hold_hours": 12,           # Max 12 hours hold
            "slippage_pct": 0.05,           # 0.05% slippage
            "fee_pct": 0.075,               # 0.075% trading fee
            "daily_loss_limit_pct": 4.0,    # 4% daily loss limit
            "drawdown_limit_pct": 15.0,     # 15% max drawdown
            "trend_filter": True,           # Enable trend filtering
            "volume_filter": True,          # Enable volume filtering
            "momentum_filter": True,        # Enable momentum filtering
        }
        
        # Performance tracking
        self.trades = []
        self.positions = {}
        self.daily_trades_count = 0
        self.last_trade_day = None
        self.daily_loss = 0.0
        
        # Risk management
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.circuit_breaker_triggered = False
        
        print("🚀 BALANCED LIVE SIMULATION 30-DAY BACKTEST")
        print(f"💰 Starting Balance: ${initial_balance:.2f}")
        print(f"⚖️ Optimal Balance Between Risk & Opportunity")
        print("=" * 70)
    
    def generate_realistic_market_data(self, days: int = 30) -> pd.DataFrame:
        """Generate realistic 30-day market data"""
        print("📊 Generating realistic 30-day market data...")
        
        # Generate 30 days of 10-minute candles
        total_candles = days * 24 * 6  # 10-minute intervals
        
        # Base prices
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
            
            # Generate realistic price movement
            returns = np.random.normal(0, 0.003, total_candles)  # 0.3% volatility per 10min
            
            # Add trending behavior
            trend_cycles = 3  # 3 trend cycles over 30 days
            trend = np.sin(np.linspace(0, trend_cycles * 2 * np.pi, total_candles)) * 0.001
            
            # Add some momentum bursts
            momentum_points = np.random.choice(total_candles, size=int(total_candles * 0.05), replace=False)
            momentum = np.zeros(total_candles)
            for point in momentum_points:
                momentum[point:min(point+20, total_candles)] += np.random.uniform(-0.002, 0.002)
            
            # Combine components
            returns = returns + trend + momentum
            
            # Generate prices
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Generate OHLC with realistic spreads
            high_spread = np.random.uniform(0.001, 0.003, total_candles)
            low_spread = np.random.uniform(0.001, 0.003, total_candles)
            
            # Generate volumes
            base_volume = np.random.lognormal(8, 0.4, total_candles)
            volume_multiplier = 1 + np.abs(returns) * 8  # Higher volume on moves
            volumes = base_volume * volume_multiplier
            
            # Create timestamps
            start_time = datetime.now() - timedelta(days=days)
            timestamps = [start_time + timedelta(minutes=10*i) for i in range(total_candles)]
            
            # Create DataFrame
            symbol_data = pd.DataFrame({
                'timestamp': timestamps,
                'symbol': symbol,
                'open': prices,
                'high': prices * (1 + high_spread),
                'low': prices * (1 - low_spread),
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
        """Calculate technical indicators for signal generation"""
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
        
        # Trend signals
        data['trend_short'] = np.where(data['close'] > data['ma_10'], 1, -1)
        data['trend_medium'] = np.where(data['close'] > data['ma_20'], 1, -1)
        
        # Volume indicators
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Momentum
        data['momentum'] = data['close'].pct_change(periods=5) * 100
        data['momentum_ma'] = data['momentum'].rolling(window=3).mean()
        
        # Volatility
        data['volatility'] = data['close'].rolling(window=20).std()
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        return data
    
    def generate_balanced_ai_signal(self, data: pd.DataFrame, current_idx: int) -> Dict:
        """Generate balanced AI trading signal"""
        if current_idx < 50:
            return {'signal': 'hold', 'confidence': 0, 'direction': None}
        
        # Get recent data
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
        macd_histogram = current_row['macd_histogram']
        
        # BALANCED SIGNAL GENERATION
        
        # 1. RSI-based signals (balanced thresholds)
        if rsi < 30:  # Oversold
            confidence += 25
            signal = 'buy'
            direction = 'long'
        elif rsi > 70:  # Overbought
            confidence += 25
            signal = 'sell'
            direction = 'short'
        
        # 2. Trend confirmation
        if signal == 'buy' and trend_short > 0:
            confidence += 20
        elif signal == 'sell' and trend_short < 0:
            confidence += 20
        
        if signal == 'buy' and trend_medium > 0:
            confidence += 15
        elif signal == 'sell' and trend_medium < 0:
            confidence += 15
        
        # 3. Volume confirmation
        if volume_ratio > 1.3:
            confidence += 15
        elif volume_ratio > 1.8:
            confidence += 25
        
        # 4. Momentum confirmation
        if signal == 'buy' and momentum > 0.3:
            confidence += 10
        elif signal == 'sell' and momentum < -0.3:
            confidence += 10
        
        # 5. MACD confirmation
        if signal == 'buy' and macd_histogram > 0:
            confidence += 10
        elif signal == 'sell' and macd_histogram < 0:
            confidence += 10
        
        # 6. Additional filters
        if self.config['trend_filter']:
            if signal == 'buy' and trend_medium < 0:
                confidence -= 15
            elif signal == 'sell' and trend_medium > 0:
                confidence -= 15
        
        if self.config['volume_filter']:
            if volume_ratio < 1.1:
                confidence -= 10
        
        if self.config['momentum_filter']:
            if abs(momentum) < 0.2:
                confidence -= 10
        
        # Apply market structure penalty for extreme RSI
        if rsi > 85 or rsi < 15:
            confidence -= 10  # Reduce confidence for extreme RSI
        
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
            'macd_histogram': macd_histogram
        }
    
    def calculate_position_size(self, signal: Dict) -> float:
        """Calculate balanced position size"""
        confidence = signal['confidence']
        
        # Base position size
        base_size = self.config['position_size_base_pct']
        
        # Confidence multiplier
        confidence_multiplier = 1.0 + (confidence - 70) / 100
        
        # Trend multiplier
        trend_score = abs(signal.get('trend_score', 0))
        trend_multiplier = 1.0 + trend_score * 0.1
        
        # Calculate position size
        position_size = base_size * confidence_multiplier * trend_multiplier
        
        # Cap at maximum
        position_size = min(position_size, self.config['position_size_max_pct'])
        
        # Reduce size based on balance
        if self.current_balance < self.initial_balance * 0.85:
            position_size *= 0.8
        
        # Reduce size after consecutive losses
        if self.consecutive_losses >= 2:
            position_size *= 0.8
        
        return position_size
    
    def calculate_leverage(self, signal: Dict) -> int:
        """Calculate balanced leverage"""
        confidence = signal['confidence']
        
        # Base leverage
        if confidence >= 85:
            leverage = self.config['leverage_max']
        elif confidence >= 80:
            leverage = int(self.config['leverage_max'] * 0.9)
        elif confidence >= 75:
            leverage = int(self.config['leverage_max'] * 0.7)
        else:
            leverage = int(self.config['leverage_max'] * 0.5)
        
        # Ensure minimum
        leverage = max(leverage, self.config['leverage_min'])
        
        # Reduce leverage after consecutive losses
        if self.consecutive_losses >= 2:
            leverage = max(self.config['leverage_min'], int(leverage * 0.8))
        
        return leverage
    
    def execute_trade(self, signal: Dict, current_time: datetime) -> bool:
        """Execute balanced trade"""
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
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': signal['confidence'],
            'max_hold_time': current_time + timedelta(hours=self.config['max_hold_hours']),
            'trend_score': signal.get('trend_score', 0)
        }
        
        self.positions[trade_id] = position
        self.daily_trades_count += 1
        
        print(f"📈 TRADE: {signal['symbol']} {signal['direction'].upper()} | "
              f"${position_value:.2f} | {leverage}x | {signal['confidence']:.1f}%")
        
        return True
    
    def manage_positions(self, current_data: pd.DataFrame, current_time: datetime) -> List[Dict]:
        """Manage open positions"""
        closed_trades = []
        positions_to_close = []
        
        for trade_id, position in self.positions.items():
            symbol_data = current_data[current_data['symbol'] == position['symbol']]
            if symbol_data.empty:
                continue
            
            current_price = symbol_data.iloc[-1]['close']
            
            # Check exit conditions
            exit_reason = None
            exit_price = current_price
            
            # Stop loss and take profit
            if position['direction'] == 'long':
                if current_price <= position['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price >= position['take_profit']:
                    exit_reason = 'take_profit'
            else:
                if current_price >= position['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price <= position['take_profit']:
                    exit_reason = 'take_profit'
            
            # Time exit
            if current_time >= position['max_hold_time']:
                exit_reason = 'time_exit'
            
            # Close position
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
        
        # Calculate P&L
        gross_pnl = position['position_size'] * leveraged_return
        
        # Apply fees
        entry_fee = position['position_size'] * self.config['fee_pct'] / 100
        exit_fee = position['position_size'] * self.config['fee_pct'] / 100
        total_fees = entry_fee + exit_fee
        
        net_pnl = gross_pnl - total_fees
        
        # Update balance
        self.current_balance += net_pnl
        
        # Track losses
        if net_pnl < 0:
            self.daily_loss += abs(net_pnl)
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        else:
            self.consecutive_losses = 0
        
        # Update drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Check circuit breaker
        if current_drawdown >= self.config['drawdown_limit_pct']:
            self.circuit_breaker_triggered = True
            print(f"🚨 CIRCUIT BREAKER! Drawdown: {current_drawdown:.2f}%")
        
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
        
        # Print result
        emoji = "✅" if net_pnl > 0 else "❌"
        print(f"{emoji} CLOSED: {position['symbol']} | ${net_pnl:.2f} | "
              f"{exit_reason} | Balance: ${self.current_balance:.2f}")
        
        return trade_record
    
    def run_backtest(self) -> Dict:
        """Run the balanced 30-day backtest"""
        print("\n🚀 STARTING BALANCED 30-DAY SIMULATION...")
        
        # Generate market data
        market_data = self.generate_realistic_market_data(days=30)
        
        # Process timestamps
        timestamps = sorted(market_data['timestamp'].unique())
        
        print(f"📊 Processing {len(timestamps)} time intervals...")
        
        # Main loop
        for i, timestamp in enumerate(timestamps):
            if i % 400 == 0:
                progress = (i / len(timestamps)) * 100
                print(f"⏳ {progress:.1f}% | Balance: ${self.current_balance:.2f} | "
                      f"Trades: {len(self.trades)} | Losses: {self.consecutive_losses}")
            
            # Get current data
            current_data = market_data[market_data['timestamp'] == timestamp]
            
            # Manage positions
            self.manage_positions(current_data, timestamp)
            
            # Skip if circuit breaker triggered
            if self.circuit_breaker_triggered:
                continue
            
            # Generate signals
            for symbol in self.config['trading_pairs']:
                symbol_data = market_data[market_data['symbol'] == symbol]
                symbol_data = symbol_data[symbol_data['timestamp'] <= timestamp].reset_index(drop=True)
                
                if len(symbol_data) < 50:
                    continue
                
                # Check if position exists
                has_position = any(pos['symbol'] == symbol for pos in self.positions.values())
                if has_position:
                    continue
                
                # Generate signal
                current_idx = len(symbol_data) - 1
                signal = self.generate_balanced_ai_signal(symbol_data, current_idx)
                
                # Execute if confident enough
                if signal['confidence'] >= self.config['ai_confidence_threshold']:
                    self.execute_trade(signal, timestamp)
        
        # Close remaining positions
        final_timestamp = timestamps[-1]
        final_data = market_data[market_data['timestamp'] == final_timestamp]
        
        for trade_id, position in list(self.positions.items()):
            symbol_data = final_data[final_data['symbol'] == position['symbol']]
            if not symbol_data.empty:
                final_price = symbol_data.iloc[0]['close']
                self.close_position(position, final_price, 'backtest_end', final_timestamp)
        
        self.positions.clear()
        
        # Calculate results
        results = self.calculate_results()
        
        print("\n🎉 BALANCED 30-DAY BACKTEST COMPLETE!")
        self.print_results(results)
        
        return results
    
    def calculate_results(self) -> Dict:
        """Calculate comprehensive results"""
        if not self.trades:
            return {
                'final_balance': self.current_balance,
                'initial_balance': self.initial_balance,
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': self.max_drawdown,
                'error': 'No trades executed'
            }
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['is_winner']])
        win_rate = (winning_trades / total_trades) * 100
        
        # P&L metrics
        total_pnl = sum(t['net_pnl'] for t in self.trades)
        total_profit = sum(t['net_pnl'] for t in self.trades if t['net_pnl'] > 0)
        total_loss = sum(t['net_pnl'] for t in self.trades if t['net_pnl'] < 0)
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Trade analysis
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
        avg_trade = total_pnl / total_trades
        avg_hold_time = np.mean([t['hold_time_minutes'] for t in self.trades])
        
        # Exit reasons
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
        
        return {
            'final_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
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
            'circuit_breaker_triggered': self.circuit_breaker_triggered
        }
    
    def print_results(self, results: Dict):
        """Print comprehensive results"""
        print("\n" + "="*80)
        print("📊 BALANCED 30-DAY LIVE SIMULATION RESULTS")
        print("="*80)
        
        print(f"💰 FINANCIAL PERFORMANCE:")
        print(f"   Initial Balance:     ${results['initial_balance']:.2f}")
        print(f"   Final Balance:       ${results['final_balance']:.2f}")
        print(f"   Total Return:        {results['total_return']:.2f}%")
        print(f"   Total P&L:           ${results['total_pnl']:.2f}")
        print(f"   Max Drawdown:        {results['max_drawdown']:.2f}%")
        
        if results['total_trades'] > 0:
            print(f"\n📈 TRADING STATISTICS:")
            print(f"   Total Trades:        {results['total_trades']}")
            print(f"   Winning Trades:      {results['winning_trades']}")
            print(f"   Win Rate:            {results['win_rate']:.2f}%")
            print(f"   Profit Factor:       {results['profit_factor']:.2f}")
            print(f"   Max Consecutive Losses: {results['max_consecutive_losses']}")
            
            print(f"\n⏱️ TRADE ANALYSIS:")
            print(f"   Average Win:         ${results['avg_win']:.2f}")
            print(f"   Average Loss:        ${results['avg_loss']:.2f}")
            print(f"   Average Trade:       ${results['avg_trade']:.2f}")
            print(f"   Avg Hold Time:       {results['avg_hold_time_minutes']:.1f} minutes")
            
            print(f"\n🚪 EXIT REASONS:")
            for reason, count in results['exit_reasons'].items():
                pct = (count / results['total_trades']) * 100
                print(f"   {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
            
            print(f"\n🔄 SYMBOL PERFORMANCE:")
            for symbol, perf in results['symbol_performance'].items():
                print(f"   {symbol}: {perf['trades']} trades, ${perf['pnl']:.2f} P&L, {perf['win_rate']:.1f}% WR")
        
        if results['circuit_breaker_triggered']:
            print(f"\n🚨 CIRCUIT BREAKER: Triggered")
        
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
        print("="*80)
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"balanced_live_simulation_30_day_results_{timestamp}.json"
        
        json_data = {
            'results': results,
            'trades': self.trades,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert datetime objects
        for trade in json_data['trades']:
            trade['entry_time'] = trade['entry_time'].isoformat()
            trade['exit_time'] = trade['exit_time'].isoformat()
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")

def main():
    """Run the balanced 30-day backtest"""
    backtest = BalancedLiveSimulation30DayBacktest(initial_balance=50.0)
    results = backtest.run_backtest()
    backtest.save_results(results)
    return results

if __name__ == "__main__":
    results = main() 