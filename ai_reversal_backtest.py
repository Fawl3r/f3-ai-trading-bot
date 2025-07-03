#!/usr/bin/env python3
"""
AI Reversal Trading Bot - 2 Month Backtest
Tests the AI reversal strategy against historical data
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Install matplotlib for charts: pip install matplotlib seaborn")

@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    position_size: float
    target_profit: float
    pnl_amount: float
    pnl_percentage: float
    exit_reason: str
    confidence: float
    hold_time_hours: float
    signals: List[str]

class AIReversalBacktester:
    """Comprehensive backtesting system for AI reversal strategy"""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Same configuration as live bot
        self.config = {
            "symbol": "SOL-USDT-SWAP",
            "base_position_size": 150.0,
            "max_position_size": 500.0,
            "profit_targets": {
                "conservative": 200,
                "aggressive": 300,
                "maximum": 500
            },
            "leverage": 10,
            "min_confidence": 70,
            "max_daily_trades": 5,
            "risk_per_trade": 0.15,
            "stop_loss_pct": 5.0,
            "max_hold_hours": 24,
        }
        
        # Backtest results
        self.trades = []
        self.daily_trades = {}
        self.balance_history = []
        
        # Performance metrics
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        
        print("🤖 AI REVERSAL BACKTESTER")
        print("📊 Testing 2 months of historical data")
        print("🎯 Same strategy as live bot")
        print("="*60)
    
    def generate_realistic_data(self, days: int = 60) -> pd.DataFrame:
        """Generate realistic SOL price data for backtesting"""
        print(f"📊 Generating {days} days of realistic SOL price data...")
        
        # Generate realistic SOL price movements
        minutes = days * 24 * 60
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=minutes,
            freq='1min'
        )
        
        # Start around realistic SOL price
        base_price = 140.0
        prices = [base_price]
        
        # Generate realistic price movements with trends and volatility
        daily_trend = 0
        for i in range(1, minutes):
            # Daily trend change
            if i % 1440 == 0:  # New day
                daily_trend = np.random.normal(0, 0.02)  # ±2% daily trend
            
            # Intraday volatility
            volatility = np.random.normal(0, 0.003)  # 0.3% per minute volatility
            
            # Add some mean reversion
            current_price = prices[-1]
            if current_price > base_price * 1.15:  # 15% above base
                volatility -= 0.002  # Pull down
            elif current_price < base_price * 0.85:  # 15% below base
                volatility += 0.002  # Pull up
            
            # Create price movement
            price_change = daily_trend/1440 + volatility  # Daily trend + intraday noise
            new_price = current_price * (1 + price_change)
            new_price = max(new_price, 10.0)  # Price floor
            prices.append(new_price)
        
        # Create realistic OHLCV data
        data = []
        for i, timestamp in enumerate(timestamps):
            close_price = prices[i]
            
            # Generate realistic OHLC from close price
            volatility_range = close_price * 0.002  # 0.2% range
            high_offset = abs(np.random.normal(0, volatility_range/2))
            low_offset = abs(np.random.normal(0, volatility_range/2))
            
            if i == 0:
                open_price = close_price
            else:
                open_price = prices[i-1]
            
            high_price = max(open_price, close_price) + high_offset
            low_price = min(open_price, close_price) - low_offset
            
            # Realistic volume
            base_volume = 5000
            volume_noise = np.random.uniform(0.5, 2.0)
            volume = base_volume * volume_noise
            
            data.append({
                'timestamp': int(timestamp.timestamp() * 1000),
                'datetime': timestamp,
                'open': round(open_price, 4),
                'high': round(high_price, 4),
                'low': round(low_price, 4),
                'close': round(close_price, 4),
                'volume': round(volume, 2)
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} realistic price candles")
        print(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
    
    def extract_features(self, data: pd.DataFrame, index: int) -> Dict:
        """Extract AI features at specific point in time"""
        if index < 50:
            return {}
        
        # Get data up to current point
        current_data = data.iloc[:index+1]
        
        try:
            high = current_data['high'].values
            low = current_data['low'].values
            close = current_data['close'].values
            volume = current_data['volume'].values
            
            features = {}
            
            # Basic price features
            features['current_price'] = close[-1]
            features['price_change_1'] = (close[-1] - close[-2]) / close[-2] * 100
            features['price_change_5'] = (close[-1] - close[-6]) / close[-6] * 100 if len(close) >= 6 else 0
            features['price_change_15'] = (close[-1] - close[-16]) / close[-16] * 100 if len(close) >= 16 else 0
            
            # Range analysis (your strategy)
            recent_high = np.max(high[-20:])
            recent_low = np.min(low[-20:])
            range_size = recent_high - recent_low
            
            if range_size > 0:
                features['distance_from_high'] = (recent_high - close[-1]) / range_size * 100
                features['distance_from_low'] = (close[-1] - recent_low) / range_size * 100
                features['range_position'] = features['distance_from_low']
            else:
                features['range_position'] = 50
            
            # Moving averages
            features['sma_5'] = np.mean(close[-5:])
            features['sma_20'] = np.mean(close[-20:])
            
            # RSI
            features['rsi'] = self._calculate_rsi(close)
            
            # Volume analysis
            features['volume_ratio'] = volume[-1] / np.mean(volume[-10:]) if len(volume) >= 10 else 1.0
            
            return features
            
        except Exception as e:
            return {}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_reversal_signals(self, features: Dict) -> Dict:
        """Same AI signal detection as live bot"""
        signals = {
            'reversal_probability': 0,
            'direction': None,
            'confidence': 0,
            'signals': [],
            'is_range_extreme': False
        }
        
        if not features:
            return signals
        
        # Range extreme detection (your strategy)
        range_pos = features.get('range_position', 50)
        
        # Near range high (short opportunity)
        if range_pos >= 85:
            signals['is_range_extreme'] = True
            signals['direction'] = 'short'
            signals['signals'].append('range_high_extreme')
            signals['reversal_probability'] += 30
        
        # Near range low (long opportunity)
        elif range_pos <= 15:
            signals['is_range_extreme'] = True
            signals['direction'] = 'long'
            signals['signals'].append('range_low_extreme')
            signals['reversal_probability'] += 30
        
        # RSI extremes
        rsi = features.get('rsi', 50)
        if rsi >= 75:
            signals['signals'].append('rsi_overbought')
            signals['reversal_probability'] += 20
            if not signals['direction']:
                signals['direction'] = 'short'
        elif rsi <= 25:
            signals['signals'].append('rsi_oversold')
            signals['reversal_probability'] += 20
            if not signals['direction']:
                signals['direction'] = 'long'
        
        # Volume confirmation
        vol_ratio = features.get('volume_ratio', 1.0)
        if vol_ratio > 1.5:
            signals['signals'].append('volume_confirmation')
            signals['reversal_probability'] += 15
        
        # Price momentum
        price_change_1 = features.get('price_change_1', 0)
        price_change_5 = features.get('price_change_5', 0)
        
        if signals['direction'] == 'short' and price_change_1 > 0 and price_change_5 > price_change_1:
            signals['signals'].append('momentum_weakening')
            signals['reversal_probability'] += 10
        elif signals['direction'] == 'long' and price_change_1 < 0 and abs(price_change_5) > abs(price_change_1):
            signals['signals'].append('momentum_weakening')
            signals['reversal_probability'] += 10
        
        signals['confidence'] = min(signals['reversal_probability'], 100)
        return signals
    
    def should_enter_trade(self, signals: Dict, features: Dict, current_date: datetime) -> bool:
        """Check if we should enter a trade"""
        # Daily trade limit
        date_key = current_date.date()
        daily_count = self.daily_trades.get(date_key, 0)
        if daily_count >= self.config['max_daily_trades']:
            return False
        
        # Must be at range extreme
        if not signals['is_range_extreme']:
            return False
        
        # Must have high AI confidence
        if signals['confidence'] < self.config['min_confidence']:
            return False
        
        # Must have clear direction
        if not signals['direction']:
            return False
        
        # Additional confirmations
        confirmations = 0
        
        # Volume confirmation
        if features.get('volume_ratio', 1.0) > 1.3:
            confirmations += 1
        
        # RSI extreme
        rsi = features.get('rsi', 50)
        if (signals['direction'] == 'short' and rsi > 70) or (signals['direction'] == 'long' and rsi < 30):
            confirmations += 1
        
        # Range position extreme
        range_pos = features.get('range_position', 50)
        if (signals['direction'] == 'short' and range_pos > 80) or (signals['direction'] == 'long' and range_pos < 20):
            confirmations += 1
        
        return confirmations >= 2
    
    def calculate_position_size(self, signals: Dict) -> Tuple[float, float]:
        """Calculate position size and target profit"""
        confidence_multiplier = signals['confidence'] / 100
        base_size = self.config['base_position_size']
        max_size = self.config['max_position_size']
        
        position_size = base_size + (max_size - base_size) * confidence_multiplier
        position_size = min(position_size, self.current_balance * self.config['risk_per_trade'])
        
        # Determine profit target
        if signals['confidence'] >= 90:
            target_profit = self.config['profit_targets']['maximum']
        elif signals['confidence'] >= 80:
            target_profit = self.config['profit_targets']['aggressive']
        else:
            target_profit = self.config['profit_targets']['conservative']
        
        return position_size, target_profit
    
    def simulate_trade(self, entry_data: Dict, data: pd.DataFrame, entry_index: int) -> Optional[BacktestTrade]:
        """Simulate a complete trade from entry to exit"""
        entry_time = entry_data['datetime']
        entry_price = entry_data['close']
        direction = entry_data['direction']
        position_size = entry_data['position_size']
        target_profit = entry_data['target_profit']
        confidence = entry_data['confidence']
        signals = entry_data['signals']
        
        # Look for exit conditions
        max_exit_index = min(entry_index + (self.config['max_hold_hours'] * 60), len(data) - 1)
        
        for i in range(entry_index + 1, max_exit_index + 1):
            current_candle = data.iloc[i]
            current_price = current_candle['close']
            hold_time = (current_candle['datetime'] - entry_time).total_seconds() / 3600
            
            # Calculate current P&L
            if direction == 'long':
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100
            
            pnl_amount = position_size * (pnl_pct / 100) * self.config['leverage']
            
            # Check exit conditions
            exit_reason = None
            
            # Target profit reached
            if pnl_amount >= target_profit:
                exit_reason = "target_profit"
            
            # Stop loss
            elif pnl_amount <= -(position_size * self.config['stop_loss_pct'] / 100):
                exit_reason = "stop_loss"
            
            # Maximum hold time
            elif hold_time >= self.config['max_hold_hours']:
                exit_reason = "time_exit"
            
            # AI reversal detection (after 1 hour)
            elif hold_time > 1:
                features = self.extract_features(data, i)
                reversal_signals = self.detect_reversal_signals(features)
                
                if (reversal_signals['confidence'] > 80 and 
                    reversal_signals['direction'] != direction and
                    pnl_amount > 0):
                    exit_reason = "ai_reversal"
            
            # Exit if condition met
            if exit_reason:
                return BacktestTrade(
                    entry_time=entry_time,
                    exit_time=current_candle['datetime'],
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=current_price,
                    position_size=position_size,
                    target_profit=target_profit,
                    pnl_amount=pnl_amount,
                    pnl_percentage=pnl_pct * self.config['leverage'],
                    exit_reason=exit_reason,
                    confidence=confidence,
                    hold_time_hours=hold_time,
                    signals=signals
                )
        
        # Force exit at end of data
        final_candle = data.iloc[max_exit_index]
        final_price = final_candle['close']
        hold_time = (final_candle['datetime'] - entry_time).total_seconds() / 3600
        
        if direction == 'long':
            pnl_pct = (final_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - final_price) / entry_price * 100
        
        pnl_amount = position_size * (pnl_pct / 100) * self.config['leverage']
        
        return BacktestTrade(
            entry_time=entry_time,
            exit_time=final_candle['datetime'],
            direction=direction,
            entry_price=entry_price,
            exit_price=final_price,
            position_size=position_size,
            target_profit=target_profit,
            pnl_amount=pnl_amount,
            pnl_percentage=pnl_pct * self.config['leverage'],
            exit_reason="data_end",
            confidence=confidence,
            hold_time_hours=hold_time,
            signals=signals
        )
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run the complete backtest"""
        print(f"\n🚀 Starting backtest on {len(data)} candles...")
        print(f"📅 Period: {data['datetime'].min()} to {data['datetime'].max()}")
        
        # Progress tracking
        total_candles = len(data)
        progress_interval = max(total_candles // 20, 1000)  # 5% intervals
        
        for i in range(50, total_candles):  # Start after 50 candles for indicators
            current_candle = data.iloc[i]
            current_time = current_candle['datetime']
            
            # Progress update
            if i % progress_interval == 0:
                progress = (i / total_candles) * 100
                print(f"📊 Progress: {progress:.1f}% - {current_time.strftime('%Y-%m-%d %H:%M')} - Trades: {self.total_trades}")
            
            # Extract features and detect signals
            features = self.extract_features(data, i)
            if not features:
                continue
            
            signals = self.detect_reversal_signals(features)
            
            # Check for entry
            if self.should_enter_trade(signals, features, current_time):
                position_size, target_profit = self.calculate_position_size(signals)
                
                # Record daily trade
                date_key = current_time.date()
                self.daily_trades[date_key] = self.daily_trades.get(date_key, 0) + 1
                
                # Create entry data
                entry_data = {
                    'datetime': current_time,
                    'close': current_candle['close'],
                    'direction': signals['direction'],
                    'position_size': position_size,
                    'target_profit': target_profit,
                    'confidence': signals['confidence'],
                    'signals': signals['signals']
                }
                
                # Simulate the trade
                trade = self.simulate_trade(entry_data, data, i)
                if trade:
                    # Update balance and stats
                    self.current_balance += trade.pnl_amount
                    self.total_profit += trade.pnl_amount
                    self.total_trades += 1
                    
                    if trade.pnl_amount > 0:
                        self.wins += 1
                    else:
                        self.losses += 1
                    
                    # Track drawdown
                    if self.current_balance > self.peak_balance:
                        self.peak_balance = self.current_balance
                    
                    drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
                    self.max_drawdown = max(self.max_drawdown, drawdown)
                    
                    self.trades.append(trade)
                    
                    # Record balance
                    self.balance_history.append({
                        'datetime': trade.exit_time,
                        'balance': self.current_balance,
                        'trade_pnl': trade.pnl_amount
                    })
                    
                    # Skip ahead to exit time to avoid overlapping trades
                    exit_index = data[data['datetime'] <= trade.exit_time].index
                    if len(exit_index) > 0:
                        i = max(i, exit_index[-1])
        
        print("✅ Backtest completed!")
        return self.generate_results()
    
    def generate_results(self) -> Dict:
        """Generate comprehensive backtest results"""
        if self.total_trades == 0:
            return {"error": "No trades executed during backtest period"}
        
        win_rate = (self.wins / self.total_trades) * 100
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate additional metrics
        winning_trades = [t for t in self.trades if t.pnl_amount > 0]
        losing_trades = [t for t in self.trades if t.pnl_amount <= 0]
        
        avg_win = np.mean([t.pnl_amount for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_amount for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl_amount for t in winning_trades) / sum(t.pnl_amount for t in losing_trades)) if losing_trades else float('inf')
        
        # Target hit analysis
        target_hits = sum(1 for t in self.trades if t.exit_reason == "target_profit")
        target_hit_rate = (target_hits / self.total_trades) * 100
        
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "total_return": total_return,
            "final_balance": self.current_balance,
            "total_profit": self.total_profit,
            "max_drawdown": self.max_drawdown,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "target_hit_rate": target_hit_rate,
            "target_hits": target_hits,
            "trades": self.trades,
            "balance_history": self.balance_history
        }
    
    def print_results(self, results: Dict):
        """Print detailed backtest results"""
        print("\n" + "="*80)
        print("🤖 AI REVERSAL BOT - 2 MONTH BACKTEST RESULTS")
        print("="*80)
        
        print(f"📊 TRADE STATISTICS:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Wins: {results['wins']} | Losses: {results['losses']}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Target Hit Rate: {results['target_hit_rate']:.1f}% ({results['target_hits']} targets hit)")
        
        print(f"\n💰 FINANCIAL PERFORMANCE:")
        print(f"   Initial Balance: ${self.initial_balance:.2f}")
        print(f"   Final Balance: ${results['final_balance']:.2f}")
        print(f"   Total Return: {results['total_return']:+.1f}%")
        print(f"   Total Profit: ${results['total_profit']:+.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.1f}%")
        
        print(f"\n📈 TRADE ANALYSIS:")
        print(f"   Average Win: ${results['avg_win']:.2f}")
        print(f"   Average Loss: ${results['avg_loss']:.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        
        # Show some example trades
        if results['trades']:
            print(f"\n📋 SAMPLE TRADES:")
            for i, trade in enumerate(results['trades'][:5]):  # Show first 5 trades
                profit_icon = "💚" if trade.pnl_amount > 0 else "❌"
                print(f"   {i+1}. {profit_icon} {trade.direction.upper()} @ ${trade.entry_price:.4f} → ${trade.exit_price:.4f}")
                print(f"      P&L: ${trade.pnl_amount:+.2f} | Exit: {trade.exit_reason} | Hold: {trade.hold_time_hours:.1f}h")
        
        # Best and worst trades
        if results['trades']:
            best_trade = max(results['trades'], key=lambda t: t.pnl_amount)
            worst_trade = min(results['trades'], key=lambda t: t.pnl_amount)
            
            print(f"\n🏆 BEST TRADE:")
            print(f"   {best_trade.direction.upper()} on {best_trade.entry_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   P&L: ${best_trade.pnl_amount:+.2f} ({best_trade.pnl_percentage:+.1f}%)")
            print(f"   Exit: {best_trade.exit_reason} after {best_trade.hold_time_hours:.1f}h")
            
            print(f"\n❌ WORST TRADE:")
            print(f"   {worst_trade.direction.upper()} on {worst_trade.entry_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   P&L: ${worst_trade.pnl_amount:+.2f} ({worst_trade.pnl_percentage:+.1f}%)")
            print(f"   Exit: {worst_trade.exit_reason} after {worst_trade.hold_time_hours:.1f}h")
        
        print("="*80)
        print("🎯 CONCLUSION:")
        if results['win_rate'] >= 70 and results['total_return'] > 0:
            print("✅ Strategy shows STRONG performance - Ready for live trading!")
        elif results['win_rate'] >= 60 and results['total_return'] > 0:
            print("⚡ Strategy shows GOOD performance - Consider optimization")
        else:
            print("⚠️ Strategy needs optimization before live trading")
        print("="*80)

def main():
    print("🤖 AI REVERSAL BOT BACKTESTER")
    print("📊 Testing your manual strategy with AI precision")
    print("📅 2 months of historical data analysis")
    
    # Get configuration
    balance = input("\n💰 Starting balance for backtest (default $1000): ").strip()
    initial_balance = float(balance) if balance else 1000.0
    
    days = input("📅 Days to backtest (default 60): ").strip()
    backtest_days = int(days) if days else 60
    
    print(f"\n🚀 Starting backtest with ${initial_balance:.2f} over {backtest_days} days...")
    
    # Create backtester
    backtester = AIReversalBacktester(initial_balance)
    
    # Generate realistic data
    data = backtester.generate_realistic_data(backtest_days)
    
    # Run backtest
    results = backtester.run_backtest(data)
    
    if "error" in results:
        print(f"❌ {results['error']}")
        return
    
    # Print results
    backtester.print_results(results)
    
    print("\n✅ Backtest completed!")
    print("🎯 This shows how your AI bot would have performed")
    print("💡 Use these results to optimize your strategy")

if __name__ == "__main__":
    main() 