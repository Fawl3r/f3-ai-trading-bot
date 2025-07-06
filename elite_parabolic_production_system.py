#!/usr/bin/env python3
"""
Elite Parabolic Production System
Final optimized version for live trading
Target: 100+ trades, 40%+ win rate, 100%+ monthly return
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

class EliteParabolicProductionSystem:
    def __init__(self, initial_balance=50.0, risk_per_trade=0.0075):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        self.positions = []
        
    def calculate_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # ATR for volatility
        df['tr'] = np.maximum(df['high'] - df['low'], 
                             np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                                       np.abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['vol_sma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma']
        
        # Price momentum (multiple timeframes)
        df['roc_3'] = df['close'].pct_change(3) * 100
        df['roc_5'] = df['close'].pct_change(5) * 100
        df['roc_10'] = df['close'].pct_change(10) * 100
        df['roc_std'] = df['roc_5'].rolling(20).std()
        
        # VWAP
        df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        df['vwap_dev'] = (df['close'] - df['vwap']) / df['atr']
        
        # Trend indicators
        df['trend_ema'] = np.where(df['close'] > df['ema_21'], 1, 
                                  np.where(df['close'] < df['ema_21'], -1, 0))
        df['trend_sma'] = np.where(df['close'] > df['sma_20'], 1, 
                                  np.where(df['close'] < df['sma_20'], -1, 0))
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        return df
    
    def generate_signals(self, df):
        """Generate balanced high-quality signals"""
        df = self.calculate_indicators(df)
        
        # Time filters
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['active_hours'] = ((df['hour'] >= 13) & (df['hour'] <= 18)) | \
                            ((df['hour'] >= 8) & (df['hour'] <= 12))
        
        # Initialize signal columns
        df['signal'] = 'none'
        df['signal_strength'] = 0.0
        df['entry_reason'] = ''
        
        # Generate signals with balanced criteria
        for i in range(50, len(df)):
            current = df.iloc[i]
            
            # Skip if ATR is too small (low volatility)
            if current['atr'] < 0.01:
                continue
                
            signal_strength = 0.0
            signal_type = 'none'
            entry_reason = ''
            
            # === PARABOLIC BURST LONG SIGNALS ===
            
            # Signal 1: Strong momentum breakout
            if (current['roc_5'] > 1.5 and 
                current['vol_ratio'] > 1.3 and 
                current['rsi'] > 50 and current['rsi'] < 75 and
                current['close'] > current['bb_upper'] and
                current['macd_hist'] > 0 and
                current['trend_ema'] >= 0):
                
                signal_strength = min(current['roc_5'] / 2.0 + current['vol_ratio'] / 2.0, 3.0)
                signal_type = 'burst_long'
                entry_reason = 'Momentum Breakout Long'
            
            # Signal 2: VWAP breakout with volume
            elif (current['vwap_dev'] > 1.5 and 
                  current['vol_ratio'] > 1.4 and 
                  current['rsi'] > 55 and
                  current['close'] > current['ema_21'] and
                  current['roc_3'] > 0.8):
                
                signal_strength = min(current['vwap_dev'] / 2.0 + current['vol_ratio'] / 2.0, 3.0)
                signal_type = 'burst_long'
                entry_reason = 'VWAP Breakout Long'
            
            # Signal 3: EMA crossover with momentum
            elif (current['close'] > current['ema_8'] and 
                  current['ema_8'] > current['ema_21'] and
                  current['roc_5'] > 1.0 and
                  current['vol_ratio'] > 1.2 and
                  current['rsi'] > 45 and current['rsi'] < 70):
                
                signal_strength = min(current['roc_5'] / 2.0 + current['vol_ratio'] / 2.0, 2.5)
                signal_type = 'burst_long'
                entry_reason = 'EMA Cross Long'
            
            # === PARABOLIC BURST SHORT SIGNALS ===
            
            # Signal 4: Strong momentum breakdown
            elif (current['roc_5'] < -1.5 and 
                  current['vol_ratio'] > 1.3 and 
                  current['rsi'] < 50 and current['rsi'] > 25 and
                  current['close'] < current['bb_lower'] and
                  current['macd_hist'] < 0 and
                  current['trend_ema'] <= 0):
                
                signal_strength = min(abs(current['roc_5']) / 2.0 + current['vol_ratio'] / 2.0, 3.0)
                signal_type = 'burst_short'
                entry_reason = 'Momentum Breakdown Short'
            
            # Signal 5: VWAP breakdown with volume
            elif (current['vwap_dev'] < -1.5 and 
                  current['vol_ratio'] > 1.4 and 
                  current['rsi'] < 45 and
                  current['close'] < current['ema_21'] and
                  current['roc_3'] < -0.8):
                
                signal_strength = min(abs(current['vwap_dev']) / 2.0 + current['vol_ratio'] / 2.0, 3.0)
                signal_type = 'burst_short'
                entry_reason = 'VWAP Breakdown Short'
            
            # Signal 6: EMA crossover down with momentum
            elif (current['close'] < current['ema_8'] and 
                  current['ema_8'] < current['ema_21'] and
                  current['roc_5'] < -1.0 and
                  current['vol_ratio'] > 1.2 and
                  current['rsi'] < 55 and current['rsi'] > 30):
                
                signal_strength = min(abs(current['roc_5']) / 2.0 + current['vol_ratio'] / 2.0, 2.5)
                signal_type = 'burst_short'
                entry_reason = 'EMA Cross Short'
            
            # === PARABOLIC FADE SIGNALS ===
            
            # Signal 7: Extreme overbought fade
            elif (current['rsi'] > 75 and 
                  current['vol_ratio'] > 1.8 and 
                  current['roc_5'] > 3.0 and
                  current['close'] > current['bb_upper'] * 1.01):
                
                signal_strength = min((current['rsi'] - 50) / 20.0 + current['vol_ratio'] / 3.0, 3.0)
                signal_type = 'fade_short'
                entry_reason = 'Extreme Overbought Fade'
            
            # Signal 8: Extreme oversold fade
            elif (current['rsi'] < 25 and 
                  current['vol_ratio'] > 1.8 and 
                  current['roc_5'] < -3.0 and
                  current['close'] < current['bb_lower'] * 0.99):
                
                signal_strength = min((50 - current['rsi']) / 20.0 + current['vol_ratio'] / 3.0, 3.0)
                signal_type = 'fade_long'
                entry_reason = 'Extreme Oversold Fade'
            
            # === ADDITIONAL MOMENTUM SIGNALS ===
            
            # Signal 9: Multi-timeframe momentum long
            elif (current['roc_3'] > 0.5 and 
                  current['roc_5'] > 1.0 and
                  current['roc_10'] > 1.5 and
                  current['vol_ratio'] > 1.1 and
                  current['rsi'] > 50 and current['rsi'] < 70 and
                  current['trend_sma'] > 0):
                
                signal_strength = min((current['roc_3'] + current['roc_5']) / 3.0 + current['vol_ratio'] / 2.0, 2.5)
                signal_type = 'burst_long'
                entry_reason = 'Multi-TF Momentum Long'
            
            # Signal 10: Multi-timeframe momentum short
            elif (current['roc_3'] < -0.5 and 
                  current['roc_5'] < -1.0 and
                  current['roc_10'] < -1.5 and
                  current['vol_ratio'] > 1.1 and
                  current['rsi'] < 50 and current['rsi'] > 30 and
                  current['trend_sma'] < 0):
                
                signal_strength = min((abs(current['roc_3']) + abs(current['roc_5'])) / 3.0 + current['vol_ratio'] / 2.0, 2.5)
                signal_type = 'burst_short'
                entry_reason = 'Multi-TF Momentum Short'
            
            # === VOLATILITY BREAKOUT SIGNALS ===
            
            # Signal 11: Volatility expansion long
            elif (current['volatility'] > 0.02 and 
                  current['roc_5'] > 0.8 and
                  current['vol_ratio'] > 1.2 and
                  current['close'] > current['sma_10'] and
                  current['rsi'] > 45):
                
                signal_strength = min(current['volatility'] * 50 + current['vol_ratio'] / 2.0, 2.0)
                signal_type = 'burst_long'
                entry_reason = 'Volatility Expansion Long'
            
            # Signal 12: Volatility expansion short
            elif (current['volatility'] > 0.02 and 
                  current['roc_5'] < -0.8 and
                  current['vol_ratio'] > 1.2 and
                  current['close'] < current['sma_10'] and
                  current['rsi'] < 55):
                
                signal_strength = min(current['volatility'] * 50 + current['vol_ratio'] / 2.0, 2.0)
                signal_type = 'burst_short'
                entry_reason = 'Volatility Expansion Short'
            
            # Apply time filter and minimum strength requirement
            if (signal_type != 'none' and 
                signal_strength >= 1.0 and 
                current['active_hours']):
                
                df.iloc[i, df.columns.get_loc('signal')] = signal_type
                df.iloc[i, df.columns.get_loc('signal_strength')] = signal_strength
                df.iloc[i, df.columns.get_loc('entry_reason')] = entry_reason
        
        return df
    
    def calculate_position_size(self, price, atr, signal_strength):
        """Calculate position size with dynamic risk adjustment"""
        # Base risk amount
        base_risk = self.balance * self.risk_per_trade
        
        # Adjust risk based on signal strength (1.0 to 1.5x)
        strength_multiplier = min(0.8 + (signal_strength / 5.0), 1.5)
        adjusted_risk = base_risk * strength_multiplier
        
        # Calculate position size
        stop_distance = atr * 1.0  # 1 ATR stop
        position_size = adjusted_risk / stop_distance
        
        return position_size, stop_distance
    
    def run_backtest(self, df):
        """Run production backtest"""
        df = self.generate_signals(df)
        
        current_position = None
        
        for i in range(len(df)):
            current_bar = df.iloc[i]
            
            # Update equity curve
            self.equity_curve.append({
                'datetime': current_bar['datetime'],
                'balance': self.balance,
                'drawdown': 0.0
            })
            
            # Check for exit conditions
            if current_position:
                exit_signal = False
                exit_reason = ''
                
                # Stop loss and take profit
                if current_position['side'] == 'long':
                    if current_bar['low'] <= current_position['stop_loss']:
                        exit_price = current_position['stop_loss']
                        exit_signal = True
                        exit_reason = 'Stop Loss'
                    elif current_bar['high'] >= current_position['take_profit']:
                        exit_price = current_position['take_profit']
                        exit_signal = True
                        exit_reason = 'Take Profit'
                else:  # short
                    if current_bar['high'] >= current_position['stop_loss']:
                        exit_price = current_position['stop_loss']
                        exit_signal = True
                        exit_reason = 'Stop Loss'
                    elif current_bar['low'] <= current_position['take_profit']:
                        exit_price = current_position['take_profit']
                        exit_signal = True
                        exit_reason = 'Take Profit'
                
                # Time-based exit (max 48 hours for momentum, 24 for fade)
                max_hold = 24 if 'fade' in current_position['signal_type'] else 48
                if i - current_position['entry_index'] >= max_hold:
                    exit_price = current_bar['close']
                    exit_signal = True
                    exit_reason = 'Time Exit'
                
                # Exit position
                if exit_signal:
                    if current_position['side'] == 'long':
                        pnl = (exit_price - current_position['entry_price']) * current_position['size']
                    else:
                        pnl = (current_position['entry_price'] - exit_price) * current_position['size']
                    
                    self.balance += pnl
                    
                    # Calculate R multiple
                    risk_amount = current_position['size'] * current_position['stop_distance']
                    r_multiple = pnl / risk_amount if risk_amount > 0 else 0
                    
                    # Record trade
                    trade = {
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_bar['datetime'],
                        'side': current_position['side'],
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'size': current_position['size'],
                        'pnl': pnl,
                        'r_multiple': r_multiple,
                        'exit_reason': exit_reason,
                        'signal_type': current_position['signal_type'],
                        'signal_strength': current_position['signal_strength'],
                        'entry_reason': current_position['entry_reason'],
                        'hold_time': i - current_position['entry_index']
                    }
                    self.trades.append(trade)
                    current_position = None
            
            # Check for new entry signals
            if current_position is None and current_bar['signal'] != 'none':
                signal_type = current_bar['signal']
                signal_strength = current_bar['signal_strength']
                
                # Calculate position size
                position_size, stop_distance = self.calculate_position_size(
                    current_bar['close'], current_bar['atr'], signal_strength
                )
                
                # Determine side and levels
                if 'long' in signal_type:
                    side = 'long'
                    entry_price = current_bar['close']
                    stop_loss = entry_price - stop_distance
                    take_profit = entry_price + (stop_distance * 4.0)  # 4:1 R:R
                else:
                    side = 'short'
                    entry_price = current_bar['close']
                    stop_loss = entry_price + stop_distance
                    take_profit = entry_price - (stop_distance * 4.0)  # 4:1 R:R
                
                # Create position
                current_position = {
                    'entry_time': current_bar['datetime'],
                    'entry_index': i,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size,
                    'side': side,
                    'stop_distance': stop_distance,
                    'signal_type': signal_type,
                    'signal_strength': signal_strength,
                    'entry_reason': current_bar['entry_reason']
                }
        
        return df
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Calculate monthly return (assume 30 days per month)
        days_trading = len(self.equity_curve) / 24  # hourly data
        monthly_return = (total_return / days_trading) * 30 if days_trading > 0 else 0
        
        # Risk metrics
        returns = [t['pnl'] / self.initial_balance for t in self.trades]
        expectancy = np.mean(returns) * 100 if returns else 0
        
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / 
                           sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        
        # R multiples
        r_multiples = [t['r_multiple'] for t in self.trades]
        avg_r = np.mean(r_multiples) if r_multiples else 0
        
        # Drawdown calculation
        equity_values = [eq['balance'] for eq in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        # Sharpe ratio
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': total_return,
            'monthly_return': monthly_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'avg_r_multiple': avg_r,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'days_trading': days_trading
        }

def generate_realistic_data():
    """Generate realistic market data with various market conditions"""
    np.random.seed(42)
    
    n_bars = 5000  # More data for better testing
    base_price = 100.0
    
    # Generate different market regimes
    returns = []
    
    # Regime 1: Trending up (500 bars)
    trend_returns = np.random.normal(0.0005, 0.015, 500)
    returns.extend(trend_returns)
    
    # Regime 2: Sideways choppy (1000 bars)
    sideways_returns = np.random.normal(0.0, 0.02, 1000)
    returns.extend(sideways_returns)
    
    # Regime 3: Volatile trending down (500 bars)
    down_returns = np.random.normal(-0.0003, 0.025, 500)
    returns.extend(down_returns)
    
    # Regime 4: Recovery trending up (1000 bars)
    recovery_returns = np.random.normal(0.0004, 0.018, 1000)
    returns.extend(recovery_returns)
    
    # Regime 5: High volatility (1000 bars)
    volatile_returns = np.random.normal(0.0001, 0.035, 1000)
    returns.extend(volatile_returns)
    
    # Regime 6: Final consolidation (1000 bars)
    final_returns = np.random.normal(0.0, 0.012, 1000)
    returns.extend(final_returns)
    
    # Add some momentum spikes throughout
    for i in range(0, len(returns), 100):
        if np.random.random() > 0.7:  # 30% chance of momentum spike
            spike_length = min(20, len(returns) - i)
            if np.random.random() > 0.5:
                # Bullish spike
                returns[i:i+spike_length] = [r + np.random.normal(0.008, 0.003) for r in returns[i:i+spike_length]]
            else:
                # Bearish spike
                returns[i:i+spike_length] = [r + np.random.normal(-0.008, 0.003) for r in returns[i:i+spike_length]]
    
    # Calculate prices
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = []
    for i in range(len(returns)):
        close = prices[i+1]
        open_price = prices[i]
        
        # Generate high/low with realistic volatility
        volatility = abs(np.random.normal(0, 0.008))
        high = max(open_price, close) * (1 + volatility + np.random.normal(0, 0.002))
        low = min(open_price, close) * (1 - volatility - np.random.normal(0, 0.002))
        
        # Volume with correlation to price movement and volatility
        price_change = abs(close - open_price) / open_price
        base_volume = 1000000
        volume_multiplier = 1 + price_change * 15 + volatility * 10 + np.random.normal(0, 0.3)
        volume = base_volume * max(volume_multiplier, 0.1)
        
        data.append({
            'datetime': datetime.now() - timedelta(hours=len(returns)-i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def main():
    """Main execution function"""
    print("🚀 ELITE PARABOLIC PRODUCTION SYSTEM")
    print("=" * 60)
    
    # Generate realistic data
    print("📊 Generating realistic market data...")
    df = generate_realistic_data()
    print(f"Generated {len(df)} bars of data")
    
    # Initialize system
    system = EliteParabolicProductionSystem(initial_balance=50.0)
    
    # Run backtest
    print("🔄 Running production backtest...")
    df_with_signals = system.run_backtest(df)
    
    # Calculate metrics
    metrics = system.calculate_metrics()
    
    # Display results
    print("\n" + "=" * 70)
    print("📈 ELITE PARABOLIC PRODUCTION RESULTS")
    print("=" * 70)
    
    print(f"\n💰 PERFORMANCE SUMMARY")
    print(f"Initial Balance:     ${metrics['initial_balance']:.2f}")
    print(f"Final Balance:       ${metrics['final_balance']:.2f}")
    print(f"Total Return:        {metrics['total_return']:.1f}%")
    print(f"Monthly Return:      {metrics['monthly_return']:.1f}%")
    print(f"Total P&L:           ${metrics['total_pnl']:.2f}")
    print(f"Days Trading:        {metrics['days_trading']:.1f}")
    
    print(f"\n📊 TRADE STATISTICS")
    print(f"Total Trades:        {metrics['total_trades']}")
    print(f"Winning Trades:      {metrics['winning_trades']}")
    print(f"Losing Trades:       {metrics['losing_trades']}")
    print(f"Win Rate:            {metrics['win_rate']:.1f}%")
    print(f"Average Win:         ${metrics['avg_win']:.2f}")
    print(f"Average Loss:        ${metrics['avg_loss']:.2f}")
    print(f"Expectancy:          {metrics['expectancy']:.3f}%")
    print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
    print(f"Avg R Multiple:      {metrics['avg_r_multiple']:.2f}R")
    
    print(f"\n⚠️  RISK METRICS")
    print(f"Max Drawdown:        {metrics['max_drawdown']:.1f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    
    # Elite gates validation
    print(f"\n🎯 ELITE GATES VALIDATION")
    gates = {
        'min_trades': metrics['total_trades'] >= 100,
        'expectancy': metrics['expectancy'] >= 0.30,
        'profit_factor': metrics['profit_factor'] >= 1.30,
        'win_rate': metrics['win_rate'] >= 30,
        'max_drawdown': metrics['max_drawdown'] <= 5,
        'sharpe_ratio': metrics['sharpe_ratio'] >= 1.0
    }
    
    print(f"Min Trades:          {'✅ PASS' if gates['min_trades'] else '❌ FAIL'} {metrics['total_trades']} (≥100)")
    print(f"Expectancy:          {'✅ PASS' if gates['expectancy'] else '❌ FAIL'} {metrics['expectancy']:.3f}% (≥0.30%)")
    print(f"Profit Factor:       {'✅ PASS' if gates['profit_factor'] else '❌ FAIL'} {metrics['profit_factor']:.2f} (≥1.30)")
    print(f"Win Rate:            {'✅ PASS' if gates['win_rate'] else '❌ FAIL'} {metrics['win_rate']:.1f}% (≥30%)")
    print(f"Max Drawdown:        {'✅ PASS' if gates['max_drawdown'] else '❌ FAIL'} {metrics['max_drawdown']:.1f}% (≤5%)")
    print(f"Sharpe Ratio:        {'✅ PASS' if gates['sharpe_ratio'] else '❌ FAIL'} {metrics['sharpe_ratio']:.2f} (≥1.0)")
    
    gates_passed = sum(gates.values())
    print(f"\nGates Passed: {gates_passed}/6")
    print(f"Status: {'✅ READY FOR LIVE TRADING' if gates_passed >= 5 else '❌ NEEDS OPTIMIZATION'}")
    
    # Trade type analysis
    burst_trades = [t for t in system.trades if 'burst' in t['signal_type']]
    fade_trades = [t for t in system.trades if 'fade' in t['signal_type']]
    
    print(f"\n🎯 TRADE TYPE ANALYSIS")
    if burst_trades:
        burst_wins = [t for t in burst_trades if t['pnl'] > 0]
        burst_wr = len(burst_wins) / len(burst_trades) * 100
        burst_pnl = sum(t['pnl'] for t in burst_trades)
        burst_avg_r = np.mean([t['r_multiple'] for t in burst_trades])
        print(f"BURST TRADES: {len(burst_trades)} trades, {burst_wr:.1f}% WR, ${burst_pnl:.2f} P&L, {burst_avg_r:.2f}R avg")
    
    if fade_trades:
        fade_wins = [t for t in fade_trades if t['pnl'] > 0]
        fade_wr = len(fade_wins) / len(fade_trades) * 100
        fade_pnl = sum(t['pnl'] for t in fade_trades)
        fade_avg_r = np.mean([t['r_multiple'] for t in fade_trades])
        print(f"FADE TRADES:  {len(fade_trades)} trades, {fade_wr:.1f}% WR, ${fade_pnl:.2f} P&L, {fade_avg_r:.2f}R avg")
    
    # Entry reason analysis
    print(f"\n🎯 ENTRY REASON ANALYSIS")
    reason_stats = {}
    for trade in system.trades:
        reason = trade['entry_reason']
        if reason not in reason_stats:
            reason_stats[reason] = {'count': 0, 'wins': 0, 'pnl': 0}
        reason_stats[reason]['count'] += 1
        if trade['pnl'] > 0:
            reason_stats[reason]['wins'] += 1
        reason_stats[reason]['pnl'] += trade['pnl']
    
    for reason, stats in reason_stats.items():
        wr = stats['wins'] / stats['count'] * 100
        print(f"{reason}: {stats['count']} trades, {wr:.1f}% WR, ${stats['pnl']:.2f} P&L")
    
    # Monthly projection
    if metrics['monthly_return'] > 0:
        print(f"\n🚀 DOUBLE-UP PROJECTION")
        print(f"Month 1: ${50:.2f} → ${50 * (1 + metrics['monthly_return']/100):.2f}")
        if metrics['monthly_return'] >= 100:
            print("✅ TARGET ACHIEVED: Double-up in 30 days!")
        else:
            months_to_double = np.log(2) / np.log(1 + metrics['monthly_return']/100)
            print(f"Months to double: {months_to_double:.1f}")
    
    # Save results
    results = {
        'metrics': metrics,
        'trades': system.trades,
        'total_signals': len(df_with_signals[df_with_signals['signal'] != 'none']),
        'gates_passed': gates_passed,
        'ready_for_live': gates_passed >= 5
    }
    
    with open('elite_production_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to elite_production_results.json")
    print(f"📊 Total signals generated: {results['total_signals']}")
    print(f"🎯 Signal to trade conversion: {metrics['total_trades']}/{results['total_signals']} ({metrics['total_trades']/results['total_signals']*100:.1f}%)")

if __name__ == "__main__":
    main() 