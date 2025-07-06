#!/usr/bin/env python3
"""
Optimized Elite Parabolic System
Focus: High-quality signals with superior risk management
Target: 40%+ win rate, 100%+ monthly return
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

class OptimizedEliteParabolicSystem:
    def __init__(self, initial_balance=50.0, risk_per_trade=0.0075):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        self.positions = []
        
    def calculate_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        # Basic indicators
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
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
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volume indicators
        df['vol_sma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma']
        df['vol_spike'] = df['volume'] > (df['vol_sma'] * 2.0)
        
        # Price momentum
        df['roc_5'] = df['close'].pct_change(5) * 100
        df['roc_10'] = df['close'].pct_change(10) * 100
        df['roc_20'] = df['close'].pct_change(20) * 100
        
        # VWAP
        df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        df['vwap_dev'] = (df['close'] - df['vwap']) / df['atr']
        
        # Trend strength
        df['trend_strength'] = np.where(
            (df['close'] > df['ema_50']) & (df['ema_50'] > df['ema_200']), 1,
            np.where((df['close'] < df['ema_50']) & (df['ema_50'] < df['ema_200']), -1, 0)
        )
        
        return df
    
    def detect_parabolic_patterns(self, df):
        """Detect high-quality parabolic patterns"""
        # Initialize pattern columns
        df['parabolic_burst'] = False
        df['parabolic_fade'] = False
        df['pattern_strength'] = 0.0
        
        # Look for parabolic acceleration patterns
        for i in range(20, len(df)):
            current_row = df.iloc[i]
            
            # Parabolic Burst Detection (Long)
            if (current_row['roc_5'] > 2.0 and  # Strong 5-period momentum
                current_row['roc_10'] > 3.0 and  # Accelerating momentum
                current_row['vol_ratio'] > 1.5 and  # Volume confirmation
                current_row['rsi'] > 55 and current_row['rsi'] < 75 and  # Not overbought
                current_row['close'] > current_row['bb_upper'] and  # Breakout
                current_row['macd_hist'] > 0 and  # MACD bullish
                current_row['trend_strength'] >= 0 and  # Uptrend or neutral
                current_row['vwap_dev'] > 1.0):  # Above VWAP
                
                df.iloc[i, df.columns.get_loc('parabolic_burst')] = True
                df.iloc[i, df.columns.get_loc('pattern_strength')] = min(
                    current_row['roc_5'] / 5.0 + current_row['vol_ratio'] / 3.0, 3.0
                )
            
            # Parabolic Burst Detection (Short)
            elif (current_row['roc_5'] < -2.0 and  # Strong downward momentum
                  current_row['roc_10'] < -3.0 and  # Accelerating down
                  current_row['vol_ratio'] > 1.5 and  # Volume confirmation
                  current_row['rsi'] < 45 and current_row['rsi'] > 25 and  # Not oversold
                  current_row['close'] < current_row['bb_lower'] and  # Breakdown
                  current_row['macd_hist'] < 0 and  # MACD bearish
                  current_row['trend_strength'] <= 0 and  # Downtrend or neutral
                  current_row['vwap_dev'] < -1.0):  # Below VWAP
                
                df.iloc[i, df.columns.get_loc('parabolic_burst')] = True
                df.iloc[i, df.columns.get_loc('pattern_strength')] = min(
                    abs(current_row['roc_5']) / 5.0 + current_row['vol_ratio'] / 3.0, 3.0
                )
            
            # Parabolic Fade Detection (Exhaustion reversal)
            elif (current_row['rsi'] > 80 and  # Extreme overbought
                  current_row['vol_ratio'] > 2.0 and  # High volume
                  current_row['roc_5'] > 5.0 and  # Extreme momentum
                  current_row['bb_width'] > 0.05 and  # High volatility
                  current_row['close'] > current_row['bb_upper'] * 1.02):  # Well above upper band
                
                df.iloc[i, df.columns.get_loc('parabolic_fade')] = True
                df.iloc[i, df.columns.get_loc('pattern_strength')] = min(
                    (current_row['rsi'] - 50) / 30.0 + current_row['vol_ratio'] / 3.0, 3.0
                )
            
            # Parabolic Fade Detection (Oversold reversal)
            elif (current_row['rsi'] < 20 and  # Extreme oversold
                  current_row['vol_ratio'] > 2.0 and  # High volume
                  current_row['roc_5'] < -5.0 and  # Extreme momentum down
                  current_row['bb_width'] > 0.05 and  # High volatility
                  current_row['close'] < current_row['bb_lower'] * 0.98):  # Well below lower band
                
                df.iloc[i, df.columns.get_loc('parabolic_fade')] = True
                df.iloc[i, df.columns.get_loc('pattern_strength')] = min(
                    (50 - current_row['rsi']) / 30.0 + current_row['vol_ratio'] / 3.0, 3.0
                )
        
        return df
    
    def generate_signals(self, df):
        """Generate high-quality trading signals"""
        df = self.calculate_indicators(df)
        df = self.detect_parabolic_patterns(df)
        
        # Time filters for optimal trading hours
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['optimal_hours'] = ((df['hour'] >= 13) & (df['hour'] <= 18)) | \
                              ((df['hour'] >= 8) & (df['hour'] <= 11))
        
        # Initialize signal columns
        df['signal'] = 'none'
        df['signal_strength'] = 0.0
        df['entry_reason'] = ''
        
        # Generate signals with strict quality filters
        for i in range(50, len(df)):
            current = df.iloc[i]
            
            # Only trade during optimal hours
            if not current['optimal_hours']:
                continue
            
            # Parabolic Burst Long
            if (current['parabolic_burst'] and 
                current['roc_5'] > 0 and 
                current['pattern_strength'] >= 1.5 and
                current['atr'] > 0):
                
                df.iloc[i, df.columns.get_loc('signal')] = 'burst_long'
                df.iloc[i, df.columns.get_loc('signal_strength')] = current['pattern_strength']
                df.iloc[i, df.columns.get_loc('entry_reason')] = 'Parabolic Burst Long'
            
            # Parabolic Burst Short
            elif (current['parabolic_burst'] and 
                  current['roc_5'] < 0 and 
                  current['pattern_strength'] >= 1.5 and
                  current['atr'] > 0):
                
                df.iloc[i, df.columns.get_loc('signal')] = 'burst_short'
                df.iloc[i, df.columns.get_loc('signal_strength')] = current['pattern_strength']
                df.iloc[i, df.columns.get_loc('entry_reason')] = 'Parabolic Burst Short'
            
            # Parabolic Fade Long (Buy oversold)
            elif (current['parabolic_fade'] and 
                  current['rsi'] < 25 and 
                  current['pattern_strength'] >= 2.0 and
                  current['atr'] > 0):
                
                df.iloc[i, df.columns.get_loc('signal')] = 'fade_long'
                df.iloc[i, df.columns.get_loc('signal_strength')] = current['pattern_strength']
                df.iloc[i, df.columns.get_loc('entry_reason')] = 'Parabolic Fade Long'
            
            # Parabolic Fade Short (Sell overbought)
            elif (current['parabolic_fade'] and 
                  current['rsi'] > 75 and 
                  current['pattern_strength'] >= 2.0 and
                  current['atr'] > 0):
                
                df.iloc[i, df.columns.get_loc('signal')] = 'fade_short'
                df.iloc[i, df.columns.get_loc('signal_strength')] = current['pattern_strength']
                df.iloc[i, df.columns.get_loc('entry_reason')] = 'Parabolic Fade Short'
        
        return df
    
    def calculate_position_size(self, price, atr, signal_strength):
        """Calculate position size based on risk management"""
        # Base risk amount
        risk_amount = self.balance * self.risk_per_trade
        
        # Adjust risk based on signal strength
        strength_multiplier = min(signal_strength / 2.0, 1.5)  # Cap at 1.5x
        adjusted_risk = risk_amount * strength_multiplier
        
        # Calculate position size
        stop_distance = atr * 1.0  # 1 ATR stop
        position_size = adjusted_risk / stop_distance
        
        return position_size, stop_distance
    
    def run_backtest(self, df):
        """Run optimized backtest"""
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
            
            # Check for exit conditions first
            if current_position:
                exit_signal = False
                exit_reason = ''
                
                # Stop loss
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
                
                # Time-based exit (max 24 hours)
                if i - current_position['entry_index'] >= 24:
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
                        'entry_reason': current_position['entry_reason']
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
        
        # Sharpe ratio approximation
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': total_return,
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
            'sharpe_ratio': sharpe
        }

def generate_sample_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Create realistic price data with trends and volatility
    n_bars = 2000
    base_price = 100.0
    
    # Generate price movements with some trending behavior
    returns = np.random.normal(0.0001, 0.02, n_bars)  # Small positive drift
    
    # Add some momentum periods
    for i in range(0, n_bars, 200):
        if np.random.random() > 0.5:
            # Bullish momentum
            returns[i:i+20] += np.random.normal(0.01, 0.005, 20)
        else:
            # Bearish momentum
            returns[i:i+20] += np.random.normal(-0.01, 0.005, 20)
    
    # Calculate prices
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = []
    for i in range(n_bars):
        close = prices[i+1]
        open_price = prices[i]
        
        # Generate high/low with some volatility
        volatility = abs(np.random.normal(0, 0.01))
        high = max(open_price, close) * (1 + volatility)
        low = min(open_price, close) * (1 - volatility)
        
        # Volume with some correlation to price movement
        price_change = abs(close - open_price) / open_price
        base_volume = 1000000
        volume = base_volume * (1 + price_change * 10 + np.random.normal(0, 0.5))
        volume = max(volume, base_volume * 0.1)
        
        data.append({
            'datetime': datetime.now() - timedelta(hours=n_bars-i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def main():
    """Main execution function"""
    print("🚀 OPTIMIZED ELITE PARABOLIC SYSTEM")
    print("=" * 50)
    
    # Generate sample data
    print("📊 Generating sample data...")
    df = generate_sample_data()
    
    # Initialize system
    system = OptimizedEliteParabolicSystem(initial_balance=50.0)
    
    # Run backtest
    print("🔄 Running optimized backtest...")
    df_with_signals = system.run_backtest(df)
    
    # Calculate metrics
    metrics = system.calculate_metrics()
    
    # Display results
    print("\n" + "=" * 60)
    print("📈 OPTIMIZED PARABOLIC SYSTEM RESULTS")
    print("=" * 60)
    
    print(f"\n💰 PERFORMANCE SUMMARY")
    print(f"Initial Balance:     ${metrics['initial_balance']:.2f}")
    print(f"Final Balance:       ${metrics['final_balance']:.2f}")
    print(f"Total Return:        {metrics['total_return']:.1f}%")
    print(f"Total P&L:           ${metrics['total_pnl']:.2f}")
    
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
    print(f"Min Trades:          {'✅ PASS' if metrics['total_trades'] >= 100 else '❌ FAIL'} {metrics['total_trades']} (≥100)")
    print(f"Expectancy:          {'✅ PASS' if metrics['expectancy'] >= 0.30 else '❌ FAIL'} {metrics['expectancy']:.3f}% (≥0.30%)")
    print(f"Profit Factor:       {'✅ PASS' if metrics['profit_factor'] >= 1.30 else '❌ FAIL'} {metrics['profit_factor']:.2f} (≥1.30)")
    print(f"Win Rate:            {'✅ PASS' if metrics['win_rate'] >= 30 else '❌ FAIL'} {metrics['win_rate']:.1f}% (≥30%)")
    print(f"Max Drawdown:        {'✅ PASS' if metrics['max_drawdown'] <= 5 else '❌ FAIL'} {metrics['max_drawdown']:.1f}% (≤5%)")
    print(f"Sharpe Ratio:        {'✅ PASS' if metrics['sharpe_ratio'] >= 1.0 else '❌ FAIL'} {metrics['sharpe_ratio']:.2f} (≥1.0)")
    
    # Count passed gates
    gates_passed = sum([
        metrics['total_trades'] >= 100,
        metrics['expectancy'] >= 0.30,
        metrics['profit_factor'] >= 1.30,
        metrics['win_rate'] >= 30,
        metrics['max_drawdown'] <= 5,
        metrics['sharpe_ratio'] >= 1.0
    ])
    
    print(f"\nGates Passed: {gates_passed}/6")
    print(f"Status: {'✅ READY FOR LIVE TRADING' if gates_passed >= 5 else '❌ NEEDS OPTIMIZATION'}")
    
    # Trade type analysis
    burst_trades = [t for t in system.trades if 'burst' in t['signal_type']]
    fade_trades = [t for t in system.trades if 'fade' in t['signal_type']]
    
    if burst_trades:
        burst_wr = len([t for t in burst_trades if t['pnl'] > 0]) / len(burst_trades) * 100
        burst_pnl = sum(t['pnl'] for t in burst_trades)
        print(f"\n🚀 BURST TRADES: {len(burst_trades)} trades, {burst_wr:.1f}% WR, ${burst_pnl:.2f} P&L")
    
    if fade_trades:
        fade_wr = len([t for t in fade_trades if t['pnl'] > 0]) / len(fade_trades) * 100
        fade_pnl = sum(t['pnl'] for t in fade_trades)
        print(f"🔄 FADE TRADES: {len(fade_trades)} trades, {fade_wr:.1f}% WR, ${fade_pnl:.2f} P&L")
    
    # Save results
    results = {
        'metrics': metrics,
        'trades': system.trades,
        'total_signals': len(df_with_signals[df_with_signals['signal'] != 'none'])
    }
    
    with open('optimized_elite_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to optimized_elite_results.json")
    print(f"📊 Total signals generated: {results['total_signals']}")
    print(f"🎯 Signal to trade conversion: {metrics['total_trades']}/{results['total_signals']} ({metrics['total_trades']/results['total_signals']*100:.1f}%)")

if __name__ == "__main__":
    main() 