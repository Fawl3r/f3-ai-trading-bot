#!/usr/bin/env python3
"""
Enhanced Parabolic Backtest - More aggressive signal generation
Target: 100+ trades for proper validation
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

# Simple technical indicators
def calculate_sma(series, period):
    return series.rolling(window=period).mean()

def calculate_ema(series, period):
    return series.ewm(span=period).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def generate_enhanced_signals(df):
    """Generate enhanced parabolic signals with more opportunities"""
    
    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    df['ema_20'] = calculate_ema(df['close'], 20)
    df['ema_50'] = calculate_ema(df['close'], 50)
    df['ema_200'] = calculate_ema(df['close'], 200)
    df['sma_10'] = calculate_sma(df['close'], 10)
    
    # Multiple timeframe ROC
    df['roc_3'] = df['close'].pct_change(3) * 100
    df['roc_5'] = df['close'].pct_change(5) * 100
    df['roc_10'] = df['close'].pct_change(10) * 100
    df['roc_std'] = df['roc_3'].rolling(20).std()  # Shorter period for more signals
    
    # VWAP analysis
    df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
    df['vwap_gap'] = (df['close'] - df['vwap']) / df['atr']
    
    # Volume analysis
    df['vol_ma_10'] = df['volume'].rolling(10).mean()
    df['vol_ma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma_20']
    df['vol_spike'] = df['volume'] > df['vol_ma_10'] * 1.5
    
    # Price patterns
    df['body_size'] = np.abs(df['close'] - df['open'])
    df['body_ratio'] = df['body_size'] / df['atr']
    df['price_change'] = df['close'].pct_change()
    
    # Momentum indicators
    df['momentum_5'] = df['close'] / df['close'].shift(5)
    df['momentum_10'] = df['close'] / df['close'].shift(10)
    
    # Trend context
    df['trend_short'] = np.where(df['close'] > df['ema_20'], 'up', 'down')
    df['trend_medium'] = np.where(df['close'] > df['ema_50'], 'up', 'down')
    df['trend_long'] = np.where(df['close'] > df['ema_200'], 'up', 'down')
    
    # Enhanced burst signals (more sensitive)
    df['burst_momentum'] = (
        (np.abs(df['roc_3']) > df['roc_std'] * 1.5) |  # Reduced threshold
        (np.abs(df['roc_5']) > df['roc_std'] * 1.2) |
        (df['momentum_5'] > 1.02) | (df['momentum_5'] < 0.98)
    )
    
    df['burst_volume'] = (
        (df['vol_ratio'] > 1.2) |  # Reduced volume threshold
        df['vol_spike']
    )
    
    df['burst_vwap'] = np.abs(df['vwap_gap']) > 1.5  # Reduced VWAP threshold
    
    # Enhanced fade signals (exhaustion)
    df['fade_rsi'] = (df['rsi'] > 65) | (df['rsi'] < 35)  # More sensitive RSI
    df['fade_volume'] = df['vol_ratio'] > 1.8
    df['fade_body'] = df['body_ratio'] < 0.4  # Small bodies indicate indecision
    
    # Divergence patterns (simplified)
    df['price_high'] = df['high'].rolling(5, center=True).max() == df['high']
    df['price_low'] = df['low'].rolling(5, center=True).min() == df['low']
    df['rsi_divergence'] = False
    
    # Look for basic divergence
    for i in range(10, len(df) - 2):
        if df.iloc[i]['price_high']:
            # Look for previous high
            for j in range(max(0, i-10), i-2):
                if df.iloc[j]['price_high']:
                    price_hh = df.iloc[i]['high'] > df.iloc[j]['high']
                    rsi_lh = df.iloc[i]['rsi'] < df.iloc[j]['rsi']
                    if price_hh and rsi_lh and df.iloc[i]['rsi'] > 60:
                        df.iloc[i, df.columns.get_loc('rsi_divergence')] = True
                    break
        
        if df.iloc[i]['price_low']:
            # Look for previous low
            for j in range(max(0, i-10), i-2):
                if df.iloc[j]['price_low']:
                    price_ll = df.iloc[i]['low'] < df.iloc[j]['low']
                    rsi_hl = df.iloc[i]['rsi'] > df.iloc[j]['rsi']
                    if price_ll and rsi_hl and df.iloc[i]['rsi'] < 40:
                        df.iloc[i, df.columns.get_loc('rsi_divergence')] = True
                    break
    
    # Time filters (more permissive)
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['ny_hours'] = (df['hour'] >= 13) & (df['hour'] <= 18)
    df['london_hours'] = (df['hour'] >= 8) & (df['hour'] <= 16)
    df['active_hours'] = df['ny_hours'] | df['london_hours']
    
    # Signal generation with multiple conditions
    df['signal'] = 'none'
    df['signal_strength'] = 0.0
    
    # Burst Long Signals
    burst_long_conditions = [
        # Strong momentum burst
        (df['burst_momentum'] & df['burst_volume'] & (df['roc_3'] > 0) & 
         (df['vwap_gap'] > 1) & df['active_hours']),
        
        # VWAP breakout
        (df['burst_vwap'] & (df['vwap_gap'] > 2) & df['vol_spike'] & 
         (df['trend_medium'] == 'up')),
        
        # Momentum continuation
        ((df['momentum_5'] > 1.015) & (df['close'] > df['ema_20']) & 
         df['vol_spike'] & (df['rsi'] < 70)),
        
        # Breakout from consolidation
        ((df['close'] > df['high'].rolling(10).max().shift(1)) & 
         df['vol_spike'] & (df['rsi'] > 50) & (df['trend_long'] == 'up'))
    ]
    
    # Burst Short Signals
    burst_short_conditions = [
        # Strong momentum burst down
        (df['burst_momentum'] & df['burst_volume'] & (df['roc_3'] < 0) & 
         (df['vwap_gap'] < -1) & df['active_hours']),
        
        # VWAP breakdown
        (df['burst_vwap'] & (df['vwap_gap'] < -2) & df['vol_spike'] & 
         (df['trend_medium'] == 'down')),
        
        # Momentum continuation down
        ((df['momentum_5'] < 0.985) & (df['close'] < df['ema_20']) & 
         df['vol_spike'] & (df['rsi'] > 30)),
        
        # Breakdown from consolidation
        ((df['close'] < df['low'].rolling(10).min().shift(1)) & 
         df['vol_spike'] & (df['rsi'] < 50) & (df['trend_long'] == 'down'))
    ]
    
    # Fade Long Signals (buy oversold)
    fade_long_conditions = [
        # RSI oversold with volume
        (df['fade_rsi'] & (df['rsi'] < 35) & df['fade_volume'] & 
         (df['trend_long'] == 'up')),
        
        # Divergence reversal
        (df['rsi_divergence'] & (df['rsi'] < 40) & df['vol_spike']),
        
        # Support bounce
        ((df['close'] < df['ema_50']) & (df['rsi'] < 40) & 
         (df['close'] > df['low'].rolling(5).min()) & df['vol_spike']),
        
        # Exhaustion after decline
        ((df['roc_5'] < -3) & (df['rsi'] < 30) & df['fade_volume'] & 
         df['fade_body'])
    ]
    
    # Fade Short Signals (sell overbought)
    fade_short_conditions = [
        # RSI overbought with volume
        (df['fade_rsi'] & (df['rsi'] > 65) & df['fade_volume'] & 
         (df['trend_long'] == 'down')),
        
        # Divergence reversal
        (df['rsi_divergence'] & (df['rsi'] > 60) & df['vol_spike']),
        
        # Resistance rejection
        ((df['close'] > df['ema_50']) & (df['rsi'] > 60) & 
         (df['close'] < df['high'].rolling(5).max()) & df['vol_spike']),
        
        # Exhaustion after rally
        ((df['roc_5'] > 3) & (df['rsi'] > 70) & df['fade_volume'] & 
         df['fade_body'])
    ]
    
    # Apply signals with strength scoring
    for i, conditions in enumerate(burst_long_conditions):
        mask = conditions
        df.loc[mask & (df['signal'] == 'none'), 'signal'] = 'burst_long'
        df.loc[mask, 'signal_strength'] = np.maximum(df.loc[mask, 'signal_strength'], 0.6 + i * 0.1)
    
    for i, conditions in enumerate(burst_short_conditions):
        mask = conditions
        df.loc[mask & (df['signal'] == 'none'), 'signal'] = 'burst_short'
        df.loc[mask, 'signal_strength'] = np.maximum(df.loc[mask, 'signal_strength'], 0.6 + i * 0.1)
    
    for i, conditions in enumerate(fade_long_conditions):
        mask = conditions
        df.loc[mask & (df['signal'] == 'none'), 'signal'] = 'fade_long'
        df.loc[mask, 'signal_strength'] = np.maximum(df.loc[mask, 'signal_strength'], 0.5 + i * 0.1)
    
    for i, conditions in enumerate(fade_short_conditions):
        mask = conditions
        df.loc[mask & (df['signal'] == 'none'), 'signal'] = 'fade_short'
        df.loc[mask, 'signal_strength'] = np.maximum(df.loc[mask, 'signal_strength'], 0.5 + i * 0.1)
    
    return df

def run_enhanced_backtest(df):
    """Run enhanced backtest with adaptive position sizing"""
    
    # Configuration
    initial_balance = 50.0
    base_risk = 0.0075  # 0.75% base risk
    risk_reward_ratio = 4.0
    commission = 0.001
    slippage = 0.0005
    max_positions = 2
    
    # Risk management
    consecutive_losses = 0
    daily_loss = 0.0
    max_daily_loss = 0.02  # 2%
    
    # Initialize
    balance = initial_balance
    peak_balance = balance
    trades = []
    equity_curve = []
    open_trades = []
    
    # Process each bar
    for i in range(50, len(df)):  # Start earlier for more opportunities
        current_bar = df.iloc[i]
        
        # Update open trades
        for trade in open_trades[:]:  # Copy list to avoid modification during iteration
            # Check for exits
            exit_triggered = False
            
            if trade['side'] == 'long':
                if current_bar['low'] <= trade['stop_loss']:
                    trade['exit_price'] = trade['stop_loss']
                    trade['exit_reason'] = 'stop_loss'
                    exit_triggered = True
                elif current_bar['high'] >= trade['take_profit']:
                    trade['exit_price'] = trade['take_profit']
                    trade['exit_reason'] = 'take_profit'
                    exit_triggered = True
            else:  # short
                if current_bar['high'] >= trade['stop_loss']:
                    trade['exit_price'] = trade['stop_loss']
                    trade['exit_reason'] = 'stop_loss'
                    exit_triggered = True
                elif current_bar['low'] <= trade['take_profit']:
                    trade['exit_price'] = trade['take_profit']
                    trade['exit_reason'] = 'take_profit'
                    exit_triggered = True
            
            # Time-based exit (max 48 hours)
            bars_open = i - trade['entry_bar']
            if bars_open >= 48:
                trade['exit_price'] = current_bar['close']
                trade['exit_reason'] = 'timeout'
                exit_triggered = True
            
            if exit_triggered:
                # Calculate PnL
                if trade['side'] == 'long':
                    trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['position_size']
                else:
                    trade['pnl'] = (trade['entry_price'] - trade['exit_price']) * trade['position_size']
                
                # Apply costs
                costs = (trade['entry_price'] + trade['exit_price']) * trade['position_size'] * (commission + slippage)
                trade['pnl'] -= costs
                
                # Calculate R multiple
                risk_per_share = abs(trade['entry_price'] - trade['stop_loss'])
                trade['r_multiple'] = trade['pnl'] / (risk_per_share * trade['position_size'])
                
                # Update balance and risk management
                balance += trade['pnl']
                peak_balance = max(peak_balance, balance)
                
                if trade['pnl'] < 0:
                    consecutive_losses += 1
                    daily_loss += abs(trade['pnl'])
                else:
                    consecutive_losses = 0
                
                # Record trade
                trade['exit_time'] = current_bar['datetime']
                trades.append(trade)
                open_trades.remove(trade)
        
        # Check for new signals
        if (current_bar['signal'] != 'none' and 
            len(open_trades) < max_positions and
            daily_loss < balance * max_daily_loss):
            
            signal_type = current_bar['signal']
            entry_price = current_bar['close']
            atr = current_bar['atr']
            signal_strength = current_bar['signal_strength']
            
            if pd.notna(atr) and atr > 0:
                # Adaptive position sizing
                risk_multiplier = 1.0
                
                # Reduce risk after consecutive losses
                if consecutive_losses > 2:
                    risk_multiplier *= 0.8 ** (consecutive_losses - 2)
                
                # Adjust risk based on signal strength
                risk_multiplier *= signal_strength
                
                # Volatility adjustment
                atr_percentile = df['atr'].rolling(100).rank(pct=True).iloc[i]
                if atr_percentile > 0.8:  # High volatility
                    risk_multiplier *= 0.7
                elif atr_percentile < 0.2:  # Low volatility
                    risk_multiplier *= 1.3
                
                risk_percent = base_risk * risk_multiplier
                risk_percent = min(risk_percent, 0.015)  # Cap at 1.5%
                
                # Calculate position parameters
                if 'long' in signal_type:
                    stop_loss = entry_price - atr
                    take_profit = entry_price + (atr * risk_reward_ratio)
                else:
                    stop_loss = entry_price + atr
                    take_profit = entry_price - (atr * risk_reward_ratio)
                
                # Position sizing
                risk_amount = balance * risk_percent
                risk_per_share = abs(entry_price - stop_loss)
                
                if risk_per_share > 0:
                    position_size = risk_amount / risk_per_share
                    
                    # Create trade
                    trade = {
                        'entry_time': current_bar['datetime'],
                        'entry_bar': i,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'position_size': position_size,
                        'side': 'long' if 'long' in signal_type else 'short',
                        'trade_type': signal_type.split('_')[0],
                        'signal_strength': signal_strength,
                        'risk_percent': risk_percent,
                        'exit_price': None,
                        'exit_time': None,
                        'pnl': 0,
                        'exit_reason': None
                    }
                    
                    open_trades.append(trade)
        
        # Record equity
        open_pnl = 0
        for trade in open_trades:
            if trade['side'] == 'long':
                unrealized_pnl = (current_bar['close'] - trade['entry_price']) * trade['position_size']
            else:
                unrealized_pnl = (trade['entry_price'] - current_bar['close']) * trade['position_size']
            open_pnl += unrealized_pnl
        
        equity_curve.append({
            'datetime': current_bar['datetime'],
            'balance': balance,
            'open_pnl': open_pnl,
            'total_equity': balance + open_pnl
        })
        
        # Reset daily loss at start of new day
        if i > 0 and current_bar['datetime'].date() != df.iloc[i-1]['datetime'].date():
            daily_loss = 0.0
    
    # Close remaining open trades
    for trade in open_trades:
        trade['exit_price'] = df.iloc[-1]['close']
        trade['exit_reason'] = 'end_of_data'
        trade['exit_time'] = df.iloc[-1]['datetime']
        
        # Calculate final PnL
        if trade['side'] == 'long':
            trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['position_size']
        else:
            trade['pnl'] = (trade['entry_price'] - trade['exit_price']) * trade['position_size']
        
        # Apply costs
        costs = (trade['entry_price'] + trade['exit_price']) * trade['position_size'] * (commission + slippage)
        trade['pnl'] -= costs
        
        # Calculate R multiple
        risk_per_share = abs(trade['entry_price'] - trade['stop_loss'])
        trade['r_multiple'] = trade['pnl'] / (risk_per_share * trade['position_size'])
        
        balance += trade['pnl']
        trades.append(trade)
    
    return trades, equity_curve

def calculate_enhanced_metrics(trades, equity_curve, initial_balance):
    """Calculate enhanced performance metrics"""
    
    if not trades:
        return {}
    
    final_balance = equity_curve[-1]['balance']
    
    # Basic metrics
    total_return = (final_balance - initial_balance) / initial_balance * 100
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    losing_trades = len([t for t in trades if t['pnl'] < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # PnL metrics
    total_pnl = sum(t['pnl'] for t in trades)
    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0
    
    # Expectancy
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
    expectancy_pct = expectancy / initial_balance * 100
    
    # Profit factor
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # R multiples
    r_multiples = [t['r_multiple'] for t in trades if 'r_multiple' in t]
    avg_r = np.mean(r_multiples) if r_multiples else 0
    
    # Drawdown analysis
    equity_df = pd.DataFrame(equity_curve)
    equity_df['peak'] = equity_df['total_equity'].expanding().max()
    equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['peak']) / equity_df['peak']
    max_drawdown = abs(equity_df['drawdown'].min()) * 100
    
    # Risk metrics
    equity_df['returns'] = equity_df['total_equity'].pct_change().fillna(0)
    volatility = equity_df['returns'].std() * np.sqrt(365 * 24)  # Annualized
    sharpe_ratio = (equity_df['returns'].mean() * 365 * 24) / volatility if volatility > 0 else 0
    
    # Time metrics
    days_trading = (equity_curve[-1]['datetime'] - equity_curve[0]['datetime']).days
    monthly_return = ((final_balance / initial_balance) ** (30 / days_trading) - 1) * 100 if days_trading > 0 else 0
    
    # Exit reason analysis
    exit_reasons = {}
    for reason in ['stop_loss', 'take_profit', 'timeout', 'end_of_data']:
        count = len([t for t in trades if t.get('exit_reason') == reason])
        exit_reasons[reason] = count
    
    return {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_return': total_return,
        'monthly_return': monthly_return,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'expectancy_pct': expectancy_pct,
        'profit_factor': profit_factor,
        'avg_r_multiple': avg_r,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'volatility': volatility * 100,
        'days_trading': days_trading,
        'exit_reasons': exit_reasons
    }

def analyze_enhanced_trades(trades):
    """Enhanced trade analysis"""
    
    analysis = {}
    
    # By trade type
    for trade_type in ['burst', 'fade']:
        type_trades = [t for t in trades if t['trade_type'] == trade_type]
        
        if type_trades:
            wins = len([t for t in type_trades if t['pnl'] > 0])
            total = len(type_trades)
            win_rate = wins / total * 100
            
            total_pnl = sum(t['pnl'] for t in type_trades)
            avg_pnl = total_pnl / total
            
            gross_profit = sum(t['pnl'] for t in type_trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in type_trades if t['pnl'] < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_r = np.mean([t['r_multiple'] for t in type_trades if 'r_multiple' in t])
            avg_strength = np.mean([t.get('signal_strength', 0.5) for t in type_trades])
            
            analysis[f'{trade_type}_trades'] = {
                'count': total,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'profit_factor': pf,
                'avg_r_multiple': avg_r,
                'avg_signal_strength': avg_strength
            }
    
    # By signal strength
    high_strength = [t for t in trades if t.get('signal_strength', 0.5) >= 0.7]
    low_strength = [t for t in trades if t.get('signal_strength', 0.5) < 0.7]
    
    for strength_type, strength_trades in [('high_strength', high_strength), ('low_strength', low_strength)]:
        if strength_trades:
            wins = len([t for t in strength_trades if t['pnl'] > 0])
            total = len(strength_trades)
            win_rate = wins / total * 100
            avg_r = np.mean([t['r_multiple'] for t in strength_trades if 'r_multiple' in t])
            
            analysis[strength_type] = {
                'count': total,
                'win_rate': win_rate,
                'avg_r_multiple': avg_r
            }
    
    return analysis

def print_enhanced_results(metrics, trade_analysis):
    """Print enhanced results"""
    
    print("=" * 80)
    print("ENHANCED PARABOLIC SYSTEM BACKTEST RESULTS ($50 START)")
    print("=" * 80)
    
    # Performance Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("-" * 40)
    print(f"Initial Balance:     ${metrics['initial_balance']:,.2f}")
    print(f"Final Balance:       ${metrics['final_balance']:,.2f}")
    print(f"Total Return:        {metrics['total_return']:+.1f}%")
    print(f"Monthly Return:      {metrics['monthly_return']:+.1f}%")
    print(f"Days Trading:        {metrics['days_trading']}")
    
    # Trade Statistics
    print("\n📈 TRADE STATISTICS")
    print("-" * 40)
    print(f"Total Trades:        {metrics['total_trades']}")
    print(f"Winning Trades:      {metrics['winning_trades']}")
    print(f"Losing Trades:       {metrics['losing_trades']}")
    print(f"Win Rate:            {metrics['win_rate']:.1f}%")
    print(f"Avg Win:             ${metrics['avg_win']:+.2f}")
    print(f"Avg Loss:            ${metrics['avg_loss']:+.2f}")
    print(f"Expectancy:          ${metrics['expectancy']:+.4f} ({metrics['expectancy_pct']:+.3f}%)")
    print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
    print(f"Avg R Multiple:      {metrics['avg_r_multiple']:+.2f}R")
    
    # Risk Metrics
    print("\n⚠️  RISK METRICS")
    print("-" * 40)
    print(f"Max Drawdown:        {metrics['max_drawdown']:.1f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"Volatility:          {metrics['volatility']:.1f}%")
    
    # Exit Analysis
    print("\n🚪 EXIT ANALYSIS")
    print("-" * 40)
    for reason, count in metrics['exit_reasons'].items():
        percentage = count / metrics['total_trades'] * 100
        print(f"{reason.replace('_', ' ').title():15} {count:3d} ({percentage:.1f}%)")
    
    # Validation Gates
    print("\n🚪 VALIDATION GATES")
    print("-" * 40)
    gates = {
        'Min Trades': (metrics['total_trades'] >= 100, f"{metrics['total_trades']} (≥100)"),
        'Expectancy': (metrics['expectancy_pct'] >= 0.30, f"{metrics['expectancy_pct']:.3f}% (≥0.30%)"),
        'Profit Factor': (metrics['profit_factor'] >= 1.30, f"{metrics['profit_factor']:.2f} (≥1.30)"),
        'Sharpe Ratio': (metrics['sharpe_ratio'] >= 1.0, f"{metrics['sharpe_ratio']:.2f} (≥1.0)"),
        'Win Rate': (metrics['win_rate'] >= 30, f"{metrics['win_rate']:.1f}% (≥30%)"),
        'Max Drawdown': (metrics['max_drawdown'] <= 5, f"{metrics['max_drawdown']:.1f}% (≤5%)")
    }
    
    passed = 0
    for gate_name, (passed_gate, value) in gates.items():
        status = "✅ PASS" if passed_gate else "❌ FAIL"
        print(f"{gate_name:15} {status} {value}")
        if passed_gate:
            passed += 1
    
    print(f"\nGates Passed: {passed}/{len(gates)}")
    
    # Trade Type Analysis
    if trade_analysis:
        print("\n🎯 TRADE TYPE ANALYSIS")
        print("-" * 40)
        for trade_type, analysis in trade_analysis.items():
            if 'trades' in trade_type and analysis['count'] > 0:
                print(f"\n{trade_type.upper()}:")
                print(f"  Count:         {analysis['count']}")
                print(f"  Win Rate:      {analysis['win_rate']:.1f}%")
                print(f"  Total PnL:     ${analysis['total_pnl']:+.2f}")
                print(f"  Avg PnL:       ${analysis['avg_pnl']:+.4f}")
                print(f"  Profit Factor: {analysis['profit_factor']:.2f}")
                print(f"  Avg R:         {analysis['avg_r_multiple']:+.2f}R")
                print(f"  Avg Strength:  {analysis['avg_signal_strength']:.2f}")
    
    # Signal Strength Analysis
    if 'high_strength' in trade_analysis and 'low_strength' in trade_analysis:
        print("\n🎯 SIGNAL STRENGTH ANALYSIS")
        print("-" * 40)
        high = trade_analysis['high_strength']
        low = trade_analysis['low_strength']
        print(f"High Strength (≥0.7): {high['count']} trades, {high['win_rate']:.1f}% WR, {high['avg_r_multiple']:+.2f}R")
        print(f"Low Strength (<0.7):  {low['count']} trades, {low['win_rate']:.1f}% WR, {low['avg_r_multiple']:+.2f}R")
    
    # Enhanced Gates
    print("\n🚀 ENHANCED PARABOLIC GATES")
    print("-" * 40)
    
    burst_analysis = trade_analysis.get('burst_trades', {})
    fade_analysis = trade_analysis.get('fade_trades', {})
    
    enhanced_gates = {}
    
    if burst_analysis.get('count', 0) > 0:
        enhanced_gates['Burst Trade PF'] = (
            burst_analysis['profit_factor'] >= 3.0,
            f"{burst_analysis['profit_factor']:.2f} (≥3.0)"
        )
    
    if fade_analysis.get('count', 0) > 0:
        enhanced_gates['Fade Trade Hit Rate'] = (
            fade_analysis['win_rate'] >= 45,
            f"{fade_analysis['win_rate']:.1f}% (≥45%)"
        )
    
    enhanced_gates['Rolling DD'] = (
        metrics['max_drawdown'] <= 5,
        f"{metrics['max_drawdown']:.1f}% (≤5%)"
    )
    
    enhanced_passed = 0
    for gate_name, (passed_gate, value) in enhanced_gates.items():
        status = "✅ PASS" if passed_gate else "❌ FAIL"
        print(f"{gate_name:20} {status} {value}")
        if passed_gate:
            enhanced_passed += 1
    
    print(f"\nEnhanced Gates Passed: {enhanced_passed}/{len(enhanced_gates)}")
    
    # Mathematical Proof
    print("\n🧮 MATHEMATICAL PROOF")
    print("-" * 40)
    risk_reward_ratio = 4.0
    breakeven_wr = 1 / (1 + risk_reward_ratio) * 100
    safety_margin = metrics['win_rate'] - breakeven_wr
    expected_value = metrics['avg_r_multiple']
    
    print(f"Risk:Reward Ratio:   1:{risk_reward_ratio}")
    print(f"Breakeven Win Rate:  {breakeven_wr:.1f}%")
    print(f"Actual Win Rate:     {metrics['win_rate']:.1f}%")
    print(f"Safety Margin:       {safety_margin:+.1f}%")
    print(f"Expected Value:      {expected_value:+.3f}R per trade")
    
    if safety_margin > 0:
        print(f"✅ EDGE CONFIRMED: {safety_margin:.1f}% above breakeven")
    else:
        print(f"❌ NO EDGE: {abs(safety_margin):.1f}% below breakeven")

def main():
    """Run the enhanced parabolic backtest"""
    
    print("Generating enhanced test data...")
    
    # Generate 6 months of hourly data for more opportunities
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=4320, freq='h')  # 180 days * 24 hours
    
    # Simulate more realistic crypto price action
    price = 50.0
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # More frequent parabolic behavior
        if i % 300 == 0:  # Major move every ~12.5 days
            change = np.random.normal(0.06, 0.02)
        elif i % 150 == 0:  # Medium move every ~6 days
            change = np.random.normal(0.03, 0.015)
        elif i % 75 == 0:  # Small move every ~3 days
            change = np.random.normal(0.015, 0.01)
        elif i % 50 == 25:  # Reversal
            change = np.random.normal(-0.02, 0.015)
        else:  # Normal price action
            change = np.random.normal(0.0003, 0.012)
        
        price *= (1 + change)
        prices.append(price)
        
        # Volume patterns
        base_volume = 1000
        if abs(change) > 0.025:  # High volume on big moves
            volume = base_volume * (1 + abs(change) * 8)
        elif abs(change) > 0.01:  # Medium volume
            volume = base_volume * (1 + abs(change) * 4)
        else:
            volume = base_volume * (1 + np.random.exponential(0.3))
        
        volumes.append(volume)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.006))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.006))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    print("Generating enhanced parabolic signals...")
    df = generate_enhanced_signals(df)
    
    signal_count = len(df[df['signal'] != 'none'])
    print(f"Generated {signal_count} signals from {len(df)} bars")
    
    print("Running enhanced backtest...")
    trades, equity_curve = run_enhanced_backtest(df)
    
    print("Calculating enhanced metrics...")
    metrics = calculate_enhanced_metrics(trades, equity_curve, 50.0)
    trade_analysis = analyze_enhanced_trades(trades)
    
    # Print results
    print_enhanced_results(metrics, trade_analysis)
    
    # Save results
    results = {
        'metrics': metrics,
        'trade_analysis': trade_analysis,
        'trade_count': len(trades),
        'signal_count': signal_count,
        'data_points': len(df)
    }
    
    with open('enhanced_parabolic_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to enhanced_parabolic_results.json")
    print(f"📊 Processed {len(df)} bars of data")
    print(f"🎯 Generated {len(trades)} trades from {signal_count} signals")
    print(f"📈 Signal conversion rate: {len(trades)/signal_count*100:.1f}%")

if __name__ == "__main__":
    main() 