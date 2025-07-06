#!/usr/bin/env python3
"""
Quick Parabolic Test - Simple backtest with $50 starting balance
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

def generate_parabolic_signals(df):
    """Generate simple parabolic signals"""
    
    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    df['ema_20'] = calculate_ema(df['close'], 20)
    df['ema_200'] = calculate_ema(df['close'], 200)
    
    # Rate of change
    df['roc_3'] = df['close'].pct_change(3) * 100
    df['roc_std'] = df['roc_3'].rolling(30).std()
    
    # VWAP
    df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
    df['vwap_gap'] = (df['close'] - df['vwap']) / df['atr']
    
    # Volume analysis
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    
    # Burst signals (momentum)
    df['burst_signal'] = (
        (df['roc_3'] > df['roc_std'] * 2) &  # Strong momentum
        (np.abs(df['vwap_gap']) > 2) &       # Away from VWAP
        (df['vol_ratio'] > 1.5)              # High volume
    )
    
    # Fade signals (exhaustion)
    df['fade_signal'] = (
        (df['rsi'] > 70) | (df['rsi'] < 30)  # Overbought/oversold
    ) & (df['vol_ratio'] > 2)  # High volume climax
    
    # Time filter (NY hours: 13-18 UTC)
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['ny_hours'] = (df['hour'] >= 13) & (df['hour'] <= 18)
    
    # Final signals
    df['signal'] = 'none'
    
    # Burst long (momentum up)
    burst_long = df['burst_signal'] & (df['vwap_gap'] > 2) & df['ny_hours']
    df.loc[burst_long, 'signal'] = 'burst_long'
    
    # Burst short (momentum down)
    burst_short = df['burst_signal'] & (df['vwap_gap'] < -2) & df['ny_hours']
    df.loc[burst_short, 'signal'] = 'burst_short'
    
    # Fade long (buy oversold)
    fade_long = df['fade_signal'] & (df['rsi'] < 30) & (df['close'] < df['ema_200'])
    df.loc[fade_long, 'signal'] = 'fade_long'
    
    # Fade short (sell overbought)
    fade_short = df['fade_signal'] & (df['rsi'] > 70) & (df['close'] > df['ema_200'])
    df.loc[fade_short, 'signal'] = 'fade_short'
    
    return df

def run_backtest(df):
    """Run simple backtest"""
    
    # Configuration
    initial_balance = 50.0
    risk_per_trade = 0.0075  # 0.75%
    risk_reward_ratio = 4.0
    commission = 0.001
    slippage = 0.0005
    
    # Initialize
    balance = initial_balance
    peak_balance = balance
    trades = []
    equity_curve = []
    
    # Process each bar
    for i in range(200, len(df)):
        current_bar = df.iloc[i]
        
        # Check for signals
        if current_bar['signal'] != 'none':
            signal_type = current_bar['signal']
            entry_price = current_bar['close']
            atr = current_bar['atr']
            
            if pd.isna(atr) or atr <= 0:
                continue
            
            # Calculate position parameters
            if 'long' in signal_type:
                stop_loss = entry_price - atr
                take_profit = entry_price + (atr * risk_reward_ratio)
            else:
                stop_loss = entry_price + atr
                take_profit = entry_price - (atr * risk_reward_ratio)
            
            # Position sizing
            risk_amount = balance * risk_per_trade
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share > 0:
                position_size = risk_amount / risk_per_share
                
                # Create trade
                trade = {
                    'entry_time': current_bar['datetime'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': position_size,
                    'side': 'long' if 'long' in signal_type else 'short',
                    'trade_type': signal_type.split('_')[0],
                    'exit_price': None,
                    'exit_time': None,
                    'pnl': 0,
                    'exit_reason': None
                }
                
                # Simulate trade execution
                # Look ahead for exit (simplified)
                for j in range(i + 1, min(i + 100, len(df))):  # Max 100 bars
                    future_bar = df.iloc[j]
                    
                    if trade['side'] == 'long':
                        # Check stop loss
                        if future_bar['low'] <= stop_loss:
                            trade['exit_price'] = stop_loss
                            trade['exit_reason'] = 'stop_loss'
                            break
                        # Check take profit
                        elif future_bar['high'] >= take_profit:
                            trade['exit_price'] = take_profit
                            trade['exit_reason'] = 'take_profit'
                            break
                    else:  # short
                        # Check stop loss
                        if future_bar['high'] >= stop_loss:
                            trade['exit_price'] = stop_loss
                            trade['exit_reason'] = 'stop_loss'
                            break
                        # Check take profit
                        elif future_bar['low'] <= take_profit:
                            trade['exit_price'] = take_profit
                            trade['exit_reason'] = 'take_profit'
                            break
                
                # If no exit found, close at current price
                if trade['exit_price'] is None:
                    trade['exit_price'] = df.iloc[min(i + 50, len(df) - 1)]['close']
                    trade['exit_reason'] = 'timeout'
                
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
                
                # Update balance
                balance += trade['pnl']
                peak_balance = max(peak_balance, balance)
                
                trades.append(trade)
        
        # Record equity
        equity_curve.append({
            'datetime': current_bar['datetime'],
            'balance': balance,
            'total_equity': balance
        })
    
    return trades, equity_curve

def calculate_metrics(trades, equity_curve, initial_balance):
    """Calculate performance metrics"""
    
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
    
    # Drawdown
    equity_df = pd.DataFrame(equity_curve)
    equity_df['peak'] = equity_df['balance'].expanding().max()
    equity_df['drawdown'] = (equity_df['balance'] - equity_df['peak']) / equity_df['peak']
    max_drawdown = abs(equity_df['drawdown'].min()) * 100
    
    # Time metrics
    days_trading = (equity_curve[-1]['datetime'] - equity_curve[0]['datetime']).days
    monthly_return = ((final_balance / initial_balance) ** (30 / days_trading) - 1) * 100 if days_trading > 0 else 0
    
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
        'days_trading': days_trading
    }

def analyze_by_type(trades):
    """Analyze trades by type"""
    
    analysis = {}
    
    for trade_type in ['burst', 'fade']:
        type_trades = [t for t in trades if t['trade_type'] == trade_type]
        
        if type_trades:
            wins = len([t for t in type_trades if t['pnl'] > 0])
            total = len(type_trades)
            win_rate = wins / total * 100
            
            total_pnl = sum(t['pnl'] for t in type_trades)
            
            gross_profit = sum(t['pnl'] for t in type_trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in type_trades if t['pnl'] < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_r = np.mean([t['r_multiple'] for t in type_trades if 'r_multiple' in t])
            
            analysis[f'{trade_type}_trades'] = {
                'count': total,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'profit_factor': pf,
                'avg_r_multiple': avg_r
            }
    
    return analysis

def print_results(metrics, trade_analysis):
    """Print results"""
    
    print("=" * 80)
    print("PARABOLIC SYSTEM BACKTEST RESULTS ($50 START)")
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
    print(f"Max Drawdown:        {metrics['max_drawdown']:.1f}%")
    
    # Validation Gates
    print("\n🚪 VALIDATION GATES")
    print("-" * 40)
    gates = {
        'Min Trades': (metrics['total_trades'] >= 100, f"{metrics['total_trades']} (≥100)"),
        'Expectancy': (metrics['expectancy_pct'] >= 0.30, f"{metrics['expectancy_pct']:.3f}% (≥0.30%)"),
        'Profit Factor': (metrics['profit_factor'] >= 1.30, f"{metrics['profit_factor']:.2f} (≥1.30)"),
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
            if analysis['count'] > 0:
                print(f"\n{trade_type.upper()}:")
                print(f"  Count:         {analysis['count']}")
                print(f"  Win Rate:      {analysis['win_rate']:.1f}%")
                print(f"  Total PnL:     ${analysis['total_pnl']:+.2f}")
                print(f"  Profit Factor: {analysis['profit_factor']:.2f}")
                print(f"  Avg R:         {analysis['avg_r_multiple']:+.2f}R")
    
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
    
    enhanced_passed = 0
    for gate_name, (passed_gate, value) in enhanced_gates.items():
        status = "✅ PASS" if passed_gate else "❌ FAIL"
        print(f"{gate_name:20} {status} {value}")
        if passed_gate:
            enhanced_passed += 1
    
    if enhanced_gates:
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
    """Run the quick parabolic test"""
    
    print("Generating test data...")
    
    # Generate 3 months of hourly data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=2159, freq='h')  # 90 days * 24 hours - 1
    
    # Simulate realistic crypto price action with parabolic moves
    price = 50.0
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Add parabolic behavior
        if i % 400 == 0:  # Major parabolic move every ~17 days
            change = np.random.normal(0.08, 0.02)  # Strong 8% move
        elif i % 200 == 0:  # Minor parabolic move every ~8 days
            change = np.random.normal(0.04, 0.015)  # 4% move
        elif i % 100 == 50:  # Exhaustion/reversal
            change = np.random.normal(-0.03, 0.02)  # -3% reversal
        else:  # Normal price action
            change = np.random.normal(0.0005, 0.015)  # Small drift
        
        price *= (1 + change)
        prices.append(price)
        
        # Volume with spikes during parabolic moves
        base_volume = 1000
        if abs(change) > 0.03:  # High volume on big moves
            volume = base_volume * (1 + abs(change) * 10)
        else:
            volume = base_volume * (1 + np.random.exponential(0.5))
        
        volumes.append(volume)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    print("Generating parabolic signals...")
    df = generate_parabolic_signals(df)
    
    print("Running backtest...")
    trades, equity_curve = run_backtest(df)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(trades, equity_curve, 50.0)
    trade_analysis = analyze_by_type(trades)
    
    # Print results
    print_results(metrics, trade_analysis)
    
    # Save results
    results = {
        'metrics': metrics,
        'trade_analysis': trade_analysis,
        'trade_count': len(trades),
        'signal_count': len(df[df['signal'] != 'none'])
    }
    
    with open('quick_parabolic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to quick_parabolic_results.json")
    print(f"📊 Processed {len(df)} bars of data")
    print(f"🎯 Generated {len(trades)} trades from {len(df[df['signal'] != 'none'])} signals")

if __name__ == "__main__":
    main() 