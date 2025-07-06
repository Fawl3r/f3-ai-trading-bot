#!/usr/bin/env python3
"""
Quick Edge System Test - Fast validation of improved system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from improved_edge_system import ImprovedEdgeSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_realistic_data(days=30, volatility=0.003):
    """Generate realistic market data"""
    candles = []
    base_price = 100
    current_price = base_price
    
    candles_per_day = 288  # 5-minute candles
    total_candles = days * candles_per_day
    
    for i in range(total_candles):
        # Create market regimes
        regime_cycle = (i % 2000) / 2000
        
        if regime_cycle < 0.3:  # Trending up
            trend = 0.0003
            vol = volatility * 0.8
        elif regime_cycle < 0.6:  # Ranging
            trend = 0
            vol = volatility * 0.6
        elif regime_cycle < 0.8:  # Trending down
            trend = -0.0002
            vol = volatility * 0.9
        else:  # High volatility
            trend = 0
            vol = volatility * 1.5
        
        # Price movement
        change = np.random.normal(trend, vol)
        new_price = current_price * (1 + change)
        
        # OHLC
        high = new_price * (1 + abs(np.random.normal(0, vol/3)))
        low = new_price * (1 - abs(np.random.normal(0, vol/3)))
        
        candle = {
            'timestamp': int((datetime.now() - timedelta(minutes=(total_candles-i)*5)).timestamp() * 1000),
            'open': current_price,
            'high': max(high, current_price, new_price),
            'low': min(low, current_price, new_price),
            'close': new_price,
            'volume': max(100, 1000 + np.random.normal(0, 300))
        }
        
        candles.append(candle)
        current_price = new_price
    
    return candles

def run_quick_backtest(system, candles, start_idx=5000):
    """Run a focused backtest"""
    trades = []
    balance = 10000
    equity_curve = [balance]
    daily_pnl = 0
    halt_trading = False
    
    for i in range(start_idx, len(candles) - 50):
        current_candles = candles[:i+1]
        current_price = candles[i]['close']
        
        # Check daily loss limit
        if daily_pnl <= -0.03:  # -3% daily limit
            halt_trading = True
            logger.warning(f"🛑 Daily loss limit hit: {daily_pnl:.2%}")
            break
        
        if halt_trading:
            continue
        
        # Generate signal with order book
        order_book = {
            'bid_volume': np.random.uniform(1000, 8000),
            'ask_volume': np.random.uniform(1000, 8000)
        }
        
        signal = system.generate_signal(current_candles, order_book)
        
        if system.should_trade(signal):
            # Position sizing
            risk_per_trade = 0.01  # 1%
            position_size = balance * risk_per_trade
            
            # Entry
            entry_price = current_price
            
            if signal['direction'] == 'long':
                stop_price = entry_price - signal['stop_distance']
                target_price = entry_price + signal['target_distance']
            else:
                stop_price = entry_price + signal['stop_distance']
                target_price = entry_price - signal['target_distance']
            
            # Simulate trade outcome
            exit_price = None
            exit_reason = None
            
            # Look ahead for exit
            for j in range(1, min(51, len(candles) - i)):
                future_candle = candles[i + j]
                
                if signal['direction'] == 'long':
                    if future_candle['low'] <= stop_price:
                        exit_price = stop_price
                        exit_reason = 'stop'
                        break
                    elif future_candle['high'] >= target_price:
                        exit_price = target_price
                        exit_reason = 'target'
                        break
                else:
                    if future_candle['high'] >= stop_price:
                        exit_price = stop_price
                        exit_reason = 'stop'
                        break
                    elif future_candle['low'] <= target_price:
                        exit_price = target_price
                        exit_reason = 'target'
                        break
            
            # Time exit if no stop/target hit
            if exit_price is None:
                exit_price = candles[min(i + 50, len(candles) - 1)]['close']
                exit_reason = 'time'
            
            # Calculate P&L
            if signal['direction'] == 'long':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            
            pnl_amount = position_size * pnl_pct
            balance += pnl_amount
            daily_pnl += pnl_pct
            equity_curve.append(balance)
            
            trade = {
                'direction': signal['direction'],
                'entry': entry_price,
                'exit': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_amount': pnl_amount,
                'reason': exit_reason,
                'edge': signal['edge'],
                'confidence': signal['confidence'],
                'atr': signal['atr']
            }
            trades.append(trade)
            
            # Log significant trades
            if abs(pnl_pct) > 0.015:  # >1.5% move
                emoji = "🚀" if pnl_pct > 0.02 else "✅" if pnl_pct > 0 else "💥" if pnl_pct < -0.015 else "❌"
                logger.info(f"{emoji} {signal['direction'].upper()}: {pnl_pct:+.2%} ({exit_reason}) - Edge: {signal['edge']:.3%}")
        
        # Reset daily P&L at market open (simplified)
        if i % 288 == 0:  # Every 24 hours
            daily_pnl = 0
            halt_trading = False
    
    return trades, balance, equity_curve

def analyze_results(trades, initial_balance, equity_curve):
    """Analyze backtest results"""
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    
    # Basic metrics
    total_trades = len(trades)
    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]
    
    win_rate = len(wins) / total_trades
    avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Profit factor
    total_wins = wins['pnl_pct'].sum() if len(wins) > 0 else 0
    total_losses = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    # Drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    # Sharpe ratio
    returns = df['pnl_pct'].values
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
    else:
        sharpe_ratio = 0
    
    # R-multiple analysis
    avg_r = df['pnl_pct'].mean() / abs(avg_loss) if avg_loss != 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_return': (equity_curve[-1] - initial_balance) / initial_balance,
        'avg_r_multiple': avg_r,
        'avg_edge': df['edge'].mean(),
        'avg_confidence': df['confidence'].mean()
    }

def main():
    print("🚀 QUICK EDGE SYSTEM VALIDATION")
    print("=" * 50)
    
    # Initialize system
    system = ImprovedEdgeSystem()
    
    # Generate test data
    print("📊 Generating market data...")
    candles = generate_realistic_data(days=60, volatility=0.003)
    
    # Train system
    print("🧠 Training models...")
    train_split = int(len(candles) * 0.6)
    training_results = system.train_models(candles[:train_split])
    
    print("\n📈 Model Performance:")
    for model, results in training_results.items():
        if 'edge_acc' in results:
            print(f"  {model}: {results['overall_acc']:.1%} overall, {results['edge_acc']:.1%} on edge trades")
        else:
            print(f"  {model}: {results['overall_acc']:.1%} overall")
    
    # Run backtest
    print("\n⚡ Running backtest...")
    trades, final_balance, equity_curve = run_quick_backtest(system, candles, train_split + 100)
    
    # Analyze results
    results = analyze_results(trades, 10000, equity_curve)
    
    if results:
        print(f"\n📊 RESULTS:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        print(f"   💰 EXPECTANCY: {results['expectancy']:.3%} per trade")
        print(f"   📈 Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Avg Win: {results['avg_win']:.2%}")
        print(f"   Avg Loss: {results['avg_loss']:.2%}")
        print(f"   Max Drawdown: {results['max_drawdown']:.1%}")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Total Return: {results['total_return']:.1%}")
        print(f"   Avg R-Multiple: {results['avg_r_multiple']:.2f}")
        print(f"   Avg Edge: {results['avg_edge']:.3%}")
        print(f"   Avg Confidence: {results['avg_confidence']:.1%}")
        
        # Validation gates
        print(f"\n🎯 VALIDATION GATES:")
        print(f"   Expectancy ≥ 0.25%: {'✅' if results['expectancy'] >= 0.0025 else '❌'} ({results['expectancy']:.3%})")
        print(f"   Profit Factor ≥ 1.3: {'✅' if results['profit_factor'] >= 1.3 else '❌'} ({results['profit_factor']:.2f})")
        print(f"   Trades ≥ 50: {'✅' if results['total_trades'] >= 50 else '❌'} ({results['total_trades']})")
        print(f"   Max DD ≤ 5%: {'✅' if results['max_drawdown'] <= 0.05 else '❌'} ({results['max_drawdown']:.1%})")
        print(f"   Sharpe ≥ 1.0: {'✅' if results['sharpe_ratio'] >= 1.0 else '❌'} ({results['sharpe_ratio']:.2f})")
        
        # Overall assessment
        gates_passed = (
            results['expectancy'] >= 0.0025 and
            results['profit_factor'] >= 1.3 and
            results['total_trades'] >= 50 and
            results['max_drawdown'] <= 0.05 and
            results['sharpe_ratio'] >= 1.0
        )
        
        if gates_passed:
            print(f"\n🎉 ALL GATES PASSED!")
            print(f"✅ System ready for shadow trading")
        else:
            print(f"\n🔄 Some gates failed, but system shows promise")
            if results['expectancy'] > 0 and results['profit_factor'] > 1.0:
                print(f"💡 Positive edge detected - proceed with caution")
        
        # R:R Analysis
        actual_rr = abs(results['avg_win'] / results['avg_loss']) if results['avg_loss'] != 0 else 0
        print(f"\n📐 Risk-Reward Analysis:")
        print(f"   Target R:R: 4.0:1")
        print(f"   Actual R:R: {actual_rr:.1f}:1")
        print(f"   Breakeven Win Rate: {1/(1+actual_rr):.1%}")
        print(f"   Actual Win Rate: {results['win_rate']:.1%}")
        
    else:
        print("❌ No trades generated - system too conservative")

if __name__ == "__main__":
    main() 