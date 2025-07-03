#!/usr/bin/env python3
"""
Lucrative $40 Profit Bot - Optimized Backtesting
AI-optimized parameters with 20-30x leverage and trailing stops
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LucrativeBacktester:
    """Optimized backtesting for maximum profitability"""
    
    def __init__(self, initial_balance: float = 500.0):
        self.initial_balance = initial_balance
        
        # AI-OPTIMIZED PARAMETER SETS
        self.strategies = {
            "ULTRA_AGGRESSIVE": {
                "leverage": 30,
                "min_confidence": 45,  # Lowered for more trades
                "position_size_pct": 25,
                "trailing_stop_pct": 0.8,
                "trailing_activation": 0.5,
                "profit_target_pct": 3.0,
                "max_hold_hours": 4,
                "volume_threshold": 1.2  # Lowered threshold
            },
            "HIGH_LEVERAGE": {
                "leverage": 25,
                "min_confidence": 50,  # Lowered for more trades
                "position_size_pct": 20,
                "trailing_stop_pct": 1.0,
                "trailing_activation": 0.8,
                "profit_target_pct": 2.5,
                "max_hold_hours": 6,
                "volume_threshold": 1.4  # Lowered threshold
            },
            "BALANCED_HIGH": {
                "leverage": 22,
                "min_confidence": 55,  # Lowered for more trades
                "position_size_pct": 18,
                "trailing_stop_pct": 1.2,
                "trailing_activation": 1.0,
                "profit_target_pct": 2.2,
                "max_hold_hours": 8,
                "volume_threshold": 1.6  # Lowered threshold
            },
            "CONSERVATIVE_HIGH": {
                "leverage": 20,
                "min_confidence": 60,  # Lowered for more trades
                "position_size_pct": 15,
                "trailing_stop_pct": 1.5,
                "trailing_activation": 1.2,
                "profit_target_pct": 2.0,
                "max_hold_hours": 12,
                "volume_threshold": 1.8  # Lowered threshold
            }
        }
        
        print("💰 LUCRATIVE BOT - OPTIMIZED BACKTESTING")
        print("🎯 20-30X LEVERAGE WITH TRAILING STOPS")
        print("🤖 AI-OPTIMIZED PARAMETERS")
        print("=" * 60)
    
    def generate_market_data(self, days: int = 60) -> pd.DataFrame:
        """Generate 2 months of realistic SOL market data"""
        print(f"📊 Generating {days} days of realistic market data...")
        
        start_price = 120.0
        total_minutes = days * 24 * 60
        np.random.seed(42)
        
        data = []
        current_price = start_price
        current_time = datetime.now() - timedelta(days=days)
        
        for i in range(total_minutes):
            # Realistic volatility based on time of day
            hour = (i // 60) % 24
            if 0 <= hour < 8:  # Asian session - lower volatility
                volatility = 0.002
            elif 8 <= hour < 16:  # European session - medium volatility
                volatility = 0.003
            else:  # US session - higher volatility
                volatility = 0.004
            
            # Trend and noise
            trend = np.sin(i / (7 * 24 * 60) * 2 * np.pi) * 0.001
            noise = np.random.normal(0, volatility)
            
            # Price movement
            price_change = trend + noise
            current_price *= (1 + price_change)
            current_price = max(80, min(200, current_price))
            
            # OHLC generation
            spread = current_price * volatility * 0.5
            high = current_price + np.random.uniform(0, spread)
            low = current_price - np.random.uniform(0, spread)
            open_price = current_price + np.random.uniform(-spread/2, spread/2)
            
            # Volume simulation
            base_volume = 1000000
            volume_mult = 1 + abs(price_change) * 100 + np.random.uniform(0.5, 2.0)
            volume = base_volume * volume_mult
            
            data.append({
                'timestamp': current_time + timedelta(minutes=i),
                'open': max(low, min(high, open_price)),
                'high': high,
                'low': low,
                'close': current_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} candles: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        return df
    
    def calculate_indicators(self, data: pd.DataFrame, idx: int) -> dict:
        """Calculate technical indicators"""
        if idx < 50:
            return {"confidence": 0}
        
        window = data.iloc[max(0, idx-50):idx+1]
        
        # RSI
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Moving averages
        ma_5 = window['close'].rolling(5).mean().iloc[-1]
        ma_20 = window['close'].rolling(20).mean().iloc[-1]
        
        # Volume
        volume_ma = window['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = window['volume'].iloc[-1] / volume_ma
        
        # Momentum
        if len(window) >= 6:
            momentum = (window['close'].iloc[-1] - window['close'].iloc[-6]) / window['close'].iloc[-6] * 100
        else:
            momentum = 0
        
        return {
            'rsi': current_rsi,
            'ma_5': ma_5,
            'ma_20': ma_20,
            'volume_ratio': volume_ratio,
            'momentum': momentum,
            'current_price': window['close'].iloc[-1]
        }
    
    def analyze_signal(self, indicators: dict, params: dict) -> dict:
        """AI signal analysis"""
        confidence = 0
        direction = "hold"
        reasons = []
        
        # RSI signals
        if indicators['rsi'] < 30:
            confidence += 30
            direction = "long"
            reasons.append("oversold")
        elif indicators['rsi'] > 70:
            confidence += 30
            direction = "short"
            reasons.append("overbought")
        
        # MA signals
        if indicators['current_price'] > indicators['ma_5'] > indicators['ma_20']:
            if direction == "long" or direction == "hold":
                confidence += 25
                direction = "long"
                reasons.append("bullish_ma")
        elif indicators['current_price'] < indicators['ma_5'] < indicators['ma_20']:
            if direction == "short" or direction == "hold":
                confidence += 25
                direction = "short"
                reasons.append("bearish_ma")
        
        # Volume confirmation
        if indicators['volume_ratio'] > params['volume_threshold']:
            confidence += 20
            reasons.append("volume_breakout")
        
        # Momentum
        if abs(indicators['momentum']) > 1.0:
            if direction == "long" and indicators['momentum'] > 0:
                confidence += 15
                reasons.append("positive_momentum")
            elif direction == "short" and indicators['momentum'] < 0:
                confidence += 15
                reasons.append("negative_momentum")
        
        return {
            "confidence": min(confidence, 95),
            "direction": direction,
            "reasons": reasons
        }
    
    def simulate_trade_with_trailing_stop(self, entry_price: float, direction: str, 
                                        data: pd.DataFrame, start_idx: int, params: dict) -> dict:
        """Simulate trade with trailing stop loss"""
        leverage = params['leverage']
        trailing_stop_pct = params['trailing_stop_pct']
        trailing_activation = params['trailing_activation']
        profit_target_pct = params['profit_target_pct']
        max_hold_hours = params['max_hold_hours']
        
        best_price = entry_price
        trailing_stop_price = None
        trailing_activated = False
        
        # Calculate targets
        if direction == 'long':
            profit_target = entry_price * (1 + profit_target_pct / 100)
            initial_stop = entry_price * (1 - trailing_stop_pct / 100)
        else:
            profit_target = entry_price * (1 - profit_target_pct / 100)
            initial_stop = entry_price * (1 + trailing_stop_pct / 100)
        
        max_candles = max_hold_hours * 60
        
        for i in range(start_idx + 1, min(start_idx + max_candles + 1, len(data))):
            candle = data.iloc[i]
            high = candle['high']
            low = candle['low']
            close = candle['close']
            
            if direction == 'long':
                # Update best price
                if high > best_price:
                    best_price = high
                
                # Activate trailing stop
                if not trailing_activated:
                    profit_pct = (best_price - entry_price) / entry_price * 100
                    if profit_pct >= trailing_activation:
                        trailing_activated = True
                        trailing_stop_price = best_price * (1 - trailing_stop_pct / 100)
                
                # Update trailing stop
                if trailing_activated:
                    new_stop = best_price * (1 - trailing_stop_pct / 100)
                    if new_stop > trailing_stop_price:
                        trailing_stop_price = new_stop
                
                # Check exits
                if high >= profit_target:
                    pnl_pct = (profit_target - entry_price) / entry_price * 100 * leverage
                    pnl_amount = (params['position_size_pct'] / 100 * self.initial_balance) * (pnl_pct / 100)
                    return {
                        'exit_price': profit_target,
                        'exit_reason': 'take_profit',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated
                    }
                
                if trailing_activated and low <= trailing_stop_price:
                    pnl_pct = (trailing_stop_price - entry_price) / entry_price * 100 * leverage
                    pnl_amount = (params['position_size_pct'] / 100 * self.initial_balance) * (pnl_pct / 100)
                    return {
                        'exit_price': trailing_stop_price,
                        'exit_reason': 'trailing_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated
                    }
                
                if not trailing_activated and low <= initial_stop:
                    pnl_pct = (initial_stop - entry_price) / entry_price * 100 * leverage
                    pnl_amount = (params['position_size_pct'] / 100 * self.initial_balance) * (pnl_pct / 100)
                    return {
                        'exit_price': initial_stop,
                        'exit_reason': 'stop_loss',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated
                    }
            
            else:  # Short
                if low < best_price:
                    best_price = low
                
                if not trailing_activated:
                    profit_pct = (entry_price - best_price) / entry_price * 100
                    if profit_pct >= trailing_activation:
                        trailing_activated = True
                        trailing_stop_price = best_price * (1 + trailing_stop_pct / 100)
                
                if trailing_activated:
                    new_stop = best_price * (1 + trailing_stop_pct / 100)
                    if new_stop < trailing_stop_price:
                        trailing_stop_price = new_stop
                
                if low <= profit_target:
                    pnl_pct = (entry_price - profit_target) / entry_price * 100 * leverage
                    pnl_amount = (params['position_size_pct'] / 100 * self.initial_balance) * (pnl_pct / 100)
                    return {
                        'exit_price': profit_target,
                        'exit_reason': 'take_profit',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated
                    }
                
                if trailing_activated and high >= trailing_stop_price:
                    pnl_pct = (entry_price - trailing_stop_price) / entry_price * 100 * leverage
                    pnl_amount = (params['position_size_pct'] / 100 * self.initial_balance) * (pnl_pct / 100)
                    return {
                        'exit_price': trailing_stop_price,
                        'exit_reason': 'trailing_stop',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated
                    }
                
                if not trailing_activated and high >= initial_stop:
                    pnl_pct = (entry_price - initial_stop) / entry_price * 100 * leverage
                    pnl_amount = (params['position_size_pct'] / 100 * self.initial_balance) * (pnl_pct / 100)
                    return {
                        'exit_price': initial_stop,
                        'exit_reason': 'stop_loss',
                        'pnl_amount': pnl_amount,
                        'hold_minutes': i - start_idx,
                        'trailing_activated': trailing_activated
                    }
        
        # Time exit
        final_price = data.iloc[min(start_idx + max_candles, len(data) - 1)]['close']
        if direction == 'long':
            pnl_pct = (final_price - entry_price) / entry_price * 100 * leverage
        else:
            pnl_pct = (entry_price - final_price) / entry_price * 100 * leverage
        
        pnl_amount = (params['position_size_pct'] / 100 * self.initial_balance) * (pnl_pct / 100)
        
        return {
            'exit_price': final_price,
            'exit_reason': 'time_exit',
            'pnl_amount': pnl_amount,
            'hold_minutes': max_candles,
            'trailing_activated': trailing_activated
        }
    
    def run_backtest(self, data: pd.DataFrame, strategy_name: str) -> dict:
        """Run backtest for a specific strategy"""
        params = self.strategies[strategy_name]
        balance = self.initial_balance
        trades = []
        wins = 0
        losses = 0
        
        print(f"\n🧪 Testing {strategy_name} Strategy:")
        print(f"   ⚡ Leverage: {params['leverage']}x")
        print(f"   🎯 Min Confidence: {params['min_confidence']}%")
        print(f"   🛡️ Trailing Stop: {params['trailing_stop_pct']}%")
        
        i = 60
        while i < len(data) - 100:
            indicators = self.calculate_indicators(data, i)
            if indicators.get('confidence', 0) == 0:
                i += 1
                continue
            
            signal = self.analyze_signal(indicators, params)
            
            if signal['confidence'] >= params['min_confidence'] and signal['direction'] != "hold":
                entry_price = indicators['current_price']
                
                # Simulate trade
                result = self.simulate_trade_with_trailing_stop(
                    entry_price, signal['direction'], data, i, params
                )
                
                balance += result['pnl_amount']
                
                if result['pnl_amount'] > 0:
                    wins += 1
                else:
                    losses += 1
                
                trade = {
                    'entry_price': entry_price,
                    'exit_price': result['exit_price'],
                    'direction': signal['direction'],
                    'pnl_amount': result['pnl_amount'],
                    'exit_reason': result['exit_reason'],
                    'hold_minutes': result['hold_minutes'],
                    'trailing_activated': result['trailing_activated'],
                    'balance': balance
                }
                trades.append(trade)
                
                i += result['hold_minutes'] + 30
            else:
                i += 5
        
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_return = ((balance - self.initial_balance) / self.initial_balance) * 100
        
        exit_reasons = {}
        for trade in trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        trailing_activations = sum(1 for t in trades if t['trailing_activated'])
        
        print(f"   📊 Results: {total_trades} trades, {win_rate:.1f}% WR, {total_return:.1f}% return")
        
        return {
            'strategy': strategy_name,
            'params': params,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_balance': balance,
            'trades': trades,
            'exit_reasons': exit_reasons,
            'trailing_activations': trailing_activations
        }
    
    def run_all_strategies(self, data: pd.DataFrame) -> dict:
        """Run all strategy backtests"""
        print("\n🚀 RUNNING ALL STRATEGY BACKTESTS")
        print("=" * 60)
        
        results = {}
        for strategy_name in self.strategies.keys():
            results[strategy_name] = self.run_backtest(data, strategy_name)
        
        return results
    
    def display_results(self, results: dict):
        """Display comprehensive results"""
        print("\n" + "="*80)
        print("🏆 LUCRATIVE BOT BACKTESTING RESULTS")
        print("="*80)
        
        # Sort by profitability
        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_return'], reverse=True)
        
        for i, (strategy, result) in enumerate(sorted_results):
            params = result['params']
            
            print(f"\n🥇 RANK #{i+1}: {strategy}")
            print(f"   📊 PERFORMANCE:")
            print(f"      🏆 Win Rate: {result['win_rate']:.1f}% ({result['wins']}W/{result['losses']}L)")
            print(f"      💰 Total Return: {result['total_return']:.1f}%")
            print(f"      💵 Final Balance: ${result['final_balance']:.2f}")
            print(f"      📈 Total Trades: {result['total_trades']}")
            
            print(f"   🔧 PARAMETERS:")
            print(f"      ⚡ Leverage: {params['leverage']}x")
            print(f"      🎯 Min Confidence: {params['min_confidence']}%")
            print(f"      💰 Position Size: {params['position_size_pct']}%")
            print(f"      🛡️ Trailing Stop: {params['trailing_stop_pct']}%")
            print(f"      ⚡ Activation: {params['trailing_activation']}%")
            print(f"      🎯 Target: {params['profit_target_pct']}%")
            
            if result['trades']:
                profitable_trades = [t for t in result['trades'] if t['pnl_amount'] > 0]
                if profitable_trades:
                    avg_win = np.mean([t['pnl_amount'] for t in profitable_trades])
                    max_win = max([t['pnl_amount'] for t in profitable_trades])
                    print(f"      💚 Avg Win: ${avg_win:.2f} | Max Win: ${max_win:.2f}")
                
                losing_trades = [t for t in result['trades'] if t['pnl_amount'] < 0]
                if losing_trades:
                    avg_loss = np.mean([t['pnl_amount'] for t in losing_trades])
                    max_loss = min([t['pnl_amount'] for t in losing_trades])
                    print(f"      ❌ Avg Loss: ${avg_loss:.2f} | Max Loss: ${max_loss:.2f}")
            
            print(f"   🛡️ TRAILING STOP:")
            print(f"      ⚡ Activations: {result['trailing_activations']}/{result['total_trades']}")
            if result['trailing_activations'] > 0:
                effectiveness = result['trailing_activations']/result['total_trades']*100
                print(f"      🎯 Effectiveness: {effectiveness:.1f}%")
            
            exit_reasons = result['exit_reasons']
            if exit_reasons:
                total_exits = sum(exit_reasons.values())
                print(f"   📤 EXIT BREAKDOWN:")
                for reason, count in exit_reasons.items():
                    pct = count/total_exits*100
                    print(f"      {reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
            
            print("-" * 80)
        
        # Recommendations
        if sorted_results:
            best_strategy = sorted_results[0][0]
            best_result = sorted_results[0][1]
            
            print(f"\n🎯 RECOMMENDED STRATEGY: {best_strategy}")
            print(f"   💰 Expected Return: {best_result['total_return']:.1f}%")
            print(f"   🏆 Win Rate: {best_result['win_rate']:.1f}%")
            print(f"   ⚡ Leverage: {best_result['params']['leverage']}x")
            print(f"   🛡️ Trailing Stop: {best_result['params']['trailing_stop_pct']}%")
            
            print(f"\n💡 OPTIMIZATION INSIGHTS:")
            
            # Analyze leverage performance
            leverage_returns = {}
            for strategy, result in results.items():
                lev = result['params']['leverage']
                if lev not in leverage_returns:
                    leverage_returns[lev] = []
                leverage_returns[lev].append(result['total_return'])
            
            best_leverage = max(leverage_returns.keys(), key=lambda x: np.mean(leverage_returns[x]))
            print(f"   ⚡ Optimal Leverage Range: {best_leverage}x")
            
            # Trailing stop analysis
            trailing_performance = {}
            for strategy, result in results.items():
                trail = result['params']['trailing_stop_pct']
                if trail not in trailing_performance:
                    trailing_performance[trail] = []
                trailing_performance[trail].append(result['win_rate'])
            
            best_trailing = max(trailing_performance.keys(), key=lambda x: np.mean(trailing_performance[x]))
            print(f"   🛡️ Optimal Trailing Stop: {best_trailing}%")
            
            print("="*80)

def main():
    """Main function"""
    print("💰 LUCRATIVE $40 PROFIT BOT - OPTIMIZED BACKTESTING")
    print("🎯 AI-OPTIMIZED 20-30X LEVERAGE WITH TRAILING STOPS")
    print("=" * 70)
    
    try:
        balance = float(input("💵 Enter starting balance (default $500): ") or "500")
    except ValueError:
        balance = 500.0
    
    try:
        days = int(input("📅 Backtest period in days (default 60): ") or "60")
    except ValueError:
        days = 60
    
    backtester = LucrativeBacktester(initial_balance=balance)
    
    # Generate market data
    data = backtester.generate_market_data(days=days)
    
    # Run all strategies
    results = backtester.run_all_strategies(data)
    
    # Display results
    backtester.display_results(results)

if __name__ == "__main__":
    main()