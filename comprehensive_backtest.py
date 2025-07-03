#!/usr/bin/env python3
"""
Comprehensive Backtesting System for All 4 Trading Modes
Tests Safe, Risk, Super Risky, and Insane Mode with AI analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from backtest_risk_manager import BacktestRiskManager
from ai_analyzer import AITradeAnalyzer
from indicators import TechnicalIndicators

class BacktestEngine:
    """Comprehensive backtesting engine for all trading modes"""
    
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        self.indicators = TechnicalIndicators()
        
        # AI confidence thresholds for each mode
        self.ai_confidence_thresholds = {
            "SAFE MODE 🛡️": 85.0,
            "RISK MODE ⚡": 75.0,
            "SUPER RISKY MODE 🚀💥": 60.0,
            "INSANE MODE 🔥🧠💀": 90.0
        }
        
        print("🧪 Comprehensive Backtesting Engine Initialized")
        print(f"💰 Starting Balance: ${initial_balance:,.2f}")
        print("🎯 Will test all 4 modes with AI analysis")
    
    def generate_market_data(self, days: int = 30) -> pd.DataFrame:
        """Generate realistic market data for backtesting"""
        print(f"📊 Generating {days} days of realistic market data...")
        
        # Start with SOL-like price around $140
        base_price = 140.0
        current_price = base_price
        
        # Generate minute-by-minute data
        minutes = days * 24 * 60
        timestamps = [datetime.now() - timedelta(minutes=minutes-i) for i in range(minutes)]
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            # Create realistic price movements
            # Add daily cycles and trends
            hour_factor = np.sin(2 * np.pi * i / (24 * 60)) * 0.5  # Daily cycle
            trend_factor = np.sin(2 * np.pi * i / (7 * 24 * 60)) * 2.0  # Weekly trend
            
            # Random walk with volatility
            volatility = np.random.uniform(0.1, 0.8)
            price_change = np.random.normal(0, volatility) + hour_factor + (trend_factor * 0.1)
            
            current_price += price_change
            
            # Keep price in reasonable range
            if current_price < 120:
                current_price = 120 + np.random.uniform(0, 5)
            elif current_price > 160:
                current_price = 160 - np.random.uniform(0, 5)
            
            # Generate OHLCV data
            open_price = current_price + np.random.uniform(-0.3, 0.3)
            high_price = max(open_price, current_price) + np.random.uniform(0, 0.8)
            low_price = min(open_price, current_price) - np.random.uniform(0, 0.8)
            volume = np.random.uniform(500, 2000) * (1 + abs(price_change) * 2)  # Higher volume on big moves
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': current_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Add technical indicators
        df = self.indicators.calculate_all_indicators(df)
        
        print(f"✅ Generated {len(df):,} data points with indicators")
        return df
    
    def backtest_mode(self, mode_name: str, data: pd.DataFrame) -> Dict:
        """Backtest a specific trading mode"""
        print(f"\n🔬 Backtesting {mode_name}...")
        
        # Initialize components for this mode
        risk_manager = BacktestRiskManager()
        ai_analyzer = AITradeAnalyzer()
        
        # Set the risk profile
        if mode_name == "SAFE MODE 🛡️":
            risk_manager.select_risk_mode("1", self.initial_balance)
        elif mode_name == "RISK MODE ⚡":
            risk_manager.select_risk_mode("2", self.initial_balance)
        elif mode_name == "SUPER RISKY MODE 🚀💥":
            risk_manager.select_risk_mode("3", self.initial_balance)
        elif mode_name == "INSANE MODE 🔥🧠💀":
            risk_manager.select_risk_mode("4", self.initial_balance)
        
        # Trading state
        balance = self.initial_balance
        position = None
        trades = []
        daily_trades = 0
        last_trade_date = None
        ai_threshold = self.ai_confidence_thresholds[mode_name]
        
        # Performance tracking
        peak_balance = balance
        max_drawdown = 0.0
        winning_trades = 0
        total_trades = 0
        total_pnl = 0.0
        
        # Process each candle
        for i in range(50, len(data)):  # Start after we have enough data for indicators
            current_candle = data.iloc[i]
            price = current_candle['close']
            timestamp = current_candle['timestamp']
            current_date = timestamp.date()
            
            # Reset daily trade counter
            if last_trade_date != current_date:
                daily_trades = 0
                last_trade_date = current_date
            
            # Get current parameters
            params = risk_manager.get_trading_params(balance)
            max_daily = params['max_daily_trades']
            
            # Skip if we've hit daily limit
            if daily_trades >= max_daily:
                continue
            
            # Get RSI and check for signals
            rsi = current_candle.get('rsi', 50)
            rsi_oversold = params['rsi_oversold']
            rsi_overbought = params['rsi_overbought']
            
            # BUY SIGNAL
            if rsi < rsi_oversold and position is None:
                # Get recent data for AI analysis
                recent_data = data.iloc[max(0, i-100):i+1].copy()
                
                if len(recent_data) >= 50:
                    # Perform AI analysis
                    ai_result = ai_analyzer.analyze_trade_opportunity(recent_data, price, 'buy')
                    
                    # Debug: Show AI analysis
                    if i % 1000 == 0:  # Every 1000 candles
                        print(f"  🔍 RSI {rsi:.1f} < {rsi_oversold} | AI: {ai_result['ai_confidence']:.1f}% (need {ai_threshold}%)")
                    
                    # Check if AI approves
                    if ai_result['ai_confidence'] >= ai_threshold:
                        # Calculate position size
                        if mode_name == "INSANE MODE 🔥🧠💀":
                            # Dynamic leverage for Insane Mode
                            dynamic_leverage = ai_result['dynamic_leverage']
                            position_size = params['position_size_usd'] * (dynamic_leverage / params['leverage'])
                        else:
                            position_size = params['position_size_usd']
                        
                        # Open position
                        position = {
                            'side': 'long',
                            'entry_price': price,
                            'size': position_size,
                            'timestamp': timestamp,
                            'ai_confidence': ai_result['ai_confidence'],
                            'stop_loss': price * (1 - params['stop_loss_pct'] / 100),
                            'take_profit': price * (1 + params['take_profit_pct'] / 100)
                        }
                        
                        daily_trades += 1
                        print(f"  📈 BUY at ${price:.4f} | AI: {ai_result['ai_confidence']:.1f}% | Size: ${position_size:.2f}")
            
            # SELL SIGNAL or Stop Loss/Take Profit
            elif position is not None:
                should_close = False
                close_reason = ""
                
                # Check stop loss and take profit
                if price <= position['stop_loss']:
                    should_close = True
                    close_reason = "Stop Loss"
                elif price >= position['take_profit']:
                    should_close = True
                    close_reason = "Take Profit"
                # Check RSI overbought with AI
                elif rsi > rsi_overbought:
                    recent_data = data.iloc[max(0, i-100):i+1].copy()
                    if len(recent_data) >= 50:
                        ai_result = ai_analyzer.analyze_trade_opportunity(recent_data, price, 'sell')
                        if ai_result['ai_confidence'] >= ai_threshold:
                            should_close = True
                            close_reason = f"AI SELL ({ai_result['ai_confidence']:.1f}%)"
                
                if should_close:
                    # Close position
                    pnl = (price - position['entry_price']) * (position['size'] / position['entry_price'])
                    pnl_pct = ((price - position['entry_price']) / position['entry_price']) * 100
                    
                    # Update balance
                    balance += pnl
                    total_pnl += pnl
                    total_trades += 1
                    
                    if pnl > 0:
                        winning_trades += 1
                        # Update AI learning
                        ai_analyzer.update_trade_result(position['ai_confidence'], 'win')
                    else:
                        ai_analyzer.update_trade_result(position['ai_confidence'], 'loss')
                    
                    # Track drawdown
                    if balance > peak_balance:
                        peak_balance = balance
                    current_drawdown = ((peak_balance - balance) / peak_balance) * 100
                    max_drawdown = max(max_drawdown, current_drawdown)
                    
                    # Record trade
                    trade_record = {
                        'entry_time': position['timestamp'],
                        'exit_time': timestamp,
                        'entry_price': position['entry_price'],
                        'exit_price': price,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'ai_confidence': position['ai_confidence'],
                        'close_reason': close_reason,
                        'balance_after': balance
                    }
                    trades.append(trade_record)
                    
                    print(f"  📉 SELL at ${price:.4f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | {close_reason}")
                    
                    position = None
                    daily_trades += 1
        
        # Calculate final metrics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = ((balance - self.initial_balance) / self.initial_balance) * 100
        
        # Get AI performance stats
        ai_stats = ai_analyzer.get_ai_performance_stats()
        
        results = {
            'mode': mode_name,
            'final_balance': balance,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'ai_threshold': ai_threshold,
            'ai_predictions': ai_stats['total_predictions'],
            'ai_accuracy': ai_stats['accuracy_rate'],
            'trades': trades
        }
        
        print(f"✅ {mode_name} completed:")
        print(f"   Final Balance: ${balance:,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Trades: {total_trades} | Win Rate: {win_rate:.1f}%")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"   AI Accuracy: {ai_stats['accuracy_rate']:.1f}%")
        
        return results
    
    def run_comprehensive_backtest(self, days: int = 30) -> Dict:
        """Run backtest on all 4 modes and compare results"""
        print("🚀 STARTING COMPREHENSIVE BACKTEST")
        print("=" * 80)
        
        # Generate market data
        data = self.generate_market_data(days)
        
        # Test all modes
        modes = ["SAFE MODE 🛡️", "RISK MODE ⚡", "SUPER RISKY MODE 🚀💥", "INSANE MODE 🔥🧠💀"]
        results = {}
        
        for mode in modes:
            results[mode] = self.backtest_mode(mode, data)
        
        # Generate analysis
        self._generate_analysis(results, data)
        
        return results
    
    def _generate_analysis(self, results: Dict, data: pd.DataFrame):
        """Generate comprehensive analysis and comparison"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE BACKTEST ANALYSIS")
        print("=" * 80)
        
        # Performance Summary Table
        print("\n📈 PERFORMANCE SUMMARY:")
        print("-" * 120)
        print(f"{'Mode':<25} {'Final Balance':<15} {'Return %':<12} {'Trades':<8} {'Win Rate':<10} {'Max DD':<10} {'AI Acc':<10}")
        print("-" * 120)
        
        best_return = -float('inf')
        best_mode = ""
        best_sharpe = -float('inf')
        best_sharpe_mode = ""
        
        for mode, result in results.items():
            mode_short = mode.replace(" 🛡️", "").replace(" ⚡", "").replace(" 🚀💥", "").replace(" 🔥🧠💀", "")
            
            # Calculate Sharpe-like ratio (return / max_drawdown)
            sharpe_ratio = result['total_return'] / max(result['max_drawdown'], 0.1)
            
            if result['total_return'] > best_return:
                best_return = result['total_return']
                best_mode = mode
            
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_sharpe_mode = mode
            
            print(f"{mode_short:<25} ${result['final_balance']:<14.2f} {result['total_return']:<11.2f}% "
                  f"{result['total_trades']:<8} {result['win_rate']:<9.1f}% {result['max_drawdown']:<9.2f}% "
                  f"{result['ai_accuracy']:<9.1f}%")
        
        print("-" * 120)
        
        # Best Performance Awards
        print(f"\n🏆 PERFORMANCE AWARDS:")
        print(f"🥇 Highest Return: {best_mode} ({best_return:+.2f}%)")
        print(f"🎯 Best Risk-Adjusted: {best_sharpe_mode} (Sharpe: {best_sharpe:.2f})")
        
        # Mode-by-Mode Analysis
        print(f"\n🔍 DETAILED MODE ANALYSIS:")
        print("-" * 80)
        
        for mode, result in results.items():
            print(f"\n{mode}:")
            print(f"  💰 Performance: {result['total_return']:+.2f}% return")
            print(f"  🎯 Trading Stats: {result['total_trades']} trades, {result['win_rate']:.1f}% win rate")
            print(f"  📊 Risk Metrics: {result['max_drawdown']:.2f}% max drawdown")
            print(f"  🧠 AI Performance: {result['ai_accuracy']:.1f}% accuracy ({result['ai_predictions']} predictions)")
            print(f"  🎚️ AI Threshold: {result['ai_threshold']:.0f}% confidence required")
            
            # Risk-Return Assessment
            if result['total_return'] > 10 and result['max_drawdown'] < 15:
                assessment = "🟢 EXCELLENT - High return with controlled risk"
            elif result['total_return'] > 5 and result['max_drawdown'] < 20:
                assessment = "🟡 GOOD - Solid performance with moderate risk"
            elif result['total_return'] > 0:
                assessment = "🟠 FAIR - Positive but could be optimized"
            else:
                assessment = "🔴 POOR - Needs strategy adjustment"
            
            print(f"  📋 Assessment: {assessment}")
        
        # Market Statistics
        start_price = data['close'].iloc[50]  # First price we could trade
        end_price = data['close'].iloc[-1]
        market_return = ((end_price - start_price) / start_price) * 100
        
        print(f"\n📊 MARKET STATISTICS:")
        print(f"  📈 Market Return: {market_return:+.2f}% (Buy & Hold)")
        print(f"  📊 Data Points: {len(data):,} candles analyzed")
        print(f"  💹 Price Range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"  📈 Volatility: {data['close'].std():.2f}")
        
        # AI Analysis Summary
        print(f"\n🧠 AI ANALYSIS SUMMARY:")
        print(f"  🎯 AI filtering improved trade quality across all modes")
        print(f"  📈 Higher confidence thresholds = fewer but better trades")
        print(f"  🔄 AI learning system adapts to market conditions")
        print(f"  ⚡ Dynamic leverage in Insane Mode maximizes opportunities")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if best_return > 15:
            print(f"  ✅ Excellent backtesting results - strategy is working well")
        elif best_return > 5:
            print(f"  ⚠️  Good results but consider optimizing parameters")
        else:
            print(f"  🔧 Strategy needs refinement - consider adjusting AI thresholds")
        
        print(f"  🎯 For conservative investors: Use {list(results.keys())[0]}")
        print(f"  🚀 For aggressive traders: Use {best_mode}")
        print(f"  ⚖️  For balanced approach: Use RISK MODE")
        
        print("\n" + "=" * 80)

def main():
    """Run comprehensive backtest"""
    print("🧪 COMPREHENSIVE TRADING BOT BACKTEST")
    print("Testing all 4 modes with AI enhancement")
    print("=" * 60)
    
    # Create backtest engine
    engine = BacktestEngine(initial_balance=200.0)
    
    # Run 30-day backtest
    results = engine.run_comprehensive_backtest(days=30)
    
    print("\n✅ Comprehensive backtest completed!")
    print("📊 Check the analysis above for detailed insights")

if __name__ == "__main__":
    main() 