#!/usr/bin/env python3
"""
Edge Validation Suite - Complete testing pipeline for improved edge system
Includes walk-forward backtesting, pass/fail gates, and shadow trading
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import argparse
import sys
import os
from typing import Dict, List, Tuple
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
import sqlite3
from collections import deque

# Import our improved edge system
from improved_edge_system import ImprovedEdgeSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Metrics for validation gates"""
    expectancy: float
    profit_factor: float
    win_rate: float
    total_trades: int
    max_drawdown: float
    sharpe_ratio: float
    avg_win: float
    avg_loss: float
    
    def passes_gates(self) -> Tuple[bool, List[str]]:
        """Check if metrics pass all validation gates"""
        failures = []
        
        if self.expectancy < 0.0025:  # 0.25%
            failures.append(f"Expectancy {self.expectancy:.3%} < 0.25%")
        
        if self.profit_factor < 1.30:
            failures.append(f"Profit Factor {self.profit_factor:.2f} < 1.30")
        
        if self.total_trades < 100:
            failures.append(f"Total Trades {self.total_trades} < 100")
        
        if self.max_drawdown > 0.05:  # 5%
            failures.append(f"Max Drawdown {self.max_drawdown:.1%} > 5%")
        
        if self.sharpe_ratio < 1.0:
            failures.append(f"Sharpe Ratio {self.sharpe_ratio:.2f} < 1.0")
        
        return len(failures) == 0, failures

class WalkForwardValidator:
    """Walk-forward validation with rolling windows"""
    
    def __init__(self, window_size: int = 5000, shift_size: int = 3000):
        self.window_size = window_size
        self.shift_size = shift_size
        self.results = []
        
    async def fetch_historical_data(self, asset: str, timeframe: str = '1m', 
                                  lookback_days: int = 60) -> List[dict]:
        """Fetch historical data for asset"""
        # In production, this would fetch from your data source
        # For testing, we'll generate synthetic data
        logger.info(f"Fetching {lookback_days} days of {timeframe} data for {asset}")
        
        candles = []
        base_prices = {'BTC': 50000, 'ETH': 3000, 'SOL': 100}
        base_price = base_prices.get(asset, 100)
        
        start_time = datetime.now() - timedelta(days=lookback_days)
        candles_per_day = 1440 if timeframe == '1m' else 288  # 1m or 5m
        
        current_price = base_price
        for i in range(lookback_days * candles_per_day):
            # Create realistic price movements
            volatility = 0.002 if asset == 'BTC' else 0.003
            change = np.random.normal(0, volatility)
            new_price = current_price * (1 + change)
            
            candle = {
                'timestamp': int((start_time + timedelta(minutes=i)).timestamp() * 1000),
                'open': current_price,
                'high': new_price * (1 + abs(np.random.normal(0, volatility/2))),
                'low': new_price * (1 - abs(np.random.normal(0, volatility/2))),
                'close': new_price,
                'volume': max(100, 1000 + np.random.normal(0, 200))
            }
            
            candles.append(candle)
            current_price = new_price
        
        return candles
    
    def run_window_backtest(self, candles: List[dict], start_idx: int, 
                          system: ImprovedEdgeSystem) -> ValidationMetrics:
        """Run backtest on a single window"""
        window_candles = candles[start_idx:start_idx + self.window_size]
        
        # Train on first 60% of window
        train_split = int(len(window_candles) * 0.6)
        system.train_models(window_candles[:train_split])
        
        # Test on remaining 40%
        trades = []
        balance = 10000
        equity_curve = [balance]
        
        for i in range(train_split + 100, len(window_candles) - 40):
            current_candles = window_candles[:i+1]
            current_price = window_candles[i]['close']
            
            # Generate signal
            order_book = {
                'bid_volume': np.random.uniform(1000, 5000),
                'ask_volume': np.random.uniform(1000, 5000)
            }
            
            signal = system.generate_signal(current_candles, order_book)
            
            if system.should_trade(signal):
                # Execute trade
                entry_price = current_price
                position_size = balance * 0.01  # 1% risk
                
                if signal['direction'] == 'long':
                    stop_price = entry_price - signal['stop_distance']
                    target_price = entry_price + signal['target_distance']
                else:
                    stop_price = entry_price + signal['stop_distance']
                    target_price = entry_price - signal['target_distance']
                
                # Simulate outcome
                exit_price = None
                exit_reason = None
                
                for j in range(1, min(41, len(window_candles) - i)):
                    future_candle = window_candles[i + j]
                    
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
                
                if exit_price is None:
                    exit_price = window_candles[min(i + 40, len(window_candles) - 1)]['close']
                    exit_reason = 'time'
                
                # Calculate P&L
                if signal['direction'] == 'long':
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                
                pnl_amount = position_size * pnl_pct
                balance += pnl_amount
                equity_curve.append(balance)
                
                trades.append({
                    'direction': signal['direction'],
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_amount': pnl_amount,
                    'reason': exit_reason
                })
        
        # Calculate metrics
        if not trades:
            return ValidationMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        df_trades = pd.DataFrame(trades)
        wins = df_trades[df_trades['pnl_pct'] > 0]
        losses = df_trades[df_trades['pnl_pct'] <= 0]
        
        win_rate = len(wins) / len(trades)
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        total_wins = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Calculate max drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Calculate Sharpe ratio
        returns = df_trades['pnl_pct'].values
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        return ValidationMetrics(
            expectancy=expectancy,
            profit_factor=profit_factor,
            win_rate=win_rate,
            total_trades=len(trades),
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss
        )
    
    async def validate_asset(self, asset: str, system: ImprovedEdgeSystem) -> Dict:
        """Run walk-forward validation for a single asset"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Validating {asset}")
        logger.info(f"{'='*50}")
        
        # Fetch data
        candles = await self.fetch_historical_data(asset, '1m', 60)
        
        # Run walk-forward windows
        window_results = []
        num_windows = (len(candles) - self.window_size) // self.shift_size + 1
        
        for i in range(num_windows):
            start_idx = i * self.shift_size
            if start_idx + self.window_size > len(candles):
                break
            
            logger.info(f"Window {i+1}/{num_windows} for {asset}")
            metrics = self.run_window_backtest(candles, start_idx, system)
            window_results.append(metrics)
            
            # Log window results
            logger.info(f"  Trades: {metrics.total_trades}")
            logger.info(f"  Expectancy: {metrics.expectancy:.3%}")
            logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
        
        # Aggregate results
        if window_results:
            avg_metrics = ValidationMetrics(
                expectancy=np.mean([m.expectancy for m in window_results]),
                profit_factor=np.mean([m.profit_factor for m in window_results]),
                win_rate=np.mean([m.win_rate for m in window_results]),
                total_trades=sum([m.total_trades for m in window_results]),
                max_drawdown=max([m.max_drawdown for m in window_results]),
                sharpe_ratio=np.mean([m.sharpe_ratio for m in window_results]),
                avg_win=np.mean([m.avg_win for m in window_results]),
                avg_loss=np.mean([m.avg_loss for m in window_results])
            )
            
            return {
                'asset': asset,
                'metrics': avg_metrics,
                'windows': len(window_results),
                'window_results': window_results
            }
        
        return None

class ShadowTrader:
    """Paper trading system for validation"""
    
    def __init__(self, system: ImprovedEdgeSystem):
        self.system = system
        self.trades = []
        self.balance = 10000
        self.positions = {}
        self.metrics_db = 'shadow_metrics.db'
        self.init_db()
        
    def init_db(self):
        """Initialize metrics database"""
        conn = sqlite3.connect(self.metrics_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS shadow_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                asset TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl_pct REAL,
                reason TEXT,
                latency_ms REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    async def run_shadow_trade(self, asset: str, candles: list, 
                             order_book: dict, latency_ms: float):
        """Execute a shadow trade"""
        signal = self.system.generate_signal(candles, order_book)
        
        if self.system.should_trade(signal) and asset not in self.positions:
            # Open position
            current_price = candles[-1]['close']
            position = {
                'asset': asset,
                'direction': signal['direction'],
                'entry_price': current_price,
                'stop_price': current_price - signal['stop_distance'] if signal['direction'] == 'long' 
                            else current_price + signal['stop_distance'],
                'target_price': current_price + signal['target_distance'] if signal['direction'] == 'long'
                              else current_price - signal['target_distance'],
                'entry_time': datetime.now(),
                'size': self.balance * 0.01  # 1% risk
            }
            self.positions[asset] = position
            
            logger.info(f"📝 SHADOW: Opened {signal['direction']} on {asset} @ {current_price:.2f}")
    
    def check_exits(self, asset: str, current_candle: dict):
        """Check if any positions should be closed"""
        if asset not in self.positions:
            return
        
        position = self.positions[asset]
        current_price = current_candle['close']
        
        exit_price = None
        exit_reason = None
        
        if position['direction'] == 'long':
            if current_candle['low'] <= position['stop_price']:
                exit_price = position['stop_price']
                exit_reason = 'stop'
            elif current_candle['high'] >= position['target_price']:
                exit_price = position['target_price']
                exit_reason = 'target'
        else:
            if current_candle['high'] >= position['stop_price']:
                exit_price = position['stop_price']
                exit_reason = 'stop'
            elif current_candle['low'] <= position['target_price']:
                exit_price = position['target_price']
                exit_reason = 'target'
        
        # Time exit after 40 candles
        if exit_price is None and (datetime.now() - position['entry_time']).seconds > 200 * 60:
            exit_price = current_price
            exit_reason = 'time'
        
        if exit_price:
            # Calculate P&L
            if position['direction'] == 'long':
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
            
            # Update balance
            pnl_amount = position['size'] * pnl_pct
            self.balance += pnl_amount
            
            # Log trade
            self.log_trade(asset, position['direction'], position['entry_price'],
                         exit_price, pnl_pct, exit_reason, 0)
            
            # Remove position
            del self.positions[asset]
            
            emoji = "✅" if pnl_pct > 0 else "❌"
            logger.info(f"{emoji} SHADOW: Closed {position['direction']} on {asset} "
                       f"{pnl_pct:+.2%} ({exit_reason})")
    
    def log_trade(self, asset: str, direction: str, entry: float, 
                  exit: float, pnl_pct: float, reason: str, latency: float):
        """Log trade to database"""
        conn = sqlite3.connect(self.metrics_db)
        conn.execute('''
            INSERT INTO shadow_trades 
            (asset, direction, entry_price, exit_price, pnl_pct, reason, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (asset, direction, entry, exit, pnl_pct, reason, latency))
        conn.commit()
        conn.close()
        
        self.trades.append({
            'asset': asset,
            'direction': direction,
            'pnl_pct': pnl_pct,
            'reason': reason
        })
    
    def get_metrics(self) -> ValidationMetrics:
        """Calculate current shadow trading metrics"""
        if not self.trades:
            return ValidationMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        df = pd.DataFrame(self.trades)
        wins = df[df['pnl_pct'] > 0]
        losses = df[df['pnl_pct'] <= 0]
        
        win_rate = len(wins) / len(self.trades)
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        total_wins = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Simple drawdown calc
        cumsum = df['pnl_pct'].cumsum()
        running_max = cumsum.cummax()
        drawdown = cumsum - running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Sharpe
        returns = df['pnl_pct'].values
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        return ValidationMetrics(
            expectancy=expectancy,
            profit_factor=profit_factor,
            win_rate=win_rate,
            total_trades=len(self.trades),
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss
        )

async def main():
    """Main validation pipeline"""
    parser = argparse.ArgumentParser(description='Edge System Validation Suite')
    parser.add_argument('--assets', type=str, default='SOL,BTC,ETH',
                       help='Comma-separated list of assets')
    parser.add_argument('--window', type=int, default=5000,
                       help='Walk-forward window size')
    parser.add_argument('--shift', type=int, default=3000,
                       help='Window shift size')
    parser.add_argument('--min_trades', type=int, default=100,
                       help='Minimum trades required')
    parser.add_argument('--metrics_out', type=str, default='backtests/edge_results.csv',
                       help='Output file for metrics')
    parser.add_argument('--paper', action='store_true',
                       help='Run in paper trading mode')
    
    args = parser.parse_args()
    
    # Initialize system
    system = ImprovedEdgeSystem()
    
    if args.paper:
        # Shadow trading mode
        logger.info("🏃 Starting Shadow Trading Mode")
        
        # Train system first
        assets = args.assets.split(',')
        validator = WalkForwardValidator()
        training_data = await validator.fetch_historical_data(assets[0], '1m', 30)
        system.train_models(training_data[:int(len(training_data)*0.8)])
        logger.info("✅ System trained for shadow trading")
        
        shadow = ShadowTrader(system)
        
        # Simulate real-time trading
        trade_count = 0
        target_trades = 200
        
        while trade_count < target_trades:
            for asset in assets:
                # Fetch latest data (simulated)
                candles = await validator.fetch_historical_data(asset, '1m', 1)
                
                # Simulate order book
                order_book = {
                    'bid_volume': np.random.uniform(1000, 5000),
                    'ask_volume': np.random.uniform(1000, 5000)
                }
                
                # Simulate latency
                latency = np.random.uniform(50, 200)
                
                # Run shadow trade
                await shadow.run_shadow_trade(asset, candles, order_book, latency)
                
                # Check exits
                shadow.check_exits(asset, candles[-1])
            
            trade_count = len(shadow.trades)
            
            if trade_count % 10 == 0 and trade_count > 0:
                metrics = shadow.get_metrics()
                logger.info(f"\n📊 Shadow Metrics ({trade_count} trades):")
                logger.info(f"  Expectancy: {metrics.expectancy:.3%}")
                logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
                logger.info(f"  Win Rate: {metrics.win_rate:.1%}")
            
            await asyncio.sleep(1)  # Simulate time passing
        
        # Final metrics
        final_metrics = shadow.get_metrics()
        passed, failures = final_metrics.passes_gates()
        
        logger.info(f"\n{'='*50}")
        logger.info("SHADOW TRADING COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Total Trades: {final_metrics.total_trades}")
        logger.info(f"Expectancy: {final_metrics.expectancy:.3%}")
        logger.info(f"Profit Factor: {final_metrics.profit_factor:.2f}")
        logger.info(f"Max Drawdown: {final_metrics.max_drawdown:.1%}")
        logger.info(f"Sharpe Ratio: {final_metrics.sharpe_ratio:.2f}")
        
        if passed:
            logger.info("\n✅ ALL GATES PASSED - Ready for live deployment!")
            with open('shadow_status.txt', 'w') as f:
                f.write('PASSED')
        else:
            logger.error("\n❌ FAILED GATES:")
            for failure in failures:
                logger.error(f"  - {failure}")
            with open('shadow_status.txt', 'w') as f:
                f.write('FAILED')
            sys.exit(1)
    
    else:
        # Walk-forward validation mode
        logger.info("🔄 Starting Walk-Forward Validation")
        logger.info(f"Window: {args.window}, Shift: {args.shift}")
        
        validator = WalkForwardValidator(args.window, args.shift)
        assets = args.assets.split(',')
        
        all_results = []
        
        for asset in assets:
            result = await validator.validate_asset(asset, system)
            if result:
                all_results.append(result)
        
        # Aggregate all results
        if all_results:
            total_trades = sum(r['metrics'].total_trades for r in all_results)
            avg_expectancy = np.mean([r['metrics'].expectancy for r in all_results])
            avg_pf = np.mean([r['metrics'].profit_factor for r in all_results])
            max_dd = max(r['metrics'].max_drawdown for r in all_results)
            avg_sharpe = np.mean([r['metrics'].sharpe_ratio for r in all_results])
            
            aggregate_metrics = ValidationMetrics(
                expectancy=avg_expectancy,
                profit_factor=avg_pf,
                win_rate=np.mean([r['metrics'].win_rate for r in all_results]),
                total_trades=total_trades,
                max_drawdown=max_dd,
                sharpe_ratio=avg_sharpe,
                avg_win=np.mean([r['metrics'].avg_win for r in all_results]),
                avg_loss=np.mean([r['metrics'].avg_loss for r in all_results])
            )
            
            # Check gates
            passed, failures = aggregate_metrics.passes_gates()
            
            # Save results
            os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
            
            results_df = pd.DataFrame([
                {
                    'asset': r['asset'],
                    'expectancy': r['metrics'].expectancy,
                    'profit_factor': r['metrics'].profit_factor,
                    'win_rate': r['metrics'].win_rate,
                    'trades': r['metrics'].total_trades,
                    'max_dd': r['metrics'].max_drawdown,
                    'sharpe': r['metrics'].sharpe_ratio
                }
                for r in all_results
            ])
            
            results_df.to_csv(args.metrics_out, index=False)
            
            # Print summary
            logger.info(f"\n{'='*50}")
            logger.info("VALIDATION COMPLETE")
            logger.info(f"{'='*50}")
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Avg Expectancy: {avg_expectancy:.3%}")
            logger.info(f"Avg Profit Factor: {avg_pf:.2f}")
            logger.info(f"Max Drawdown: {max_dd:.1%}")
            logger.info(f"Avg Sharpe: {avg_sharpe:.2f}")
            
            if passed:
                logger.info("\n✅ ALL VALIDATION GATES PASSED!")
                logger.info("🚀 System ready for shadow trading")
                with open('validation_status.txt', 'w') as f:
                    f.write('PASSED')
                sys.exit(0)
            else:
                logger.error("\n❌ VALIDATION FAILED:")
                for failure in failures:
                    logger.error(f"  - {failure}")
                with open('validation_status.txt', 'w') as f:
                    f.write('FAILED')
                sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 