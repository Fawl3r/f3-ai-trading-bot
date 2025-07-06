#!/usr/bin/env python3
"""
Test Advanced AI 70%+ Win Rate System
Comprehensive Validation & Analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from advanced_ai_70wr_system import AdvancedAI70WRSystem
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAIBacktester:
    """Comprehensive backtester for Advanced AI System"""
    
    def __init__(self, initial_balance: float = 50.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades = []
        self.positions = {}
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
        # Initialize AI system
        self.ai_system = AdvancedAI70WRSystem()
        
        # Trading parameters optimized for 70%+ WR
        self.position_size_pct = 0.005  # 0.5% of balance per trade (very conservative)
        self.stop_loss_pct = 0.008      # 0.8% stop loss (tight)
        self.take_profit_pct = 0.012    # 1.2% take profit (conservative R:R 1.5:1)
        self.max_holding_time_hours = 1.5  # 1.5 hours max
        
    def generate_realistic_data(self, symbol: str = 'ETH', days: int = 60) -> List[Dict]:
        """Generate highly realistic market data with various regimes"""
        candles = []
        
        base_prices = {'ETH': 3500, 'BTC': 65000, 'AVAX': 45}
        base_price = base_prices.get(symbol, 3500)
        current_price = base_price
        
        start_time = datetime.now() - timedelta(days=days)
        current_time = start_time
        
        # Define market regimes with realistic transitions
        total_candles = days * 288  # 5-minute candles
        regime_schedule = [
            (0.0, 0.15, 'STRONG_BULL'),     # 15% strong bull
            (0.15, 0.3, 'BULL_TREND'),      # 15% bull trend  
            (0.3, 0.5, 'RANGE_BOUND'),      # 20% range bound
            (0.5, 0.65, 'BEAR_TREND'),      # 15% bear trend
            (0.65, 0.8, 'STRONG_BEAR'),     # 15% strong bear
            (0.8, 0.9, 'HIGH_VOLATILITY'),  # 10% high volatility
            (0.9, 1.0, 'RANGE_BOUND')       # 10% range bound
        ]
        
        candle_count = 0
        
        while current_time < datetime.now():
            # Determine current regime
            progress = candle_count / total_candles
            current_regime = 'RANGE_BOUND'
            
            for start_pct, end_pct, regime in regime_schedule:
                if start_pct <= progress < end_pct:
                    current_regime = regime
                    break
            
            # Generate price movement based on regime
            if current_regime == 'STRONG_BULL':
                base_drift = 0.0008  # Strong upward drift
                volatility = 0.006
            elif current_regime == 'BULL_TREND':
                base_drift = 0.0004  # Moderate upward drift
                volatility = 0.008
            elif current_regime == 'STRONG_BEAR':
                base_drift = -0.0008  # Strong downward drift
                volatility = 0.006
            elif current_regime == 'BEAR_TREND':
                base_drift = -0.0004  # Moderate downward drift
                volatility = 0.008
            elif current_regime == 'HIGH_VOLATILITY':
                base_drift = 0.0001  # Slight drift
                volatility = 0.018   # High volatility
            else:  # RANGE_BOUND
                base_drift = 0.0001  # Minimal drift
                volatility = 0.005   # Low volatility
            
            # Add mean reversion
            deviation = (current_price - base_price) / base_price
            mean_reversion = -deviation * 0.05  # 5% mean reversion strength
            
            # Generate price change
            random_component = np.random.normal(0, volatility)
            price_change = base_drift + mean_reversion + random_component
            
            # Apply price change
            open_price = current_price
            close_price = current_price * (1 + price_change)
            
            # Generate realistic OHLC
            if price_change > 0:  # Bullish candle
                high_price = close_price * (1 + abs(price_change) * np.random.uniform(0.2, 0.8))
                low_price = open_price * (1 - abs(price_change) * np.random.uniform(0.1, 0.4))
            else:  # Bearish candle
                high_price = open_price * (1 + abs(price_change) * np.random.uniform(0.1, 0.4))
                low_price = close_price * (1 - abs(price_change) * np.random.uniform(0.2, 0.8))
            
            # Ensure OHLC consistency
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume (correlated with volatility and price movement)
            base_volume = 1000 if symbol == 'ETH' else 5000
            volume_multiplier = 1 + (abs(price_change) * 20) + (volatility * 10)
            volume = base_volume * volume_multiplier * np.random.lognormal(0, 0.3)
            
            candle = {
                'timestamp': int(current_time.timestamp() * 1000),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 1)
            }
            
            candles.append(candle)
            current_price = close_price
            current_time += timedelta(minutes=5)
            candle_count += 1
        
        logger.info(f"Generated {len(candles)} realistic candles for {symbol}")
        return candles
    
    def calculate_position_size(self, price: float, signal) -> float:
        """Calculate optimal position size"""
        # Base position value
        base_value = self.current_balance * self.position_size_pct
        
        # Adjust based on signal quality
        quality_multiplier = signal.quality_score
        confidence_multiplier = signal.confidence
        
        # Risk adjustment
        risk_multiplier = 1.0 - signal.risk_score
        
        # Final position value
        position_value = base_value * quality_multiplier * confidence_multiplier * risk_multiplier
        position_size = position_value / price
        
        # Ensure minimum viable trade
        min_position_size = 0.001
        position_size = max(position_size, min_position_size)
        
        # Ensure we don't exceed balance
        max_affordable = (self.current_balance * 0.98) / price
        position_size = min(position_size, max_affordable)
        
        return round(position_size, 6)
    
    def execute_trade(self, signal, price: float, timestamp: datetime) -> Dict:
        """Execute trade with advanced parameters"""
        position_size = self.calculate_position_size(price, signal)
        
        if position_size <= 0:
            return None
        
        # Dynamic stop loss and take profit based on signal quality
        base_stop = self.stop_loss_pct
        base_profit = self.take_profit_pct
        
        # Adjust based on signal quality and market regime
        if signal.market_regime in ['STRONG_BULL', 'STRONG_BEAR']:
            profit_multiplier = 1.5  # Wider targets in strong trends
            stop_multiplier = 0.8    # Tighter stops
        elif signal.volatility_regime == 'HIGH_VOL':
            profit_multiplier = 1.2
            stop_multiplier = 1.2    # Wider stops in high vol
        else:
            profit_multiplier = 1.0
            stop_multiplier = 1.0
        
        # Quality-based adjustments
        quality_adj = 0.5 + (signal.quality_score * 0.5)
        profit_multiplier *= quality_adj
        
        if signal.direction == 'long':
            stop_loss = price * (1 - base_stop * stop_multiplier)
            take_profit = price * (1 + base_profit * profit_multiplier)
        else:
            stop_loss = price * (1 + base_stop * stop_multiplier)
            take_profit = price * (1 - base_profit * profit_multiplier)
        
        trade = {
            'id': len(self.trades) + 1,
            'entry_time': timestamp,
            'side': signal.direction,
            'entry_price': price,
            'quantity': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'ai_confidence': signal.confidence,
            'quality_score': signal.quality_score,
            'market_regime': signal.market_regime,
            'volatility_regime': signal.volatility_regime,
            'risk_score': signal.risk_score,
            'status': 'open'
        }
        
        self.positions[trade['id']] = trade
        
        logger.info(f"🎯 Advanced Trade: {signal.direction} {position_size:.6f} @ ${price:.2f} "
                   f"(Conf: {signal.confidence:.3f}, Qual: {signal.quality_score:.3f}, "
                   f"Regime: {signal.market_regime})")
        
        return trade
    
    def check_exit_conditions(self, trade: Dict, current_price: float, timestamp: datetime) -> tuple:
        """Check advanced exit conditions"""
        # Standard stop/profit checks
        if trade['side'] == 'long':
            if current_price <= trade['stop_loss']:
                return 'stop_loss', trade['stop_loss']
            elif current_price >= trade['take_profit']:
                return 'take_profit', trade['take_profit']
        else:
            if current_price >= trade['stop_loss']:
                return 'stop_loss', trade['stop_loss']
            elif current_price <= trade['take_profit']:
                return 'take_profit', trade['take_profit']
        
        # Time-based exit
        holding_time = (timestamp - trade['entry_time']).total_seconds() / 3600
        if holding_time > self.max_holding_time_hours:
            return 'time_exit', current_price
        
        return None, None
    
    def close_trade(self, trade_id: int, exit_reason: str, exit_price: float, exit_time: datetime):
        """Close trade and update statistics"""
        trade = self.positions[trade_id]
        
        # Calculate P&L
        if trade['side'] == 'long':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']
        
        pnl_pct = (pnl / (trade['entry_price'] * trade['quantity'])) * 100
        
        # Update balance
        self.current_balance += pnl
        
        # Track drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Record trade
        holding_time = (exit_time - trade['entry_time']).total_seconds() / 60
        
        trade_result = {
            **trade,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_time_minutes': holding_time,
            'status': 'closed'
        }
        
        self.trades.append(trade_result)
        del self.positions[trade_id]
        
        result_emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"{result_emoji} Closed {trade['side']}: PnL ${pnl:.2f} ({pnl_pct:.2f}%) - {exit_reason}")
    
    def run_backtest(self, symbol: str = 'ETH', days: int = 30) -> Dict:
        """Run comprehensive backtest"""
        logger.info(f"🚀 Starting Advanced AI Backtest for {symbol} ({days} days)")
        
        # Generate realistic data
        candles = self.generate_realistic_data(symbol, days)
        
        # Train AI system
        logger.info("🧠 Training Advanced AI System...")
        training_results = self.ai_system.train_advanced_models(candles[:int(len(candles)*0.7)])
        
        # Start backtesting from 70% point (use last 30% for testing)
        test_start = int(len(candles) * 0.7)
        test_candles = candles[test_start:]
        
        logger.info(f"Testing on {len(test_candles)} candles...")
        
        for i in range(200, len(test_candles)):  # Need enough history for features
            current_candle = test_candles[i]
            current_time = datetime.fromtimestamp(current_candle['timestamp'] / 1000)
            current_price = current_candle['close']
            
            # Check exit conditions for open positions
            positions_to_close = []
            for trade_id, trade in self.positions.items():
                exit_reason, exit_price = self.check_exit_conditions(trade, current_price, current_time)
                if exit_reason:
                    positions_to_close.append((trade_id, exit_reason, exit_price))
            
            # Close positions
            for trade_id, exit_reason, exit_price in positions_to_close:
                self.close_trade(trade_id, exit_reason, exit_price, current_time)
            
            # Generate new signals (only if no open positions for ultra-selective approach)
            if not self.positions:
                # Use all available history for signal generation
                signal_history = candles[:test_start + i + 1]
                signal = self.ai_system.generate_signal(signal_history)
                
                if self.ai_system.should_trade_advanced(signal):
                    self.execute_trade(signal, current_price, current_time)
        
        # Close any remaining positions
        final_time = datetime.fromtimestamp(test_candles[-1]['timestamp'] / 1000)
        final_price = test_candles[-1]['close']
        
        for trade_id in list(self.positions.keys()):
            self.close_trade(trade_id, 'backtest_end', final_price, final_time)
        
        return self.calculate_results()
    
    def calculate_results(self) -> Dict:
        """Calculate comprehensive results"""
        if not self.trades:
            return {'total_trades': 0, 'win_rate': 0.0, 'message': 'No trades executed'}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        total_return_pct = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Advanced metrics
        pnls = [t['pnl'] for t in self.trades]
        avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Performance by regime
        regime_performance = {}
        for regime in ['STRONG_BULL', 'BULL_TREND', 'BEAR_TREND', 'STRONG_BEAR', 'RANGE_BOUND', 'HIGH_VOLATILITY']:
            regime_trades = [t for t in self.trades if t['market_regime'] == regime]
            if regime_trades:
                regime_wins = len([t for t in regime_trades if t['pnl'] > 0])
                regime_wr = regime_wins / len(regime_trades)
                regime_performance[regime] = {
                    'trades': len(regime_trades),
                    'win_rate': regime_wr,
                    'avg_pnl': np.mean([t['pnl'] for t in regime_trades])
                }
        
        # Quality analysis
        quality_ranges = [(0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        quality_performance = {}
        
        for low, high in quality_ranges:
            quality_trades = [t for t in self.trades if low <= t['quality_score'] < high]
            if quality_trades:
                quality_wins = len([t for t in quality_trades if t['pnl'] > 0])
                quality_wr = quality_wins / len(quality_trades)
                quality_performance[f"{low:.1f}-{high:.1f}"] = {
                    'trades': len(quality_trades),
                    'win_rate': quality_wr,
                    'avg_pnl': np.mean([t['pnl'] for t in quality_trades])
                }
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': self.max_drawdown * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'regime_performance': regime_performance,
            'quality_performance': quality_performance,
            'avg_holding_time': np.mean([t['holding_time_minutes'] for t in self.trades]),
            'avg_confidence': np.mean([t['ai_confidence'] for t in self.trades]),
            'avg_quality': np.mean([t['quality_score'] for t in self.trades])
        }

def main():
    """Run comprehensive advanced AI testing"""
    print("🚀 ADVANCED AI 70%+ WIN RATE VALIDATION")
    print("=" * 70)
    
    test_configs = [
        {"name": "Advanced 14-Day", "days": 14, "balance": 50},
        {"name": "Advanced 30-Day", "days": 30, "balance": 50},
        {"name": "Advanced 60-Day", "days": 60, "balance": 100},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n🧠 Running {config['name']} Test...")
        print("-" * 50)
        
        try:
            backtester = AdvancedAIBacktester(config['balance'])
            results = backtester.run_backtest('ETH', config['days'])
            
            print(f"💰 RESULTS FOR {config['name']}:")
            print(f"   Initial Balance: ${results['initial_balance']:.2f}")
            print(f"   Final Balance: ${results['final_balance']:.2f}")
            print(f"   Total Return: {results['total_return_pct']:.2f}%")
            print(f"   Total Trades: {results['total_trades']}")
            print(f"   🎯 WIN RATE: {results['win_rate']:.1%}")
            print(f"   Max Drawdown: {results['max_drawdown_pct']:.2f}%")
            print(f"   Profit Factor: {results['profit_factor']:.2f}")
            print(f"   Avg Quality: {results['avg_quality']:.3f}")
            
            # Win rate status
            if results['win_rate'] >= 0.70:
                status = "🎯 TARGET ACHIEVED!"
                print(f"   Status: {status}")
            elif results['win_rate'] >= 0.60:
                status = "🔄 Close to Target"
                print(f"   Status: {status}")
            else:
                status = "❌ Below Target"
                print(f"   Status: {status}")
            
            # Quality analysis
            if results['quality_performance']:
                print(f"\n🎯 QUALITY ANALYSIS:")
                for quality_range, stats in results['quality_performance'].items():
                    print(f"   Quality {quality_range}: {stats['trades']} trades, {stats['win_rate']:.1%} WR")
            
            all_results.append({
                'config': config,
                'results': results,
                'status': status
            })
            
        except Exception as e:
            logger.error(f"Error in {config['name']}: {str(e)}")
            continue
    
    # Final analysis
    if all_results:
        print(f"\n🎯 FINAL ADVANCED AI ANALYSIS")
        print("=" * 70)
        
        best_wr = max(all_results, key=lambda x: x['results']['win_rate'])
        
        print(f"🏆 BEST WIN RATE: {best_wr['config']['name']}")
        print(f"   Win Rate: {best_wr['results']['win_rate']:.1%}")
        print(f"   Return: {best_wr['results']['total_return_pct']:.2f}%")
        print(f"   Quality: {best_wr['results']['avg_quality']:.3f}")
        
        # Overall statistics
        total_trades = sum(r['results']['total_trades'] for r in all_results)
        total_wins = sum(r['results']['winning_trades'] for r in all_results)
        overall_wr = total_wins / total_trades if total_trades > 0 else 0
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Overall Win Rate: {overall_wr:.1%}")
        print(f"   Tests Achieving 70%+: {len([r for r in all_results if r['results']['win_rate'] >= 0.70])}/{len(all_results)}")
        
        # Save results
        with open('advanced_ai_70wr_results.json', 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'target_win_rate': 0.70,
                'overall_win_rate': overall_wr,
                'results': all_results
            }, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to 'advanced_ai_70wr_results.json'")
        
        # Final verdict
        if overall_wr >= 0.70:
            print(f"\n🎯 SUCCESS! Advanced AI achieved 70%+ win rate target!")
            print(f"🚀 Ready for live trading with paper trading validation")
        elif overall_wr >= 0.60:
            print(f"\n🔄 PROMISING! Advanced AI shows strong potential")
            print(f"📈 Consider further optimization and larger sample size")
        else:
            print(f"\n⚠️  NEEDS WORK! Advanced AI requires further development")
            print(f"🔧 Consider ensemble refinement and feature engineering")

if __name__ == "__main__":
    main() 