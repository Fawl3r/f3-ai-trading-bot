#!/usr/bin/env python3
"""
Focused ETH 5m Backtest - Validate 94%+ Win Rates
$50 Starting Balance Test
"""

import asyncio
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_main_trading_bot import EnhancedMainTradingBot
from advanced_risk_management import AdvancedRiskManager, RiskMetrics
from advanced_top_bottom_detector import AdvancedTopBottomDetector
from enhanced_execution_layer import EnhancedExecutionLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    """Individual trade result"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    holding_time_minutes: float
    exit_reason: str
    market_regime: str

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_time: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    trades: List[TradeResult]

class FocusedETHBacktest:
    """Focused backtest for ETH 5m trading validation"""
    
    def __init__(self, initial_balance: float = 50.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades = []
        self.positions = {}
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
        # Initialize components
        self.risk_manager = AdvancedRiskManager()
        self.detector = AdvancedTopBottomDetector()
        # Note: execution_layer not needed for backtest simulation
        
        # Trading parameters optimized for ETH 5m
        self.max_position_size = 0.02  # 2% of balance per trade
        self.stop_loss_pct = 0.015     # 1.5% stop loss
        self.take_profit_pct = 0.025   # 2.5% take profit
        self.min_trade_size = 0.001    # Minimum ETH trade size
        
    def generate_realistic_eth_data(self, days: int = 7) -> List[Dict]:
        """Generate realistic ETH 5m candle data"""
        candles = []
        start_time = datetime.now() - timedelta(days=days)
        current_time = start_time
        
        # ETH realistic price range
        base_price = 3500.0
        current_price = base_price
        
        while current_time < datetime.now():
            # Generate realistic price movement
            volatility = np.random.normal(0, 0.008)  # 0.8% volatility
            price_change = current_price * volatility
            
            # OHLC generation
            open_price = current_price
            close_price = current_price + price_change
            
            # High/Low with realistic wicks
            high_wick = abs(np.random.normal(0, 0.003)) * current_price
            low_wick = abs(np.random.normal(0, 0.003)) * current_price
            
            high_price = max(open_price, close_price) + high_wick
            low_price = min(open_price, close_price) - low_wick
            
            # Volume (realistic for ETH)
            volume = np.random.lognormal(8, 1.5)  # Log-normal distribution
            
            candle = {
                'timestamp': int(current_time.timestamp() * 1000),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 4)
            }
            
            candles.append(candle)
            current_price = close_price
            current_time += timedelta(minutes=5)
        
        logger.info(f"Generated {len(candles)} ETH 5m candles")
        return candles
    
    def calculate_indicators(self, candles: List[Dict], period: int = 20) -> Dict:
        """Calculate technical indicators"""
        if len(candles) < period:
            return {}
        
        closes = [c['close'] for c in candles[-period:]]
        highs = [c['high'] for c in candles[-period:]]
        lows = [c['low'] for c in candles[-period:]]
        
        # Simple moving averages
        sma_20 = np.mean(closes)
        sma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else sma_20
        
        # RSI calculation
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 1
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # ATR
        tr_values = []
        for i in range(1, len(candles[-period:])):
            h = highs[i]
            l = lows[i]
            c_prev = closes[i-1]
            
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            tr_values.append(tr)
        
        atr = np.mean(tr_values) if tr_values else 0
        
        return {
            'sma_20': sma_20,
            'sma_10': sma_10,
            'rsi': rsi,
            'atr': atr,
            'current_price': closes[-1]
        }
    
    def generate_trading_signal(self, candles: List[Dict], indicators: Dict) -> Optional[str]:
        """Generate trading signals using the enhanced strategy"""
        if len(candles) < 50:
            return None
        
        try:
            # Get market structure analysis
            market_structure = self.detector.analyze_market_structure(candles[-50:])
            
            # Get swing points
            swing_points = self.detector.detect_swing_points(candles[-50:])
            
            # Risk metrics
            risk_metrics = self.calculate_risk_metrics(candles[-50:])
            
            current_price = indicators['current_price']
            sma_20 = indicators['sma_20']
            sma_10 = indicators['sma_10']
            rsi = indicators['rsi']
            atr = indicators['atr']
            
            # Enhanced signal logic
            bullish_signals = 0
            bearish_signals = 0
            
            # Trend following
            if sma_10 > sma_20:
                bullish_signals += 1
            elif sma_10 < sma_20:
                bearish_signals += 1
            
            # RSI momentum
            if 30 < rsi < 70:  # Avoid extreme overbought/oversold
                if rsi > 55:
                    bullish_signals += 1
                elif rsi < 45:
                    bearish_signals += 1
            
            # Market structure
            if market_structure.get('trend') == 'bullish':
                bullish_signals += 2
            elif market_structure.get('trend') == 'bearish':
                bearish_signals += 2
            
            # Swing point confirmation
            recent_highs = [sp.price for sp in swing_points.get('highs', [])[-3:]]
            recent_lows = [sp.price for sp in swing_points.get('lows', [])[-3:]]
            
            if recent_highs and current_price > max(recent_highs):
                bullish_signals += 1
            if recent_lows and current_price < min(recent_lows):
                bearish_signals += 1
            
            # Volatility filter
            if atr > 0:
                volatility_pct = (atr / current_price) * 100
                if volatility_pct > 2.0:  # Too volatile
                    return None
            
            # Signal generation
            if bullish_signals >= 3 and bullish_signals > bearish_signals:
                return 'long'
            elif bearish_signals >= 3 and bearish_signals > bullish_signals:
                return 'short'
            
            return None
            
        except Exception as e:
            logger.warning(f"Error generating signal: {e}")
            return None
    
    def calculate_risk_metrics(self, candles: List[Dict]) -> RiskMetrics:
        """Calculate risk metrics for position sizing"""
        try:
            if len(candles) < 20:
                return RiskMetrics(
                    atr_1m=0.0, atr_1h=0.0, volatility_1m=0.0,
                    rsi_14=50.0, vwap_distance=0.0, obi=0.0,
                    funding_rate=0.0, open_interest_change=0.0
                )
            
            # Calculate ATR
            atr = self.risk_manager.calculate_atr(candles[-20:], 14)
            
            # Calculate volatility
            closes = [c['close'] for c in candles[-20:]]
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * np.sqrt(288)  # 5m periods in a day
            
            return RiskMetrics(
                atr_1m=atr * 0.5,
                atr_1h=atr,
                volatility_1m=volatility,
                rsi_14=50.0,
                vwap_distance=0.0,
                obi=0.0,
                funding_rate=0.0,
                open_interest_change=0.0
            )
            
        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                atr_1m=0.0, atr_1h=0.0, volatility_1m=0.0,
                rsi_14=50.0, vwap_distance=0.0, obi=0.0,
                funding_rate=0.0, open_interest_change=0.0
            )
    
    def calculate_position_size(self, price: float, risk_metrics: RiskMetrics) -> float:
        """Calculate optimal position size"""
        # Risk-based position sizing
        account_risk = self.current_balance * self.max_position_size
        
        # ATR-based stop distance
        stop_distance = max(risk_metrics.atr_1h * 2, price * self.stop_loss_pct)
        
        # Position size based on risk
        if stop_distance > 0:
            position_value = account_risk / (stop_distance / price)
            position_size = position_value / price
        else:
            position_size = account_risk / price
        
        # Ensure minimum trade size
        position_size = max(position_size, self.min_trade_size)
        
        # Ensure we don't exceed balance
        max_affordable = (self.current_balance * 0.95) / price
        position_size = min(position_size, max_affordable)
        
        return round(position_size, 6)
    
    def execute_trade(self, signal: str, price: float, timestamp: datetime, 
                     position_size: float, market_regime: str) -> Optional[Dict]:
        """Execute a trade"""
        if signal not in ['long', 'short']:
            return None
        
        trade_value = position_size * price
        
        # Check if we have enough balance
        if trade_value > self.current_balance * 0.95:
            logger.warning(f"Insufficient balance for trade: {trade_value} > {self.current_balance}")
            return None
        
        # Calculate stop loss and take profit
        if signal == 'long':
            stop_loss = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)
        else:
            stop_loss = price * (1 + self.stop_loss_pct)
            take_profit = price * (1 - self.take_profit_pct)
        
        trade = {
            'id': len(self.trades) + 1,
            'entry_time': timestamp,
            'symbol': 'ETH',
            'side': signal,
            'entry_price': price,
            'quantity': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'market_regime': market_regime,
            'status': 'open'
        }
        
        self.positions[trade['id']] = trade
        logger.info(f"Opened {signal} position: {position_size:.6f} ETH @ ${price:.2f}")
        
        return trade
    
    def check_exit_conditions(self, trade: Dict, current_price: float, 
                            timestamp: datetime) -> Optional[Tuple[str, float]]:
        """Check if trade should be exited"""
        if trade['side'] == 'long':
            if current_price <= trade['stop_loss']:
                return 'stop_loss', trade['stop_loss']
            elif current_price >= trade['take_profit']:
                return 'take_profit', trade['take_profit']
        else:  # short
            if current_price >= trade['stop_loss']:
                return 'stop_loss', trade['stop_loss']
            elif current_price <= trade['take_profit']:
                return 'take_profit', trade['take_profit']
        
        # Time-based exit (max 4 hours for 5m strategy)
        holding_time = (timestamp - trade['entry_time']).total_seconds() / 3600
        if holding_time > 4:
            return 'time_exit', current_price
        
        return None
    
    def close_trade(self, trade_id: int, exit_reason: str, exit_price: float, 
                   exit_time: datetime):
        """Close a trade and record results"""
        trade = self.positions[trade_id]
        
        # Calculate PnL
        if trade['side'] == 'long':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']
        
        pnl_pct = (pnl / (trade['entry_price'] * trade['quantity'])) * 100
        
        # Update balance
        self.current_balance += pnl
        
        # Track peak balance and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Record trade result
        holding_time = (exit_time - trade['entry_time']).total_seconds() / 60
        
        trade_result = TradeResult(
            entry_time=trade['entry_time'],
            exit_time=exit_time,
            symbol=trade['symbol'],
            side=trade['side'],
            entry_price=trade['entry_price'],
            exit_price=exit_price,
            quantity=trade['quantity'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_time_minutes=holding_time,
            exit_reason=exit_reason,
            market_regime=trade['market_regime']
        )
        
        self.trades.append(trade_result)
        del self.positions[trade_id]
        
        logger.info(f"Closed {trade['side']} position: PnL ${pnl:.2f} ({pnl_pct:.2f}%) - {exit_reason}")
    
    def run_backtest(self, days: int = 7) -> BacktestResults:
        """Run the focused ETH backtest"""
        logger.info(f"Starting focused ETH 5m backtest with ${self.initial_balance} balance")
        
        # Generate realistic data
        candles = self.generate_realistic_eth_data(days)
        
        # Market regime simulation
        market_regimes = ['BULL_TREND', 'BEAR_TREND', 'RANGE_BOUND', 'HIGH_VOLATILITY']
        current_regime_idx = 0
        regime_change_interval = len(candles) // 4
        
        # Process each candle
        for i, candle in enumerate(candles[50:], 50):  # Start after enough data
            current_time = datetime.fromtimestamp(candle['timestamp'] / 1000)
            current_price = candle['close']
            
            # Update market regime
            current_regime = market_regimes[min(current_regime_idx, len(market_regimes) - 1)]
            if i % regime_change_interval == 0 and current_regime_idx < len(market_regimes) - 1:
                current_regime_idx += 1
            
            # Check exit conditions for open positions
            positions_to_close = []
            for trade_id, trade in self.positions.items():
                exit_condition = self.check_exit_conditions(trade, current_price, current_time)
                if exit_condition:
                    exit_reason, exit_price = exit_condition
                    positions_to_close.append((trade_id, exit_reason, exit_price))
            
            # Close positions
            for trade_id, exit_reason, exit_price in positions_to_close:
                self.close_trade(trade_id, exit_reason, exit_price, current_time)
            
            # Generate new signals if no open positions
            if not self.positions:
                indicators = self.calculate_indicators(candles[:i+1])
                if indicators:
                    signal = self.generate_trading_signal(candles[:i+1], indicators)
                    
                    if signal:
                        risk_metrics = self.calculate_risk_metrics(candles[:i+1])
                        position_size = self.calculate_position_size(current_price, risk_metrics)
                        
                        if position_size > 0:
                            self.execute_trade(signal, current_price, current_time, 
                                             position_size, current_regime)
        
        # Close any remaining positions
        final_time = datetime.fromtimestamp(candles[-1]['timestamp'] / 1000)
        final_price = candles[-1]['close']
        
        for trade_id in list(self.positions.keys()):
            self.close_trade(trade_id, 'backtest_end', final_price, final_time)
        
        return self.calculate_results()
    
    def calculate_results(self) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        if not self.trades:
            return BacktestResults(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, total_pnl=0.0, total_return_pct=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, max_drawdown_pct=0.0,
                profit_factor=0.0, avg_win=0.0, avg_loss=0.0,
                avg_holding_time=0.0, largest_win=0.0, largest_loss=0.0,
                consecutive_wins=0, consecutive_losses=0, trades=[]
            )
        
        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        total_pnl = sum(t.pnl for t in self.trades)
        total_return_pct = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Win/Loss statistics
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = [t.pnl_pct for t in self.trades]
        if returns and len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                consecutive_wins = max(consecutive_wins, current_streak)
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                consecutive_losses = max(consecutive_losses, abs(current_streak))
        
        # Average holding time
        avg_holding_time = np.mean([t.holding_time_minutes for t in self.trades])
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=self.max_drawdown * 100,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_holding_time=avg_holding_time,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            trades=self.trades
        )

def main():
    """Run focused ETH backtest"""
    print("🚀 FOCUSED ETH 5M BACKTEST - $50 ACCOUNT")
    print("=" * 50)
    
    # Test different scenarios
    scenarios = [
        {"balance": 50, "days": 7, "name": "1 Week Test"},
        {"balance": 50, "days": 14, "name": "2 Week Test"},
        {"balance": 50, "days": 30, "name": "1 Month Test"}
    ]
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n📊 Running {scenario['name']} (${scenario['balance']} balance, {scenario['days']} days)")
        print("-" * 40)
        
        backtest = FocusedETHBacktest(scenario['balance'])
        results = backtest.run_backtest(scenario['days'])
        
        # Display results
        print(f"💰 RESULTS FOR {scenario['name']}:")
        print(f"   Initial Balance: ${scenario['balance']:.2f}")
        print(f"   Final Balance: ${backtest.current_balance:.2f}")
        print(f"   Total Return: {results.total_return_pct:.2f}%")
        print(f"   Total PnL: ${results.total_pnl:.2f}")
        print(f"   Total Trades: {results.total_trades}")
        print(f"   Win Rate: {results.win_rate:.1%}")
        print(f"   Winning Trades: {results.winning_trades}")
        print(f"   Losing Trades: {results.losing_trades}")
        print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"   Profit Factor: {results.profit_factor:.2f}")
        print(f"   Max Drawdown: {results.max_drawdown_pct:.2f}%")
        print(f"   Avg Win: ${results.avg_win:.2f}")
        print(f"   Avg Loss: ${results.avg_loss:.2f}")
        print(f"   Largest Win: ${results.largest_win:.2f}")
        print(f"   Largest Loss: ${results.largest_loss:.2f}")
        print(f"   Avg Holding Time: {results.avg_holding_time:.1f} minutes")
        print(f"   Consecutive Wins: {results.consecutive_wins}")
        print(f"   Consecutive Losses: {results.consecutive_losses}")
        
        # Trade breakdown by market regime
        regime_stats = {}
        for trade in results.trades:
            regime = trade.market_regime
            if regime not in regime_stats:
                regime_stats[regime] = {'wins': 0, 'losses': 0, 'pnl': 0}
            
            if trade.pnl > 0:
                regime_stats[regime]['wins'] += 1
            else:
                regime_stats[regime]['losses'] += 1
            regime_stats[regime]['pnl'] += trade.pnl
        
        print(f"\n📈 PERFORMANCE BY MARKET REGIME:")
        for regime, stats in regime_stats.items():
            total_regime_trades = stats['wins'] + stats['losses']
            regime_win_rate = stats['wins'] / total_regime_trades if total_regime_trades > 0 else 0
            print(f"   {regime}: {stats['wins']}W/{stats['losses']}L ({regime_win_rate:.1%}) PnL: ${stats['pnl']:.2f}")
        
        all_results.append({
            'scenario': scenario['name'],
            'results': results,
            'final_balance': backtest.current_balance
        })
    
    # Summary
    print(f"\n🎯 SUMMARY - ETH 5M FOCUSED BACKTEST")
    print("=" * 50)
    
    for result in all_results:
        print(f"{result['scenario']}: ${result['final_balance']:.2f} ({result['results'].total_return_pct:.2f}%) - {result['results'].win_rate:.1%} win rate")
    
    # Save detailed results
    with open('focused_eth_backtest_results.json', 'w') as f:
        json.dump({
            'summary': {
                'test_date': datetime.now().isoformat(),
                'scenarios_tested': len(scenarios),
                'results': [{
                    'scenario': r['scenario'],
                    'final_balance': r['final_balance'],
                    'return_pct': r['results'].total_return_pct,
                    'win_rate': r['results'].win_rate,
                    'total_trades': r['results'].total_trades,
                    'sharpe_ratio': r['results'].sharpe_ratio,
                    'max_drawdown': r['results'].max_drawdown_pct
                } for r in all_results]
            }
        }, f, indent=2)
    
    print(f"\n✅ Detailed results saved to 'focused_eth_backtest_results.json'")

if __name__ == "__main__":
    main() 