#!/usr/bin/env python3
"""
Multi-Asset 5m Backtest - ETH, AVAX, DOGE
$50 Starting Balance - Validation Test
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
class AssetConfig:
    """Configuration for each trading asset"""
    symbol: str
    base_price: float
    volatility: float
    min_trade_size: float
    stop_loss_pct: float
    take_profit_pct: float
    max_position_pct: float

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

class MultiAssetBacktest:
    """Multi-asset backtest for ETH, AVAX, DOGE validation"""
    
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
        
        # Asset configurations
        self.assets = {
            'ETH': AssetConfig(
                symbol='ETH',
                base_price=3500.0,
                volatility=0.008,  # 0.8% per 5m
                min_trade_size=0.001,
                stop_loss_pct=0.015,  # 1.5%
                take_profit_pct=0.025,  # 2.5%
                max_position_pct=0.4  # 40% of balance
            ),
            'AVAX': AssetConfig(
                symbol='AVAX',
                base_price=45.0,
                volatility=0.012,  # 1.2% per 5m
                min_trade_size=0.01,
                stop_loss_pct=0.02,  # 2%
                take_profit_pct=0.03,  # 3%
                max_position_pct=0.3  # 30% of balance
            ),
            'DOGE': AssetConfig(
                symbol='DOGE',
                base_price=0.35,
                volatility=0.015,  # 1.5% per 5m
                min_trade_size=1.0,
                stop_loss_pct=0.025,  # 2.5%
                take_profit_pct=0.035,  # 3.5%
                max_position_pct=0.3  # 30% of balance
            )
        }
        
        # Portfolio allocation
        self.max_simultaneous_positions = 2
        self.portfolio_heat = 0.8  # Max 80% of balance at risk
        
    def generate_realistic_data(self, symbol: str, days: int = 7) -> List[Dict]:
        """Generate realistic candle data for each asset"""
        config = self.assets[symbol]
        candles = []
        start_time = datetime.now() - timedelta(days=days)
        current_time = start_time
        
        current_price = config.base_price
        
        while current_time < datetime.now():
            # Generate realistic price movement
            volatility = np.random.normal(0, config.volatility)
            price_change = current_price * volatility
            
            # OHLC generation
            open_price = current_price
            close_price = current_price + price_change
            
            # High/Low with realistic wicks
            high_wick = abs(np.random.normal(0, config.volatility * 0.5)) * current_price
            low_wick = abs(np.random.normal(0, config.volatility * 0.5)) * current_price
            
            high_price = max(open_price, close_price) + high_wick
            low_price = min(open_price, close_price) - low_wick
            
            # Volume (asset-specific)
            if symbol == 'ETH':
                volume = np.random.lognormal(8, 1.5)
            elif symbol == 'AVAX':
                volume = np.random.lognormal(6, 1.2)
            else:  # DOGE
                volume = np.random.lognormal(10, 2.0)
            
            candle = {
                'timestamp': int(current_time.timestamp() * 1000),
                'open': round(open_price, 8),
                'high': round(high_price, 8),
                'low': round(low_price, 8),
                'close': round(close_price, 8),
                'volume': round(volume, 4)
            }
            
            candles.append(candle)
            current_price = close_price
            current_time += timedelta(minutes=5)
        
        logger.info(f"Generated {len(candles)} {symbol} 5m candles")
        return candles
    
    def calculate_indicators(self, candles: List[Dict], period: int = 20) -> Dict:
        """Calculate technical indicators"""
        if len(candles) < period:
            return {}
        
        closes = [c['close'] for c in candles[-period:]]
        highs = [c['high'] for c in candles[-period:]]
        lows = [c['low'] for c in candles[-period:]]
        volumes = [c['volume'] for c in candles[-period:]]
        
        # Simple moving averages
        sma_20 = np.mean(closes)
        sma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else sma_20
        sma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else sma_20
        
        # EMA calculation
        ema_12 = closes[-1]  # Start with current price
        ema_26 = closes[-1]
        
        if len(closes) >= 12:
            multiplier_12 = 2 / (12 + 1)
            for price in closes[-12:]:
                ema_12 = (price * multiplier_12) + (ema_12 * (1 - multiplier_12))
        
        if len(closes) >= 26:
            multiplier_26 = 2 / (26 + 1)
            for price in closes[-26:]:
                ema_26 = (price * multiplier_26) + (ema_26 * (1 - multiplier_26))
        
        # MACD
        macd_line = ema_12 - ema_26
        
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
        
        # Volume analysis
        avg_volume = np.mean(volumes)
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
        
        # Bollinger Bands
        bb_std = np.std(closes)
        bb_upper = sma_20 + (bb_std * 2)
        bb_lower = sma_20 - (bb_std * 2)
        bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        return {
            'sma_20': sma_20,
            'sma_10': sma_10,
            'sma_5': sma_5,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'macd': macd_line,
            'rsi': rsi,
            'atr': atr,
            'current_price': closes[-1],
            'volume_ratio': volume_ratio,
            'bb_position': bb_position
        }
    
    def generate_trading_signal(self, symbol: str, candles: List[Dict], 
                              indicators: Dict) -> Optional[str]:
        """Generate trading signals using enhanced multi-factor analysis"""
        if len(candles) < 50:
            return None
        
        try:
            # Get market structure analysis
            market_structure = self.detector.analyze_market_structure(candles[-50:])
            
            # Get swing points
            swing_points = self.detector.detect_swing_points(candles[-50:])
            
            current_price = indicators['current_price']
            
            # Multi-factor signal scoring
            bullish_score = 0
            bearish_score = 0
            
            # Trend following signals
            if indicators['sma_5'] > indicators['sma_10'] > indicators['sma_20']:
                bullish_score += 2
            elif indicators['sma_5'] < indicators['sma_10'] < indicators['sma_20']:
                bearish_score += 2
            
            # MACD momentum
            if indicators['macd'] > 0:
                bullish_score += 1
            else:
                bearish_score += 1
            
            # RSI momentum (avoid extremes)
            if 45 < indicators['rsi'] < 65:
                bullish_score += 1
            elif 35 < indicators['rsi'] < 55:
                bearish_score += 1
            elif indicators['rsi'] > 80 or indicators['rsi'] < 20:
                return None  # Avoid extreme conditions
            
            # Market structure
            structure_trend = market_structure.get('trend', 'neutral')
            if structure_trend == 'bullish':
                bullish_score += 3
            elif structure_trend == 'bearish':
                bearish_score += 3
            
            # Volume confirmation
            if indicators['volume_ratio'] > 1.2:  # Above average volume
                if bullish_score > bearish_score:
                    bullish_score += 1
                elif bearish_score > bullish_score:
                    bearish_score += 1
            
            # Bollinger Bands position
            if indicators['bb_position'] < 0.2:  # Near lower band
                bullish_score += 1
            elif indicators['bb_position'] > 0.8:  # Near upper band
                bearish_score += 1
            
            # Swing point confirmation
            recent_highs = [sp.price for sp in swing_points.get('highs', [])[-3:]]
            recent_lows = [sp.price for sp in swing_points.get('lows', [])[-3:]]
            
            if recent_highs and current_price > max(recent_highs):
                bullish_score += 1
            if recent_lows and current_price < min(recent_lows):
                bearish_score += 1
            
            # Asset-specific adjustments
            config = self.assets[symbol]
            min_score = 4 if symbol == 'ETH' else 3  # Higher threshold for ETH
            
            # Signal generation
            if bullish_score >= min_score and bullish_score > bearish_score + 1:
                return 'long'
            elif bearish_score >= min_score and bearish_score > bullish_score + 1:
                return 'short'
            
            return None
            
        except Exception as e:
            logger.warning(f"Error generating signal for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, price: float, 
                              risk_metrics: RiskMetrics) -> float:
        """Calculate optimal position size for each asset"""
        config = self.assets[symbol]
        
        # Available capital for this position
        available_capital = self.current_balance * config.max_position_pct
        
        # Risk-based position sizing
        account_risk = available_capital * 0.02  # 2% risk per trade
        
        # ATR-based stop distance
        stop_distance = max(risk_metrics.atr_1h * 1.5, price * config.stop_loss_pct)
        
        # Position size based on risk
        if stop_distance > 0:
            position_value = account_risk / (stop_distance / price)
            position_size = position_value / price
        else:
            position_size = available_capital / price
        
        # Ensure minimum trade size
        position_size = max(position_size, config.min_trade_size)
        
        # Ensure we don't exceed available capital
        max_affordable = available_capital / price
        position_size = min(position_size, max_affordable)
        
        return round(position_size, 8)
    
    def can_open_position(self, symbol: str, position_value: float) -> bool:
        """Check if we can open a new position"""
        # Check maximum simultaneous positions
        if len(self.positions) >= self.max_simultaneous_positions:
            return False
        
        # Check portfolio heat
        current_exposure = sum(pos['entry_price'] * pos['quantity'] 
                             for pos in self.positions.values())
        total_exposure = current_exposure + position_value
        
        if total_exposure > self.current_balance * self.portfolio_heat:
            return False
        
        return True
    
    def execute_trade(self, symbol: str, signal: str, price: float, 
                     timestamp: datetime, position_size: float, 
                     market_regime: str) -> Optional[Dict]:
        """Execute a trade"""
        if signal not in ['long', 'short']:
            return None
        
        trade_value = position_size * price
        config = self.assets[symbol]
        
        # Check if we can open this position
        if not self.can_open_position(symbol, trade_value):
            return None
        
        # Calculate stop loss and take profit
        if signal == 'long':
            stop_loss = price * (1 - config.stop_loss_pct)
            take_profit = price * (1 + config.take_profit_pct)
        else:
            stop_loss = price * (1 + config.stop_loss_pct)
            take_profit = price * (1 - config.take_profit_pct)
        
        trade = {
            'id': len(self.trades) + 1,
            'entry_time': timestamp,
            'symbol': symbol,
            'side': signal,
            'entry_price': price,
            'quantity': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'market_regime': market_regime,
            'status': 'open'
        }
        
        self.positions[trade['id']] = trade
        logger.info(f"Opened {signal} {symbol}: {position_size:.6f} @ ${price:.6f}")
        
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
        
        # Time-based exit (max 6 hours for 5m strategy)
        holding_time = (timestamp - trade['entry_time']).total_seconds() / 3600
        if holding_time > 6:
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
        
        logger.info(f"Closed {trade['side']} {trade['symbol']}: PnL ${pnl:.2f} ({pnl_pct:.2f}%) - {exit_reason}")
    
    def run_backtest(self, days: int = 14) -> Dict:
        """Run the multi-asset backtest"""
        logger.info(f"Starting multi-asset backtest with ${self.initial_balance} balance")
        
        # Generate data for all assets
        asset_data = {}
        for symbol in self.assets.keys():
            asset_data[symbol] = self.generate_realistic_data(symbol, days)
        
        # Ensure all assets have the same number of candles
        min_candles = min(len(data) for data in asset_data.values())
        for symbol in asset_data:
            asset_data[symbol] = asset_data[symbol][:min_candles]
        
        # Market regime simulation
        market_regimes = ['BULL_TREND', 'BEAR_TREND', 'RANGE_BOUND', 'HIGH_VOLATILITY']
        regime_change_interval = min_candles // 4
        
        # Process each time period
        for i in range(50, min_candles):  # Start after enough data
            current_regime_idx = min(i // regime_change_interval, len(market_regimes) - 1)
            current_regime = market_regimes[current_regime_idx]
            
            # Process each asset
            for symbol in self.assets.keys():
                candles = asset_data[symbol]
                current_candle = candles[i]
                current_time = datetime.fromtimestamp(current_candle['timestamp'] / 1000)
                current_price = current_candle['close']
                
                # Check exit conditions for open positions of this asset
                positions_to_close = []
                for trade_id, trade in self.positions.items():
                    if trade['symbol'] == symbol:
                        exit_condition = self.check_exit_conditions(trade, current_price, current_time)
                        if exit_condition:
                            exit_reason, exit_price = exit_condition
                            positions_to_close.append((trade_id, exit_reason, exit_price))
                
                # Close positions
                for trade_id, exit_reason, exit_price in positions_to_close:
                    self.close_trade(trade_id, exit_reason, exit_price, current_time)
                
                # Generate new signals if no open position for this asset
                has_open_position = any(pos['symbol'] == symbol for pos in self.positions.values())
                
                if not has_open_position:
                    indicators = self.calculate_indicators(candles[:i+1])
                    if indicators:
                        signal = self.generate_trading_signal(symbol, candles[:i+1], indicators)
                        
                        if signal:
                            # Calculate risk metrics
                            risk_metrics = RiskMetrics(
                                atr_1m=indicators['atr'] * 0.5,
                                atr_1h=indicators['atr'],
                                volatility_1m=indicators['atr'] / current_price,
                                rsi_14=indicators['rsi'],
                                vwap_distance=0.0,
                                obi=0.0,
                                funding_rate=0.0,
                                open_interest_change=0.0
                            )
                            
                            position_size = self.calculate_position_size(symbol, current_price, risk_metrics)
                            
                            if position_size > 0:
                                self.execute_trade(symbol, signal, current_price, current_time, 
                                                 position_size, current_regime)
        
        # Close any remaining positions
        final_time = datetime.now()
        for symbol in self.assets.keys():
            final_price = asset_data[symbol][-1]['close']
            for trade_id in list(self.positions.keys()):
                if self.positions[trade_id]['symbol'] == symbol:
                    self.close_trade(trade_id, 'backtest_end', final_price, final_time)
        
        return self.calculate_results()
    
    def calculate_results(self) -> Dict:
        """Calculate comprehensive results by asset and overall"""
        results = {
            'overall': self.calculate_overall_results(),
            'by_asset': {}
        }
        
        # Calculate results by asset
        for symbol in self.assets.keys():
            asset_trades = [t for t in self.trades if t.symbol == symbol]
            results['by_asset'][symbol] = self.calculate_asset_results(asset_trades)
        
        return results
    
    def calculate_overall_results(self) -> Dict:
        """Calculate overall portfolio results"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return_pct': 0.0,
                'final_balance': self.current_balance,
                'max_drawdown_pct': 0.0
            }
        
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_return_pct = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'total_pnl': sum(t.pnl for t in self.trades),
            'max_drawdown_pct': self.max_drawdown * 100,
            'avg_holding_time': np.mean([t.holding_time_minutes for t in self.trades])
        }
    
    def calculate_asset_results(self, asset_trades: List[TradeResult]) -> Dict:
        """Calculate results for a specific asset"""
        if not asset_trades:
            return {
                'trades': 0,
                'win_rate': 0.0,
                'pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        winning_trades = [t for t in asset_trades if t.pnl > 0]
        losing_trades = [t for t in asset_trades if t.pnl < 0]
        
        return {
            'trades': len(asset_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(asset_trades),
            'total_pnl': sum(t.pnl for t in asset_trades),
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            'largest_win': max([t.pnl for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t.pnl for t in losing_trades]) if losing_trades else 0,
            'avg_holding_time': np.mean([t.holding_time_minutes for t in asset_trades])
        }

def main():
    """Run multi-asset backtest"""
    print("🚀 MULTI-ASSET 5M BACKTEST - ETH, AVAX, DOGE")
    print("💰 Starting Balance: $50")
    print("=" * 60)
    
    # Run different test periods
    test_periods = [7, 14, 30]
    all_results = []
    
    for days in test_periods:
        print(f"\n📊 Running {days}-day backtest...")
        print("-" * 40)
        
        backtest = MultiAssetBacktest(50.0)
        results = backtest.run_backtest(days)
        
        overall = results['overall']
        by_asset = results['by_asset']
        
        print(f"🎯 OVERALL RESULTS ({days} days):")
        print(f"   Initial Balance: ${overall['initial_balance']:.2f}")
        print(f"   Final Balance: ${overall['final_balance']:.2f}")
        print(f"   Total Return: {overall['total_return_pct']:.2f}%")
        print(f"   Total PnL: ${overall['total_pnl']:.2f}")
        print(f"   Total Trades: {overall['total_trades']}")
        print(f"   Win Rate: {overall['win_rate']:.1%}")
        print(f"   Max Drawdown: {overall['max_drawdown_pct']:.2f}%")
        
        print(f"\n📈 RESULTS BY ASSET:")
        for symbol, asset_results in by_asset.items():
            if asset_results['trades'] > 0:
                print(f"   {symbol}:")
                print(f"     Trades: {asset_results['trades']} ({asset_results['win_rate']:.1%} win rate)")
                print(f"     PnL: ${asset_results['total_pnl']:.2f}")
                print(f"     Avg Win: ${asset_results['avg_win']:.2f}")
                print(f"     Avg Loss: ${asset_results['avg_loss']:.2f}")
                print(f"     Best Trade: ${asset_results['largest_win']:.2f}")
                print(f"     Worst Trade: ${asset_results['largest_loss']:.2f}")
            else:
                print(f"   {symbol}: No trades")
        
        all_results.append({
            'days': days,
            'results': results
        })
    
    # Summary comparison
    print(f"\n🎯 SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Period':<10} {'Final Balance':<15} {'Return %':<10} {'Win Rate':<10} {'Trades':<8}")
    print("-" * 60)
    
    for result in all_results:
        overall = result['results']['overall']
        print(f"{result['days']} days{'':<4} ${overall['final_balance']:<14.2f} {overall['total_return_pct']:<9.2f}% {overall['win_rate']:<9.1%} {overall['total_trades']:<8}")
    
    # Save results
    with open('multi_asset_50_backtest_results.json', 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'initial_balance': 50.0,
            'assets_tested': ['ETH', 'AVAX', 'DOGE'],
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Detailed results saved to 'multi_asset_50_backtest_results.json'")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    best_result = max(all_results, key=lambda x: x['results']['overall']['total_return_pct'])
    print(f"   Best performing period: {best_result['days']} days")
    print(f"   Best return: {best_result['results']['overall']['total_return_pct']:.2f}%")
    
    # Asset performance analysis
    asset_performance = {}
    for result in all_results:
        for symbol, asset_data in result['results']['by_asset'].items():
            if symbol not in asset_performance:
                asset_performance[symbol] = []
            asset_performance[symbol].append(asset_data.get('total_pnl', 0))
    
    print(f"\n📊 ASSET PERFORMANCE SUMMARY:")
    for symbol, pnls in asset_performance.items():
        avg_pnl = np.mean(pnls)
        total_pnl = sum(pnls)
        print(f"   {symbol}: Avg PnL ${avg_pnl:.2f}, Total PnL ${total_pnl:.2f}")

if __name__ == "__main__":
    main() 