#!/usr/bin/env python3
"""
Simple Parabolic Backtest System
Comprehensive backtesting with parabolic detection - no heavy dependencies
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import our simplified modules
from features.parabola_detector_simple import SimpleParabolaDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    position_size: float
    side: str  # 'long' or 'short'
    trade_type: str  # 'burst', 'fade', 'normal'
    signal_confidence: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    r_multiple: Optional[float] = None
    bars_held: Optional[int] = None
    exit_reason: Optional[str] = None

class SimpleRiskManager:
    """Simplified risk manager"""
    
    def __init__(self, config: Dict = None):
        default_config = {
            'base_risk_percent': 0.0075,
            'max_risk_percent': 0.015,
            'max_daily_loss': 0.02,
            'max_drawdown': 0.05,
            'consecutive_loss_limit': 5,
        }
        self.config = {**default_config, **(config or {})}
        self.state = {
            'consecutive_losses': 0,
            'daily_loss': 0.0,
            'max_drawdown': 0.0,
            'mode': 'normal'
        }
        self.trade_history = []
    
    def calculate_volatility_weighted_risk(self, current_atr: float, reference_atr: float = None) -> float:
        """Calculate volatility-weighted risk"""
        if reference_atr is None:
            reference_atr = current_atr
        
        volatility_ratio = current_atr / reference_atr if reference_atr > 0 else 1.0
        risk_percent = self.config['base_risk_percent'] * volatility_ratio
        return min(risk_percent, self.config['max_risk_percent'])
    
    def calculate_position_size(self, balance: float, entry_price: float, stop_loss: float, risk_percent: float) -> float:
        """Calculate position size"""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0
        
        risk_amount = balance * risk_percent
        position_size = risk_amount / risk_per_share
        
        # Apply consecutive loss penalty
        if self.state['consecutive_losses'] > 2:
            penalty = 0.8 ** (self.state['consecutive_losses'] - 2)
            position_size *= penalty
        
        return position_size
    
    def validate_trade(self, trade_type: str, balance: float) -> Tuple[bool, str]:
        """Validate if trade should be taken"""
        if self.state['daily_loss'] >= self.config['max_daily_loss']:
            return False, "Daily loss limit exceeded"
        
        if self.state['max_drawdown'] >= self.config['max_drawdown']:
            return False, "Maximum drawdown exceeded"
        
        return True, "Trade validated"
    
    def record_trade(self, trade_result: Dict):
        """Record trade result"""
        self.trade_history.append(trade_result)
        
        pnl = trade_result.get('pnl', 0)
        if pnl < 0:
            self.state['consecutive_losses'] += 1
        else:
            self.state['consecutive_losses'] = 0

class SimpleParabolicBacktest:
    """Simple parabolic backtesting system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.parabola_detector = SimpleParabolaDetector()
        self.risk_manager = SimpleRiskManager(self.config.get('risk_config', {}))
        self.trades = []
        self.equity_curve = []
        
    def _default_config(self) -> Dict:
        """Default backtest configuration"""
        return {
            'initial_balance': 50.0,
            'commission': 0.001,  # 0.1% commission
            'slippage': 0.0005,   # 0.05% slippage
            'risk_per_trade': 0.0075,  # 0.75% risk per trade
            'risk_reward_ratio': 4.0,  # 4:1 R:R
            'max_positions': 2,
            'time_filter_enabled': True,
            'pyramid_enabled': True,
            'adaptive_risk': True,
            'risk_config': {
                'base_risk_percent': 0.0075,
                'max_risk_percent': 0.015,
                'max_daily_loss': 0.02,
                'max_drawdown': 0.05,
            }
        }
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with parabolic features"""
        logger.info("Preparing data with parabolic features...")
        df = self.parabola_detector.process_complete_analysis(df)
        return df
    
    def generate_signals(self, df: pd.DataFrame, idx: int) -> List[Dict]:
        """Generate trading signals for current bar"""
        
        if idx < 200:  # Need enough data for indicators
            return []
        
        signals = []
        current_bar = df.iloc[idx]
        
        # Burst signals
        if current_bar.get('burst_long_filtered', False):
            signals.append({
                'type': 'burst_long',
                'confidence': current_bar.get('burst_strength', 0.5),
                'entry_price': current_bar['close'],
                'atr': current_bar['atr_14']
            })
        
        if current_bar.get('burst_short_filtered', False):
            signals.append({
                'type': 'burst_short',
                'confidence': current_bar.get('burst_strength', 0.5),
                'entry_price': current_bar['close'],
                'atr': current_bar['atr_14']
            })
        
        # Fade signals
        if current_bar.get('fade_long_filtered', False):
            signals.append({
                'type': 'fade_long',
                'confidence': current_bar.get('fade_strength', 0.5),
                'entry_price': current_bar['close'],
                'atr': current_bar['atr_14']
            })
        
        if current_bar.get('fade_short_filtered', False):
            signals.append({
                'type': 'fade_short',
                'confidence': current_bar.get('fade_strength', 0.5),
                'entry_price': current_bar['close'],
                'atr': current_bar['atr_14']
            })
        
        return signals
    
    def calculate_position_parameters(self, signal: Dict, balance: float, reference_atr: float = None) -> Tuple[float, float, float]:
        """Calculate position size, stop loss, and take profit"""
        
        entry_price = signal['entry_price']
        atr = signal['atr']
        signal_type = signal['type']
        
        # Calculate stop loss and take profit
        if 'long' in signal_type:
            stop_loss = entry_price - atr  # 1 ATR stop
            take_profit = entry_price + (atr * self.config['risk_reward_ratio'])  # 4 ATR target
        else:
            stop_loss = entry_price + atr  # 1 ATR stop
            take_profit = entry_price - (atr * self.config['risk_reward_ratio'])  # 4 ATR target
        
        # Calculate risk percentage
        if self.config['adaptive_risk']:
            risk_percent = self.risk_manager.calculate_volatility_weighted_risk(atr, reference_atr)
        else:
            risk_percent = self.config['risk_per_trade']
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            balance, entry_price, stop_loss, risk_percent
        )
        
        return position_size, stop_loss, take_profit
    
    def execute_trade(self, signal: Dict, balance: float, current_time: datetime, reference_atr: float = None) -> Optional[Trade]:
        """Execute a trade based on signal"""
        
        # Validate trade
        is_valid, reason = self.risk_manager.validate_trade(signal['type'], balance)
        
        if not is_valid:
            logger.debug(f"Trade rejected: {reason}")
            return None
        
        # Check position limits
        open_positions = len([t for t in self.trades if t.exit_time is None])
        if open_positions >= self.config['max_positions']:
            return None
        
        # Calculate position parameters
        position_size, stop_loss, take_profit = self.calculate_position_parameters(signal, balance, reference_atr)
        
        if position_size <= 0:
            return None
        
        # Create trade
        trade = Trade(
            entry_time=current_time,
            exit_time=None,
            entry_price=signal['entry_price'],
            exit_price=None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            side='long' if 'long' in signal['type'] else 'short',
            trade_type=signal['type'].split('_')[0],
            signal_confidence=signal['confidence']
        )
        
        return trade
    
    def update_trades(self, df: pd.DataFrame, idx: int) -> List[Trade]:
        """Update open trades and check for exits"""
        
        current_bar = df.iloc[idx]
        current_price = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']
        current_time = current_bar.get('datetime', datetime.now())
        
        closed_trades = []
        
        for trade in self.trades:
            if trade.exit_time is not None:
                continue
            
            # Check for exit conditions
            exit_price = None
            exit_reason = None
            
            if trade.side == 'long':
                # Check stop loss
                if current_low <= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = 'stop_loss'
                # Check take profit
                elif current_high >= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = 'take_profit'
            else:  # short
                # Check stop loss
                if current_high >= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = 'stop_loss'
                # Check take profit
                elif current_low <= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = 'take_profit'
            
            # Execute exit if triggered
            if exit_price is not None:
                trade.exit_time = current_time
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                
                # Calculate PnL
                if trade.side == 'long':
                    trade.pnl = (exit_price - trade.entry_price) * trade.position_size
                else:
                    trade.pnl = (trade.entry_price - exit_price) * trade.position_size
                
                # Apply commission and slippage
                commission = (trade.entry_price + exit_price) * trade.position_size * self.config['commission']
                slippage = (trade.entry_price + exit_price) * trade.position_size * self.config['slippage']
                trade.pnl -= (commission + slippage)
                
                # Calculate metrics
                trade.pnl_pct = trade.pnl / (trade.entry_price * trade.position_size)
                
                # Calculate R multiple
                risk_per_share = abs(trade.entry_price - trade.stop_loss)
                trade.r_multiple = trade.pnl / (risk_per_share * trade.position_size)
                
                # Calculate bars held
                entry_idx = df[df['datetime'] <= trade.entry_time].index[-1] if 'datetime' in df.columns else idx - 1
                trade.bars_held = idx - entry_idx
                
                # Record with risk manager
                self.risk_manager.record_trade({
                    'pnl': trade.pnl,
                    'trade_type': trade.trade_type,
                    'timestamp': trade.exit_time,
                    'r_multiple': trade.r_multiple
                })
                
                closed_trades.append(trade)
        
        return closed_trades
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run complete backtest"""
        
        logger.info("Starting simple parabolic backtest...")
        
        # Prepare data
        df = self.prepare_data(df)
        
        # Calculate reference ATR for volatility weighting
        reference_atr = df['atr_14'].median()
        
        # Initialize
        balance = self.config['initial_balance']
        peak_balance = balance
        self.trades = []
        self.equity_curve = []
        
        # Run backtest
        for idx in range(200, len(df)):
            current_bar = df.iloc[idx]
            current_time = current_bar.get('datetime', datetime.now())
            
            # Update existing trades
            closed_trades = self.update_trades(df, idx)
            
            # Update balance
            for trade in closed_trades:
                balance += trade.pnl
                peak_balance = max(peak_balance, balance)
            
            # Generate new signals
            signals = self.generate_signals(df, idx)
            
            # Execute new trades
            for signal in signals:
                trade = self.execute_trade(signal, balance, current_time, reference_atr)
                if trade:
                    self.trades.append(trade)
            
            # Record equity
            open_pnl = 0
            for trade in self.trades:
                if trade.exit_time is None:
                    if trade.side == 'long':
                        unrealized_pnl = (current_bar['close'] - trade.entry_price) * trade.position_size
                    else:
                        unrealized_pnl = (trade.entry_price - current_bar['close']) * trade.position_size
                    open_pnl += unrealized_pnl
            
            self.equity_curve.append({
                'datetime': current_time,
                'balance': balance,
                'open_pnl': open_pnl,
                'total_equity': balance + open_pnl,
                'peak_balance': peak_balance
            })
        
        # Close any remaining open trades
        for trade in self.trades:
            if trade.exit_time is None:
                trade.exit_time = df.iloc[-1].get('datetime', datetime.now())
                trade.exit_price = df.iloc[-1]['close']
                trade.exit_reason = 'end_of_data'
                
                # Calculate final PnL
                if trade.side == 'long':
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
                else:
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size
                
                # Apply costs
                commission = (trade.entry_price + trade.exit_price) * trade.position_size * self.config['commission']
                slippage = (trade.entry_price + trade.exit_price) * trade.position_size * self.config['slippage']
                trade.pnl -= (commission + slippage)
                
                # Calculate metrics
                trade.pnl_pct = trade.pnl / (trade.entry_price * trade.position_size)
                risk_per_share = abs(trade.entry_price - trade.stop_loss)
                trade.r_multiple = trade.pnl / (risk_per_share * trade.position_size)
                
                balance += trade.pnl
        
        # Calculate results
        equity_df = pd.DataFrame(self.equity_curve)
        metrics = self.calculate_metrics(equity_df)
        trade_analysis = self.analyze_trades()
        
        results = {
            'trades': self.trades,
            'equity_curve': equity_df,
            'metrics': metrics,
            'trade_analysis': trade_analysis
        }
        
        logger.info("Backtest complete")
        return results
    
    def calculate_metrics(self, equity_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive backtest metrics"""
        
        if len(equity_df) == 0:
            return {}
        
        initial_balance = self.config['initial_balance']
        final_balance = equity_df.iloc[-1]['total_equity']
        
        # Basic metrics
        total_return = (final_balance - initial_balance) / initial_balance
        
        # Calculate returns
        equity_df['returns'] = equity_df['total_equity'].pct_change().fillna(0)
        
        # Risk metrics
        volatility = equity_df['returns'].std() * np.sqrt(252 * 24)  # Annualized for hourly data
        sharpe_ratio = (equity_df['returns'].mean() * 252 * 24) / volatility if volatility > 0 else 0
        
        # Drawdown
        equity_df['peak'] = equity_df['total_equity'].expanding().max()
        equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Trade metrics
        completed_trades = [t for t in self.trades if t.exit_time is not None]
        
        if completed_trades:
            total_trades = len(completed_trades)
            winning_trades = len([t for t in completed_trades if t.pnl > 0])
            losing_trades = len([t for t in completed_trades if t.pnl < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # PnL metrics
            total_pnl = sum(t.pnl for t in completed_trades)
            avg_win = np.mean([t.pnl for t in completed_trades if t.pnl > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t.pnl for t in completed_trades if t.pnl < 0]) if losing_trades > 0 else 0
            
            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            expectancy_pct = expectancy / initial_balance * 100
            
            # Profit factor
            gross_profit = sum(t.pnl for t in completed_trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in completed_trades if t.pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # R multiples
            r_multiples = [t.r_multiple for t in completed_trades if t.r_multiple is not None]
            avg_r = np.mean(r_multiples) if r_multiples else 0
            
        else:
            total_trades = winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = expectancy = expectancy_pct = 0
            profit_factor = avg_r = 0
        
        # Time-based metrics
        if len(equity_df) > 1:
            days_trading = (equity_df.iloc[-1]['datetime'] - equity_df.iloc[0]['datetime']).days
            monthly_return = ((final_balance / initial_balance) ** (30 / days_trading) - 1) * 100 if days_trading > 0 else 0
        else:
            days_trading = 0
            monthly_return = 0
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return': total_return * 100,
            'monthly_return': monthly_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'expectancy_pct': expectancy_pct,
            'profit_factor': profit_factor,
            'avg_r_multiple': avg_r,
            'max_drawdown': abs(max_drawdown) * 100,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility * 100,
            'days_trading': days_trading
        }
    
    def analyze_trades(self) -> Dict:
        """Analyze trades by type"""
        
        completed_trades = [t for t in self.trades if t.exit_time is not None]
        
        if not completed_trades:
            return {}
        
        analysis = {}
        
        # Analyze by trade type
        for trade_type in ['burst', 'fade', 'normal']:
            type_trades = [t for t in completed_trades if t.trade_type == trade_type]
            
            if type_trades:
                wins = len([t for t in type_trades if t.pnl > 0])
                total = len(type_trades)
                win_rate = wins / total * 100
                
                total_pnl = sum(t.pnl for t in type_trades)
                avg_pnl = total_pnl / total
                
                gross_profit = sum(t.pnl for t in type_trades if t.pnl > 0)
                gross_loss = abs(sum(t.pnl for t in type_trades if t.pnl < 0))
                pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                avg_r = np.mean([t.r_multiple for t in type_trades if t.r_multiple is not None])
                
                analysis[f'{trade_type}_trades'] = {
                    'count': total,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl,
                    'profit_factor': pf,
                    'avg_r_multiple': avg_r
                }
        
        return analysis
    
    def print_results(self, results: Dict):
        """Print comprehensive backtest results"""
        
        print("=" * 80)
        print("SIMPLE PARABOLIC BACKTEST RESULTS")
        print("=" * 80)
        
        # Performance Summary
        print("\n📊 PERFORMANCE SUMMARY")
        print("-" * 40)
        metrics = results['metrics']
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
        if results['trade_analysis']:
            print("\n🎯 TRADE TYPE ANALYSIS")
            print("-" * 40)
            for trade_type, analysis in results['trade_analysis'].items():
                if analysis['count'] > 0:
                    print(f"\n{trade_type.upper()}:")
                    print(f"  Count:         {analysis['count']}")
                    print(f"  Win Rate:      {analysis['win_rate']:.1f}%")
                    print(f"  Total PnL:     ${analysis['total_pnl']:+.2f}")
                    print(f"  Avg PnL:       ${analysis['avg_pnl']:+.4f}")
                    print(f"  Profit Factor: {analysis['profit_factor']:.2f}")
                    print(f"  Avg R:         {analysis['avg_r_multiple']:+.2f}R")
        
        # Enhanced Gates for Parabolic System
        print("\n🚀 ENHANCED PARABOLIC GATES")
        print("-" * 40)
        
        burst_analysis = results['trade_analysis'].get('burst_trades', {})
        fade_analysis = results['trade_analysis'].get('fade_trades', {})
        
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
        breakeven_wr = 1 / (1 + self.config['risk_reward_ratio']) * 100
        safety_margin = metrics['win_rate'] - breakeven_wr
        expected_value = metrics['avg_r_multiple']
        
        print(f"Risk:Reward Ratio:   1:{self.config['risk_reward_ratio']}")
        print(f"Breakeven Win Rate:  {breakeven_wr:.1f}%")
        print(f"Actual Win Rate:     {metrics['win_rate']:.1f}%")
        print(f"Safety Margin:       {safety_margin:+.1f}%")
        print(f"Expected Value:      {expected_value:+.3f}R per trade")
        
        if safety_margin > 0:
            print(f"✅ EDGE CONFIRMED: {safety_margin:.1f}% above breakeven")
        else:
            print(f"❌ NO EDGE: {abs(safety_margin):.1f}% below breakeven")

def main():
    """Run the simple parabolic backtest"""
    
    # Generate comprehensive test data
    np.random.seed(42)
    
    # Create 3 months of hourly data
    dates = pd.date_range('2024-01-01', periods=2160, freq='H')  # 90 days * 24 hours
    
    # Simulate realistic crypto price action with parabolic moves
    price = 50.0  # Starting at $50
    prices = [price]
    volumes = []
    
    for i in range(len(dates) - 1):
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
        'datetime': dates[:-1],  # Remove last date to match other arrays
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': volumes
    })
    
    # Run backtest
    config = {
        'initial_balance': 50.0,
        'risk_per_trade': 0.0075,  # 0.75% risk
        'risk_reward_ratio': 4.0,
        'max_positions': 2,
        'pyramid_enabled': True,
        'adaptive_risk': True,
        'risk_config': {
            'base_risk_percent': 0.0075,
            'max_risk_percent': 0.015,
        }
    }
    
    backtest = SimpleParabolicBacktest(config)
    results = backtest.run_backtest(df)
    
    # Print results
    backtest.print_results(results)
    
    # Save results
    results_data = {
        'config': config,
        'metrics': results['metrics'],
        'trade_analysis': results['trade_analysis'],
        'trades': [
            {
                'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'r_multiple': t.r_multiple,
                'trade_type': t.trade_type,
                'side': t.side,
                'exit_reason': t.exit_reason
            }
            for t in results['trades']
        ]
    }
    
    with open('simple_parabolic_backtest_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to simple_parabolic_backtest_results.json")
    print(f"📊 Processed {len(df)} bars of data")
    print(f"🎯 Generated {len(results['trades'])} trades")

if __name__ == "__main__":
    main() 