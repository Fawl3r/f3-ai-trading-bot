#!/usr/bin/env python3
"""
Elite Parabolic Backtest System
Comprehensive backtesting with parabolic detection and enhanced risk management
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from features.parabola_detector import ParabolaDetector
from models.parabolic_transformer import ParabolicPatternTrainer
from risk_manager_enhanced import EnhancedRiskManager

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

@dataclass
class BacktestResults:
    """Backtest results container"""
    trades: List[Trade]
    equity_curve: pd.DataFrame
    metrics: Dict
    trade_analysis: Dict
    risk_metrics: Dict

class EliteParabolicBacktest:
    """Elite parabolic backtesting system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.parabola_detector = ParabolaDetector()
        self.transformer_trainer = ParabolicPatternTrainer()
        self.risk_manager = EnhancedRiskManager(self.config.get('risk_config', {}))
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
            'use_transformer': True,
            'transformer_threshold': 0.6,
            'time_filter_enabled': True,
            'burst_time_filter': True,
            'fade_time_filter': False,
            'pyramid_enabled': True,
            'adaptive_risk': True,
            'risk_config': {
                'base_risk_percent': 0.0075,
                'max_risk_percent': 0.015,
                'max_daily_loss': 0.02,
                'max_drawdown': 0.05,
                'drawdown_switch_threshold': 0.04,
                'drawdown_recovery_threshold': 0.02,
            }
        }
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with all features and signals"""
        
        logger.info("Preparing data with parabolic features...")
        
        # Add parabolic features
        df = self.parabola_detector.process_complete_analysis(df)
        
        # Train transformer if enabled
        if self.config['use_transformer'] and len(df) > 1000:
            logger.info("Training parabolic transformer...")
            try:
                self.transformer_trainer.train(df, epochs=30, batch_size=16)
                
                # Get transformer predictions
                predictions, confidences = self.transformer_trainer.predict(df)
                
                # Add transformer features
                df['transformer_burst_prob'] = predictions[:, 1]  # Burst probability
                df['transformer_fade_prob'] = predictions[:, 2]   # Fade probability
                df['transformer_confidence'] = confidences.flatten()
                
                logger.info("Transformer training complete")
                
            except Exception as e:
                logger.warning(f"Transformer training failed: {e}")
                # Use dummy values
                df['transformer_burst_prob'] = 0.33
                df['transformer_fade_prob'] = 0.33
                df['transformer_confidence'] = 0.5
        else:
            # Use dummy values
            df['transformer_burst_prob'] = 0.33
            df['transformer_fade_prob'] = 0.33
            df['transformer_confidence'] = 0.5
        
        # Initialize ATR reference for risk manager
        self.risk_manager.calculate_atr_reference(df)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, idx: int) -> List[Dict]:
        """Generate trading signals for current bar"""
        
        if idx < 200:  # Need enough data for indicators
            return []
        
        signals = []
        current_bar = df.iloc[idx]
        
        # Get transformer probabilities
        burst_prob = current_bar.get('transformer_burst_prob', 0.33)
        fade_prob = current_bar.get('transformer_fade_prob', 0.33)
        confidence = current_bar.get('transformer_confidence', 0.5)
        
        # Enhanced signal generation with transformer
        if self.config['use_transformer']:
            # Burst signal
            if (burst_prob > self.config['transformer_threshold'] and
                current_bar.get('burst_long_filtered', False)):
                
                signals.append({
                    'type': 'burst_long',
                    'confidence': burst_prob * confidence,
                    'entry_price': current_bar['close'],
                    'atr': current_bar['atr_14']
                })
            
            if (burst_prob > self.config['transformer_threshold'] and
                current_bar.get('burst_short_filtered', False)):
                
                signals.append({
                    'type': 'burst_short',
                    'confidence': burst_prob * confidence,
                    'entry_price': current_bar['close'],
                    'atr': current_bar['atr_14']
                })
            
            # Fade signal
            if (fade_prob > self.config['transformer_threshold'] and
                current_bar.get('fade_long_filtered', False)):
                
                signals.append({
                    'type': 'fade_long',
                    'confidence': fade_prob * confidence,
                    'entry_price': current_bar['close'],
                    'atr': current_bar['atr_14']
                })
            
            if (fade_prob > self.config['transformer_threshold'] and
                current_bar.get('fade_short_filtered', False)):
                
                signals.append({
                    'type': 'fade_short',
                    'confidence': fade_prob * confidence,
                    'entry_price': current_bar['close'],
                    'atr': current_bar['atr_14']
                })
        
        else:
            # Original signal logic
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
    
    def calculate_position_parameters(self, signal: Dict, balance: float) -> Tuple[float, float, float]:
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
        
        # Calculate risk percentage with volatility adjustment
        if self.config['adaptive_risk']:
            risk_percent = self.risk_manager.calculate_volatility_weighted_risk(atr)
        else:
            risk_percent = self.config['risk_per_trade']
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            balance, entry_price, stop_loss, risk_percent, signal_type.split('_')[0]
        )
        
        return position_size, stop_loss, take_profit
    
    def execute_trade(self, signal: Dict, balance: float, current_time: datetime) -> Optional[Trade]:
        """Execute a trade based on signal"""
        
        # Validate trade with risk manager
        is_valid, reason = self.risk_manager.validate_trade(
            signal['type'], balance, current_time
        )
        
        if not is_valid:
            logger.debug(f"Trade rejected: {reason}")
            return None
        
        # Check position limits
        open_positions = len([t for t in self.trades if t.exit_time is None])
        if open_positions >= self.config['max_positions']:
            return None
        
        # Calculate position parameters
        position_size, stop_loss, take_profit = self.calculate_position_parameters(signal, balance)
        
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
            
            # Check for pyramid opportunity
            if (exit_price is None and 
                self.config['pyramid_enabled'] and 
                trade.trade_type == 'burst'):
                
                # Calculate current PnL in R terms
                if trade.side == 'long':
                    current_pnl_r = (current_price - trade.entry_price) / (trade.entry_price - trade.stop_loss)
                else:
                    current_pnl_r = (trade.entry_price - current_price) / (trade.stop_loss - trade.entry_price)
                
                bars_since_entry = idx - df[df['datetime'] <= trade.entry_time].index[-1]
                
                can_pyramid, additional_size = self.risk_manager.check_pyramid_opportunity(
                    current_pnl_r, bars_since_entry, trade.position_size
                )
                
                if can_pyramid:
                    # Add to position (simplified - just increase position size)
                    trade.position_size += additional_size
                    # Adjust stop loss (trail by 1 ATR)
                    atr = current_bar['atr_14']
                    if trade.side == 'long':
                        trade.stop_loss = max(trade.stop_loss, current_price - atr)
                    else:
                        trade.stop_loss = min(trade.stop_loss, current_price + atr)
            
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
                entry_idx = df[df['datetime'] <= trade.entry_time].index[-1]
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
    
    def run_backtest(self, df: pd.DataFrame) -> BacktestResults:
        """Run complete backtest"""
        
        logger.info("Starting elite parabolic backtest...")
        
        # Prepare data
        df = self.prepare_data(df)
        
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
                trade = self.execute_trade(signal, balance, current_time)
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
        
        # Create results
        equity_df = pd.DataFrame(self.equity_curve)
        metrics = self.calculate_metrics(equity_df)
        trade_analysis = self.analyze_trades()
        risk_metrics = self.risk_manager.get_risk_metrics(balance, peak_balance)
        
        results = BacktestResults(
            trades=self.trades,
            equity_curve=equity_df,
            metrics=metrics,
            trade_analysis=trade_analysis,
            risk_metrics=risk_metrics.__dict__
        )
        
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
        days_trading = (equity_df.iloc[-1]['datetime'] - equity_df.iloc[0]['datetime']).days
        monthly_return = ((final_balance / initial_balance) ** (30 / days_trading) - 1) * 100 if days_trading > 0 else 0
        
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
        """Analyze trades by type and pattern"""
        
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
    
    def print_results(self, results: BacktestResults):
        """Print comprehensive backtest results"""
        
        print("=" * 80)
        print("ELITE PARABOLIC BACKTEST RESULTS")
        print("=" * 80)
        
        # Performance Summary
        print("\n📊 PERFORMANCE SUMMARY")
        print("-" * 40)
        metrics = results.metrics
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
        if results.trade_analysis:
            print("\n🎯 TRADE TYPE ANALYSIS")
            print("-" * 40)
            for trade_type, analysis in results.trade_analysis.items():
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
        
        burst_analysis = results.trade_analysis.get('burst_trades', {})
        fade_analysis = results.trade_analysis.get('fade_trades', {})
        
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
            results.risk_metrics['current_drawdown'] * 100 <= 5,
            f"{results.risk_metrics['current_drawdown'] * 100:.1f}% (≤5%)"
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
    """Run the elite parabolic backtest"""
    
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
        'datetime': dates,
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
        'use_transformer': True,
        'transformer_threshold': 0.55,  # Slightly lower threshold
        'pyramid_enabled': True,
        'adaptive_risk': True,
        'risk_config': {
            'base_risk_percent': 0.0075,
            'max_risk_percent': 0.015,
            'drawdown_switch_threshold': 0.04,
            'drawdown_recovery_threshold': 0.02,
        }
    }
    
    backtest = EliteParabolicBacktest(config)
    results = backtest.run_backtest(df)
    
    # Print results
    backtest.print_results(results)
    
    # Save results
    results_data = {
        'config': config,
        'metrics': results.metrics,
        'trade_analysis': results.trade_analysis,
        'risk_metrics': results.risk_metrics,
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
            for t in results.trades
        ]
    }
    
    with open('elite_parabolic_backtest_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to elite_parabolic_backtest_results.json")
    print(f"📊 Processed {len(df)} bars of data")
    print(f"🎯 Generated {len(results.trades)} trades")

if __name__ == "__main__":
    main() 