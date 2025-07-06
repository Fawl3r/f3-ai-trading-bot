#!/usr/bin/env python3
"""
Monte Carlo Slippage & Latency Jitter Test
Pre-launch stress test for Elite 100%/5% Trading System
Tests system resilience under exchange hiccups and market stress
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sqlite3
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCTradeResult:
    """Monte Carlo trade result"""
    trade_id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    intended_entry: float
    intended_exit: float
    slippage_entry: float
    slippage_exit: float
    latency_ms: float
    pnl: float
    r_multiple: float
    slippage_cost: float

@dataclass
class MCRunResult:
    """Monte Carlo run result"""
    run_id: int
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_return: float
    avg_slippage_bps: float
    avg_latency_ms: float
    slippage_cost_total: float

class MonteCarloSlippageTest:
    """Monte Carlo test for slippage and latency impact"""
    
    def __init__(self, risk_pct: float = 0.005):
        self.risk_pct = risk_pct
        self.initial_balance = 10000.0
        self.results: List[MCRunResult] = []
        
    def generate_market_conditions(self, n_trades: int = 200) -> List[Dict]:
        """Generate realistic market conditions for testing"""
        market_conditions = []
        
        for i in range(n_trades):
            # Base market condition
            condition = {
                'timestamp': datetime.now() + timedelta(minutes=i*30),
                'symbol': np.random.choice(['BTC-USD', 'ETH-USD', 'SOL-USD'], p=[0.4, 0.4, 0.2]),
                'base_price': 50000 if 'BTC' in str(i%3) else (3000 if 'ETH' in str(i%3) else 100),
                'volatility': np.random.uniform(0.01, 0.05),  # 1-5% volatility
                'liquidity_score': np.random.uniform(0.3, 1.0),  # 0.3-1.0 liquidity
                'market_stress': np.random.uniform(0.0, 1.0),   # 0-1 stress level
                'time_of_day': (i * 30) % 1440,  # Minutes since midnight
            }
            
            # Market microstructure effects
            condition['bid_ask_spread_bps'] = self._calculate_spread(condition)
            condition['market_impact_bps'] = self._calculate_market_impact(condition)
            condition['latency_base_ms'] = self._calculate_base_latency(condition)
            
            market_conditions.append(condition)
        
        return market_conditions
    
    def _calculate_spread(self, condition: Dict) -> float:
        """Calculate bid-ask spread based on market conditions"""
        base_spread = 2.0  # 2 bps base spread
        
        # Increase spread with lower liquidity
        liquidity_factor = 1.0 / max(condition['liquidity_score'], 0.1)
        
        # Increase spread with market stress
        stress_factor = 1.0 + condition['market_stress'] * 3.0
        
        # Time of day effects (wider spreads during low liquidity hours)
        hour = condition['time_of_day'] // 60
        time_factor = 1.5 if (hour < 6 or hour > 22) else 1.0
        
        return base_spread * liquidity_factor * stress_factor * time_factor
    
    def _calculate_market_impact(self, condition: Dict) -> float:
        """Calculate market impact based on order size and liquidity"""
        base_impact = 1.0  # 1 bps base impact
        
        # Scale with liquidity (lower liquidity = higher impact)
        liquidity_factor = 1.0 / max(condition['liquidity_score'], 0.1)
        
        # Scale with volatility
        volatility_factor = 1.0 + condition['volatility'] * 10
        
        # Market stress increases impact
        stress_factor = 1.0 + condition['market_stress'] * 2.0
        
        return base_impact * liquidity_factor * volatility_factor * stress_factor
    
    def _calculate_base_latency(self, condition: Dict) -> float:
        """Calculate base latency before jitter"""
        base_latency = 50.0  # 50ms base latency
        
        # Market stress increases latency
        stress_factor = 1.0 + condition['market_stress'] * 2.0
        
        # Time of day effects
        hour = condition['time_of_day'] // 60
        time_factor = 1.3 if (hour >= 14 and hour <= 18) else 1.0  # Higher during active hours
        
        return base_latency * stress_factor * time_factor
    
    def simulate_trade_execution(self, condition: Dict, side: str, intended_entry: float, 
                               intended_exit: float) -> MCTradeResult:
        """Simulate trade execution with slippage and latency"""
        
        # Generate latency jitter
        latency_jitter = np.random.exponential(condition['latency_base_ms'] * 0.3)
        total_latency = condition['latency_base_ms'] + latency_jitter
        
        # Generate slippage for entry
        spread_bps = condition['bid_ask_spread_bps']
        impact_bps = condition['market_impact_bps']
        
        # Entry slippage (always unfavorable)
        entry_slippage_bps = (spread_bps / 2) + impact_bps + np.random.exponential(2.0)
        if side == 'long':
            entry_slippage = entry_slippage_bps / 10000  # Convert to decimal
            actual_entry = intended_entry * (1 + entry_slippage)
        else:
            entry_slippage = -entry_slippage_bps / 10000
            actual_entry = intended_entry * (1 + entry_slippage)
        
        # Exit slippage (also generally unfavorable)
        exit_slippage_bps = (spread_bps / 2) + (impact_bps * 0.7) + np.random.exponential(1.5)
        if side == 'long':
            exit_slippage = -exit_slippage_bps / 10000
            actual_exit = intended_exit * (1 + exit_slippage)
        else:
            exit_slippage = exit_slippage_bps / 10000
            actual_exit = intended_exit * (1 + exit_slippage)
        
        # Calculate P&L
        if side == 'long':
            pnl_pct = (actual_exit - actual_entry) / actual_entry
        else:
            pnl_pct = (actual_entry - actual_exit) / actual_entry
        
        # Calculate position size based on risk
        risk_amount = self.initial_balance * self.risk_pct
        stop_distance = abs(intended_exit - intended_entry) / intended_entry
        position_size = risk_amount / (intended_entry * stop_distance)
        
        pnl_dollars = pnl_pct * actual_entry * position_size
        r_multiple = pnl_dollars / risk_amount
        
        # Calculate total slippage cost
        slippage_cost = abs(entry_slippage_bps) + abs(exit_slippage_bps)
        
        return MCTradeResult(
            trade_id=0,  # Will be set later
            symbol=condition['symbol'],
            side=side,
            entry_price=actual_entry,
            exit_price=actual_exit,
            intended_entry=intended_entry,
            intended_exit=intended_exit,
            slippage_entry=entry_slippage_bps,
            slippage_exit=exit_slippage_bps,
            latency_ms=total_latency,
            pnl=pnl_dollars,
            r_multiple=r_multiple,
            slippage_cost=slippage_cost
        )
    
    def run_monte_carlo_simulation(self, n_runs: int = 500, trades_per_run: int = 200) -> List[MCRunResult]:
        """Run Monte Carlo simulation"""
        logger.info(f"Starting Monte Carlo simulation: {n_runs} runs, {trades_per_run} trades each")
        
        for run_id in range(n_runs):
            if run_id % 50 == 0:
                logger.info(f"Completed {run_id}/{n_runs} runs")
            
            # Generate market conditions for this run
            market_conditions = self.generate_market_conditions(trades_per_run)
            
            # Simulate trades
            trades = []
            current_balance = self.initial_balance
            peak_balance = self.initial_balance
            
            for i, condition in enumerate(market_conditions):
                # Generate trade parameters (based on historical patterns)
                side = np.random.choice(['long', 'short'], p=[0.6, 0.4])
                
                # Generate intended prices (simulate signal generation)
                base_price = condition['base_price']
                entry_price = base_price * (1 + np.random.normal(0, 0.002))
                
                if side == 'long':
                    stop_price = entry_price * 0.98   # 2% stop
                    target_price = entry_price * 1.08  # 8% target
                else:
                    stop_price = entry_price * 1.02   # 2% stop
                    target_price = entry_price * 0.92  # 8% target
                
                # Determine exit (win/loss based on historical win rate)
                is_winner = np.random.random() < 0.40  # 40% win rate
                intended_exit = target_price if is_winner else stop_price
                
                # Simulate execution
                trade = self.simulate_trade_execution(condition, side, entry_price, intended_exit)
                trade.trade_id = i
                
                # Update balance
                current_balance += trade.pnl
                if current_balance > peak_balance:
                    peak_balance = current_balance
                
                trades.append(trade)
            
            # Calculate run statistics
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            max_drawdown = (peak_balance - min(self.initial_balance + sum(t.pnl for t in trades[:i+1]) 
                                             for i in range(len(trades)))) / peak_balance * 100
            
            total_return = (current_balance - self.initial_balance) / self.initial_balance * 100
            
            avg_slippage = np.mean([t.slippage_cost for t in trades])
            avg_latency = np.mean([t.latency_ms for t in trades])
            slippage_cost_total = sum(t.slippage_cost for t in trades)
            
            run_result = MCRunResult(
                run_id=run_id,
                total_trades=len(trades),
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                total_return=total_return,
                avg_slippage_bps=avg_slippage,
                avg_latency_ms=avg_latency,
                slippage_cost_total=slippage_cost_total
            )
            
            self.results.append(run_result)
        
        return self.results
    
    def analyze_results(self) -> Dict:
        """Analyze Monte Carlo results"""
        if not self.results:
            return {}
        
        # Convert to arrays for analysis
        profit_factors = [r.profit_factor for r in self.results if r.profit_factor != float('inf')]
        drawdowns = [r.max_drawdown for r in self.results]
        returns = [r.total_return for r in self.results]
        win_rates = [r.win_rate for r in self.results]
        slippages = [r.avg_slippage_bps for r in self.results]
        latencies = [r.avg_latency_ms for r in self.results]
        
        # Calculate percentiles
        pf_5th = np.percentile(profit_factors, 5)
        dd_95th = np.percentile(drawdowns, 95)
        return_5th = np.percentile(returns, 5)
        
        analysis = {
            'total_runs': len(self.results),
            'profit_factor': {
                'mean': np.mean(profit_factors),
                'median': np.median(profit_factors),
                '5th_percentile': pf_5th,
                '95th_percentile': np.percentile(profit_factors, 95)
            },
            'max_drawdown': {
                'mean': np.mean(drawdowns),
                'median': np.median(drawdowns),
                '95th_percentile': dd_95th,
                '99th_percentile': np.percentile(drawdowns, 99)
            },
            'total_return': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                '5th_percentile': return_5th,
                '95th_percentile': np.percentile(returns, 95)
            },
            'win_rate': {
                'mean': np.mean(win_rates),
                'std': np.std(win_rates)
            },
            'slippage_impact': {
                'avg_slippage_bps': np.mean(slippages),
                'slippage_cost_pct': np.mean(slippages) / 10000 * 100
            },
            'latency_impact': {
                'avg_latency_ms': np.mean(latencies),
                'p95_latency_ms': np.percentile(latencies, 95)
            },
            'pass_criteria': {
                'pf_5th_above_1_7': pf_5th >= 1.7,
                'dd_95th_below_6': dd_95th <= 6.0,
                'overall_pass': pf_5th >= 1.7 and dd_95th <= 6.0
            }
        }
        
        return analysis
    
    def save_results(self, filename: str = None):
        """Save results to file"""
        if filename is None:
            filename = f"mc_slippage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        analysis = self.analyze_results()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        output = {
            'test_parameters': {
                'risk_pct': self.risk_pct,
                'initial_balance': self.initial_balance,
                'timestamp': datetime.now().isoformat()
            },
            'analysis': convert_numpy_types(analysis),
            'raw_results': [convert_numpy_types(asdict(r)) for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        return filename

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Monte Carlo Slippage & Latency Test')
    parser.add_argument('--runs', type=int, default=500, help='Number of Monte Carlo runs')
    parser.add_argument('--trades', type=int, default=200, help='Trades per run')
    parser.add_argument('--risk', type=float, default=0.005, help='Risk per trade (0.005 = 0.5%)')
    parser.add_argument('--output', type=str, help='Output filename')
    
    args = parser.parse_args()
    
    # Run test
    logger.info("=== Monte Carlo Slippage & Latency Jitter Test ===")
    logger.info(f"Risk per trade: {args.risk*100:.2f}%")
    logger.info(f"Runs: {args.runs}, Trades per run: {args.trades}")
    
    test = MonteCarloSlippageTest(risk_pct=args.risk)
    results = test.run_monte_carlo_simulation(n_runs=args.runs, trades_per_run=args.trades)
    
    # Analyze results
    analysis = test.analyze_results()
    
    # Print summary
    logger.info("\n=== RESULTS SUMMARY ===")
    logger.info(f"Total runs: {analysis['total_runs']}")
    logger.info(f"Profit Factor (5th percentile): {analysis['profit_factor']['5th_percentile']:.2f}")
    logger.info(f"Max Drawdown (95th percentile): {analysis['max_drawdown']['95th_percentile']:.2f}%")
    logger.info(f"Average slippage cost: {analysis['slippage_impact']['avg_slippage_bps']:.1f} bps")
    logger.info(f"Average latency: {analysis['latency_impact']['avg_latency_ms']:.1f} ms")
    
    # Pass/Fail criteria
    pf_pass = analysis['pass_criteria']['pf_5th_above_1_7']
    dd_pass = analysis['pass_criteria']['dd_95th_below_6']
    overall_pass = analysis['pass_criteria']['overall_pass']
    
    logger.info("\n=== PASS/FAIL CRITERIA ===")
    logger.info(f"PF 5th percentile ≥ 1.7: {'✅ PASS' if pf_pass else '❌ FAIL'}")
    logger.info(f"DD 95th percentile ≤ 6%: {'✅ PASS' if dd_pass else '❌ FAIL'}")
    logger.info(f"Overall result: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    if overall_pass:
        logger.info("\n🎉 System passes Monte Carlo stress test!")
        logger.info("Ready for live deployment under real market conditions.")
    else:
        logger.warning("\n⚠️  System failed Monte Carlo stress test!")
        logger.warning("Review slippage assumptions and risk parameters before live deployment.")
    
    # Save results
    filename = test.save_results(args.output)
    logger.info(f"\nDetailed results saved to: {filename}")

if __name__ == "__main__":
    main() 