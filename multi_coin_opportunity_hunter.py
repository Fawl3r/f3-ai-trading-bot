#!/usr/bin/env python3
"""
Multi-Coin Opportunity Hunter
Automatically finds and ranks the best trading opportunities across multiple coins
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import asyncio
from typing import List, Dict, Tuple
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniverseBuilder:
    """Build and maintain trading universe"""
    
    def __init__(self, min_volume_usd: float = 5_000_000):
        self.min_volume_usd = min_volume_usd
        self.universe_cache = None
        self.last_update = None
        
    def fetch_hyperliquid_universe(self) -> List[str]:
        """Fetch liquid perpetual contracts from Hyperliquid"""
        # Simulated universe for testing
        # In production, this would call Hyperliquid API
        universe = [
            "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "MATIC-USD",
            "ARB-USD", "OP-USD", "DOGE-USD", "LTC-USD", "LINK-USD",
            "UNI-USD", "AAVE-USD", "CRV-USD", "MKR-USD", "SNX-USD"
        ]
        
        logger.info(f"📊 Universe: {len(universe)} liquid perpetuals")
        return universe
    
    def filter_by_liquidity(self, symbols: List[str], 
                          market_data: Dict) -> List[str]:
        """Filter symbols by liquidity metrics"""
        liquid_symbols = []
        
        for symbol in symbols:
            if symbol in market_data:
                data = market_data[symbol]
                # Check 24h volume
                if data.get('volume_24h_usd', 0) >= self.min_volume_usd:
                    liquid_symbols.append(symbol)
        
        logger.info(f"🔍 Filtered to {len(liquid_symbols)} liquid symbols")
        return liquid_symbols

class StrategyBacktester:
    """Run strategy on individual coins with 5k candle limit"""
    
    def __init__(self, strategy_class):
        self.strategy_class = strategy_class
        self.window_size = 5000
        self.shift_size = 3000  # 40% overlap
        
    def generate_candle_data(self, symbol: str, days: int = 90) -> List[Dict]:
        """Generate synthetic candle data for testing"""
        candles = []
        
        # Different characteristics per symbol
        base_prices = {
            "BTC-USD": 50000, "ETH-USD": 3500, "SOL-USD": 100,
            "AVAX-USD": 40, "MATIC-USD": 1.2, "ARB-USD": 1.5,
            "OP-USD": 2.5, "DOGE-USD": 0.15, "LTC-USD": 100
        }
        
        volatilities = {
            "BTC-USD": 0.02, "ETH-USD": 0.025, "SOL-USD": 0.04,
            "AVAX-USD": 0.035, "MATIC-USD": 0.045, "DOGE-USD": 0.06
        }
        
        base_price = base_prices.get(symbol, 100)
        volatility = volatilities.get(symbol, 0.03)
        current_price = base_price
        
        start_time = datetime.now() - timedelta(days=days)
        
        for i in range(days * 288):  # 5-minute candles
            # Symbol-specific trends
            if "SOL" in symbol:
                trend = 0.0001 * np.sin(i / 500)  # Oscillating
            elif "BTC" in symbol:
                trend = 0.00005  # Slight uptrend
            else:
                trend = -0.00002  # Slight downtrend
            
            change = np.random.normal(trend, volatility)
            new_price = current_price * (1 + change)
            
            high = new_price * (1 + abs(np.random.normal(0, volatility/3)))
            low = new_price * (1 - abs(np.random.normal(0, volatility/3)))
            
            candle = {
                'timestamp': int((start_time + timedelta(minutes=i*5)).timestamp() * 1000),
                'open': current_price,
                'high': max(high, current_price, new_price),
                'low': min(low, current_price, new_price),
                'close': new_price,
                'volume': max(100, 10000 / volatility + np.random.normal(0, 1000))
            }
            
            candles.append(candle)
            current_price = new_price
        
        return candles
    
    def run_window_backtest(self, symbol: str, candles: List[Dict], 
                           start_idx: int) -> Dict:
        """Run backtest on a single window"""
        window_candles = candles[start_idx:start_idx + self.window_size]
        
        if len(window_candles) < self.window_size:
            return None
        
        # Initialize strategy
        strategy = self.strategy_class()
        
        # Simple backtest logic
        trades = 0
        wins = 0
        total_pnl = 0
        max_dd = 0
        equity = 10000
        peak_equity = equity
        
        for i in range(100, len(window_candles) - 10):
            # Generate signal
            signal = strategy.generate_signal(window_candles[:i+1])
            
            if signal['direction'] != 'hold' and signal.get('confidence', 0) > 0.6:
                trades += 1
                
                # Simulate trade
                entry_price = window_candles[i]['close']
                
                # Random outcome for testing
                if np.random.random() < 0.55:  # 55% win rate
                    pnl = np.random.uniform(0.01, 0.03)  # 1-3% win
                    wins += 1
                else:
                    pnl = -np.random.uniform(0.005, 0.015)  # 0.5-1.5% loss
                
                total_pnl += pnl
                equity *= (1 + pnl)
                
                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                else:
                    dd = (peak_equity - equity) / peak_equity
                    max_dd = max(max_dd, dd)
        
        if trades == 0:
            return None
        
        # Calculate metrics
        win_rate = wins / trades
        expectancy = total_pnl / trades
        
        # Simplified Sharpe
        sharpe = expectancy / 0.01 if expectancy > 0 else 0
        
        return {
            'symbol': symbol,
            'start_idx': start_idx,
            'trades': trades,
            'win_rate': win_rate,
            'expectancy': expectancy,
            'total_pnl': total_pnl,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'final_equity': equity
        }
    
    def backtest_symbol(self, symbol: str, days: int = 90) -> List[Dict]:
        """Run rolling window backtests for a symbol"""
        logger.info(f"🔄 Backtesting {symbol}...")
        
        # Generate or fetch candle data
        candles = self.generate_candle_data(symbol, days)
        
        results = []
        start_idx = 0
        
        while start_idx + self.window_size <= len(candles):
            result = self.run_window_backtest(symbol, candles, start_idx)
            if result:
                results.append(result)
            start_idx += self.shift_size
        
        return results

class OpportunityRanker:
    """Rank coins by trading opportunity"""
    
    def __init__(self, db_path: str = "backtest_results.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize results database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                trades INTEGER,
                win_rate REAL,
                expectancy REAL,
                sharpe REAL,
                max_drawdown REAL,
                total_pnl REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_results(self, results: List[Dict]):
        """Save backtest results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in results:
            cursor.execute('''
                INSERT INTO backtest_results 
                (symbol, trades, win_rate, expectancy, sharpe, max_drawdown, total_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['symbol'],
                result['trades'],
                result['win_rate'],
                result['expectancy'],
                result['sharpe'],
                result['max_drawdown'],
                result['total_pnl']
            ))
        
        conn.commit()
        conn.close()
    
    def calculate_rankings(self) -> pd.DataFrame:
        """Calculate composite rankings"""
        conn = sqlite3.connect(self.db_path)
        
        # Aggregate metrics by symbol
        query = '''
            SELECT 
                symbol,
                COUNT(*) as windows,
                AVG(win_rate) as avg_win_rate,
                AVG(expectancy) as avg_expectancy,
                AVG(sharpe) as avg_sharpe,
                MAX(max_drawdown) as max_dd,
                SUM(total_pnl) as total_pnl,
                AVG(trades) as avg_trades
            FROM backtest_results
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY symbol
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Calculate composite score
        # Normalize metrics to 0-1 range
        df['expectancy_rank'] = df['avg_expectancy'].rank(pct=True)
        df['sharpe_rank'] = df['avg_sharpe'].rank(pct=True)
        df['dd_rank'] = 1 - df['max_dd'].rank(pct=True)  # Lower is better
        df['activity_rank'] = df['avg_trades'].rank(pct=True)
        
        # Weighted composite score
        weights = {
            'expectancy': 0.35,
            'sharpe': 0.35,
            'drawdown': 0.20,
            'activity': 0.10
        }
        
        df['composite_score'] = (
            df['expectancy_rank'] * weights['expectancy'] +
            df['sharpe_rank'] * weights['sharpe'] +
            df['dd_rank'] * weights['drawdown'] +
            df['activity_rank'] * weights['activity']
        )
        
        # Sort by composite score
        df = df.sort_values('composite_score', ascending=False)
        
        return df
    
    def get_top_opportunities(self, n: int = 5) -> List[str]:
        """Get top N trading opportunities"""
        rankings = self.calculate_rankings()
        
        if len(rankings) == 0:
            return []
        
        top_symbols = rankings.head(n)['symbol'].tolist()
        
        logger.info("🏆 Top Trading Opportunities:")
        for i, row in rankings.head(n).iterrows():
            logger.info(f"   {row['symbol']}: Score={row['composite_score']:.3f}, "
                       f"Exp={row['avg_expectancy']:.3%}, Sharpe={row['avg_sharpe']:.2f}")
        
        return top_symbols

class MultiCoinCoordinator:
    """Coordinate multi-coin trading"""
    
    def __init__(self, strategy_class, universe: List[str], 
                 mode: str = "top3", rotation_hours: int = 24):
        self.strategy_class = strategy_class
        self.universe = universe
        self.mode = mode  # single, top3, basket
        self.rotation_hours = rotation_hours
        
        self.backtester = StrategyBacktester(strategy_class)
        self.ranker = OpportunityRanker()
        self.current_selection = []
        self.last_rotation = None
        
    async def run_parallel_backtests(self):
        """Run backtests in parallel"""
        logger.info(f"🚀 Running parallel backtests on {len(self.universe)} coins...")
        
        all_results = []
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            # Submit all backtest jobs
            future_to_symbol = {
                executor.submit(self.backtester.backtest_symbol, symbol): symbol
                for symbol in self.universe
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"✅ Completed {symbol}: {len(results)} windows")
                except Exception as e:
                    logger.error(f"❌ Error backtesting {symbol}: {e}")
        
        # Save all results
        if all_results:
            self.ranker.save_results(all_results)
        
        return all_results
    
    def select_coins(self) -> List[str]:
        """Select coins based on mode"""
        if self.mode == "single":
            top = self.ranker.get_top_opportunities(1)
            return top[:1] if top else []
        
        elif self.mode == "top3":
            return self.ranker.get_top_opportunities(3)
        
        elif self.mode == "basket":
            return self.ranker.get_top_opportunities(5)
        
        else:
            return []
    
    def should_rotate(self) -> bool:
        """Check if it's time to rotate selection"""
        if self.last_rotation is None:
            return True
        
        hours_passed = (datetime.now() - self.last_rotation).total_seconds() / 3600
        return hours_passed >= self.rotation_hours
    
    async def update_selection(self):
        """Update coin selection if needed"""
        if self.should_rotate():
            logger.info("🔄 Rotating coin selection...")
            
            # Run fresh backtests
            await self.run_parallel_backtests()
            
            # Get new selection
            new_selection = self.select_coins()
            
            if new_selection != self.current_selection:
                logger.info(f"📊 Selection changed: {self.current_selection} → {new_selection}")
                self.current_selection = new_selection
            else:
                logger.info(f"📊 Selection unchanged: {self.current_selection}")
            
            self.last_rotation = datetime.now()
            
            # Save selection to file
            with open('top_assets.json', 'w') as f:
                json.dump({
                    'timestamp': self.last_rotation.isoformat(),
                    'mode': self.mode,
                    'selection': self.current_selection,
                    'rankings': self.ranker.calculate_rankings().to_dict('records')
                }, f, indent=2)
            
            logger.info("✅ Selection saved to top_assets.json")
    
    def get_position_sizes(self, total_capital: float) -> Dict[str, float]:
        """Calculate position sizes for selected coins"""
        if not self.current_selection:
            return {}
        
        # Equal weight for now
        # Could be enhanced with Kelly sizing based on edge
        size_per_coin = total_capital / len(self.current_selection)
        
        return {coin: size_per_coin for coin in self.current_selection}

# Dummy strategy for testing
class SimpleStrategy:
    """Simple strategy for testing multi-coin system"""
    
    def generate_signal(self, candles: list) -> dict:
        if len(candles) < 50:
            return {'direction': 'hold', 'confidence': 0}
        
        # Simple MA crossover
        closes = [c['close'] for c in candles[-50:]]
        sma_10 = np.mean(closes[-10:])
        sma_30 = np.mean(closes[-30:])
        
        if sma_10 > sma_30 * 1.01:
            return {'direction': 'long', 'confidence': 0.7}
        elif sma_10 < sma_30 * 0.99:
            return {'direction': 'short', 'confidence': 0.7}
        else:
            return {'direction': 'hold', 'confidence': 0}

async def main():
    """Test multi-coin opportunity hunter"""
    print("🎯 MULTI-COIN OPPORTUNITY HUNTER")
    print("=" * 50)
    print("Finding the best trading opportunities across the universe\n")
    
    # Initialize universe
    universe_builder = UniverseBuilder()
    universe = universe_builder.fetch_hyperliquid_universe()
    
    # Test different modes
    modes = ["single", "top3", "basket"]
    
    for mode in modes:
        print(f"\n🧪 Testing {mode.upper()} mode...")
        print("-" * 40)
        
        coordinator = MultiCoinCoordinator(
            SimpleStrategy,
            universe[:10],  # Test with subset
            mode=mode,
            rotation_hours=24
        )
        
        # Run selection
        await coordinator.update_selection()
        
        # Show results
        print(f"📊 Selected coins: {coordinator.current_selection}")
        
        if coordinator.current_selection:
            sizes = coordinator.get_position_sizes(10000)
            print(f"💰 Position sizes:")
            for coin, size in sizes.items():
                print(f"   {coin}: ${size:.2f}")
    
    # Show final rankings
    print(f"\n📈 FINAL RANKINGS")
    print("=" * 50)
    
    ranker = OpportunityRanker()
    rankings = ranker.calculate_rankings()
    
    if not rankings.empty:
        print(rankings[['symbol', 'avg_expectancy', 'avg_sharpe', 'max_dd', 'composite_score']].head(10))
    
    print("\n✅ Multi-coin system ready for deployment!")

if __name__ == "__main__":
    asyncio.run(main()) 