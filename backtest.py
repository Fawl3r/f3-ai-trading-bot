import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Dict
from strategy import AdvancedTradingStrategy, TradingSignal
from indicators import TechnicalIndicators
from okx_client import OKXClient
from config import INSTRUMENT_ID, POSITION_SIZE_USD, LEVERAGE

class Backtester:
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.strategy = AdvancedTradingStrategy()
        self.trades = []
        self.equity_curve = []
        
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical data from OKX"""
        print(f"Fetching {days} days of historical data...")
        
        try:
            client = OKXClient()
            all_data = []
            
            # OKX limits to 300 candles per request for 1m data
            # Calculate number of requests needed
            candles_per_day = 1440  # 1440 minutes in a day
            total_candles = days * candles_per_day
            requests_needed = (total_candles // 300) + 1
            
            for i in range(requests_needed):
                if i == 0:
                    # First request - most recent data
                    response = client.get_candlesticks(
                        inst_id=INSTRUMENT_ID,
                        bar='1m',
                        limit=300
                    )
                else:
                    # Subsequent requests with before parameter
                    before_ts = all_data[-1][0]  # Use timestamp of oldest candle
                    response = client.get_candlesticks(
                        inst_id=INSTRUMENT_ID,
                        bar='1m',
                        limit=300,
                        # Note: OKX API might need 'before' parameter implementation
                    )
                
                if response.get('code') == '0' and response.get('data'):
                    all_data.extend(response['data'])
                    print(f"Fetched batch {i+1}/{requests_needed}")
                else:
                    print(f"Error fetching data: {response}")
                    break
                    
                if len(all_data) >= total_candles:
                    break
            
            # Convert to DataFrame
            df_data = []
            for candle in reversed(all_data):  # OKX returns newest first, we want oldest first
                df_data.append({
                    'timestamp': int(candle[0]),
                    'datetime': datetime.fromtimestamp(int(candle[0]) / 1000),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            df = pd.DataFrame(df_data)
            print(f"Successfully loaded {len(df)} candles")
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            # Return sample data for testing
            return self._generate_sample_data(days)
    
    def _generate_sample_data(self, days: int) -> pd.DataFrame:
        """Generate sample data for testing when API is unavailable"""
        print("Generating sample data for testing...")
        
        periods = days * 1440  # 1 minute candles
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             periods=periods, freq='1min')
        
        # Generate realistic SOL price movement
        np.random.seed(42)
        price_start = 150.0
        returns = np.random.normal(0, 0.002, periods)  # 0.2% volatility per minute
        prices = [price_start]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[:-1])
        
        # Generate OHLC from prices
        data = []
        for i, price in enumerate(prices):
            noise = np.random.normal(0, 0.001, 4)
            high = price * (1 + abs(noise[0]))
            low = price * (1 - abs(noise[1]))
            open_price = price * (1 + noise[2])
            close = price * (1 + noise[3])
            volume = np.random.normal(100000, 20000)
            
            data.append({
                'timestamp': int(dates[i].timestamp() * 1000),
                'datetime': dates[i],
                'open': max(open_price, 0.1),
                'high': max(high, open_price, close),
                'low': min(low, open_price, close),
                'close': max(close, 0.1),
                'volume': max(volume, 1000)
            })
        
        return pd.DataFrame(data)
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run backtest on historical data"""
        print("Running backtest...")
        
        position = None
        entry_price = 0
        entry_time = None
        
        for i in range(50, len(df)):  # Start after enough data for indicators
            current_slice = df.iloc[:i+1].copy()
            current_price = current_slice['close'].iloc[-1]
            current_time = current_slice['datetime'].iloc[-1]
            
            # Update equity curve
            if position:
                if position == 'long':
                    unrealized_pnl = (current_price - entry_price) / entry_price * POSITION_SIZE_USD * LEVERAGE
                else:  # short
                    unrealized_pnl = (entry_price - current_price) / entry_price * POSITION_SIZE_USD * LEVERAGE
                current_equity = self.balance + unrealized_pnl
            else:
                current_equity = self.balance
            
            self.equity_curve.append({
                'datetime': current_time,
                'equity': current_equity,
                'price': current_price
            })
            
            # Generate signal
            signal = self.strategy.generate_signal(current_slice)
            
            if signal and signal.signal_type in ['buy', 'sell', 'close']:
                # Close existing position
                if position:
                    if position == 'long':
                        pnl = (current_price - entry_price) / entry_price * POSITION_SIZE_USD * LEVERAGE
                    else:  # short
                        pnl = (entry_price - current_price) / entry_price * POSITION_SIZE_USD * LEVERAGE
                    
                    self.balance += pnl
                    
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'side': position,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': (pnl / POSITION_SIZE_USD) * 100,
                        'signal_confidence': signal.confidence if signal else 0,
                        'duration': (current_time - entry_time).total_seconds() / 60  # minutes
                    }
                    self.trades.append(trade)
                    
                    print(f"{current_time}: Closed {position} | P&L: ${pnl:.2f} ({trade['pnl_pct']:.2f}%)")
                    position = None
                
                # Open new position
                if signal.signal_type in ['buy', 'sell']:
                    position = 'long' if signal.signal_type == 'buy' else 'short'
                    entry_price = current_price
                    entry_time = current_time
                    
                    print(f"{current_time}: Opened {position} @ ${entry_price:.4f} | Confidence: {signal.confidence:.2%}")
        
        # Close final position if any
        if position:
            if position == 'long':
                pnl = (df['close'].iloc[-1] - entry_price) / entry_price * POSITION_SIZE_USD * LEVERAGE
            else:
                pnl = (entry_price - df['close'].iloc[-1]) / entry_price * POSITION_SIZE_USD * LEVERAGE
            
            self.balance += pnl
            
            trade = {
                'entry_time': entry_time,
                'exit_time': df['datetime'].iloc[-1],
                'side': position,
                'entry_price': entry_price,
                'exit_price': df['close'].iloc[-1],
                'pnl': pnl,
                'pnl_pct': (pnl / POSITION_SIZE_USD) * 100,
                'signal_confidence': 0,
                'duration': (df['datetime'].iloc[-1] - entry_time).total_seconds() / 60
            }
            self.trades.append(trade)
        
        return self._calculate_results()
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100
        
        # P&L stats
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float('inf')
        
        # Equity curve stats
        equity_df = pd.DataFrame(self.equity_curve)
        peak_equity = equity_df['equity'].max()
        max_drawdown = ((peak_equity - equity_df['equity'].min()) / peak_equity) * 100
        
        # Return calculations
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'avg_trade_duration': trades_df['duration'].mean(),
            'trades': trades_df,
            'equity_curve': equity_df
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"Initial Balance:     ${results['initial_balance']:,.2f}")
        print(f"Final Balance:       ${results['final_balance']:,.2f}")
        print(f"Total Return:        {results['total_return']:,.2f}%")
        print(f"Total P&L:           ${results['total_pnl']:,.2f}")
        print()
        
        print(f"Total Trades:        {results['total_trades']}")
        print(f"Winning Trades:      {results['winning_trades']}")
        print(f"Losing Trades:       {results['losing_trades']}")
        print(f"Win Rate:            {results['win_rate']:.2f}%")
        print()
        
        print(f"Average Win:         ${results['avg_win']:.2f}")
        print(f"Average Loss:        ${results['avg_loss']:.2f}")
        print(f"Profit Factor:       {results['profit_factor']:.2f}")
        print(f"Max Drawdown:        {results['max_drawdown']:.2f}%")
        print(f"Avg Trade Duration:  {results['avg_trade_duration']:.1f} minutes")
        
        print("\n" + "="*60)
        
        # Show recent trades
        if not results['trades'].empty:
            print("RECENT TRADES:")
            recent_trades = results['trades'].tail(10)
            for _, trade in recent_trades.iterrows():
                print(f"{trade['entry_time'].strftime('%m-%d %H:%M')} | "
                      f"{trade['side'].upper():5} | "
                      f"${trade['entry_price']:7.4f} -> ${trade['exit_price']:7.4f} | "
                      f"P&L: ${trade['pnl']:6.2f} ({trade['pnl_pct']:+5.2f}%)")
    
    def plot_results(self, results: Dict):
        """Plot backtest results"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot equity curve
            equity_df = results['equity_curve']
            ax1.plot(equity_df['datetime'], equity_df['equity'], label='Equity', linewidth=2)
            ax1.axhline(y=self.initial_balance, color='r', linestyle='--', alpha=0.7, label='Initial Balance')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Balance ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot price with trades
            ax2.plot(equity_df['datetime'], equity_df['price'], label='SOL Price', alpha=0.7)
            
            trades_df = results['trades']
            
            # Mark entry points
            long_entries = trades_df[trades_df['side'] == 'long']
            short_entries = trades_df[trades_df['side'] == 'short']
            
            if not long_entries.empty:
                ax2.scatter(long_entries['entry_time'], long_entries['entry_price'], 
                           color='green', marker='^', s=50, label='Long Entry', zorder=5)
            
            if not short_entries.empty:
                ax2.scatter(short_entries['entry_time'], short_entries['entry_price'], 
                           color='red', marker='v', s=50, label='Short Entry', zorder=5)
            
            ax2.set_title('Price Chart with Trade Entries')
            ax2.set_ylabel('Price ($)')
            ax2.set_xlabel('Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Chart saved as 'backtest_results.png'")
            
        except ImportError:
            print("Matplotlib not available for plotting")

def main():
    """Run backtest"""
    print("OKX SOL-USD Perpetual Strategy Backtest")
    print("="*50)
    
    # Initialize backtester
    backtester = Backtester(initial_balance=10000)
    
    # Get historical data
    df = backtester.get_historical_data(days=7)  # Test with 7 days
    
    if df.empty:
        print("Failed to get historical data")
        return
    
    # Run backtest
    results = backtester.run_backtest(df)
    
    if 'error' in results:
        print(f"Backtest error: {results['error']}")
        return
    
    # Print and plot results
    backtester.print_results(results)
    backtester.plot_results(results)
    
    # Additional analysis
    if results['win_rate'] >= 80:
        print(f"\n✅ TARGET ACHIEVED: {results['win_rate']:.1f}% win rate (target: 80%+)")
    else:
        print(f"\n❌ TARGET MISSED: {results['win_rate']:.1f}% win rate (target: 80%+)")
    
    print(f"\nStrategy generated {results['total_trades']} trades over {len(df)} candles")
    print(f"Trade frequency: {(results['total_trades'] / len(df)) * 100:.2f}% of candles")

if __name__ == "__main__":
    main() 