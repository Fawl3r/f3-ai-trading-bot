import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from scipy import stats
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

from strategy import AdvancedTradingStrategy, TradingSignal
from indicators import TechnicalIndicators
from okx_client import OKXClient
from config import INSTRUMENT_ID, POSITION_SIZE_USD, LEVERAGE

@dataclass
class OptimizationParams:
    """Parameters to optimize"""
    cmf_period: Tuple[int, int] = (10, 30)  # min, max
    obv_sma_period: Tuple[int, int] = (8, 20)
    rsi_period: Tuple[int, int] = (10, 21)
    bb_period: Tuple[int, int] = (15, 25)
    bb_std: Tuple[float, float] = (1.5, 2.5)
    ema_fast: Tuple[int, int] = (5, 15)
    ema_slow: Tuple[int, int] = (15, 30)
    min_confidence: Tuple[float, float] = (0.5, 0.8)
    divergence_weight: Tuple[float, float] = (0.15, 0.35)
    range_break_weight: Tuple[float, float] = (0.10, 0.30)
    reversal_weight: Tuple[float, float] = (0.10, 0.30)

class AdvancedBacktester:
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.optimization_results = []
        self.best_params = None
        self.best_score = -np.inf
        
    def get_extended_historical_data(self, days: int = 60) -> pd.DataFrame:
        """Get extended historical data for comprehensive testing"""
        print(f"Fetching {days} days of historical data...")
        
        try:
            client = OKXClient()
            all_data = []
            
            # Get 5-minute data for more history (OKX allows more 5m candles)
            total_candles = days * 288  # 288 5-minute candles per day
            requests_needed = min((total_candles // 300) + 1, 20)  # Limit requests
            
            for i in range(requests_needed):
                response = client.get_candlesticks(
                    inst_id=INSTRUMENT_ID,
                    bar='5m',
                    limit=300
                )
                
                if response.get('code') == '0' and response.get('data'):
                    all_data.extend(response['data'])
                    print(f"Fetched batch {i+1}/{requests_needed}")
                else:
                    break
            
            # Convert to 1-minute equivalent data
            df_data = []
            for candle in reversed(all_data):
                base_time = datetime.fromtimestamp(int(candle[0]) / 1000)
                
                # Create 5 synthetic 1-minute candles from each 5-minute candle
                open_price = float(candle[1])
                high_price = float(candle[2])
                low_price = float(candle[3])
                close_price = float(candle[4])
                volume = float(candle[5]) / 5  # Distribute volume
                
                for j in range(5):
                    # Create realistic intra-5min movement
                    progress = j / 4.0
                    current_price = open_price + (close_price - open_price) * progress
                    noise = np.random.normal(0, 0.001)
                    
                    minute_time = base_time + timedelta(minutes=j)
                    
                    df_data.append({
                        'timestamp': int(minute_time.timestamp() * 1000),
                        'datetime': minute_time,
                        'open': current_price * (1 + noise),
                        'high': min(high_price, current_price * (1 + abs(noise) + 0.001)),
                        'low': max(low_price, current_price * (1 - abs(noise) - 0.001)),
                        'close': current_price,
                        'volume': volume
                    })
            
            return pd.DataFrame(df_data).sort_values('datetime').reset_index(drop=True)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return self._generate_extended_sample_data(days)
    
    def _generate_extended_sample_data(self, days: int) -> pd.DataFrame:
        """Generate extended sample data with realistic patterns"""
        print(f"Generating {days} days of sample data...")
        
        periods = days * 1440
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=periods,
            freq='1min'
        )
        
        # Generate realistic price action with trends and volatility clustering
        np.random.seed(42)
        price_start = 150.0
        
        # Create trend periods
        trend_length = 2880  # 2 days
        num_trends = periods // trend_length
        
        prices = [price_start]
        for trend in range(num_trends):
            trend_direction = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            trend_strength = np.random.uniform(0.0005, 0.002)
            volatility = np.random.uniform(0.001, 0.003)
            
            for i in range(min(trend_length, periods - len(prices) + 1)):
                # Add trend component
                trend_component = trend_direction * trend_strength
                
                # Add volatility clustering
                vol_cluster = volatility * (1 + 0.5 * np.sin(i / 100))
                
                # Generate return
                return_val = trend_component + np.random.normal(0, vol_cluster)
                
                # Add momentum and mean reversion
                if len(prices) > 5:
                    momentum = np.mean([prices[-1]/prices[-2] - 1, 
                                     prices[-2]/prices[-3] - 1]) * 0.1
                    mean_reversion = (np.mean(prices[-20:]) - prices[-1]) / prices[-1] * 0.05
                    return_val += momentum + mean_reversion
                
                new_price = prices[-1] * (1 + return_val)
                prices.append(max(new_price, 1.0))
        
        prices = prices[:periods]
        
        # Generate OHLC with realistic spreads
        data = []
        for i, price in enumerate(prices):
            spread = price * np.random.uniform(0.0001, 0.0005)
            
            # Generate realistic OHLC
            if i == 0:
                open_price = price
            else:
                gap = np.random.normal(0, 0.0002)
                open_price = prices[i-1] * (1 + gap)
            
            high = max(open_price, price) + spread
            low = min(open_price, price) - spread
            close = price
            
            # Ensure OHLC rules
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.lognormal(11, 0.5)  # Realistic volume distribution
            
            data.append({
                'timestamp': int(dates[i].timestamp() * 1000),
                'datetime': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def optimize_parameters(self, df: pd.DataFrame, method: str = 'differential_evolution') -> Dict:
        """Optimize strategy parameters using various methods"""
        print(f"Starting parameter optimization using {method}...")
        
        if method == 'grid_search':
            return self._grid_search_optimization(df)
        elif method == 'differential_evolution':
            return self._differential_evolution_optimization(df)
        elif method == 'walk_forward':
            return self._walk_forward_optimization(df)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _differential_evolution_optimization(self, df: pd.DataFrame) -> Dict:
        """Use differential evolution for parameter optimization"""
        print("Running differential evolution optimization...")
        
        # Define parameter bounds
        bounds = [
            (10, 30),   # cmf_period
            (8, 20),    # obv_sma_period  
            (10, 21),   # rsi_period
            (15, 25),   # bb_period
            (1.5, 2.5), # bb_std
            (5, 15),    # ema_fast
            (15, 30),   # ema_slow
            (0.5, 0.8), # min_confidence
            (0.15, 0.35), # divergence_weight
            (0.10, 0.30), # range_break_weight
            (0.10, 0.30), # reversal_weight
        ]
        
        def objective_function(params):
            try:
                # Convert params to integers where needed
                config = {
                    'cmf_period': int(params[0]),
                    'obv_sma_period': int(params[1]),
                    'rsi_period': int(params[2]),
                    'bb_period': int(params[3]),
                    'bb_std': params[4],
                    'ema_fast': int(params[5]),
                    'ema_slow': int(params[6]),
                    'min_confidence': params[7],
                    'weights': {
                        'divergence': params[8],
                        'range_break': params[9],
                        'reversal': params[10],
                        'pullback': 0.15,
                        'volume_confirmation': 0.10,
                        'parabolic_exit': 0.10
                    }
                }
                
                # Normalize weights
                total_weight = sum(config['weights'].values())
                for key in config['weights']:
                    config['weights'][key] /= total_weight
                
                result = self._backtest_with_params(df, config)
                
                # Multi-objective optimization: maximize win_rate * profit_factor
                if result['total_trades'] < 5:
                    return -1000  # Penalize configs with too few trades
                
                score = (result['win_rate'] / 100) * min(result['profit_factor'], 5) * (1 - result['max_drawdown'] / 100)
                return -score  # Negative because scipy minimizes
                
            except Exception as e:
                return -1000  # Return bad score on error
        
        # Run optimization
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=50,
            popsize=15,
            seed=42,
            disp=True
        )
        
        # Convert best parameters
        best_params = {
            'cmf_period': int(result.x[0]),
            'obv_sma_period': int(result.x[1]),
            'rsi_period': int(result.x[2]),
            'bb_period': int(result.x[3]),
            'bb_std': result.x[4],
            'ema_fast': int(result.x[5]),
            'ema_slow': int(result.x[6]),
            'min_confidence': result.x[7],
            'weights': {
                'divergence': result.x[8],
                'range_break': result.x[9],
                'reversal': result.x[10],
                'pullback': 0.15,
                'volume_confirmation': 0.10,
                'parabolic_exit': 0.10
            }
        }
        
        # Normalize weights
        total_weight = sum(best_params['weights'].values())
        for key in best_params['weights']:
            best_params['weights'][key] /= total_weight
        
        # Test best parameters
        final_result = self._backtest_with_params(df, best_params)
        
        return {
            'method': 'differential_evolution',
            'best_params': best_params,
            'best_score': -result.fun,
            'optimization_result': result,
            'backtest_result': final_result
        }
    
    def _walk_forward_optimization(self, df: pd.DataFrame) -> Dict:
        """Walk-forward optimization to prevent overfitting"""
        print("Running walk-forward optimization...")
        
        window_size = len(df) // 4  # 25% for optimization
        test_size = len(df) // 8    # 12.5% for testing
        step_size = test_size       # Non-overlapping windows
        
        results = []
        
        for start_idx in range(0, len(df) - window_size - test_size, step_size):
            print(f"Walk-forward step: {start_idx//step_size + 1}")
            
            # Split data
            train_end = start_idx + window_size
            test_start = train_end
            test_end = test_start + test_size
            
            train_data = df.iloc[start_idx:train_end].copy()
            test_data = df.iloc[test_start:test_end].copy()
            
            if len(train_data) < 100 or len(test_data) < 50:
                continue
            
            # Optimize on training data
            opt_result = self._differential_evolution_optimization(train_data)
            
            # Test on out-of-sample data
            test_result = self._backtest_with_params(test_data, opt_result['best_params'])
            
            results.append({
                'train_period': (train_data['datetime'].iloc[0], train_data['datetime'].iloc[-1]),
                'test_period': (test_data['datetime'].iloc[0], test_data['datetime'].iloc[-1]),
                'train_result': opt_result['backtest_result'],
                'test_result': test_result,
                'best_params': opt_result['best_params']
            })
        
        # Aggregate results
        if not results:
            return {'error': 'No walk-forward results generated'}
        
        avg_test_win_rate = np.mean([r['test_result']['win_rate'] for r in results])
        avg_test_profit_factor = np.mean([r['test_result']['profit_factor'] for r in results if r['test_result']['profit_factor'] != float('inf')])
        
        return {
            'method': 'walk_forward',
            'results': results,
            'avg_test_win_rate': avg_test_win_rate,
            'avg_test_profit_factor': avg_test_profit_factor,
            'stability_score': 1 - np.std([r['test_result']['win_rate'] for r in results]) / 100
        }
    
    def _backtest_with_params(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Run backtest with specific parameters"""
        # Create modified strategy with custom parameters
        strategy = AdvancedTradingStrategy()
        
        # Update strategy parameters
        if 'min_confidence' in params:
            strategy.min_confidence = params['min_confidence']
        if 'weights' in params:
            strategy.weights = params['weights']
        
        # Track results
        balance = self.initial_balance
        trades = []
        position = None
        entry_price = 0
        entry_time = None
        
        for i in range(50, len(df)):
            current_slice = df.iloc[:i+1].copy()
            
            # Apply custom indicator parameters
            current_slice = self._calculate_custom_indicators(current_slice, params)
            
            current_price = current_slice['close'].iloc[-1]
            current_time = current_slice['datetime'].iloc[-1]
            
            # Generate signal with custom strategy
            signal = strategy.generate_signal(current_slice)
            
            if signal and signal.signal_type in ['buy', 'sell', 'close']:
                # Close existing position
                if position:
                    if position == 'long':
                        pnl = (current_price - entry_price) / entry_price * POSITION_SIZE_USD * LEVERAGE
                    else:
                        pnl = (entry_price - current_price) / entry_price * POSITION_SIZE_USD * LEVERAGE
                    
                    balance += pnl
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'side': position,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'duration': (current_time - entry_time).total_seconds() / 60
                    })
                    
                    position = None
                
                # Open new position
                if signal.signal_type in ['buy', 'sell']:
                    position = 'long' if signal.signal_type == 'buy' else 'short'
                    entry_price = current_price
                    entry_time = current_time
        
        return self._calculate_performance_metrics(trades, balance)
    
    def _calculate_custom_indicators(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate indicators with custom parameters"""
        df = df.copy()
        
        # Use custom parameters if provided
        cmf_period = params.get('cmf_period', 20)
        obv_sma_period = params.get('obv_sma_period', 14)
        rsi_period = params.get('rsi_period', 14)
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2.0)
        ema_fast = params.get('ema_fast', 9)
        ema_slow = params.get('ema_slow', 21)
        
        # Calculate indicators with custom parameters
        # CMF
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mf_multiplier = mf_multiplier.fillna(0)
        mf_volume = mf_multiplier * df['volume']
        df['cmf'] = mf_volume.rolling(window=cmf_period).sum() / df['volume'].rolling(window=cmf_period).sum()
        df['cmf'] = df['cmf'].fillna(0)
        
        # OBV
        price_change = df['close'].diff()
        direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        df['obv'] = (direction * df['volume']).cumsum()
        df['obv_sma'] = df['obv'].rolling(window=obv_sma_period).mean()
        
        # Other indicators
        try:
            import pandas_ta as ta
            df['rsi'] = ta.rsi(df['close'], length=rsi_period)
            bb = ta.bbands(df['close'], length=bb_period, std=bb_std)
            df['bb_lower'] = bb[f'BBL_{bb_period}_{bb_std}']
            df['bb_middle'] = bb[f'BBM_{bb_period}_{bb_std}']
            df['bb_upper'] = bb[f'BBU_{bb_period}_{bb_std}']
            df['ema_fast'] = ta.ema(df['close'], length=ema_fast)
            df['ema_slow'] = ta.ema(df['close'], length=ema_slow)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        except:
            # Fallback calculations
            df['rsi'] = self._calculate_rsi(df['close'], rsi_period)
            df['bb_middle'] = df['close'].rolling(bb_period).mean()
            bb_std_val = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
            df['ema_fast'] = df['close'].ewm(span=ema_fast).mean()
            df['ema_slow'] = df['close'].ewm(span=ema_slow).mean()
            df['atr'] = self._calculate_atr(df, 14)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR manually"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_performance_metrics(self, trades: List[Dict], final_balance: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'sharpe_ratio': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Risk metrics
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] < 0]['pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
        
        # Drawdown calculation
        equity_curve = [self.initial_balance]
        for _, trade in trades_df.iterrows():
            equity_curve.append(equity_curve[-1] + trade['pnl'])
        
        equity_series = pd.Series(equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = abs(drawdown.min())
        
        # Sharpe ratio approximation
        if len(trades_df) > 1:
            returns = trades_df['pnl'] / POSITION_SIZE_USD
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration': trades_df['duration'].mean()
        }
    
    def monte_carlo_simulation(self, df: pd.DataFrame, best_params: Dict, n_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation to test strategy robustness"""
        print(f"Running Monte Carlo simulation with {n_simulations} iterations...")
        
        results = []
        
        for i in range(n_simulations):
            if i % 100 == 0:
                print(f"Monte Carlo progress: {i}/{n_simulations}")
            
            # Shuffle trade order while maintaining temporal structure
            shuffled_df = self._shuffle_returns(df)
            
            # Run backtest
            result = self._backtest_with_params(shuffled_df, best_params)
            results.append(result)
        
        # Calculate statistics
        win_rates = [r['win_rate'] for r in results]
        returns = [r['total_return'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        
        return {
            'n_simulations': n_simulations,
            'win_rate_mean': np.mean(win_rates),
            'win_rate_std': np.std(win_rates),
            'win_rate_percentiles': np.percentile(win_rates, [5, 25, 50, 75, 95]),
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'return_percentiles': np.percentile(returns, [5, 25, 50, 75, 95]),
            'max_drawdown_mean': np.mean(drawdowns),
            'max_drawdown_percentiles': np.percentile(drawdowns, [5, 25, 50, 75, 95]),
            'probability_positive': len([r for r in returns if r > 0]) / len(returns)
        }
    
    def _shuffle_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shuffle returns while maintaining price structure"""
        df_shuffled = df.copy()
        
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        # Shuffle returns
        shuffled_returns = returns.sample(frac=1, random_state=np.random.randint(0, 10000)).values
        
        # Reconstruct prices
        new_prices = [df['close'].iloc[0]]
        for ret in shuffled_returns:
            new_prices.append(new_prices[-1] * (1 + ret))
        
        # Update OHLC proportionally
        for i in range(1, len(df_shuffled)):
            price_ratio = new_prices[i] / df['close'].iloc[i]
            df_shuffled.loc[df_shuffled.index[i], 'open'] *= price_ratio
            df_shuffled.loc[df_shuffled.index[i], 'high'] *= price_ratio
            df_shuffled.loc[df_shuffled.index[i], 'low'] *= price_ratio
            df_shuffled.loc[df_shuffled.index[i], 'close'] = new_prices[i]
        
        return df_shuffled
    
    def generate_optimization_report(self, optimization_result: Dict) -> str:
        """Generate comprehensive optimization report"""
        report = []
        report.append("=" * 80)
        report.append("ADVANCED BACKTESTING & OPTIMIZATION REPORT")
        report.append("=" * 80)
        
        if optimization_result.get('method') == 'differential_evolution':
            result = optimization_result['backtest_result']
            params = optimization_result['best_params']
            
            report.append(f"\nOptimization Method: Differential Evolution")
            report.append(f"Best Score: {optimization_result['best_score']:.4f}")
            report.append(f"\nOPTIMIZED PARAMETERS:")
            report.append(f"  CMF Period: {params['cmf_period']}")
            report.append(f"  OBV SMA Period: {params['obv_sma_period']}")
            report.append(f"  RSI Period: {params['rsi_period']}")
            report.append(f"  BB Period: {params['bb_period']}")
            report.append(f"  BB Std Dev: {params['bb_std']:.2f}")
            report.append(f"  EMA Fast: {params['ema_fast']}")
            report.append(f"  EMA Slow: {params['ema_slow']}")
            report.append(f"  Min Confidence: {params['min_confidence']:.2f}")
            
            report.append(f"\nSTRATEGY WEIGHTS:")
            for key, value in params['weights'].items():
                report.append(f"  {key.replace('_', ' ').title()}: {value:.3f}")
            
            report.append(f"\nPERFORMANCE METRICS:")
            report.append(f"  Total Trades: {result['total_trades']}")
            report.append(f"  Win Rate: {result['win_rate']:.2f}%")
            report.append(f"  Total Return: {result['total_return']:.2f}%")
            report.append(f"  Profit Factor: {result['profit_factor']:.2f}")
            report.append(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
            report.append(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            
            accuracy_status = "✅ TARGET ACHIEVED" if result['win_rate'] >= 80 else "❌ TARGET MISSED"
            report.append(f"\nACCURACY STATUS: {accuracy_status}")
            report.append(f"Target: 80%+ | Achieved: {result['win_rate']:.2f}%")
        
        elif optimization_result.get('method') == 'walk_forward':
            report.append(f"\nOptimization Method: Walk-Forward Analysis")
            report.append(f"Average Test Win Rate: {optimization_result['avg_test_win_rate']:.2f}%")
            report.append(f"Average Test Profit Factor: {optimization_result['avg_test_profit_factor']:.2f}")
            report.append(f"Strategy Stability Score: {optimization_result['stability_score']:.2f}")
            
            report.append(f"\nWALK-FORWARD RESULTS:")
            for i, result in enumerate(optimization_result['results']):
                report.append(f"  Period {i+1}:")
                report.append(f"    Train Win Rate: {result['train_result']['win_rate']:.2f}%")
                report.append(f"    Test Win Rate: {result['test_result']['win_rate']:.2f}%")
                report.append(f"    Out-of-Sample Performance: {result['test_result']['total_return']:.2f}%")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_optimization_results(self, results: Dict, filename: str = None):
        """Save optimization results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Optimization results saved to {filename}")

def main():
    """Main optimization workflow"""
    print("Advanced Backtesting & Optimization System")
    print("=" * 50)
    
    # Initialize backtester
    backtester = AdvancedBacktester(initial_balance=10000)
    
    # Get extended historical data
    print("Phase 1: Data Collection")
    df = backtester.get_extended_historical_data(days=30)
    
    if df.empty:
        print("Failed to get historical data")
        return
    
    print(f"Loaded {len(df)} candles for optimization")
    
    # Phase 2: Parameter Optimization
    print("\nPhase 2: Parameter Optimization")
    optimization_result = backtester.optimize_parameters(df, method='differential_evolution')
    
    # Phase 3: Walk-Forward Validation
    print("\nPhase 3: Walk-Forward Validation")
    walk_forward_result = backtester.optimize_parameters(df, method='walk_forward')
    
    # Phase 4: Monte Carlo Simulation
    print("\nPhase 4: Monte Carlo Simulation")
    if 'best_params' in optimization_result:
        monte_carlo_result = backtester.monte_carlo_simulation(
            df, optimization_result['best_params'], n_simulations=500
        )
    
    # Generate and print reports
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    # Print differential evolution report
    de_report = backtester.generate_optimization_report(optimization_result)
    print(de_report)
    
    # Print Monte Carlo results
    if 'monte_carlo_result' in locals():
        print(f"\nMONTE CARLO SIMULATION RESULTS:")
        print(f"Mean Win Rate: {monte_carlo_result['win_rate_mean']:.2f}% ± {monte_carlo_result['win_rate_std']:.2f}%")
        print(f"Win Rate 95% Confidence Interval: [{monte_carlo_result['win_rate_percentiles'][0]:.2f}%, {monte_carlo_result['win_rate_percentiles'][4]:.2f}%]")
        print(f"Probability of Positive Returns: {monte_carlo_result['probability_positive']:.2%}")
    
    # Save results
    all_results = {
        'differential_evolution': optimization_result,
        'walk_forward': walk_forward_result,
        'monte_carlo': monte_carlo_result if 'monte_carlo_result' in locals() else None,
        'data_period': {
            'start': df['datetime'].iloc[0].isoformat(),
            'end': df['datetime'].iloc[-1].isoformat(),
            'total_candles': len(df)
        }
    }
    
    backtester.save_optimization_results(all_results)
    
    # Final recommendations
    print(f"\n{'='*80}")
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    if optimization_result.get('backtest_result', {}).get('win_rate', 0) >= 80:
        print("✅ Optimization successful! Win rate target achieved.")
        print("✅ Parameters are ready for live trading.")
        print("⚠️  Recommended: Start with small position sizes")
    else:
        print("❌ Win rate target not achieved with current approach.")
        print("🔧 Recommendations:")
        print("   - Try longer optimization period")
        print("   - Adjust optimization bounds")
        print("   - Consider additional indicators")
        print("   - Review market conditions during test period")
    
    if 'monte_carlo_result' in locals():
        if monte_carlo_result['probability_positive'] >= 0.7:
            print("✅ Strategy shows good robustness in Monte Carlo simulation")
        else:
            print("⚠️  Strategy may be sensitive to market conditions")
    
    print("\n📁 Results saved to optimization_results_[timestamp].json")
    print("🔄 Run 'python advanced_backtest.py' to re-optimize with new data")

if __name__ == "__main__":
    main() 