#!/usr/bin/env python3
"""
Balanced 75% Win Rate Backtest
Optimized for 75-85% win rate with realistic trade frequency
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Balanced75AI:
    """Balanced AI for 75% win rate target"""
    
    def extract_features(self, data: pd.DataFrame) -> dict:
        """Extract comprehensive features"""
        if len(data) < 100:
            return {}
        
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            features = {}
            features['current_price'] = close[-1]
            
            # Range analysis (multiple timeframes)
            range_positions = []
            for period in [20, 40, 60]:
                if len(high) >= period:
                    recent_high = np.max(high[-period:])
                    recent_low = np.min(low[-period:])
                    range_size = recent_high - recent_low
                    
                    if range_size > 0:
                        range_pos = (close[-1] - recent_low) / range_size * 100
                        range_positions.append(range_pos)
            
            features['range_position'] = np.mean(range_positions) if range_positions else 50
            features['range_consensus'] = len([r for r in range_positions if r > 85 or r < 15])
            features['extreme_range'] = any(r > 88 or r < 12 for r in range_positions) if range_positions else False
            
            # RSI analysis
            features['rsi_14'] = self._calculate_rsi(close, 14)
            features['rsi_extreme'] = (features['rsi_14'] > 80 or features['rsi_14'] < 20)
            
            # Volume analysis
            if len(volume) >= 30:
                vol_sma_20 = np.mean(volume[-20:])
                features['volume_ratio'] = volume[-1] / vol_sma_20
                features['high_volume'] = volume[-1] > vol_sma_20 * 2.0
            else:
                features['volume_ratio'] = 1.0
                features['high_volume'] = False
            
            # Moving averages
            ma_periods = [5, 10, 20, 50]
            ma_values = []
            
            for period in ma_periods:
                if len(close) >= period:
                    ma = np.mean(close[-period:])
                    ma_values.append(ma)
            
            # MA alignment
            if len(ma_values) >= 4:
                ascending = all(ma_values[i] <= ma_values[i+1] for i in range(len(ma_values)-1))
                descending = all(ma_values[i] >= ma_values[i+1] for i in range(len(ma_values)-1))
                features['ma_alignment'] = ascending or descending
                features['trend_strength'] = 100 if ascending else -100 if descending else 0
            else:
                features['ma_alignment'] = False
                features['trend_strength'] = 0
            
            # Momentum
            if len(close) >= 20:
                momentum_5 = (close[-1] - close[-6]) / close[-6] * 100
                momentum_10 = (close[-1] - close[-11]) / close[-11] * 100
                features['momentum_exhaustion'] = (abs(momentum_5) > 6 and abs(momentum_10) < 2)
            else:
                features['momentum_exhaustion'] = False
            
            # Support/Resistance
            features['support_strength'] = self._calculate_support_strength(low, close[-1])
            features['resistance_strength'] = self._calculate_resistance_strength(high, close[-1])
            
            return features
            
        except Exception as e:
            return {}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_support_strength(self, lows: np.ndarray, current_price: float) -> float:
        """Calculate support strength"""
        if len(lows) < 30:
            return 0
        
        support_levels = []
        for i in range(len(lows) - 8):
            window = lows[i:i+8]
            if lows[i+4] == np.min(window):
                support_levels.append(lows[i+4])
        
        if not support_levels:
            return 0
        
        distances = [abs(current_price - level) / current_price for level in support_levels]
        closest_distance = min(distances) if distances else 1.0
        
        return max(0, 100 * (1 - closest_distance * 15))
    
    def _calculate_resistance_strength(self, highs: np.ndarray, current_price: float) -> float:
        """Calculate resistance strength"""
        if len(highs) < 30:
            return 0
        
        resistance_levels = []
        for i in range(len(highs) - 8):
            window = highs[i:i+8]
            if highs[i+4] == np.max(window):
                resistance_levels.append(highs[i+4])
        
        if not resistance_levels:
            return 0
        
        distances = [abs(current_price - level) / current_price for level in resistance_levels]
        closest_distance = min(distances) if distances else 1.0
        
        return max(0, 100 * (1 - closest_distance * 15))
    
    def detect_75_signals(self, features: dict) -> dict:
        """Detect signals for 75% win rate target"""
        if not features:
            return {'confidence': 0, 'direction': None, 'signals': [], 'quality_score': 0}
        
        signals = []
        confidence = 0
        direction = None
        quality_score = 0
        
        range_pos = features.get('range_position', 50)
        rsi_14 = features.get('rsi_14', 50)
        
        # STRICT SHORT CONDITIONS (but achievable)
        if range_pos > 85:  # High but not ultra-extreme
            short_signals = []
            short_confidence = 0
            
            # Strong RSI
            if rsi_14 > 80:
                short_signals.append("Strong RSI (>80)")
                short_confidence += 20
            
            # Range consensus
            if features.get('range_consensus', 0) >= 2:
                short_signals.append("Range Consensus")
                short_confidence += 15
            
            # High volume
            if features.get('high_volume', False):
                short_signals.append("High Volume")
                short_confidence += 15
            
            # Momentum exhaustion
            if features.get('momentum_exhaustion', False):
                short_signals.append("Momentum Exhaustion")
                short_confidence += 15
            
            # Resistance
            if features.get('resistance_strength', 0) > 60:
                short_signals.append("Strong Resistance")
                short_confidence += 15
            
            # MA alignment
            if features.get('ma_alignment', False) and features.get('trend_strength', 0) > 30:
                short_signals.append("Counter-Trend Setup")
                short_confidence += 10
            
            # Extreme range position
            if features.get('extreme_range', False):
                short_signals.append("Extreme Range Position")
                short_confidence += 10
            
            if short_confidence >= 65 and len(short_signals) >= 3:  # Balanced requirements
                direction = 'short'
                confidence = short_confidence
                signals = short_signals
        
        # STRICT LONG CONDITIONS (but achievable)
        elif range_pos < 15:  # Low but not ultra-extreme
            long_signals = []
            long_confidence = 0
            
            # Strong RSI
            if rsi_14 < 20:
                long_signals.append("Strong RSI (<20)")
                long_confidence += 20
            
            # Range consensus
            if features.get('range_consensus', 0) >= 2:
                long_signals.append("Range Consensus")
                long_confidence += 15
            
            # High volume
            if features.get('high_volume', False):
                long_signals.append("High Volume")
                long_confidence += 15
            
            # Momentum exhaustion
            if features.get('momentum_exhaustion', False):
                long_signals.append("Momentum Exhaustion")
                long_confidence += 15
            
            # Support
            if features.get('support_strength', 0) > 60:
                long_signals.append("Strong Support")
                long_confidence += 15
            
            # MA alignment
            if features.get('ma_alignment', False) and features.get('trend_strength', 0) < -30:
                long_signals.append("Counter-Trend Setup")
                long_confidence += 10
            
            # Extreme range position
            if features.get('extreme_range', False):
                long_signals.append("Extreme Range Position")
                long_confidence += 10
            
            if long_confidence >= 65 and len(long_signals) >= 3:  # Balanced requirements
                direction = 'long'
                confidence = long_confidence
                signals = long_signals
        
        # Quality scoring
        if direction:
            quality_factors = 0
            
            if features.get('extreme_range', False):
                quality_factors += 20
            
            if features.get('rsi_extreme', False):
                quality_factors += 20
            
            if features.get('high_volume', False):
                quality_factors += 15
            
            support_res = max(features.get('support_strength', 0), features.get('resistance_strength', 0))
            if support_res > 70:
                quality_factors += 20
            
            if features.get('ma_alignment', False):
                quality_factors += 15
            
            if features.get('momentum_exhaustion', False):
                quality_factors += 10
            
            quality_score = min(100, quality_factors)
        
        return {
            'confidence': confidence,
            'direction': direction,
            'signals': signals,
            'quality_score': quality_score,
            'is_extreme': features.get('extreme_range', False)
        }

class Balanced75Backtest:
    """Balanced backtest for 75% win rate"""
    
    def __init__(self):
        self.ai = Balanced75AI()
        
        # Balanced configuration for 75% win rate
        self.config = {
            "initial_balance": 1000.0,
            "base_position_size": 100.0,
            "max_position_size": 200.0,
            "profit_targets": {
                "conservative": 120,
                "aggressive": 160,
                "maximum": 200
            },
            "leverage": 6,
            "min_confidence": 75,      # Balanced
            "min_quality_score": 60,   # Balanced
            "max_daily_trades": 2,     # Slightly more trades
            "risk_per_trade": 0.08,
            "stop_loss_pct": 2.5,      # Balanced
            "max_hold_hours": 8,       # Balanced
        }
        
        self.reset_stats()
    
    def reset_stats(self):
        """Reset all statistics"""
        self.balance = self.config["initial_balance"]
        self.trades = []
        self.daily_trades = {}
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0
    
    def generate_realistic_data(self, days: int = 60) -> pd.DataFrame:
        """Generate realistic market data"""
        print(f"🔄 Generating {days} days of realistic market data...")
        
        # Start with realistic SOL price
        start_price = 140.0
        minutes_per_day = 1440
        total_minutes = days * minutes_per_day
        
        # Market characteristics
        daily_volatility = 0.06  # 6% daily volatility
        trend_strength = 0.0001  # Slight upward bias
        noise_factor = 0.002     # 0.2% noise per minute
        
        # Generate base price movement
        returns = np.random.normal(trend_strength, noise_factor, total_minutes)
        
        # Add realistic market cycles
        cycle_length = 1440  # Daily cycle
        for i in range(total_minutes):
            # Add daily volatility cycle
            cycle_position = (i % cycle_length) / cycle_length
            volatility_multiplier = 1 + 0.4 * np.sin(2 * np.pi * cycle_position)
            returns[i] *= volatility_multiplier
            
            # Add weekly trends
            week_position = (i / (cycle_length * 7)) % 1
            trend_multiplier = 1 + 0.2 * np.sin(2 * np.pi * week_position)
            returns[i] *= trend_multiplier
        
        # Generate prices
        prices = [start_price]
        for return_val in returns:
            new_price = prices[-1] * (1 + return_val)
            prices.append(max(new_price, 80.0))  # Floor price
        
        prices = prices[1:]  # Remove initial price
        
        # Generate realistic OHLCV data
        data = []
        for i, close_price in enumerate(prices):
            # Generate realistic high/low based on volatility
            volatility = abs(returns[i]) * 1.5
            high = close_price * (1 + volatility * np.random.uniform(0.3, 1.2))
            low = close_price * (1 - volatility * np.random.uniform(0.3, 1.2))
            
            # Open price based on previous close
            if i == 0:
                open_price = start_price
            else:
                open_price = prices[i-1] * (1 + np.random.normal(0, 0.0008))
            
            # Ensure OHLC logic
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate realistic volume
            base_volume = 800000
            volume_multiplier = 1 + abs(returns[i]) * 8  # Higher volume on big moves
            volume = base_volume * volume_multiplier * np.random.uniform(0.6, 1.8)
            
            data.append({
                'timestamp': i,
                'open': round(open_price, 4),
                'high': round(high, 4),
                'low': round(low, 4),
                'close': round(close_price, 4),
                'volume': int(volume)
            })
        
        df = pd.DataFrame(data)
        
        print(f"✅ Generated {len(df)} minutes of data")
        print(f"📊 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"📈 Total return: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.1f}%")
        
        return df
    
    def should_enter_trade(self, signals: dict, features: dict, current_day: int) -> bool:
        """Balanced entry conditions for 75% win rate"""
        # Must have good confidence
        if signals['confidence'] < self.config['min_confidence']:
            return False
        
        # Must have good quality score
        if signals.get('quality_score', 0) < self.config['min_quality_score']:
            return False
        
        # Must have clear direction
        if not signals['direction']:
            return False
        
        # Must have at least 3 signals
        if len(signals.get('signals', [])) < 3:
            return False
        
        # Check daily trade limit
        if self.daily_trades.get(current_day, 0) >= self.config['max_daily_trades']:
            return False
        
        # Balanced confirmations (need at least 3 of 4)
        confirmations = 0
        
        # 1. Strong RSI
        rsi_14 = features.get('rsi_14', 50)
        if ((signals['direction'] == 'short' and rsi_14 > 80) or 
            (signals['direction'] == 'long' and rsi_14 < 20)):
            confirmations += 1
        
        # 2. High volume
        if features.get('high_volume', False):
            confirmations += 1
        
        # 3. Range consensus
        if features.get('range_consensus', 0) >= 2:
            confirmations += 1
        
        # 4. Support/resistance
        if ((signals['direction'] == 'long' and features.get('support_strength', 0) > 60) or
            (signals['direction'] == 'short' and features.get('resistance_strength', 0) > 60)):
            confirmations += 1
        
        return confirmations >= 3
    
    def run_balanced_backtest(self, data: pd.DataFrame) -> dict:
        """Run balanced backtest for 75% win rate"""
        print("\n🎯 Running Balanced 75% Win Rate Backtest...")
        print("🏆 Target: 75-85% Win Rate")
        print("=" * 60)
        
        self.reset_stats()
        
        # Skip first 100 candles for proper analysis
        for i in range(100, len(data) - 300):  # Leave room for trade simulation
            current_day = i // 1440  # Minutes per day
            
            # Get analysis window
            analysis_data = data.iloc[i-99:i+1]
            features = self.ai.extract_features(analysis_data)
            
            if not features:
                continue
            
            signals = self.ai.detect_75_signals(features)
            
            # Check for entry
            if self.should_enter_trade(signals, features, current_day):
                
                # Calculate position details
                confidence_factor = min(signals['confidence'] / 100, 0.8)
                quality_factor = min(signals.get('quality_score', 50) / 100, 0.6)
                
                base_size = self.config['base_position_size']
                max_size = self.config['max_position_size']
                
                size_multiplier = (confidence_factor + quality_factor) / 2
                position_size = base_size + (max_size - base_size) * size_multiplier
                position_size = min(position_size, self.balance * self.config['risk_per_trade'])
                
                # Profit targets
                if signals['confidence'] >= 90 and signals.get('quality_score', 0) >= 80:
                    target_profit = self.config['profit_targets']['maximum']
                elif signals['confidence'] >= 80 and signals.get('quality_score', 0) >= 70:
                    target_profit = self.config['profit_targets']['aggressive']
                else:
                    target_profit = self.config['profit_targets']['conservative']
                
                # Simulate trade
                entry_price = data.iloc[i]['close']
                direction = signals['direction']
                
                max_hold_minutes = int(self.config['max_hold_hours'] * 60)
                stop_loss_amount = -(position_size * self.config['stop_loss_pct'] / 100)
                
                # Simulate price movement
                trade_result = None
                for j in range(i + 1, min(i + max_hold_minutes + 1, len(data))):
                    current_price = data.iloc[j]['close']
                    hold_minutes = j - i
                    
                    # Calculate P&L
                    if direction == 'long':
                        pnl_pct = (current_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - current_price) / entry_price * 100
                    
                    pnl_amount = position_size * (pnl_pct / 100) * self.config['leverage']
                    
                    # Check exit conditions
                    
                    # 1. Target profit reached
                    if pnl_amount >= target_profit:
                        trade_result = {
                            'exit_price': current_price,
                            'exit_reason': 'target_profit',
                            'pnl_amount': pnl_amount,
                            'pnl_pct': pnl_pct * self.config['leverage'],
                            'hold_minutes': hold_minutes,
                            'success': True
                        }
                        break
                    
                    # 2. Partial profit at 70% of target
                    if pnl_amount >= target_profit * 0.7 and hold_minutes > 90:
                        trade_result = {
                            'exit_price': current_price,
                            'exit_reason': 'partial_profit',
                            'pnl_amount': pnl_amount,
                            'pnl_pct': pnl_pct * self.config['leverage'],
                            'hold_minutes': hold_minutes,
                            'success': True
                        }
                        break
                    
                    # 3. Stop loss
                    if pnl_amount <= stop_loss_amount:
                        trade_result = {
                            'exit_price': current_price,
                            'exit_reason': 'stop_loss',
                            'pnl_amount': pnl_amount,
                            'pnl_pct': pnl_pct * self.config['leverage'],
                            'hold_minutes': hold_minutes,
                            'success': False
                        }
                        break
                
                # Time exit if no other exit triggered
                if trade_result is None:
                    final_price = data.iloc[min(i + max_hold_minutes, len(data) - 1)]['close']
                    if direction == 'long':
                        final_pnl_pct = (final_price - entry_price) / entry_price * 100
                    else:
                        final_pnl_pct = (entry_price - final_price) / entry_price * 100
                    
                    final_pnl_amount = position_size * (final_pnl_pct / 100) * self.config['leverage']
                    
                    trade_result = {
                        'exit_price': final_price,
                        'exit_reason': 'time_exit',
                        'pnl_amount': final_pnl_amount,
                        'pnl_pct': final_pnl_pct * self.config['leverage'],
                        'hold_minutes': max_hold_minutes,
                        'success': final_pnl_amount > 0
                    }
                
                # Record trade
                trade = {
                    'entry_time': i,
                    'entry_price': entry_price,
                    'direction': direction,
                    'position_size': position_size,
                    'target_profit': target_profit,
                    'confidence': signals['confidence'],
                    'quality_score': signals['quality_score'],
                    'signals': signals['signals'],
                    **trade_result
                }
                
                self.trades.append(trade)
                self.total_trades += 1
                self.balance += trade_result['pnl_amount']
                self.total_profit += trade_result['pnl_amount']
                
                if trade_result['success']:
                    self.wins += 1
                else:
                    self.losses += 1
                
                # Update daily trades
                self.daily_trades[current_day] = self.daily_trades.get(current_day, 0) + 1
                
                # Skip ahead to avoid overlapping trades
                i += max(30, trade_result['hold_minutes'] // 2)  # Skip at least 30 minutes
        
        # Calculate final statistics
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        total_return = ((self.balance - self.config["initial_balance"]) / self.config["initial_balance"]) * 100
        
        profit_trades = [t for t in self.trades if t['success']]
        loss_trades = [t for t in self.trades if not t['success']]
        
        avg_win = np.mean([t['pnl_amount'] for t in profit_trades]) if profit_trades else 0
        avg_loss = np.mean([abs(t['pnl_amount']) for t in loss_trades]) if loss_trades else 0
        profit_factor = (avg_win * len(profit_trades)) / (avg_loss * len(loss_trades)) if loss_trades else float('inf')
        
        max_balance = self.config["initial_balance"]
        max_drawdown = 0
        running_balance = self.config["initial_balance"]
        
        for trade in self.trades:
            running_balance += trade['pnl_amount']
            max_balance = max(max_balance, running_balance)
            drawdown = (max_balance - running_balance) / max_balance * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_balance': self.balance,
            'total_profit': self.total_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trades': self.trades
        }
    
    def print_results(self, results: dict):
        """Print balanced backtest results"""
        print("\n" + "=" * 80)
        print("🎯 BALANCED 75% WIN RATE BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"📊 TRADE STATISTICS:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Wins: {results['wins']} | Losses: {results['losses']}")
        print(f"   🎯 Win Rate: {results['win_rate']:.1f}% (TARGET: 75-85%)")
        
        if results['win_rate'] >= 75:
            print("   🎉 TARGET ACHIEVED! Win rate 75%+ reached!")
        elif results['win_rate'] >= 70:
            print("   👍 Close to target! Very good performance.")
        else:
            print("   ⚠️ Below target. Consider adjusting conditions.")
        
        print(f"\n💰 FINANCIAL PERFORMANCE:")
        print(f"   Initial Balance: ${results['final_balance'] - results['total_profit']:.2f}")
        print(f"   Final Balance: ${results['final_balance']:.2f}")
        print(f"   Total Profit: ${results['total_profit']:+.2f}")
        print(f"   Total Return: {results['total_return']:+.1f}%")
        print(f"   Average Win: ${results['avg_win']:.2f}")
        print(f"   Average Loss: ${results['avg_loss']:.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.1f}%")
        
        if results['total_trades'] > 0:
            print(f"\n🎯 STRATEGY ANALYSIS:")
            target_hits = len([t for t in results['trades'] if t['exit_reason'] == 'target_profit'])
            partial_profits = len([t for t in results['trades'] if t['exit_reason'] == 'partial_profit'])
            stop_losses = len([t for t in results['trades'] if t['exit_reason'] == 'stop_loss'])
            time_exits = len([t for t in results['trades'] if t['exit_reason'] == 'time_exit'])
            
            print(f"   Target Hits: {target_hits} ({target_hits/results['total_trades']*100:.1f}%)")
            print(f"   Partial Profits: {partial_profits} ({partial_profits/results['total_trades']*100:.1f}%)")
            print(f"   Stop Losses: {stop_losses} ({stop_losses/results['total_trades']*100:.1f}%)")
            print(f"   Time Exits: {time_exits} ({time_exits/results['total_trades']*100:.1f}%)")
            
            print(f"\n📈 QUALITY METRICS:")
            avg_confidence = np.mean([t['confidence'] for t in results['trades']])
            avg_quality = np.mean([t['quality_score'] for t in results['trades']])
            avg_hold_time = np.mean([t['hold_minutes'] for t in results['trades']]) / 60
            
            print(f"   Average Confidence: {avg_confidence:.1f}%")
            print(f"   Average Quality Score: {avg_quality:.1f}%")
            print(f"   Average Hold Time: {avg_hold_time:.1f} hours")
            print(f"   Trades Per Day: {results['total_trades'] / 60:.1f}")
            
            print("\n🎯 BALANCED CONDITIONS SUMMARY:")
            print("   ✅ 75%+ AI Confidence Required")
            print("   ✅ 60%+ Quality Score Required")
            print("   ✅ Extreme Range Positions Preferred")
            print("   ✅ 3+ Signal Confirmations Required")
            print("   ✅ Balanced Technical Alignment")
            print("   ✅ Maximum 2 Trades Per Day")
            print("   ✅ Conservative Position Sizing")
            print("   ✅ Balanced Risk Management")
        
        print("=" * 80)

def main():
    """Main backtest execution"""
    print("🎯 BALANCED 75% WIN RATE BACKTEST")
    print("🏆 Optimized for 75-85% Win Rate Target")
    print("=" * 60)
    
    try:
        # Create backtest instance
        backtest = Balanced75Backtest()
        
        # Generate realistic data
        data = backtest.generate_realistic_data(days=60)
        
        # Run balanced backtest
        results = backtest.run_balanced_backtest(data)
        
        # Print comprehensive results
        backtest.print_results(results)
        
        # Save results
        print(f"\n💾 Backtest completed!")
        print(f"🎯 Win Rate Achieved: {results['win_rate']:.1f}%")
        
        if results['win_rate'] >= 75:
            print("🎉 SUCCESS! 75%+ win rate target achieved!")
        elif results['win_rate'] >= 70:
            print("👍 Very close! Consider minor adjustments.")
        else:
            print("📊 Need to optimize conditions further.")
        
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 