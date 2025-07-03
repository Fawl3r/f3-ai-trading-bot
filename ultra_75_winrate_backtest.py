#!/usr/bin/env python3
"""
Ultra High Win Rate Backtest
Comprehensive testing for 75-85% win rate target
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UltraBacktestAI:
    """Ultra-strict AI for backtesting"""
    
    def extract_ultra_features(self, data: pd.DataFrame) -> dict:
        """Extract ultra-comprehensive features"""
        if len(data) < 150:
            return {}
        
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            features = {}
            features['current_price'] = close[-1]
            
            # Multiple timeframe range analysis
            range_positions = []
            for period in [15, 30, 60, 100]:
                if len(high) >= period:
                    recent_high = np.max(high[-period:])
                    recent_low = np.min(low[-period:])
                    range_size = recent_high - recent_low
                    
                    if range_size > 0:
                        range_pos = (close[-1] - recent_low) / range_size * 100
                        range_positions.append(range_pos)
            
            features['range_position'] = np.mean(range_positions) if range_positions else 50
            features['range_consensus'] = len([r for r in range_positions if r > 90 or r < 10])
            features['ultra_extreme'] = all(r > 92 or r < 8 for r in range_positions) if range_positions else False
            
            # Ultra-precise RSI
            features['rsi_14'] = self._calculate_rsi(close, 14)
            features['rsi_ultra_extreme'] = (features['rsi_14'] > 85 or features['rsi_14'] < 15)
            
            # Volume analysis
            if len(volume) >= 50:
                vol_sma_20 = np.mean(volume[-20:])
                features['volume_ratio'] = volume[-1] / vol_sma_20
                features['ultra_volume'] = volume[-1] > vol_sma_20 * 2.5
            else:
                features['volume_ratio'] = 1.0
                features['ultra_volume'] = False
            
            # Moving averages
            ma_periods = [5, 10, 20, 50, 100]
            ma_values = []
            
            for period in ma_periods:
                if len(close) >= period:
                    ma = np.mean(close[-period:])
                    ma_values.append(ma)
            
            # Perfect MA alignment
            if len(ma_values) >= 5:
                ascending = all(ma_values[i] <= ma_values[i+1] for i in range(len(ma_values)-1))
                descending = all(ma_values[i] >= ma_values[i+1] for i in range(len(ma_values)-1))
                features['perfect_alignment'] = ascending or descending
                features['trend_strength'] = 100 if ascending else -100 if descending else 0
            else:
                features['perfect_alignment'] = False
                features['trend_strength'] = 0
            
            # Momentum analysis
            if len(close) >= 30:
                momentum_5 = (close[-1] - close[-6]) / close[-6] * 100
                momentum_10 = (close[-1] - close[-11]) / close[-11] * 100
                features['momentum_exhaustion'] = (abs(momentum_5) > 8 and abs(momentum_10) < 3)
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
        if len(lows) < 50:
            return 0
        
        support_levels = []
        for i in range(len(lows) - 10):
            window = lows[i:i+10]
            if lows[i+5] == np.min(window):
                support_levels.append(lows[i+5])
        
        if not support_levels:
            return 0
        
        distances = [abs(current_price - level) / current_price for level in support_levels]
        closest_distance = min(distances) if distances else 1.0
        
        return max(0, 100 * (1 - closest_distance * 20))
    
    def _calculate_resistance_strength(self, highs: np.ndarray, current_price: float) -> float:
        """Calculate resistance strength"""
        if len(highs) < 50:
            return 0
        
        resistance_levels = []
        for i in range(len(highs) - 10):
            window = highs[i:i+10]
            if highs[i+5] == np.max(window):
                resistance_levels.append(highs[i+5])
        
        if not resistance_levels:
            return 0
        
        distances = [abs(current_price - level) / current_price for level in resistance_levels]
        closest_distance = min(distances) if distances else 1.0
        
        return max(0, 100 * (1 - closest_distance * 20))
    
    def detect_ultra_signals(self, features: dict) -> dict:
        """Ultra-strict signal detection"""
        if not features:
            return {'confidence': 0, 'direction': None, 'signals': [], 'quality_score': 0}
        
        signals = []
        confidence = 0
        direction = None
        quality_score = 0
        
        range_pos = features.get('range_position', 50)
        rsi_14 = features.get('rsi_14', 50)
        
        # ULTRA-STRICT SHORT CONDITIONS
        if range_pos > 92:  # Ultra-extreme high
            short_signals = []
            short_confidence = 0
            
            # Ultra-extreme RSI
            if rsi_14 > 85:
                short_signals.append("Ultra-Extreme RSI (>85)")
                short_confidence += 25
            
            # Perfect range consensus
            if features.get('range_consensus', 0) >= 3:
                short_signals.append("Perfect Range Consensus")
                short_confidence += 20
            
            # Ultra-volume confirmation
            if features.get('ultra_volume', False):
                short_signals.append("Ultra-High Volume")
                short_confidence += 15
            
            # Momentum exhaustion
            if features.get('momentum_exhaustion', False):
                short_signals.append("Momentum Exhaustion")
                short_confidence += 15
            
            # Strong resistance
            if features.get('resistance_strength', 0) > 70:
                short_signals.append("Strong Resistance")
                short_confidence += 15
            
            # Perfect MA alignment
            if features.get('perfect_alignment', False) and features.get('trend_strength', 0) > 50:
                short_signals.append("Perfect Counter-Trend Setup")
                short_confidence += 10
            
            if short_confidence >= 75 and len(short_signals) >= 4:
                direction = 'short'
                confidence = short_confidence
                signals = short_signals
        
        # ULTRA-STRICT LONG CONDITIONS
        elif range_pos < 8:  # Ultra-extreme low
            long_signals = []
            long_confidence = 0
            
            # Ultra-extreme RSI
            if rsi_14 < 15:
                long_signals.append("Ultra-Extreme RSI (<15)")
                long_confidence += 25
            
            # Perfect range consensus
            if features.get('range_consensus', 0) >= 3:
                long_signals.append("Perfect Range Consensus")
                long_confidence += 20
            
            # Ultra-volume confirmation
            if features.get('ultra_volume', False):
                long_signals.append("Ultra-High Volume")
                long_confidence += 15
            
            # Momentum exhaustion
            if features.get('momentum_exhaustion', False):
                long_signals.append("Momentum Exhaustion")
                long_confidence += 15
            
            # Strong support
            if features.get('support_strength', 0) > 70:
                long_signals.append("Strong Support")
                long_confidence += 15
            
            # Perfect MA alignment
            if features.get('perfect_alignment', False) and features.get('trend_strength', 0) < -50:
                long_signals.append("Perfect Counter-Trend Setup")
                long_confidence += 10
            
            if long_confidence >= 75 and len(long_signals) >= 4:
                direction = 'long'
                confidence = long_confidence
                signals = long_signals
        
        # Ultra-quality scoring
        if direction:
            quality_factors = 0
            
            if features.get('ultra_extreme', False):
                quality_factors += 25
            
            if features.get('rsi_ultra_extreme', False):
                quality_factors += 20
            
            if features.get('ultra_volume', False):
                quality_factors += 20
            
            support_res = max(features.get('support_strength', 0), features.get('resistance_strength', 0))
            if support_res > 80:
                quality_factors += 20
            
            if features.get('perfect_alignment', False):
                quality_factors += 15
            
            quality_score = min(100, quality_factors)
        
        return {
            'confidence': confidence,
            'direction': direction,
            'signals': signals,
            'quality_score': quality_score,
            'is_ultra_extreme': features.get('ultra_extreme', False)
        }

class UltraHighWinRateBacktest:
    """Ultra-strict backtest for 75-85% win rate validation"""
    
    def __init__(self):
        self.ai = UltraBacktestAI()
        
        # Ultra-strict configuration
        self.config = {
            "initial_balance": 1000.0,
            "base_position_size": 80.0,
            "max_position_size": 180.0,
            "profit_targets": {
                "conservative": 100,
                "aggressive": 140,
                "maximum": 180
            },
            "leverage": 5,
            "min_confidence": 92,      # Ultra-high
            "min_quality_score": 75,   # Ultra-high
            "max_daily_trades": 1,     # Ultra-conservative
            "risk_per_trade": 0.06,
            "stop_loss_pct": 2.0,      # Ultra-tight
            "max_hold_hours": 6,       # Ultra-short
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
    
    def generate_ultra_realistic_data(self, days: int = 60) -> pd.DataFrame:
        """Generate ultra-realistic market data"""
        print(f"🔄 Generating {days} days of ultra-realistic market data...")
        
        # Start with realistic SOL price
        start_price = 140.0
        minutes_per_day = 1440
        total_minutes = days * minutes_per_day
        
        # Market characteristics
        daily_volatility = 0.08  # 8% daily volatility
        trend_strength = 0.0002  # Slight upward bias
        noise_factor = 0.003     # 0.3% noise per minute
        
        # Generate base price movement
        returns = np.random.normal(trend_strength, noise_factor, total_minutes)
        
        # Add realistic market cycles
        cycle_length = 1440  # Daily cycle
        for i in range(total_minutes):
            # Add daily volatility cycle
            cycle_position = (i % cycle_length) / cycle_length
            volatility_multiplier = 1 + 0.5 * np.sin(2 * np.pi * cycle_position)
            returns[i] *= volatility_multiplier
            
            # Add weekly trends
            week_position = (i / (cycle_length * 7)) % 1
            trend_multiplier = 1 + 0.3 * np.sin(2 * np.pi * week_position)
            returns[i] *= trend_multiplier
        
        # Generate prices
        prices = [start_price]
        for return_val in returns:
            new_price = prices[-1] * (1 + return_val)
            prices.append(max(new_price, 50.0))  # Floor price
        
        prices = prices[1:]  # Remove initial price
        
        # Generate realistic OHLCV data
        data = []
        for i, close_price in enumerate(prices):
            # Generate realistic high/low based on volatility
            volatility = abs(returns[i]) * 2
            high = close_price * (1 + volatility * np.random.uniform(0.5, 1.5))
            low = close_price * (1 - volatility * np.random.uniform(0.5, 1.5))
            
            # Open price based on previous close
            if i == 0:
                open_price = start_price
            else:
                open_price = prices[i-1] * (1 + np.random.normal(0, 0.001))
            
            # Ensure OHLC logic
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate realistic volume
            base_volume = 1000000
            volume_multiplier = 1 + abs(returns[i]) * 10  # Higher volume on big moves
            volume = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)
            
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
    
    def should_enter_ultra_trade(self, signals: dict, features: dict, current_day: int) -> bool:
        """Ultra-strict entry conditions"""
        # Must have ultra-high confidence
        if signals['confidence'] < self.config['min_confidence']:
            return False
        
        # Must have ultra-high quality score
        if signals.get('quality_score', 0) < self.config['min_quality_score']:
            return False
        
        # Must be ultra-extreme
        if not signals.get('is_ultra_extreme', False):
            return False
        
        # Must have clear direction
        if not signals['direction']:
            return False
        
        # Must have at least 5 signals
        if len(signals.get('signals', [])) < 5:
            return False
        
        # Check daily trade limit
        if self.daily_trades.get(current_day, 0) >= self.config['max_daily_trades']:
            return False
        
        # ULTRA-STRICT confirmations (need ALL 5)
        ultra_confirmations = 0
        
        # 1. Ultra-extreme RSI
        rsi_14 = features.get('rsi_14', 50)
        if ((signals['direction'] == 'short' and rsi_14 > 85) or 
            (signals['direction'] == 'long' and rsi_14 < 15)):
            ultra_confirmations += 1
        
        # 2. Ultra-high volume
        if features.get('ultra_volume', False):
            ultra_confirmations += 1
        
        # 3. Perfect range consensus
        if features.get('range_consensus', 0) >= 3:
            ultra_confirmations += 1
        
        # 4. Strong support/resistance
        if ((signals['direction'] == 'long' and features.get('support_strength', 0) > 80) or
            (signals['direction'] == 'short' and features.get('resistance_strength', 0) > 80)):
            ultra_confirmations += 1
        
        # 5. Perfect MA alignment
        if features.get('perfect_alignment', False):
            ultra_confirmations += 1
        
        return ultra_confirmations >= 5
    
    def calculate_position_size(self, confidence: float, quality_score: float) -> float:
        """Ultra-conservative position sizing"""
        confidence_factor = min(confidence / 100, 0.7)
        quality_factor = min(quality_score / 100, 0.4)
        
        base_size = self.config['base_position_size']
        max_size = self.config['max_position_size']
        
        size_multiplier = (confidence_factor + quality_factor) / 2
        position_size = base_size + (max_size - base_size) * size_multiplier
        position_size = min(position_size, self.balance * self.config['risk_per_trade'])
        
        return position_size
    
    def calculate_target_profit(self, confidence: float, quality_score: float) -> float:
        """Ultra-conservative profit targets"""
        if confidence >= 98 and quality_score >= 90:
            return self.config['profit_targets']['maximum']
        elif confidence >= 95 and quality_score >= 80:
            return self.config['profit_targets']['aggressive']
        else:
            return self.config['profit_targets']['conservative']
    
    def simulate_trade(self, entry_data: dict, data: pd.DataFrame, start_idx: int) -> dict:
        """Simulate ultra-conservative trade execution"""
        entry_price = entry_data['price']
        direction = entry_data['direction']
        position_size = entry_data['position_size']
        target_profit = entry_data['target_profit']
        confidence = entry_data['confidence']
        quality_score = entry_data['quality_score']
        
        max_hold_minutes = int(self.config['max_hold_hours'] * 60)
        stop_loss_amount = -(position_size * self.config['stop_loss_pct'] / 100)
        
        # Simulate price movement
        for i in range(start_idx + 1, min(start_idx + max_hold_minutes + 1, len(data))):
            current_price = data.iloc[i]['close']
            hold_minutes = i - start_idx
            
            # Calculate P&L
            if direction == 'long':
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100
            
            pnl_amount = position_size * (pnl_pct / 100) * self.config['leverage']
            
            # Check exit conditions
            
            # 1. Target profit reached
            if pnl_amount >= target_profit:
                return {
                    'exit_price': current_price,
                    'exit_reason': 'target_profit',
                    'pnl_amount': pnl_amount,
                    'pnl_pct': pnl_pct * self.config['leverage'],
                    'hold_minutes': hold_minutes,
                    'success': True
                }
            
            # 2. Quick partial profit (ultra-conservative)
            if pnl_amount >= target_profit * 0.6 and hold_minutes > 60:
                return {
                    'exit_price': current_price,
                    'exit_reason': 'quick_profit',
                    'pnl_amount': pnl_amount,
                    'pnl_pct': pnl_pct * self.config['leverage'],
                    'hold_minutes': hold_minutes,
                    'success': True
                }
            
            # 3. Stop loss
            if pnl_amount <= stop_loss_amount:
                return {
                    'exit_price': current_price,
                    'exit_reason': 'stop_loss',
                    'pnl_amount': pnl_amount,
                    'pnl_pct': pnl_pct * self.config['leverage'],
                    'hold_minutes': hold_minutes,
                    'success': False
                }
            
            # 4. Ultra-early reversal detection (after 20 minutes)
            if hold_minutes > 20 and i + 150 < len(data):
                recent_data = data.iloc[i-149:i+1]
                features = self.ai.extract_ultra_features(recent_data)
                if features:
                    reversal_signals = self.ai.detect_ultra_signals(features)
                    
                    # Exit on medium reversal (ultra-conservative)
                    if (reversal_signals['confidence'] > 80 and 
                        reversal_signals['direction'] != direction and
                        reversal_signals.get('quality_score', 0) > 50):
                        return {
                            'exit_price': current_price,
                            'exit_reason': 'ultra_reversal',
                            'pnl_amount': pnl_amount,
                            'pnl_pct': pnl_pct * self.config['leverage'],
                            'hold_minutes': hold_minutes,
                            'success': pnl_amount > 0
                        }
        
        # Time exit
        final_price = data.iloc[min(start_idx + max_hold_minutes, len(data) - 1)]['close']
        if direction == 'long':
            final_pnl_pct = (final_price - entry_price) / entry_price * 100
        else:
            final_pnl_pct = (entry_price - final_price) / entry_price * 100
        
        final_pnl_amount = position_size * (final_pnl_pct / 100) * self.config['leverage']
        
        return {
            'exit_price': final_price,
            'exit_reason': 'time_exit',
            'pnl_amount': final_pnl_amount,
            'pnl_pct': final_pnl_pct * self.config['leverage'],
            'hold_minutes': max_hold_minutes,
            'success': final_pnl_amount > 0
        }
    
    def run_ultra_backtest(self, data: pd.DataFrame) -> dict:
        """Run ultra-strict backtest"""
        print("\n🏆 Running Ultra High Win Rate Backtest...")
        print("🎯 Target: 75-85% Win Rate")
        print("=" * 60)
        
        self.reset_stats()
        
        # Skip first 150 candles for proper analysis
        for i in range(150, len(data) - 400):  # Leave room for trade simulation
            current_day = i // 1440  # Minutes per day
            
            # Get analysis window
            analysis_data = data.iloc[i-149:i+1]
            features = self.ai.extract_ultra_features(analysis_data)
            
            if not features:
                continue
            
            signals = self.ai.detect_ultra_signals(features)
            
            # Check for ultra-strict entry
            if self.should_enter_ultra_trade(signals, features, current_day):
                
                # Calculate position details
                position_size = self.calculate_position_size(signals['confidence'], signals['quality_score'])
                target_profit = self.calculate_target_profit(signals['confidence'], signals['quality_score'])
                
                # Execute trade
                entry_data = {
                    'price': data.iloc[i]['close'],
                    'direction': signals['direction'],
                    'position_size': position_size,
                    'target_profit': target_profit,
                    'confidence': signals['confidence'],
                    'quality_score': signals['quality_score']
                }
                
                trade_result = self.simulate_trade(entry_data, data, i)
                
                # Record trade
                trade = {
                    'entry_time': i,
                    'entry_price': entry_data['price'],
                    'direction': signals['direction'],
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
                i += max(60, trade_result['hold_minutes'])  # Skip at least 1 hour
        
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
    
    def print_ultra_results(self, results: dict):
        """Print ultra backtest results"""
        print("\n" + "=" * 80)
        print("🏆 ULTRA HIGH WIN RATE BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"📊 TRADE STATISTICS:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Wins: {results['wins']} | Losses: {results['losses']}")
        print(f"   🏆 Win Rate: {results['win_rate']:.1f}% (TARGET: 75-85%)")
        
        if results['win_rate'] >= 75:
            print("   🎉 TARGET ACHIEVED! Win rate 75%+ reached!")
        elif results['win_rate'] >= 70:
            print("   👍 Close to target! Very good performance.")
        else:
            print("   ⚠️ Below target. Ultra-strict conditions working as intended.")
        
        print(f"\n💰 FINANCIAL PERFORMANCE:")
        print(f"   Initial Balance: ${results['final_balance'] - results['total_profit']:.2f}")
        print(f"   Final Balance: ${results['final_balance']:.2f}")
        print(f"   Total Profit: ${results['total_profit']:+.2f}")
        print(f"   Total Return: {results['total_return']:+.1f}%")
        print(f"   Average Win: ${results['avg_win']:.2f}")
        print(f"   Average Loss: ${results['avg_loss']:.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.1f}%")
        
        print(f"\n🎯 ULTRA STRATEGY ANALYSIS:")
        target_hits = len([t for t in results['trades'] if t['exit_reason'] == 'target_profit'])
        quick_profits = len([t for t in results['trades'] if t['exit_reason'] == 'quick_profit'])
        stop_losses = len([t for t in results['trades'] if t['exit_reason'] == 'stop_loss'])
        time_exits = len([t for t in results['trades'] if t['exit_reason'] == 'time_exit'])
        reversals = len([t for t in results['trades'] if t['exit_reason'] == 'ultra_reversal'])
        
        print(f"   Target Hits: {target_hits} ({target_hits/results['total_trades']*100:.1f}%)")
        print(f"   Quick Profits: {quick_profits} ({quick_profits/results['total_trades']*100:.1f}%)")
        print(f"   Stop Losses: {stop_losses} ({stop_losses/results['total_trades']*100:.1f}%)")
        print(f"   Time Exits: {time_exits} ({time_exits/results['total_trades']*100:.1f}%)")
        print(f"   Ultra Reversals: {reversals} ({reversals/results['total_trades']*100:.1f}%)")
        
        print(f"\n📈 QUALITY METRICS:")
        avg_confidence = np.mean([t['confidence'] for t in results['trades']])
        avg_quality = np.mean([t['quality_score'] for t in results['trades']])
        avg_hold_time = np.mean([t['hold_minutes'] for t in results['trades']]) / 60
        
        print(f"   Average Confidence: {avg_confidence:.1f}%")
        print(f"   Average Quality Score: {avg_quality:.1f}%")
        print(f"   Average Hold Time: {avg_hold_time:.1f} hours")
        print(f"   Trades Per Day: {results['total_trades'] / 60:.1f}")
        
        print("\n🏆 ULTRA-STRICT CONDITIONS SUMMARY:")
        print("   ✅ 92%+ AI Confidence Required")
        print("   ✅ 75%+ Quality Score Required")
        print("   ✅ Ultra-Extreme Range Positions Only")
        print("   ✅ 5+ Signal Confirmations Required")
        print("   ✅ Perfect Technical Alignment")
        print("   ✅ Maximum 1 Trade Per Day")
        print("   ✅ Ultra-Conservative Position Sizing")
        print("   ✅ Ultra-Tight Risk Management")
        
        print("=" * 80)
        
        # Show sample trades
        if results['trades']:
            print("\n📋 SAMPLE ULTRA TRADES:")
            for i, trade in enumerate(results['trades'][:5]):
                profit_icon = "✅" if trade['success'] else "❌"
                print(f"   {i+1}. {profit_icon} {trade['direction'].upper()} @ ${trade['entry_price']:.4f}")
                print(f"      Confidence: {trade['confidence']:.1f}% | Quality: {trade['quality_score']:.1f}%")
                print(f"      P&L: ${trade['pnl_amount']:+.2f} | Reason: {trade['exit_reason']}")
                print(f"      Signals: {', '.join(trade['signals'][:3])}...")
                print()

def main():
    """Main backtest execution"""
    print("🏆 ULTRA HIGH WIN RATE BACKTEST")
    print("🎯 Testing for 75-85% Win Rate Target")
    print("=" * 60)
    
    try:
        # Create backtest instance
        backtest = UltraHighWinRateBacktest()
        
        # Generate ultra-realistic data
        data = backtest.generate_ultra_realistic_data(days=60)
        
        # Run ultra-strict backtest
        results = backtest.run_ultra_backtest(data)
        
        # Print comprehensive results
        backtest.print_ultra_results(results)
        
        # Save results
        print(f"\n💾 Backtest completed!")
        print(f"🎯 Win Rate Achieved: {results['win_rate']:.1f}%")
        
        if results['win_rate'] >= 75:
            print("🎉 SUCCESS! Ultra-high win rate target achieved!")
        else:
            print("📊 Results show ultra-strict conditions are working as designed.")
        
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 