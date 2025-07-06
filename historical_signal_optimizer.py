#!/usr/bin/env python3
"""
Historical Signal Optimizer
Analyzes 5000 candles to optimize signal detection thresholds
"""

import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from hyperliquid.info import Info
from hyperliquid.utils import constants
import os
from dotenv import load_dotenv

class HistoricalSignalOptimizer:
    def __init__(self):
        load_dotenv()
        testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
        base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.info = Info(base_url, skip_ws=True)
        
        self.symbols = ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX']
        self.data_stats = {}
        
    def get_extended_history(self, symbol, num_candles=5000):
        """Get extended historical data"""
        try:
            print(f"📊 Fetching {num_candles} candles for {symbol}...")
            
            # Get data in chunks (API limits)
            all_candles = []
            end_time = int(time.time() * 1000)
            
            # Use 1h candles to get more history (5000h = ~208 days)
            chunk_size = 1000  # Get 1000 candles at a time
            
            for chunk in range(0, num_candles, chunk_size):
                remaining = min(chunk_size, num_candles - chunk)
                start_time = end_time - (remaining * 60 * 60 * 1000)  # 1h candles
                
                try:
                    candles = self.info.candles_snapshot(symbol, "1h", start_time, end_time)
                    if candles:
                        all_candles.extend(candles)
                        print(f"   Got {len(candles)} candles, total: {len(all_candles)}")
                        end_time = start_time  # Move backwards in time
                        time.sleep(0.1)  # Rate limiting
                    else:
                        break
                except Exception as e:
                    print(f"   Error getting chunk for {symbol}: {e}")
                    break
            
            if len(all_candles) < 100:
                print(f"   ❌ Insufficient data for {symbol}: {len(all_candles)} candles")
                return None
                
            # Sort chronologically
            all_candles.sort(key=lambda x: x['T'])
            print(f"   ✅ Got {len(all_candles)} total candles for {symbol}")
            return all_candles
            
        except Exception as e:
            print(f"❌ Error getting history for {symbol}: {e}")
            return None
    
    def analyze_momentum_patterns(self, candles):
        """Analyze momentum patterns in historical data"""
        if not candles or len(candles) < 20:
            return None
            
        prices = [float(c['c']) for c in candles]
        volumes = [float(c['v']) for c in candles]
        
        momentum_data = []
        
        # Analyze each candle for momentum signals
        for i in range(10, len(candles) - 5):  # Leave buffer for forward/backward analysis
            current_price = prices[i]
            
            # Calculate momentum over different periods
            price_1h = prices[i-1] if i >= 1 else current_price
            price_3h = prices[i-3] if i >= 3 else current_price
            price_6h = prices[i-6] if i >= 6 else current_price
            
            momentum_1h = (current_price - price_1h) / price_1h * 100
            momentum_3h = (current_price - price_3h) / price_3h * 100
            momentum_6h = (current_price - price_6h) / price_6h * 100
            
            # Volume analysis
            current_volume = volumes[i]
            avg_volume_10 = np.mean(volumes[max(0, i-10):i]) if i >= 10 else current_volume
            volume_ratio = current_volume / avg_volume_10 if avg_volume_10 > 0 else 1.0
            
            # Price range/volatility
            recent_prices = prices[max(0, i-5):i+1]
            price_range = (max(recent_prices) - min(recent_prices)) / current_price * 100
            
            # Calculate future performance (next 3-24 hours)
            future_prices = prices[i+1:min(len(prices), i+25)]  # Next 24 hours
            if future_prices:
                max_future = max(future_prices)
                min_future = min(future_prices)
                max_gain = (max_future - current_price) / current_price * 100
                max_loss = (min_future - current_price) / current_price * 100
                
                # Determine if this was a good signal
                good_long = max_gain >= 2.0  # 2%+ gain available
                good_short = max_loss <= -2.0  # 2%+ drop available
                
                momentum_data.append({
                    'momentum_1h': momentum_1h,
                    'momentum_3h': momentum_3h,
                    'momentum_6h': momentum_6h,
                    'volume_ratio': volume_ratio,
                    'price_range': price_range,
                    'max_gain': max_gain,
                    'max_loss': max_loss,
                    'good_long': good_long,
                    'good_short': good_short,
                    'price': current_price,
                    'timestamp': candles[i]['t']
                })
        
        return momentum_data
    
    def find_optimal_thresholds(self, all_momentum_data):
        """Find optimal thresholds from historical data"""
        print("\n🔍 ANALYZING HISTORICAL PATTERNS...")
        
        # Combine all data
        combined_data = []
        for symbol_data in all_momentum_data.values():
            if symbol_data:
                combined_data.extend(symbol_data)
        
        if not combined_data:
            return None
        
        df = pd.DataFrame(combined_data)
        
        print(f"📊 Total data points: {len(df)}")
        
        # Analyze distributions
        print(f"\n📈 MOMENTUM DISTRIBUTIONS:")
        print(f"1h momentum: {df['momentum_1h'].describe()}")
        print(f"3h momentum: {df['momentum_3h'].describe()}")
        print(f"6h momentum: {df['momentum_6h'].describe()}")
        print(f"Volume ratio: {df['volume_ratio'].describe()}")
        print(f"Price range: {df['price_range'].describe()}")
        
        # Find signals that led to good opportunities
        good_long_signals = df[df['good_long'] == True]
        good_short_signals = df[df['good_short'] == True]
        
        print(f"\n🎯 PROFITABLE OPPORTUNITIES:")
        print(f"Good long signals: {len(good_long_signals)} ({len(good_long_signals)/len(df)*100:.1f}%)")
        print(f"Good short signals: {len(good_short_signals)} ({len(good_short_signals)/len(df)*100:.1f}%)")
        
        # Analyze characteristics of profitable signals
        if len(good_long_signals) > 0:
            print(f"\n🟢 PROFITABLE LONG SIGNAL CHARACTERISTICS:")
            print(f"1h momentum: min={good_long_signals['momentum_1h'].min():.3f}, "
                  f"mean={good_long_signals['momentum_1h'].mean():.3f}, "
                  f"25%ile={good_long_signals['momentum_1h'].quantile(0.25):.3f}")
            print(f"3h momentum: min={good_long_signals['momentum_3h'].min():.3f}, "
                  f"mean={good_long_signals['momentum_3h'].mean():.3f}, "
                  f"25%ile={good_long_signals['momentum_3h'].quantile(0.25):.3f}")
            print(f"Volume ratio: min={good_long_signals['volume_ratio'].min():.2f}, "
                  f"mean={good_long_signals['volume_ratio'].mean():.2f}, "
                  f"25%ile={good_long_signals['volume_ratio'].quantile(0.25):.2f}")
            print(f"Price range: min={good_long_signals['price_range'].min():.3f}, "
                  f"mean={good_long_signals['price_range'].mean():.3f}, "
                  f"25%ile={good_long_signals['price_range'].quantile(0.25):.3f}")
        
        if len(good_short_signals) > 0:
            print(f"\n🔴 PROFITABLE SHORT SIGNAL CHARACTERISTICS:")
            print(f"1h momentum: max={good_short_signals['momentum_1h'].max():.3f}, "
                  f"mean={good_short_signals['momentum_1h'].mean():.3f}, "
                  f"75%ile={good_short_signals['momentum_1h'].quantile(0.75):.3f}")
            print(f"3h momentum: max={good_short_signals['momentum_3h'].max():.3f}, "
                  f"mean={good_short_signals['momentum_3h'].mean():.3f}, "
                  f"75%ile={good_short_signals['momentum_3h'].quantile(0.75):.3f}")
            print(f"Volume ratio: min={good_short_signals['volume_ratio'].min():.2f}, "
                  f"mean={good_short_signals['volume_ratio'].mean():.2f}, "
                  f"25%ile={good_short_signals['volume_ratio'].quantile(0.25):.2f}")
        
        # Calculate optimal thresholds (use 25th percentile for conservative approach)
        optimal_thresholds = {}
        
        if len(good_long_signals) > 10:
            optimal_thresholds['min_momentum_1h_long'] = max(0.1, good_long_signals['momentum_1h'].quantile(0.25))
            optimal_thresholds['min_momentum_3h_long'] = max(0.2, good_long_signals['momentum_3h'].quantile(0.25))
            optimal_thresholds['min_volume_ratio_long'] = max(1.1, good_long_signals['volume_ratio'].quantile(0.25))
            optimal_thresholds['min_price_range_long'] = max(0.1, good_long_signals['price_range'].quantile(0.25))
        
        if len(good_short_signals) > 10:
            optimal_thresholds['max_momentum_1h_short'] = min(-0.1, good_short_signals['momentum_1h'].quantile(0.75))
            optimal_thresholds['max_momentum_3h_short'] = min(-0.2, good_short_signals['momentum_3h'].quantile(0.75))
            optimal_thresholds['min_volume_ratio_short'] = max(1.1, good_short_signals['volume_ratio'].quantile(0.25))
            optimal_thresholds['min_price_range_short'] = max(0.1, good_short_signals['price_range'].quantile(0.25))
        
        # Test different confidence thresholds
        print(f"\n🎯 TESTING CONFIDENCE THRESHOLDS:")
        
        for conf_threshold in [40, 45, 50, 55, 60, 65, 70]:
            # Simulate signals with this threshold
            signals_generated = 0
            profitable_signals = 0
            
            for _, row in df.iterrows():
                # Simple confidence calculation (you can make this more sophisticated)
                confidence = (
                    30 +  # Base
                    abs(row['momentum_1h']) * 10 +
                    abs(row['momentum_3h']) * 8 +
                    (row['volume_ratio'] - 1) * 20 +
                    row['price_range'] * 5
                )
                confidence = min(95, confidence)
                
                if confidence >= conf_threshold:
                    signals_generated += 1
                    if row['good_long'] or row['good_short']:
                        profitable_signals += 1
            
            win_rate = (profitable_signals / signals_generated * 100) if signals_generated > 0 else 0
            signal_frequency = signals_generated / len(df) * 100
            
            print(f"  {conf_threshold}% threshold: {signals_generated} signals ({signal_frequency:.1f}%), "
                  f"win rate: {win_rate:.1f}%")
        
        return optimal_thresholds
    
    async def run_optimization(self):
        """Run the full optimization process"""
        print("🚀 HISTORICAL SIGNAL OPTIMIZATION")
        print("=" * 60)
        
        all_momentum_data = {}
        
        # Get historical data for each symbol
        for symbol in self.symbols:
            candles = self.get_extended_history(symbol, 2000)  # Start with 2000 for speed
            if candles:
                momentum_data = self.analyze_momentum_patterns(candles)
                all_momentum_data[symbol] = momentum_data
                print(f"✅ {symbol}: {len(momentum_data) if momentum_data else 0} data points")
            else:
                print(f"❌ {symbol}: No data")
        
        # Find optimal thresholds
        optimal_thresholds = self.find_optimal_thresholds(all_momentum_data)
        
        if optimal_thresholds:
            print(f"\n🎯 OPTIMAL THRESHOLDS FOUND:")
            for key, value in optimal_thresholds.items():
                print(f"  {key}: {value:.3f}")
            
            # Generate optimized signal detector code
            self.generate_optimized_detector(optimal_thresholds)
        
        return optimal_thresholds
    
    def generate_optimized_detector(self, thresholds):
        """Generate optimized signal detector code"""
        print(f"\n🔧 GENERATING OPTIMIZED SIGNAL DETECTOR...")
        
        # Extract key thresholds
        min_vol = thresholds.get('min_volume_ratio_long', 1.2)
        min_mom_1h = thresholds.get('min_momentum_1h_long', 0.2)
        min_mom_3h = thresholds.get('min_momentum_3h_long', 0.3)
        min_range = thresholds.get('min_price_range_long', 0.15)
        
        code = f'''
# 🎯 OPTIMIZED THRESHOLDS (from {len(self.symbols)} symbols, 2000+ candles each)
self.min_volume_ratio = {min_vol:.2f}      # Volume spike threshold
self.min_momentum_1h = {min_mom_1h:.3f}    # 1h momentum threshold  
self.min_momentum_3h = {min_mom_3h:.3f}    # 3h momentum threshold
self.min_price_range = {min_range:.3f}     # Price range threshold
self.confidence_threshold = 50.0           # Recommended: 50% for balanced signals
'''
        
        print(code)
        
        # Save to file
        with open('optimized_thresholds.txt', 'w') as f:
            f.write(f"# Optimized Signal Detection Thresholds\n")
            f.write(f"# Generated: {datetime.now()}\n")
            f.write(f"# Data: {len(self.symbols)} symbols, 2000+ candles each\n\n")
            f.write(code)
        
        print("💾 Saved to optimized_thresholds.txt")

# Run the optimization
async def main():
    optimizer = HistoricalSignalOptimizer()
    await optimizer.run_optimization()

if __name__ == "__main__":
    asyncio.run(main()) 