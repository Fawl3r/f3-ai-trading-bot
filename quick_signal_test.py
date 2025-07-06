#!/usr/bin/env python3
"""Quick signal test with optimized thresholds"""

import time
import numpy as np
from hyperliquid.info import Info
from hyperliquid.utils import constants
import os
from dotenv import load_dotenv

def test_optimized_detection():
    load_dotenv()
    testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
    base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    
    print('🎯 TESTING OPTIMIZED SIGNAL DETECTION')
    print('📊 Using 50% confidence threshold (82.4% historical win rate)')
    print('=' * 60)
    
    info = Info(base_url, skip_ws=True)
    
    # Optimized thresholds from historical analysis
    min_volume_ratio = 1.10
    min_momentum_1h = 0.100
    min_momentum_3h = 0.200
    min_price_range = 1.022
    confidence_threshold = 50.0
    
    signals_found = 0
    
    for symbol in ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE']:
        print(f'\n🔍 Testing {symbol}...')
        
        try:
            # Get price
            all_mids = info.all_mids()
            if symbol not in all_mids:
                print(f'   ❌ {symbol} not found in market data')
                continue
            
            current_price = float(all_mids[symbol])
            print(f'   💰 Current price: ${current_price:,.2f}')
            
            # Get candles for momentum calculation
            end_time = int(time.time() * 1000)
            start_time = end_time - (60 * 60 * 1000)  # 1 hour
            
            candles = info.candles_snapshot(symbol, "5m", start_time, end_time)
            if not candles or len(candles) < 6:
                print(f'   ❌ Insufficient candle data')
                continue
            
            # Extract data
            prices = [float(c['c']) for c in candles]
            volumes = [float(c['v']) for c in candles]
            
            # Calculate momentum (optimized)
            momentum_1h = (prices[-1] - prices[-12]) / prices[-12] * 100 if len(prices) >= 12 else 0
            momentum_3h = (prices[-1] - prices[-6]) / prices[-6] * 100 if len(prices) >= 6 else 0
            momentum_5m = (prices[-1] - prices[-2]) / prices[-2] * 100 if len(prices) >= 2 else 0
            
            # Volume analysis
            current_volume = volumes[-1] if volumes else 0
            avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price range
            recent_prices = prices[-6:] if len(prices) >= 6 else prices
            price_range = (max(recent_prices) - min(recent_prices)) / current_price * 100
            
            # Optimized confidence calculation
            confidence = 30  # Base
            
            if abs(momentum_1h) >= min_momentum_1h:
                confidence += abs(momentum_1h) * 12
            if abs(momentum_3h) >= min_momentum_3h:
                confidence += abs(momentum_3h) * 10
            if abs(momentum_5m) >= 0.1:
                confidence += abs(momentum_5m) * 8
                
            if volume_ratio >= min_volume_ratio:
                confidence += (volume_ratio - 1) * 25
                
            if price_range >= min_price_range:
                confidence += price_range * 6
            
            confidence = min(95, confidence)
            
            print(f'   📊 Momentum 1h: {momentum_1h:.3f}%')
            print(f'   📊 Momentum 3h: {momentum_3h:.3f}%')
            print(f'   📊 Volume ratio: {volume_ratio:.2f}x')
            print(f'   📊 Price range: {price_range:.3f}%')
            print(f'   🎯 Confidence: {confidence:.1f}%')
            
            if confidence >= confidence_threshold:
                print(f'   ✅ SIGNAL GENERATED! ({confidence:.1f}% >= {confidence_threshold}%)')
                signals_found += 1
                
                # Determine direction
                if momentum_1h > 0 and momentum_3h > 0:
                    direction = "BUY"
                elif momentum_1h < 0 and momentum_3h < 0:
                    direction = "SELL"
                else:
                    direction = "MIXED"
                    
                print(f'   📈 Direction: {direction}')
            else:
                print(f'   ❌ No signal ({confidence:.1f}% < {confidence_threshold}%)')
                
        except Exception as e:
            print(f'   ❌ Error testing {symbol}: {e}')
    
    print(f'\n' + '=' * 60)
    print(f'📊 RESULTS: {signals_found}/{len(["BTC", "ETH", "SOL", "AVAX", "DOGE"])} signals found')
    
    if signals_found > 0:
        print('✅ SUCCESS! Optimized thresholds are generating signals!')
        print('🚀 Ready to deploy optimized bot!')
    else:
        print('❌ No signals found - may need to adjust thresholds further')
        print('💡 Current market may be in low volatility period')

if __name__ == "__main__":
    test_optimized_detection() 