#!/usr/bin/env python3
"""
Adaptive Signal Detector
Adjusts thresholds based on current market volatility conditions
"""

import time
import numpy as np
from datetime import datetime
from hyperliquid.info import Info
from hyperliquid.utils import constants
import os
from dotenv import load_dotenv

class AdaptiveSignalDetector:
    def __init__(self):
        load_dotenv()
        testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
        base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.info = Info(base_url, skip_ws=True)
        
        # Base thresholds (from historical analysis)
        self.base_volume_ratio = 1.10
        self.base_momentum_1h = 0.100
        self.base_momentum_3h = 0.200
        self.base_price_range = 1.022
        self.base_confidence_threshold = 50.0
        
        # Current adaptive thresholds (will be adjusted)
        self.current_volume_ratio = self.base_volume_ratio
        self.current_momentum_1h = self.base_momentum_1h
        self.current_momentum_3h = self.base_momentum_3h
        self.current_price_range = self.base_price_range
        self.current_confidence_threshold = self.base_confidence_threshold
        
    def analyze_market_volatility(self, symbols=['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE']):
        """Analyze current market volatility to adapt thresholds"""
        total_volume_ratios = []
        total_price_ranges = []
        total_momentums = []
        
        print("🔍 ANALYZING CURRENT MARKET VOLATILITY...")
        
        for symbol in symbols:
            try:
                # Get market data
                all_mids = self.info.all_mids()
                if symbol not in all_mids:
                    continue
                
                current_price = float(all_mids[symbol])
                
                # Get candles
                end_time = int(time.time() * 1000)
                start_time = end_time - (60 * 60 * 1000)  # 1 hour
                
                candles = self.info.candles_snapshot(symbol, "5m", start_time, end_time)
                if not candles or len(candles) < 6:
                    continue
                
                prices = [float(c['c']) for c in candles]
                volumes = [float(c['v']) for c in candles]
                
                # Calculate metrics
                momentum_1h = abs(prices[-1] - prices[-12]) / prices[-12] * 100 if len(prices) >= 12 else 0
                
                current_volume = volumes[-1] if volumes else 0
                avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else current_volume
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                recent_prices = prices[-6:] if len(prices) >= 6 else prices
                price_range = (max(recent_prices) - min(recent_prices)) / current_price * 100
                
                total_volume_ratios.append(volume_ratio)
                total_price_ranges.append(price_range)
                total_momentums.append(momentum_1h)
                
                print(f"   {symbol}: Vol={volume_ratio:.2f}x, Range={price_range:.3f}%, Mom={momentum_1h:.3f}%")
                
            except Exception as e:
                print(f"   Error analyzing {symbol}: {e}")
        
        if not total_volume_ratios:
            print("❌ No market data available")
            return
        
        # Calculate market volatility metrics
        avg_volume_ratio = np.mean(total_volume_ratios)
        avg_price_range = np.mean(total_price_ranges)
        avg_momentum = np.mean(total_momentums)
        
        print(f"\n📊 CURRENT MARKET STATE:")
        print(f"   Average Volume Ratio: {avg_volume_ratio:.2f}x")
        print(f"   Average Price Range: {avg_price_range:.3f}%")
        print(f"   Average Momentum: {avg_momentum:.3f}%")
        
        # Determine market condition and adapt thresholds
        if avg_volume_ratio < 0.5 and avg_price_range < 0.5:
            condition = "EXTREMELY LOW VOLATILITY"
            volatility_multiplier = 0.3  # Very aggressive
        elif avg_volume_ratio < 0.8 and avg_price_range < 0.8:
            condition = "LOW VOLATILITY"
            volatility_multiplier = 0.5  # Aggressive
        elif avg_volume_ratio < 1.2 and avg_price_range < 1.2:
            condition = "MODERATE VOLATILITY"
            volatility_multiplier = 0.7  # Moderate
        else:
            condition = "HIGH VOLATILITY"
            volatility_multiplier = 1.0  # Conservative (use base thresholds)
        
        print(f"\n🎯 MARKET CONDITION: {condition}")
        print(f"🔧 Volatility Multiplier: {volatility_multiplier:.1f}")
        
        # Adapt thresholds
        self.current_volume_ratio = self.base_volume_ratio * volatility_multiplier
        self.current_momentum_1h = self.base_momentum_1h * volatility_multiplier
        self.current_momentum_3h = self.base_momentum_3h * volatility_multiplier
        self.current_price_range = self.base_price_range * volatility_multiplier
        self.current_confidence_threshold = self.base_confidence_threshold * (0.6 + 0.4 * volatility_multiplier)
        
        print(f"\n🎯 ADAPTED THRESHOLDS:")
        print(f"   Volume Ratio: {self.current_volume_ratio:.2f}x (was {self.base_volume_ratio:.2f}x)")
        print(f"   Momentum 1h: {self.current_momentum_1h:.3f}% (was {self.base_momentum_1h:.3f}%)")
        print(f"   Momentum 3h: {self.current_momentum_3h:.3f}% (was {self.base_momentum_3h:.3f}%)")
        print(f"   Price Range: {self.current_price_range:.3f}% (was {self.base_price_range:.3f}%)")
        print(f"   Confidence: {self.current_confidence_threshold:.1f}% (was {self.base_confidence_threshold:.1f}%)")
        
    def detect_signal_adaptive(self, symbol):
        """Detect signals using adaptive thresholds"""
        try:
            # Get market data
            all_mids = self.info.all_mids()
            if symbol not in all_mids:
                return None
            
            current_price = float(all_mids[symbol])
            
            # Get candles
            end_time = int(time.time() * 1000)
            start_time = end_time - (60 * 60 * 1000)
            
            candles = self.info.candles_snapshot(symbol, "5m", start_time, end_time)
            if not candles or len(candles) < 6:
                return None
            
            prices = [float(c['c']) for c in candles]
            volumes = [float(c['v']) for c in candles]
            
            # Calculate momentum
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
            
            # ADAPTIVE confidence calculation
            confidence = 25  # Lower base for adaptive mode
            
            # Apply adaptive scoring
            if abs(momentum_1h) >= self.current_momentum_1h:
                confidence += abs(momentum_1h) * 15  # Higher weight in low volatility
            if abs(momentum_3h) >= self.current_momentum_3h:
                confidence += abs(momentum_3h) * 12
            if abs(momentum_5m) >= 0.05:  # Lower 5m threshold
                confidence += abs(momentum_5m) * 10
                
            if volume_ratio >= self.current_volume_ratio:
                confidence += (volume_ratio - self.current_volume_ratio + 1) * 30  # Higher weight
                
            if price_range >= self.current_price_range:
                confidence += price_range * 8  # Higher weight
            
            # Bonus for any momentum in low volatility periods
            if self.current_confidence_threshold < 40:  # Low volatility mode
                if abs(momentum_1h) > 0.05 or abs(momentum_5m) > 0.1:
                    confidence += 10  # Momentum bonus
            
            confidence = min(95, confidence)
            
            return {
                'symbol': symbol,
                'price': current_price,
                'momentum_1h': momentum_1h,
                'momentum_3h': momentum_3h,
                'momentum_5m': momentum_5m,
                'volume_ratio': volume_ratio,
                'price_range': price_range,
                'confidence': confidence,
                'signal': confidence >= self.current_confidence_threshold,
                'direction': 'BUY' if momentum_1h > 0 and momentum_3h >= 0 else 'SELL' if momentum_1h < 0 and momentum_3h <= 0 else 'MIXED'
            }
            
        except Exception as e:
            print(f"Error detecting signal for {symbol}: {e}")
            return None
    
    def test_adaptive_detection(self):
        """Test adaptive signal detection"""
        print("🚀 ADAPTIVE SIGNAL DETECTION TEST")
        print("=" * 60)
        
        # First, analyze market and adapt thresholds
        self.analyze_market_volatility()
        
        print(f"\n🔍 TESTING WITH ADAPTIVE THRESHOLDS...")
        print("=" * 60)
        
        signals_found = 0
        
        for symbol in ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE']:
            print(f"\n📊 Testing {symbol}...")
            
            result = self.detect_signal_adaptive(symbol)
            if result:
                print(f"   💰 Price: ${result['price']:,.2f}")
                print(f"   📊 Momentum 1h: {result['momentum_1h']:.3f}%")
                print(f"   📊 Volume ratio: {result['volume_ratio']:.2f}x")
                print(f"   📊 Price range: {result['price_range']:.3f}%")
                print(f"   🎯 Confidence: {result['confidence']:.1f}%")
                
                if result['signal']:
                    print(f"   ✅ ADAPTIVE SIGNAL! Direction: {result['direction']}")
                    signals_found += 1
                else:
                    print(f"   ❌ No signal ({result['confidence']:.1f}% < {self.current_confidence_threshold:.1f}%)")
        
        print(f"\n" + "=" * 60)
        print(f"📊 ADAPTIVE RESULTS: {signals_found}/5 signals found")
        
        if signals_found > 0:
            print("✅ SUCCESS! Adaptive thresholds are working!")
        else:
            print("❌ Still no signals - market may be exceptionally quiet")
            print("💡 Consider using 1-minute candles or waiting for volatility")

if __name__ == "__main__":
    detector = AdaptiveSignalDetector()
    detector.test_adaptive_detection() 