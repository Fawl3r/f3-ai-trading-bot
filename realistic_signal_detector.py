#!/usr/bin/env python3
"""
Realistic Signal Detector for Current Market Conditions
Designed to work in sideways/low volatility markets
"""

import asyncio
import time
import numpy as np
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

@dataclass
class RealisticSignal:
    symbol: str
    action: str  # 'BUY' or 'SELL'
    confidence: float
    target_price: float
    stop_loss: float
    reason: str
    timestamp: datetime

class RealisticSignalDetector:
    def __init__(self, info):
        self.info = info
        
        # 🎯 REALISTIC THRESHOLDS FOR CURRENT MARKET
        self.min_volume_ratio = 1.2      # 20% volume increase (was 1.5)
        self.min_momentum_5m = 0.2       # 0.2% 5m momentum (was 0.5%)
        self.min_momentum_10m = 0.4      # 0.4% 10m momentum (was 1.0%)
        self.min_price_range = 0.15      # 0.15% price range (was 0.3%)
        self.confidence_threshold = 55.0  # 55% threshold for current market
        
    async def detect_opportunity(self, symbol: str) -> Optional[RealisticSignal]:
        """Realistic opportunity detection for current market conditions"""
        try:
            # Get market data
            all_mids = self.info.all_mids()
            if symbol not in all_mids:
                return None
            
            current_price = float(all_mids[symbol])
            
            # Get 30 minutes of 5m candles for better sensitivity
            end_time = int(time.time() * 1000)
            start_time = end_time - (30 * 60 * 1000)  # 30 minutes
            
            candles = self.info.candles_snapshot(symbol, "5m", start_time, end_time)
            if not candles or len(candles) < 4:
                return None
            
            # Extract data
            prices = [float(c['c']) for c in candles]
            volumes = [float(c['v']) for c in candles]
            
            # Calculate momentum over multiple timeframes
            price_5m_ago = prices[-2] if len(prices) >= 2 else current_price
            price_10m_ago = prices[-3] if len(prices) >= 3 else current_price
            price_15m_ago = prices[-4] if len(prices) >= 4 else current_price
            
            momentum_5m = (current_price - price_5m_ago) / price_5m_ago * 100
            momentum_10m = (current_price - price_10m_ago) / price_10m_ago * 100
            momentum_15m = (current_price - price_15m_ago) / price_15m_ago * 100
            
            # Volume analysis
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price volatility
            price_range = (max(prices[-4:]) - min(prices[-4:])) / current_price * 100
            
            # Enhanced momentum detection
            momentum_strength = max(abs(momentum_5m), abs(momentum_10m), abs(momentum_15m))
            momentum_consistency = sum([1 for m in [momentum_5m, momentum_10m, momentum_15m] if abs(m) > 0.1])
            
            # BULLISH SIGNALS - More realistic criteria
            if (momentum_5m > self.min_momentum_5m and 
                momentum_10m > self.min_momentum_10m and 
                volume_ratio > self.min_volume_ratio and 
                price_range > self.min_price_range):
                
                confidence = min(95.0, 
                    40 +  # Base confidence
                    (momentum_5m * 8) +    # 5m momentum boost
                    (momentum_10m * 6) +   # 10m momentum boost
                    (volume_ratio * 8) +   # Volume boost
                    (price_range * 3) +    # Volatility boost
                    (momentum_consistency * 5)  # Consistency bonus
                )
                
                if confidence >= self.confidence_threshold:
                    return RealisticSignal(
                        symbol=symbol,
                        action="BUY",
                        confidence=confidence,
                        target_price=current_price * 1.04,  # 4% target
                        stop_loss=current_price * 0.98,    # 2% stop
                        reason=f"Bull: 5m{momentum_5m:+.2f}%, 10m{momentum_10m:+.2f}%, Vol{volume_ratio:.1f}x",
                        timestamp=datetime.now()
                    )
            
            # BEARISH SIGNALS - More realistic criteria
            elif (momentum_5m < -self.min_momentum_5m and 
                  momentum_10m < -self.min_momentum_10m and 
                  volume_ratio > self.min_volume_ratio and 
                  price_range > self.min_price_range):
                
                confidence = min(95.0, 
                    40 +  # Base confidence
                    (abs(momentum_5m) * 8) +
                    (abs(momentum_10m) * 6) +
                    (volume_ratio * 8) +
                    (price_range * 3) +
                    (momentum_consistency * 5)
                )
                
                if confidence >= self.confidence_threshold:
                    return RealisticSignal(
                        symbol=symbol,
                        action="SELL",
                        confidence=confidence,
                        target_price=current_price * 0.96,  # 4% target
                        stop_loss=current_price * 1.02,    # 2% stop
                        reason=f"Bear: 5m{momentum_5m:+.2f}%, 10m{momentum_10m:+.2f}%, Vol{volume_ratio:.1f}x",
                        timestamp=datetime.now()
                    )
            
            # BREAKOUT SIGNALS - For range-bound markets
            elif (volume_ratio > 1.8 and price_range > 0.4 and momentum_strength > 0.3):
                direction = "BUY" if momentum_5m + momentum_10m > 0 else "SELL"
                confidence = min(85.0, 50 + (volume_ratio * 10) + (price_range * 5) + (momentum_strength * 8))
                
                if confidence >= self.confidence_threshold:
                    multiplier = 1.03 if direction == "BUY" else 0.97
                    stop_multiplier = 0.985 if direction == "BUY" else 1.015
                    
                    return RealisticSignal(
                        symbol=symbol,
                        action=direction,
                        confidence=confidence,
                        target_price=current_price * multiplier,
                        stop_loss=current_price * stop_multiplier,
                        reason=f"Breakout: Vol{volume_ratio:.1f}x, Range{price_range:.2f}%",
                        timestamp=datetime.now()
                    )
            
            # Return weak signal for logging (ALWAYS return something)
            weak_confidence = min(59.0, 25 + (momentum_strength * 15) + (volume_ratio * 8) + (price_range * 5))
            
            return RealisticSignal(
                symbol=symbol,
                action="HOLD",
                confidence=weak_confidence,
                target_price=current_price,
                stop_loss=current_price,
                reason=f"Weak: 5m{momentum_5m:+.2f}%, Vol{volume_ratio:.1f}x, Range{price_range:.2f}%",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Signal detection error for {symbol}: {e}")
            return None

# Test the realistic detector
async def test_realistic_detector():
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
    base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    info = Info(base_url, skip_ws=True)
    
    detector = RealisticSignalDetector(info)
    
    print("🎯 Testing Realistic Signal Detector...")
    print(f"📊 Thresholds: Vol≥{detector.min_volume_ratio:.1f}x, Mom≥{detector.min_momentum_5m:.1f}%, Conf≥{detector.confidence_threshold:.0f}%")
    print("=" * 60)
    
    for symbol in ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX']:
        signal = await detector.detect_opportunity(symbol)
        if signal:
            color = "🟢" if signal.confidence >= detector.confidence_threshold else "🟡"
            print(f"{color} {symbol}: {signal.confidence:.1f}% - {signal.reason}")
            if signal.confidence >= detector.confidence_threshold:
                print(f"   🚨 SIGNAL: {signal.action} @ ${signal.target_price:.2f}")
        else:
            print(f"❌ {symbol}: Failed to analyze")

if __name__ == "__main__":
    asyncio.run(test_realistic_detector()) 