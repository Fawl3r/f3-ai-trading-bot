#!/usr/bin/env python3
"""
Simple Working Signal Detector
Replaces the complex logic with reliable momentum detection
"""

import asyncio
import time
import numpy as np
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

@dataclass
class SimpleSignal:
    symbol: str
    action: str  # 'BUY' or 'SELL'
    confidence: float
    target_price: float
    stop_loss: float
    reason: str
    timestamp: datetime

class SimpleSignalDetector:
    def __init__(self, info):
        self.info = info
        
    async def detect_opportunity(self, symbol: str) -> Optional[SimpleSignal]:
        """Simplified but working opportunity detection"""
        try:
            # Get market data
            all_mids = self.info.all_mids()
            if symbol not in all_mids:
                return None
            
            current_price = float(all_mids[symbol])
            
            # Get simple candles data (1 hour lookback)
            end_time = int(time.time() * 1000)
            start_time = end_time - (60 * 60 * 1000)  # 1 hour
            
            candles = self.info.candles_snapshot(symbol, "5m", start_time, end_time)
            if not candles or len(candles) < 6:
                return None
            
            # Extract closing prices
            prices = [float(c['c']) for c in candles]
            volumes = [float(c['v']) for c in candles]
            
            # Simple momentum calculation
            price_5m_ago = prices[-2] if len(prices) >= 2 else current_price
            price_10m_ago = prices[-3] if len(prices) >= 3 else current_price
            price_15m_ago = prices[-4] if len(prices) >= 4 else current_price
            
            # Calculate momentum over different timeframes
            momentum_5m = (current_price - price_5m_ago) / price_5m_ago * 100
            momentum_10m = (current_price - price_10m_ago) / price_10m_ago * 100
            momentum_15m = (current_price - price_15m_ago) / price_15m_ago * 100
            
            # Volume analysis
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price volatility (range of recent prices)
            price_range = (max(prices[-6:]) - min(prices[-6:])) / current_price * 100
            
            # BULLISH SIGNALS
            if (momentum_5m > 0.5 and momentum_10m > 1.0 and 
                volume_ratio > 1.5 and price_range > 0.3):
                
                confidence = min(90.0, 
                    50 + (momentum_5m * 3) + (momentum_10m * 2) + 
                    (volume_ratio * 5) + (price_range * 2)
                )
                
                if confidence >= 75.0:
                    return SimpleSignal(
                        symbol=symbol,
                        action="BUY",
                        confidence=confidence,
                        target_price=current_price * 1.06,  # 6% target
                        stop_loss=current_price * 0.97,    # 3% stop
                        reason=f"5m: +{momentum_5m:.2f}%, 10m: +{momentum_10m:.2f}%, Vol: {volume_ratio:.1f}x",
                        timestamp=datetime.now()
                    )
            
            # BEARISH SIGNALS  
            elif (momentum_5m < -0.5 and momentum_10m < -1.0 and 
                  volume_ratio > 1.5 and price_range > 0.3):
                
                confidence = min(90.0, 
                    50 + (abs(momentum_5m) * 3) + (abs(momentum_10m) * 2) + 
                    (volume_ratio * 5) + (price_range * 2)
                )
                
                if confidence >= 75.0:
                    return SimpleSignal(
                        symbol=symbol,
                        action="SELL",
                        confidence=confidence,
                        target_price=current_price * 0.94,  # 6% target
                        stop_loss=current_price * 1.03,    # 3% stop
                        reason=f"5m: {momentum_5m:.2f}%, 10m: {momentum_10m:.2f}%, Vol: {volume_ratio:.1f}x",
                        timestamp=datetime.now()
                    )
            
            # Return weak signal for logging
            max_momentum = max(abs(momentum_5m), abs(momentum_10m))
            confidence = min(74.0, 30 + (max_momentum * 5) + (volume_ratio * 3))
            
            return SimpleSignal(
                symbol=symbol,
                action="HOLD",
                confidence=confidence,
                target_price=current_price,
                stop_loss=current_price,
                reason=f"Weak momentum: {momentum_5m:.2f}%/5m, {momentum_10m:.2f}%/10m",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Signal detection error for {symbol}: {e}")
            return None

# Test the detector
async def test_detector():
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
    base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    info = Info(base_url, skip_ws=True)
    
    detector = SimpleSignalDetector(info)
    
    print("🔍 Testing Simple Signal Detector...")
    for symbol in ['BTC', 'ETH', 'SOL']:
        signal = await detector.detect_opportunity(symbol)
        if signal:
            print(f"📊 {symbol}: {signal.confidence:.1f}% - {signal.reason}")
        else:
            print(f"❌ {symbol}: Failed to analyze")

if __name__ == "__main__":
    asyncio.run(test_detector()) 