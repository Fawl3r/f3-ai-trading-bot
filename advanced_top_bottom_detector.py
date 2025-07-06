#!/usr/bin/env python3
"""
🎯 ADVANCED TOP/BOTTOM & LIQUIDITY ZONE DETECTOR
High-performance swing point and liquidity zone detection for sniper entries/exits

FEATURES:
✅ Swing High/Low Detection (5-bar pivot method)
✅ Order Book Liquidity Zone Analysis  
✅ Volume Cluster Detection
✅ Multi-timeframe Confluence
✅ Integration with existing bot logic
"""

import numpy as np
import pandas as pd
import requests
import time
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SwingPoint:
    """Swing high/low point with metadata"""
    price: float
    timestamp: int
    type: str  # 'high' or 'low'
    strength: float  # 0-100, based on volume and price movement
    volume: float
    distance_from_current: float  # % distance from current price

@dataclass
class LiquidityZone:
    """Liquidity zone with order book and volume data"""
    price_level: float
    volume_cluster: float
    order_book_depth: float
    zone_type: str  # 'bid_cluster', 'ask_cluster', 'mixed'
    strength: float  # 0-100, based on volume and depth
    distance_from_current: float  # % distance from current price

class AdvancedTopBottomDetector:
    """Advanced swing point and liquidity zone detector"""
    
    def __init__(self, api_url: str = "https://api.hyperliquid.xyz/info"):
        self.api_url = api_url
        self.min_swing_strength = 0.3  # Minimum strength for swing points
        self.min_liquidity_strength = 0.2  # Minimum strength for liquidity zones
        self.max_distance_pct = 5.0  # Maximum distance % for valid zones
        
    def detect_swing_points(self, candles: List[Dict], lookback: int = 50) -> Dict[str, List[SwingPoint]]:
        """
        Detect swing highs and lows using 5-bar pivot method
        
        Args:
            candles: List of candle data with 'h', 'l', 'c', 'v', 't' keys or 'high', 'low', 'close', 'volume', 'timestamp' keys
            lookback: Number of candles to analyze
            
        Returns:
            Dict with 'highs' and 'lows' lists of SwingPoint objects
        """
        if len(candles) < 10:
            return {'highs': [], 'lows': []}
        
        # Convert to DataFrame for easier analysis
        recent_candles = candles[-lookback:]
        df = pd.DataFrame(recent_candles)
        
        # Check if candles are in formatted or raw format
        sample_candle = recent_candles[0]
        if 'high' in sample_candle:
            # Formatted candles
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['timestamp'] = df.get('timestamp', df.index).astype(int)
        else:
            # Raw candles
            df['high'] = df['h'].astype(float)
            df['low'] = df['l'].astype(float)
            df['close'] = df['c'].astype(float)
            df['volume'] = df['v'].astype(float)
            df['timestamp'] = df['t'].astype(int)
        
        swing_highs = []
        swing_lows = []
        current_price = df['close'].iloc[-1]
        
        # 5-bar pivot detection (2 bars before, current, 2 bars after)
        for i in range(2, len(df) - 2):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_volume = df['volume'].iloc[i]
            current_timestamp = df['timestamp'].iloc[i]
            
            # Check for swing high (5-bar pivot)
            if (current_high > df['high'].iloc[i-1] and current_high > df['high'].iloc[i-2] and
                current_high > df['high'].iloc[i+1] and current_high > df['high'].iloc[i+2]):
                
                # Calculate strength based on volume and price movement
                price_movement = (current_high - df['low'].iloc[i]) / df['low'].iloc[i] * 100
                volume_ratio = current_volume / df['volume'].iloc[i-5:i+5].mean() if i >= 5 else 1.0
                strength = min(100, (price_movement * 10 + volume_ratio * 50) / 2)
                
                if strength >= self.min_swing_strength * 100:
                    distance_pct = abs(current_high - current_price) / current_price * 100
                    
                    swing_highs.append(SwingPoint(
                        price=current_high,
                        timestamp=current_timestamp,
                        type='high',
                        strength=strength,
                        volume=current_volume,
                        distance_from_current=distance_pct
                    ))
            
            # Check for swing low (5-bar pivot)
            if (current_low < df['low'].iloc[i-1] and current_low < df['low'].iloc[i-2] and
                current_low < df['low'].iloc[i+1] and current_low < df['low'].iloc[i+2]):
                
                # Calculate strength based on volume and price movement
                price_movement = (df['high'].iloc[i] - current_low) / current_low * 100
                volume_ratio = current_volume / df['volume'].iloc[i-5:i+5].mean() if i >= 5 else 1.0
                strength = min(100, (price_movement * 10 + volume_ratio * 50) / 2)
                
                if strength >= self.min_swing_strength * 100:
                    distance_pct = abs(current_low - current_price) / current_price * 100
                    
                    swing_lows.append(SwingPoint(
                        price=current_low,
                        timestamp=current_timestamp,
                        type='low',
                        strength=strength,
                        volume=current_volume,
                        distance_from_current=distance_pct
                    ))
        
        # Sort by strength and filter by distance
        swing_highs = [h for h in swing_highs if h.distance_from_current <= self.max_distance_pct]
        swing_lows = [l for l in swing_lows if l.distance_from_current <= self.max_distance_pct]
        
        swing_highs.sort(key=lambda x: x.strength, reverse=True)
        swing_lows.sort(key=lambda x: x.strength, reverse=True)
        
        logger.info(f"🎯 Detected {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
        
        return {
            'highs': swing_highs[:5],  # Top 5 strongest
            'lows': swing_lows[:5]     # Top 5 strongest
        }
    
    def analyze_liquidity_zones(self, symbol: str, current_price: float) -> List[LiquidityZone]:
        """
        Analyze order book for liquidity zones and volume clusters
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            current_price: Current market price
            
        Returns:
            List of LiquidityZone objects
        """
        try:
            # Get order book data
            payload = {"type": "l2Book", "coin": symbol}
            response = requests.post(self.api_url, json=payload, timeout=3)
            
            if response.status_code != 200:
                logger.warning(f"Failed to get order book for {symbol}")
                return []
            
            book_data = response.json()
            if len(book_data) < 2:
                return []
            
            bids = book_data[0] if len(book_data) > 0 else []
            asks = book_data[1] if len(book_data) > 1 else []
            
            liquidity_zones = []
            
            # Analyze bid clusters (support zones)
            bid_clusters = self._find_volume_clusters(bids, 'bid', current_price)
            for cluster in bid_clusters:
                if cluster['strength'] >= self.min_liquidity_strength * 100:
                    liquidity_zones.append(LiquidityZone(
                        price_level=cluster['price'],
                        volume_cluster=cluster['volume'],
                        order_book_depth=cluster['depth'],
                        zone_type='bid_cluster',
                        strength=cluster['strength'],
                        distance_from_current=cluster['distance_pct']
                    ))
            
            # Analyze ask clusters (resistance zones)
            ask_clusters = self._find_volume_clusters(asks, 'ask', current_price)
            for cluster in ask_clusters:
                if cluster['strength'] >= self.min_liquidity_strength * 100:
                    liquidity_zones.append(LiquidityZone(
                        price_level=cluster['price'],
                        volume_cluster=cluster['volume'],
                        order_book_depth=cluster['depth'],
                        zone_type='ask_cluster',
                        strength=cluster['strength'],
                        distance_from_current=cluster['distance_pct']
                    ))
            
            # Sort by strength and filter by distance
            liquidity_zones = [z for z in liquidity_zones if z.distance_from_current <= self.max_distance_pct]
            liquidity_zones.sort(key=lambda x: x.strength, reverse=True)
            
            logger.info(f"💧 Detected {len(liquidity_zones)} liquidity zones for {symbol}")
            
            return liquidity_zones[:10]  # Top 10 strongest zones
            
        except Exception as e:
            logger.error(f"Liquidity zone analysis error for {symbol}: {e}")
            return []
    
    def _find_volume_clusters(self, orders: List[Dict], order_type: str, current_price: float) -> List[Dict]:
        """Find volume clusters in order book"""
        if not orders:
            return []
        
        clusters = []
        cluster_threshold = 0.001  # 0.1% price difference for clustering
        
        # Group orders by price proximity
        grouped_orders = {}
        for order in orders[:20]:  # Analyze top 20 orders
            price = float(order['px'])
            size = float(order['sz'])
            
            # Find existing cluster or create new one
            cluster_found = False
            for cluster_price in grouped_orders:
                if abs(price - cluster_price) / cluster_price <= cluster_threshold:
                    grouped_orders[cluster_price]['volume'] += size
                    grouped_orders[cluster_price]['orders'].append({'price': price, 'size': size})
                    cluster_found = True
                    break
            
            if not cluster_found:
                grouped_orders[price] = {
                    'volume': size,
                    'orders': [{'price': price, 'size': size}],
                    'avg_price': price
                }
        
        # Calculate cluster strength and distance
        for price, cluster_data in grouped_orders.items():
            total_volume = cluster_data['volume']
            avg_price = cluster_data['avg_price']
            distance_pct = abs(avg_price - current_price) / current_price * 100
            
            # Calculate strength based on volume and proximity
            volume_score = min(100, total_volume * 100)  # Normalize volume
            proximity_score = max(0, 100 - distance_pct * 10)  # Closer = higher score
            strength = (volume_score + proximity_score) / 2
            
            clusters.append({
                'price': avg_price,
                'volume': total_volume,
                'depth': len(cluster_data['orders']),
                'strength': strength,
                'distance_pct': distance_pct
            })
        
        return clusters
    
    def get_entry_exit_signals(self, symbol: str, candles: List[Dict], current_price: float) -> Dict:
        """
        Get comprehensive entry/exit signals based on swing points and liquidity zones
        
        Args:
            symbol: Trading symbol
            candles: Historical candle data
            current_price: Current market price
            
        Returns:
            Dict with entry/exit signals and confidence scores
        """
        # Detect swing points
        swing_points = self.detect_swing_points(candles)
        
        # Analyze liquidity zones
        liquidity_zones = self.analyze_liquidity_zones(symbol, current_price)
        
        # Find nearest swing points
        nearest_high = min(swing_points['highs'], key=lambda x: x.distance_from_current) if swing_points['highs'] else None
        nearest_low = min(swing_points['lows'], key=lambda x: x.distance_from_current) if swing_points['lows'] else None
        
        # Find nearest liquidity zones
        bid_zones = [z for z in liquidity_zones if z.zone_type == 'bid_cluster']
        ask_zones = [z for z in liquidity_zones if z.zone_type == 'ask_cluster']
        
        nearest_bid_zone = min(bid_zones, key=lambda x: x.distance_from_current) if bid_zones else None
        nearest_ask_zone = min(ask_zones, key=lambda x: x.distance_from_current) if ask_zones else None
        
        # Generate signals
        signals = {
            'long_entry': False,
            'short_entry': False,
            'long_exit': False,
            'short_exit': False,
            'confidence': 0,
            'reason': '',
            'swing_points': swing_points,
            'liquidity_zones': liquidity_zones
        }
        
        # Long entry: Near swing low + bid cluster
        if nearest_low and nearest_low.distance_from_current <= 1.0:  # Within 1%
            if nearest_bid_zone and nearest_bid_zone.distance_from_current <= 1.0:
                signals['long_entry'] = True
                signals['confidence'] = min(100, (nearest_low.strength + nearest_bid_zone.strength) / 2)
                signals['reason'] = f"Near swing low ({nearest_low.price:.2f}) + bid cluster ({nearest_bid_zone.price_level:.2f})"
        
        # Short entry: Near swing high + ask cluster
        if nearest_high and nearest_high.distance_from_current <= 1.0:  # Within 1%
            if nearest_ask_zone and nearest_ask_zone.distance_from_current <= 1.0:
                signals['short_entry'] = True
                signals['confidence'] = min(100, (nearest_high.strength + nearest_ask_zone.strength) / 2)
                signals['reason'] = f"Near swing high ({nearest_high.price:.2f}) + ask cluster ({nearest_ask_zone.price_level:.2f})"
        
        # Exit signals (opposite zones)
        if nearest_ask_zone and nearest_ask_zone.distance_from_current <= 0.5:  # Very close
            signals['long_exit'] = True
        if nearest_bid_zone and nearest_bid_zone.distance_from_current <= 0.5:  # Very close
            signals['short_exit'] = True
        
        return signals
    
    def get_market_structure(self, candles: List[Dict]) -> Dict:
        """
        Analyze overall market structure for trend confirmation
        
        Args:
            candles: Historical candle data
            
        Returns:
            Dict with market structure analysis
        """
        if len(candles) < 20:
            return {'trend': 'neutral', 'structure_score': 0, 'strength': 0}
        
        recent_candles = candles[-20:]
        
        # Check if candles are in formatted or raw format
        sample_candle = recent_candles[0]
        if 'high' in sample_candle:
            # Formatted candles
            df = pd.DataFrame(recent_candles)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
        else:
            # Raw candles
            df = pd.DataFrame(recent_candles)
            df['high'] = df['h'].astype(float)
            df['low'] = df['l'].astype(float)
            df['close'] = df['c'].astype(float)
        
        # Find recent swing points
        swing_points = self.detect_swing_points(candles, lookback=20)
        
        structure_score = 0
        
        # Analyze higher highs and higher lows
        if len(swing_points['highs']) >= 2:
            highs = sorted(swing_points['highs'], key=lambda x: x.timestamp)
            if highs[-1].price > highs[-2].price:
                structure_score += 1
            else:
                structure_score -= 1
        
        if len(swing_points['lows']) >= 2:
            lows = sorted(swing_points['lows'], key=lambda x: x.timestamp)
            if lows[-1].price > lows[-2].price:
                structure_score += 1
            else:
                structure_score -= 1
        
        # Determine trend
        if structure_score >= 1:
            trend = 'bullish'
        elif structure_score <= -1:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {
            'trend': trend,
            'structure_score': structure_score,
            'strength': abs(structure_score),
            'swing_points': swing_points
        }

    def analyze_market_structure(self, candles: List[Dict], current_price: float = None) -> Dict:
        """
        Alias for get_market_structure for compatibility with integration and tests.
        Args:
            candles: Historical candle data
            current_price: (Unused, for compatibility)
        Returns:
            Dict with market structure analysis
        """
        return self.get_market_structure(candles)

# Example usage and testing
if __name__ == "__main__":
    detector = AdvancedTopBottomDetector()
    
    # Test with sample data
    sample_candles = [
        {'h': 50000, 'l': 49000, 'c': 49500, 'v': 1000, 't': 1000000},
        {'h': 51000, 'l': 49500, 'c': 50500, 'v': 1200, 't': 1000060},
        {'h': 52000, 'l': 50000, 'c': 51500, 'v': 1500, 't': 1000120},
        {'h': 51500, 'l': 50500, 'c': 51000, 'v': 800, 't': 1000180},
        {'h': 52500, 'l': 51000, 'c': 52000, 'v': 2000, 't': 1000240},
    ]
    
    # Test swing point detection
    swing_points = detector.detect_swing_points(sample_candles)
    print(f"Swing points detected: {len(swing_points['highs'])} highs, {len(swing_points['lows'])} lows")
    
    # Test market structure
    structure = detector.get_market_structure(sample_candles)
    print(f"Market structure: {structure['trend']} (score: {structure['structure_score']})") 