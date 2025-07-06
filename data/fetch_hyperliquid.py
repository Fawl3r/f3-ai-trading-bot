#!/usr/bin/env python3
"""
Hyperliquid Data Fetcher - Elite Trading Bot Data Pipeline
Handles 5000-candle limit with smart pagination and L2 order book data
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import os
import logging
from typing import List, Dict, Optional
import asyncio
import websockets
import aiohttp
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperliquidDataFetcher:
    """Professional-grade Hyperliquid data fetcher for elite trading"""
    
    def __init__(self):
        self.base_url = "https://api.hyperliquid.xyz"
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_request = 0
        self.min_interval = 0.1  # 100ms between requests
        
        # Cache for efficiency
        self.orderbook_cache = {}
        
    def _rate_limit(self):
        """Respect API rate limits"""
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()
    
    async def fetch_candles(self, coin: str, interval: str = "1m", 
                          days_back: int = 60) -> List[Dict]:
        """
        Fetch historical candles with pagination to handle 5000-bar limit
        """
        logger.info(f"📊 Fetching {days_back} days of {interval} data for {coin}")
        
        all_candles = []
        end_time = int(datetime.now().timestamp() * 1000)
        
        # Calculate how many windows we need
        bars_per_day = 1440 if interval == "1m" else 288  # 1m or 5m
        total_bars_needed = days_back * bars_per_day
        windows_needed = (total_bars_needed // 4000) + 1  # Use 4000 to be safe
        
        for window in range(windows_needed):
            # Calculate window times
            window_end = end_time - (window * 4000 * 60 * 1000)  # 4000 minutes back
            window_start = window_end - (4000 * 60 * 1000)
            
            if window_start < (datetime.now() - timedelta(days=days_back)).timestamp() * 1000:
                break
            
            self._rate_limit()
            
            payload = {
                "type": "candles",
                "coin": coin,
                "interval": interval,
                "startTime": window_start,
                "endTime": window_end
            }
            
            try:
                response = requests.post(f"{self.base_url}/info", json=payload)
                response.raise_for_status()
                
                window_data = response.json()
                if window_data and len(window_data) > 0:
                    all_candles.extend(window_data)
                    logger.info(f"  Window {window + 1}: {len(window_data)} candles")
                
            except Exception as e:
                logger.error(f"Error fetching window {window}: {e}")
                continue
        
        # Sort by timestamp and remove duplicates
        all_candles.sort(key=lambda x: x['t'])
        unique_candles = []
        seen_timestamps = set()
        
        for candle in all_candles:
            if candle['t'] not in seen_timestamps:
                unique_candles.append(candle)
                seen_timestamps.add(candle['t'])
        
        logger.info(f"✅ Fetched {len(unique_candles)} unique candles for {coin}")
        return unique_candles
    
    def save_candles(self, coin: str, candles: List[Dict], interval: str = "1m"):
        """Save candles to parquet for fast loading"""
        if not candles:
            return
        
        # Convert to standardized format
        df_data = []
        for candle in candles:
            df_data.append({
                'timestamp': candle['t'],
                'open': float(candle['o']),
                'high': float(candle['h']),
                'low': float(candle['l']),
                'close': float(candle['c']),
                'volume': float(candle['v']),
                'datetime': pd.to_datetime(candle['t'], unit='ms')
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Save to parquet
        coin_dir = self.data_dir / coin / interval
        coin_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = coin_dir / f"{coin}_{interval}_candles.parquet"
        df.to_parquet(filepath, compression='snappy')
        
        logger.info(f"💾 Saved {len(df)} candles to {filepath}")
        return filepath
    
    async def fetch_orderbook_snapshot(self, coin: str) -> Optional[Dict]:
        """Fetch current L2 order book snapshot"""
        self._rate_limit()
        
        payload = {
            "type": "l2Book",
            "coin": coin
        }
        
        try:
            response = requests.post(f"{self.base_url}/info", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching orderbook for {coin}: {e}")
            return None
    
    async def stream_orderbook(self, coins: List[str], duration_minutes: int = 60):
        """Stream live order book data for feature engineering"""
        logger.info(f"📡 Streaming orderbook for {coins} for {duration_minutes} minutes")
        
        orderbook_data = {coin: [] for coin in coins}
        
        async with websockets.connect(self.ws_url) as websocket:
            # Subscribe to order books
            for coin in coins:
                subscribe_msg = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "l2Book",
                        "coin": coin
                    }
                }
                await websocket.send(json.dumps(subscribe_msg))
            
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            while time.time() < end_time:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if 'data' in data and 'coin' in data['data']:
                        coin = data['data']['coin']
                        if coin in orderbook_data:
                            # Add timestamp
                            data['data']['timestamp'] = int(time.time() * 1000)
                            orderbook_data[coin].append(data['data'])
                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    break
        
        # Save orderbook data
        for coin, data in orderbook_data.items():
            if data:
                df = pd.DataFrame(data)
                filepath = self.data_dir / coin / f"{coin}_orderbook_stream.parquet"
                df.to_parquet(filepath, compression='snappy')
                logger.info(f"💾 Saved {len(df)} orderbook snapshots for {coin}")
        
        return orderbook_data
    
    def calculate_orderbook_features(self, orderbook_data: Dict) -> Dict:
        """Calculate advanced order book features for ML"""
        if not orderbook_data or 'levels' not in orderbook_data:
            return {}
        
        levels = orderbook_data['levels']
        bids = [(float(level['px']), float(level['sz'])) for level in levels if level['n'] > 0]
        asks = [(float(level['px']), float(level['sz'])) for level in levels if level['n'] < 0]
        
        if not bids or not asks:
            return {}
        
        # Sort bids (highest first) and asks (lowest first)
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        
        best_bid, best_ask = bids[0][0], asks[0][0]
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        # Volume calculations
        bid_volume_5 = sum(size for _, size in bids[:5])
        ask_volume_5 = sum(size for _, size in asks[:5])
        total_volume = bid_volume_5 + ask_volume_5
        
        # Order book imbalance
        obi = (bid_volume_5 - ask_volume_5) / total_volume if total_volume > 0 else 0
        
        # Spread metrics
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        
        # Depth ratio (how much volume within 0.1% of mid)
        depth_threshold = mid_price * 0.001  # 0.1%
        bid_depth = sum(size for price, size in bids if price >= mid_price - depth_threshold)
        ask_depth = sum(size for price, size in asks if price <= mid_price + depth_threshold)
        depth_ratio = bid_depth / ask_depth if ask_depth > 0 else 0
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_bps': spread_bps,
            'mid_price': mid_price,
            'obi': obi,
            'bid_volume_5': bid_volume_5,
            'ask_volume_5': ask_volume_5,
            'depth_ratio': depth_ratio,
            'total_volume': total_volume
        }
    
    async def fetch_complete_dataset(self, coins: List[str], days_back: int = 60) -> Dict:
        """Fetch complete dataset for all coins"""
        logger.info(f"🚀 Fetching complete dataset for {coins}")
        
        results = {}
        
        for coin in coins:
            logger.info(f"\n📈 Processing {coin}...")
            
            # Fetch candles
            candles = await self.fetch_candles(coin, "1m", days_back)
            candle_file = self.save_candles(coin, candles, "1m")
            
            # Fetch current orderbook
            orderbook = await self.fetch_orderbook_snapshot(coin)
            orderbook_features = self.calculate_orderbook_features(orderbook) if orderbook else {}
            
            results[coin] = {
                'candles_file': candle_file,
                'candle_count': len(candles),
                'orderbook_features': orderbook_features,
                'last_price': candles[-1]['c'] if candles else None
            }
            
            logger.info(f"✅ {coin}: {len(candles)} candles, last price: {results[coin]['last_price']}")
        
        return results

async def main():
    """Main data fetching pipeline"""
    fetcher = HyperliquidDataFetcher()
    
    # Elite trading pairs for double-up strategy
    elite_coins = ['SOL', 'BTC', 'ETH', 'AVAX', 'MATIC']
    
    # Fetch complete dataset
    results = await fetcher.fetch_complete_dataset(elite_coins, days_back=90)
    
    # Summary
    print(f"\n{'='*50}")
    print("HYPERLIQUID DATA FETCH COMPLETE")
    print(f"{'='*50}")
    
    total_candles = sum(r['candle_count'] for r in results.values())
    print(f"📊 Total candles: {total_candles:,}")
    print(f"💾 Data saved to: {fetcher.data_dir}")
    
    for coin, data in results.items():
        print(f"  {coin}: {data['candle_count']:,} candles @ ${data['last_price']}")
    
    print(f"\n🎯 Ready for elite feature engineering!")

if __name__ == "__main__":
    asyncio.run(main()) 