#!/usr/bin/env python3
"""
FINAL ADAPTIVE HYPERLIQUID TRADING BOT
Integrates adaptive signal detection with real trading capabilities
Automatically adjusts thresholds based on market volatility
"""

import asyncio
import logging
import time
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List
from dotenv import load_dotenv

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding to handle emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('final_adaptive_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    symbol: str
    action: str  # 'BUY' or 'SELL'
    confidence: float
    current_price: float
    target_price: float
    stop_loss: float
    reason: str

class FinalAdaptiveTradingBot:
    def __init__(self):
        # Load configuration
        self.testnet = os.getenv('HYPERLIQUID_TESTNET', 'false').lower() == 'true'
        self.private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
        self.account_address = os.getenv('HYPERLIQUID_ACCOUNT_ADDRESS')
        
        if not self.private_key or not self.account_address:
            raise ValueError("Missing required environment variables")
            
        # Initialize Hyperliquid connection
        base_url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
        self.info = Info(base_url, skip_ws=True)
        
        # Initialize exchange for trading (if needed)
        # Note: Removed testnet parameter due to API changes
        try:
            self.exchange = Exchange(self.private_key, base_url, account_address=self.account_address)
        except Exception as e:
            logger.warning(f"Exchange initialization failed: {e}")
            self.exchange = None
        
        # Trading parameters
        self.symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE']
        self.daily_trade_limit = 10
        self.trades_today = 0
        self.last_reset = datetime.now().date()
        
        # Historical optimized thresholds (from 9,935 data points)
        self.base_min_volume_ratio = 1.10
        self.base_min_momentum_1h = 0.100
        self.base_min_momentum_3h = 0.200
        self.base_min_price_range = 1.022
        self.base_confidence_threshold = 50.0
        
        # Adaptive parameters
        self.volatility_multiplier = 1.0
        self.current_confidence_threshold = self.base_confidence_threshold
        self.market_condition = "NORMAL"
        
        logger.info("FINAL ADAPTIVE HYPERLIQUID BOT INITIALIZED")
        
    def analyze_market_volatility(self):
        """Analyze current market volatility and adjust thresholds"""
        logger.info("Analyzing market volatility...")
        
        volatility_data = []
        
        for symbol in self.symbols[:3]:  # Quick sample of 3 symbols
            try:
                # Get recent candles with proper API signature
                end_time = int(time.time() * 1000)
                start_time = end_time - (100 * 60 * 1000)  # 100 minutes ago
                candles = self.info.candles_snapshot(symbol, "1m", start_time, end_time)
                if not candles or len(candles) < 50:
                    continue
                    
                # Calculate volatility metrics
                recent_candles = candles[-20:]  # Last 20 minutes
                volumes = [float(c['v']) for c in recent_candles]
                prices = [float(c['c']) for c in recent_candles]
                
                if len(volumes) > 1 and len(prices) > 1:
                    # Volume ratio (current vs average)
                    current_vol = volumes[-1]
                    avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
                    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0
                    
                    # Price range
                    price_range = (max(prices) - min(prices)) / min(prices) * 100
                    
                    # Momentum
                    momentum = (prices[-1] - prices[0]) / prices[0] * 100
                    
                    volatility_data.append({
                        'volume_ratio': vol_ratio,
                        'price_range': price_range,
                        'momentum': abs(momentum)
                    })
                    
            except Exception as e:
                logger.warning(f"Error analyzing {symbol} volatility: {e}")
                continue
        
        if volatility_data:
            # Calculate average market volatility
            avg_vol_ratio = sum(d['volume_ratio'] for d in volatility_data) / len(volatility_data)
            avg_price_range = sum(d['price_range'] for d in volatility_data) / len(volatility_data)
            avg_momentum = sum(d['momentum'] for d in volatility_data) / len(volatility_data)
            
            # Determine market condition and multiplier
            if avg_vol_ratio < 0.3 and avg_price_range < 0.5 and avg_momentum < 0.2:
                self.market_condition = "EXTREMELY LOW VOLATILITY"
                self.volatility_multiplier = 0.3  # Very aggressive
            elif avg_vol_ratio < 0.6 and avg_price_range < 1.0 and avg_momentum < 0.5:
                self.market_condition = "LOW VOLATILITY"
                self.volatility_multiplier = 0.5  # Aggressive
            elif avg_vol_ratio < 1.0 and avg_price_range < 2.0 and avg_momentum < 1.0:
                self.market_condition = "MODERATE VOLATILITY"
                self.volatility_multiplier = 0.7  # Moderate
            else:
                self.market_condition = "HIGH VOLATILITY"
                self.volatility_multiplier = 1.0  # Conservative
                
            # Adjust confidence threshold
            self.current_confidence_threshold = self.base_confidence_threshold * self.volatility_multiplier
            
            logger.info(f"Market: {self.market_condition}")
            logger.info(f"Adapted threshold: {self.current_confidence_threshold:.1f}%")

    async def detect_adaptive_opportunity(self, symbol: str) -> Optional[TradingSignal]:
        """Detect trading opportunities with adaptive thresholds"""
        try:
            # Get current price and recent data
            all_mids = self.info.all_mids()
            current_price = float(all_mids[symbol])
            
            # Get candlestick data for analysis with proper API signature
            end_time = int(time.time() * 1000)
            start_time_1h = end_time - (50 * 60 * 60 * 1000)  # 50 hours ago
            start_time_1m = end_time - (100 * 60 * 1000)      # 100 minutes ago
            
            candles_1h = self.info.candles_snapshot(symbol, "1h", start_time_1h, end_time)
            candles_1m = self.info.candles_snapshot(symbol, "1m", start_time_1m, end_time)
            
            if not candles_1h or not candles_1m or len(candles_1h) < 10 or len(candles_1m) < 50:
                return None
            
            # Calculate adaptive metrics
            momentum_1h = self._calculate_momentum(candles_1h, 1)
            momentum_3h = self._calculate_momentum(candles_1h, 3)
            volume_ratio = self._calculate_volume_ratio(candles_1m)
            price_range = self._calculate_price_range(candles_1m, 20)
            
            # Apply adaptive thresholds
            adapted_vol_threshold = self.base_min_volume_ratio * self.volatility_multiplier
            adapted_momentum_threshold = self.base_min_momentum_1h * self.volatility_multiplier
            adapted_range_threshold = self.base_min_price_range * self.volatility_multiplier
            
            # Calculate adaptive confidence
            confidence = self._calculate_adaptive_confidence(
                momentum_1h, momentum_3h, volume_ratio, price_range,
                adapted_momentum_threshold, adapted_vol_threshold, adapted_range_threshold
            )
            
            # Check if signal meets adaptive threshold
            if confidence >= self.current_confidence_threshold:
                # Determine direction
                if momentum_1h > 0 and momentum_3h > 0:
                    action = "BUY"
                    target_price = current_price * 1.02  # 2% target
                    stop_loss = current_price * 0.99   # 1% stop loss
                elif momentum_1h < 0 and momentum_3h < 0:
                    action = "SELL" 
                    target_price = current_price * 0.98  # 2% target
                    stop_loss = current_price * 1.01   # 1% stop loss
                else:
                    action = "HOLD"
                    target_price = current_price
                    stop_loss = current_price
                
                reason = f"{self.market_condition}: Mom={momentum_1h:.3f}%, Vol={volume_ratio:.1f}x"
                
                return TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    current_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    reason=reason
                )
            else:
                logger.info(f"    X {symbol}: No signal ({confidence:.1f}% < {self.current_confidence_threshold:.1f}%)")
                return None
                
        except Exception as e:
            logger.error(f"Error detecting opportunity for {symbol}: {e}")
            return None

    def _calculate_adaptive_confidence(self, momentum_1h, momentum_3h, volume_ratio, price_range,
                                     mom_threshold, vol_threshold, range_threshold):
        """Calculate confidence with adaptive weighting"""
        confidence = 30.0  # Base confidence
        
        # Momentum factors (adaptive)
        if abs(momentum_1h) >= mom_threshold:
            confidence += min(20, abs(momentum_1h) * 100)
        if abs(momentum_3h) >= mom_threshold * 2:
            confidence += min(15, abs(momentum_3h) * 50)
            
        # Volume factor (adaptive)
        if volume_ratio >= vol_threshold:
            confidence += min(25, volume_ratio * 10)
            
        # Price range factor (adaptive)
        if price_range >= range_threshold:
            confidence += min(15, price_range * 5)
        
        return min(confidence, 95.0)  # Cap at 95%

    async def execute_signal(self, signal: TradingSignal):
        """Execute REAL TRADING signal with safety checks"""
        logger.info(f"REAL TRADE SIGNAL FOUND!")
        logger.info(f"   {signal.symbol} {signal.action} - {signal.confidence:.1f}%")
        logger.info(f"   Target: ${signal.target_price:.2f} | Stop: ${signal.stop_loss:.2f}")
        logger.info(f"   {signal.reason}")
        
        if not self.exchange:
            logger.error("Exchange not initialized - cannot execute real trades")
            return False
            
        try:
            # Get current balance for position sizing
            user_state = self.info.user_state(self.account_address)
            balance = float(user_state['marginSummary']['accountValue'])
            logger.info(f"   Current balance: ${balance:.2f}")
            
            # Handle HOLD signals first (no trade needed)
            if signal.action == "HOLD":
                logger.info("   HOLD signal - no trade executed (market consolidation)")
                return True
            
            # Calculate position size (use 10% of balance per trade)
            risk_per_trade = balance * 0.10  # 10% max risk
            price_difference = abs(signal.current_price - signal.stop_loss)
            
            if price_difference > 0:
                position_size_usd = min(risk_per_trade, balance * 0.20)  # Max 20% per trade
                position_size_coins = position_size_usd / signal.current_price
                
                # Format size according to Hyperliquid requirements
                formatted_size = self._format_trade_size(position_size_coins)
                
                logger.info(f"   Position size: {formatted_size} {signal.symbol} (${position_size_usd:.2f})")
                
                # Create order
                if signal.action == "BUY":
                    order_price = signal.current_price * 1.001  # Slight premium for fills
                    is_buy = True
                elif signal.action == "SELL":
                    order_price = signal.current_price * 0.999  # Slight discount for fills
                    is_buy = False
                
                # Place the order
                logger.info(f"   PLACING REAL ORDER: {signal.action} {formatted_size} {signal.symbol} @ ${order_price:.2f}")
                
                order_result = self.exchange.order(
                    signal.symbol,                                    # coin
                    is_buy,                                          # is_buy  
                    float(formatted_size),                          # size
                    float(round(order_price, 2)),                   # limit_px
                    {'limit': {'tif': 'Ioc'}}                       # order_type (Immediate or Cancel)
                )
                
                logger.info(f"   Order result: {order_result}")
                
                # Check if order was successful
                order_success = self._verify_order_success(order_result)
                
                if order_success:
                    logger.info(f"   SUCCESS: Real trade executed!")
                    # Wait a moment then verify position
                    await asyncio.sleep(2)
                    await self._verify_position(signal.symbol)
                    self.trades_today += 1
                    return True
                else:
                    logger.error(f"   FAILED: Order was rejected")
                    return False
                    
            else:
                logger.error("   Invalid price difference for position sizing")
                return False
                
        except Exception as e:
            logger.error(f"   TRADE EXECUTION FAILED: {e}")
            return False
    
    def _format_trade_size(self, size: float) -> str:
        """Format trade size according to Hyperliquid requirements"""
        # Ensure size is a float
        size = float(size)
        if size >= 1:
            return f"{size:.6f}"
        else:
            return f"{size:.8f}"
    
    def _verify_order_success(self, order_result) -> bool:
        """Verify if order was successfully placed"""
        if isinstance(order_result, bool):
            return order_result
        elif isinstance(order_result, dict) and 'status' in order_result:
            if order_result['status'] == 'ok':
                # Check for errors in response
                response_data = order_result.get('response', {}).get('data', {})
                statuses = response_data.get('statuses', [])
                if statuses and any('error' in status for status in statuses):
                    error_msg = statuses[0].get('error', 'Unknown error')
                    logger.error(f"   Order rejected: {error_msg}")
                    return False
                return True
        return False
    
    async def _verify_position(self, symbol: str):
        """Verify position was opened"""
        try:
            user_state = self.info.user_state(self.account_address)
            positions = user_state.get('assetPositions', [])
            
            for pos in positions:
                if pos.get('position', {}).get('coin') == symbol:
                    position_size = float(pos['position']['szi'])
                    if abs(position_size) > 0.000001:  # Account for precision
                        logger.info(f"   POSITION CONFIRMED: {position_size:.6f} {symbol}")
                        return True
            
            logger.warning(f"   Position verification: No active position found for {symbol}")
            return False
            
        except Exception as e:
            logger.error(f"   Position verification failed: {e}")
            return False

    def _calculate_momentum(self, candles: List[dict], periods: int) -> float:
        """Calculate momentum over specified periods"""
        if len(candles) < periods + 1:
            return 0.0
        current_price = float(candles[-1]['c'])
        past_price = float(candles[-(periods+1)]['c'])
        return (current_price - past_price) / past_price * 100

    def _calculate_volume_ratio(self, candles: List[dict]) -> float:
        """Calculate current volume vs average volume ratio"""
        if len(candles) < 20:
            return 0.0
        volumes = [float(c['v']) for c in candles]
        current_vol = volumes[-1]
        avg_vol = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else 1
        return current_vol / avg_vol if avg_vol > 0 else 0

    def _calculate_price_range(self, candles: List[dict], periods: int) -> float:
        """Calculate price range over specified periods"""
        if len(candles) < periods:
            return 0.0
        recent_candles = candles[-periods:]
        highs = [float(c['h']) for c in recent_candles]
        lows = [float(c['l']) for c in recent_candles]
        price_range = (max(highs) - min(lows)) / min(lows) * 100
        return price_range

    async def run_adaptive_bot(self):
        """Main adaptive trading loop"""
        logger.info("Starting adaptive trading loop...")
        
        while True:
            try:
                # Reset daily trade count
                if datetime.now().date() > self.last_reset:
                    self.trades_today = 0
                    self.last_reset = datetime.now().date()
                    logger.info("Daily trade count reset")
                
                # Check trade limit
                if self.trades_today >= self.daily_trade_limit:
                    logger.info(f"Daily trade limit reached ({self.trades_today}/{self.daily_trade_limit})")
                    await asyncio.sleep(3600)  # Wait 1 hour
                    continue
                
                # Analyze market volatility every scan
                self.analyze_market_volatility()
                
                logger.info(f"Scanning {len(self.symbols)} symbols for adaptive opportunities...")
                
                # Scan all symbols
                for symbol in self.symbols:
                    signal = await self.detect_adaptive_opportunity(symbol)
                    if signal:
                        # Get current price for logging
                        all_mids = self.info.all_mids()
                        price = float(all_mids[symbol])
                        logger.info(f"    ${price:,.2f}")
                        
                        await self.execute_signal(signal)
                        break  # Process one signal at a time
                
                logger.info(f"Scan complete | Trades: {self.trades_today}/{self.daily_trade_limit}")
                logger.info(f"Waiting 30s...")
                
                # Wait with countdown
                for remaining in range(30, 0, -5):
                    logger.info(f"    Next scan in {remaining}s...")
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)

async def main():
    bot = FinalAdaptiveTradingBot()
    await bot.run_adaptive_bot()

if __name__ == "__main__":
    asyncio.run(main()) 