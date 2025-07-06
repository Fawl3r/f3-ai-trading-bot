#!/usr/bin/env python3
"""
FINAL ADAPTIVE HYPERLIQUID TRADING BOT WITH ENHANCED TOP/BOTTOM & LIQUIDITY ZONE DETECTION
Integrates adaptive signal detection with real trading capabilities
Automatically adjusts thresholds based on market volatility
Enhanced with swing point and liquidity zone analysis for sniper entries/exits
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
from eth_account import Account as eth_account

# Import enhanced detection features
from advanced_top_bottom_detector import AdvancedTopBottomDetector

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
    enhanced_features: Dict = None  # New field for enhanced features

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
        # Note: Using correct eth_account pattern for Exchange initialization
        try:
            account = eth_account.from_key(self.private_key)
            self.exchange = Exchange(account, base_url)
            logger.info("✅ Exchange initialized for REAL TRADING")
        except Exception as e:
            logger.warning(f"Exchange initialization failed: {e}")
            self.exchange = None
        
        # Initialize enhanced top/bottom detector
        self.enhanced_detector = AdvancedTopBottomDetector()
        logger.info("🎯 Enhanced Top/Bottom & Liquidity Zone Detector initialized")
        
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
        
        # Enhanced feature parameters
        self.use_enhanced_features = True  # Toggle for enhanced features
        self.enhanced_confidence_boost = 15  # Confidence boost for enhanced signals
        self.min_swing_distance = 1.0  # Maximum distance % to swing points
        self.min_liquidity_distance = 1.0  # Maximum distance % to liquidity zones
        
        logger.info("FINAL ADAPTIVE HYPERLIQUID BOT WITH ENHANCED FEATURES INITIALIZED")

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

    def analyze_enhanced_features(self, symbol: str, candles: List[dict], current_price: float) -> Dict:
        """Analyze enhanced top/bottom and liquidity zone features"""
        try:
            if not self.use_enhanced_features:
                return {'enhanced_confidence': 0, 'swing_points': {}, 'liquidity_zones': [], 'market_structure': {}}
            
            # Get enhanced signals from detector
            enhanced_signals = self.enhanced_detector.get_entry_exit_signals(symbol, candles, current_price)
            
            # Get market structure
            market_structure = self.enhanced_detector.get_market_structure(candles)
            
            # Calculate enhanced confidence boost
            enhanced_confidence = enhanced_signals['confidence']
            
            # Add market structure bonus
            if market_structure['trend'] == 'bullish' and enhanced_signals['long_entry']:
                enhanced_confidence += self.enhanced_confidence_boost
            elif market_structure['trend'] == 'bearish' and enhanced_signals['short_entry']:
                enhanced_confidence += self.enhanced_confidence_boost
            
            return {
                'enhanced_confidence': enhanced_confidence,
                'swing_points': enhanced_signals['swing_points'],
                'liquidity_zones': enhanced_signals['liquidity_zones'],
                'market_structure': market_structure,
                'long_entry': enhanced_signals['long_entry'],
                'short_entry': enhanced_signals['short_entry'],
                'reason': enhanced_signals['reason']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing enhanced features for {symbol}: {e}")
            return {
                'enhanced_confidence': 0,
                'swing_points': {},
                'liquidity_zones': [],
                'market_structure': {},
                'long_entry': False,
                'short_entry': False,
                'reason': f"Error: {e}"
            }

    async def detect_adaptive_opportunity(self, symbol: str) -> Optional[TradingSignal]:
        """Detect trading opportunities with adaptive thresholds and enhanced features"""
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
            
            # Calculate adaptive metrics (original logic)
            momentum_1h = self._calculate_momentum(candles_1h, 1)
            momentum_3h = self._calculate_momentum(candles_1h, 3)
            volume_ratio = self._calculate_volume_ratio(candles_1m)
            price_range = self._calculate_price_range(candles_1m, 20)
            
            # Apply adaptive thresholds
            adapted_vol_threshold = self.base_min_volume_ratio * self.volatility_multiplier
            adapted_momentum_threshold = self.base_min_momentum_1h * self.volatility_multiplier
            adapted_range_threshold = self.base_min_price_range * self.volatility_multiplier
            
            # Calculate base adaptive confidence (original logic)
            base_confidence = self._calculate_adaptive_confidence(
                momentum_1h, momentum_3h, volume_ratio, price_range,
                adapted_momentum_threshold, adapted_vol_threshold, adapted_range_threshold
            )
            
            # Analyze enhanced features
            enhanced_features = self.analyze_enhanced_features(symbol, candles_1m, current_price)
            enhanced_confidence = enhanced_features['enhanced_confidence']
            
            # Combine base confidence with enhanced confidence
            total_confidence = (base_confidence + enhanced_confidence) / 2
            
            # Check if signal meets adaptive threshold
            if total_confidence >= self.current_confidence_threshold:
                # Determine direction with enhanced logic
                action = "HOLD"
                target_price = current_price
                stop_loss = current_price
                
                # Enhanced direction logic
                if enhanced_features['long_entry'] and momentum_1h > 0:
                    action = "BUY"
                    target_price = current_price * 1.02  # 2% target
                    stop_loss = current_price * 0.99   # 1% stop loss
                elif enhanced_features['short_entry'] and momentum_1h < 0:
                    action = "SELL" 
                    target_price = current_price * 0.98  # 2% target
                    stop_loss = current_price * 1.01   # 1% stop loss
                elif momentum_1h > 0 and momentum_3h > 0:
                    action = "BUY"
                    target_price = current_price * 1.02  # 2% target
                    stop_loss = current_price * 0.99   # 1% stop loss
                elif momentum_1h < 0 and momentum_3h < 0:
                    action = "SELL" 
                    target_price = current_price * 0.98  # 2% target
                    stop_loss = current_price * 1.01   # 1% stop loss
                
                # Enhanced reason with top/bottom and liquidity zone info
                enhanced_reason = enhanced_features['reason'] if enhanced_features['reason'] else ""
                base_reason = f"{self.market_condition}: Mom={momentum_1h:.3f}%, Vol={volume_ratio:.1f}x"
                
                if enhanced_reason:
                    reason = f"{base_reason} | Enhanced: {enhanced_reason}"
                else:
                    reason = base_reason
                
                return TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=total_confidence,
                    current_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    reason=reason,
                    enhanced_features=enhanced_features
                )
            else:
                logger.info(f"    X {symbol}: No signal (Base: {base_confidence:.1f}%, Enhanced: {enhanced_confidence:.1f}%, Total: {total_confidence:.1f}% < {self.current_confidence_threshold:.1f}%)")
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
        """Execute REAL TRADING signal with safety checks and enhanced features"""
        logger.info(f"REAL TRADE SIGNAL FOUND!")
        logger.info(f"   {signal.symbol} {signal.action} - {signal.confidence:.1f}%")
        
        # Log enhanced features if available
        if signal.enhanced_features:
            enhanced = signal.enhanced_features
            logger.info(f"   🎯 Enhanced Features:")
            logger.info(f"      Swing Points: {len(enhanced['swing_points'].get('highs', []))} highs, {len(enhanced['swing_points'].get('lows', []))} lows")
            logger.info(f"      Liquidity Zones: {len(enhanced['liquidity_zones'])} zones")
            logger.info(f"      Market Structure: {enhanced['market_structure'].get('trend', 'unknown')}")
            if enhanced['reason']:
                logger.info(f"      Enhanced Reason: {enhanced['reason']}")
        
        # Safe formatting with None checks
        try:
            target_str = f"${signal.target_price:.2f}" if signal.target_price is not None else "None"
            stop_str = f"${signal.stop_loss:.2f}" if signal.stop_loss is not None else "None"
            logger.info(f"   Target: {target_str} | Stop: {stop_str}")
            logger.info(f"   {signal.reason}")
        except Exception as e:
            logger.error(f"   Error formatting signal info: {e}")
            logger.info(f"   Signal values - target: {signal.target_price}, stop: {signal.stop_loss}")
        
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
            
            # Get current market price for accurate calculations
            all_mids = self.info.all_mids()
            current_market_price = float(all_mids[signal.symbol])
            
            # Calculate position size (minimum $11 for Hyperliquid)
            risk_per_trade = balance * 0.25  # 25% max risk to meet minimum
            price_difference = abs(current_market_price - signal.stop_loss)
            
            if price_difference > 0:
                position_size_usd = max(11.0, min(risk_per_trade, balance * 0.30))  # Min $11, max 30%
                position_size_coins = position_size_usd / current_market_price
                
                # Format size according to Hyperliquid requirements
                formatted_size = self._format_trade_size(position_size_coins, signal.symbol)
                
                logger.info(f"   Position size: {formatted_size} {signal.symbol} (${position_size_usd:.2f})")
                
                # Create order with current market price (use market order for best execution)
                if signal.action == "BUY":
                    is_buy = True
                elif signal.action == "SELL":
                    is_buy = False
                
                logger.info(f"   Market price: ${current_market_price:.2f}")
                
                # Place the order (market order for guaranteed execution)
                logger.info(f"   PLACING REAL MARKET ORDER: {signal.action} {formatted_size} {signal.symbol}")
                
                try:
                    # Use small slippage for better fill rates on IOC orders
                    if is_buy:
                        market_price = current_market_price * 1.0005  # 0.05% above for guaranteed fill
                    else:
                        market_price = current_market_price * 0.9995  # 0.05% below for guaranteed fill
                    
                    logger.info(f"   DEBUG: Calling exchange.order with params:")
                    logger.info(f"      Symbol: {signal.symbol}")
                    logger.info(f"      Is_buy: {is_buy}")
                    logger.info(f"      Size: {float(formatted_size)}")
                    logger.info(f"      Limit_px: {market_price:.2f} (IOC market execution)")
                    
                    # Get asset index for the symbol (Hyperliquid uses indices, not names)
                    symbol_to_index = {
                        "BTC": 0,
                        "ETH": 1, 
                        "ATOM": 2,
                        "SOL": 5,
                        "AVAX": 6,
                        "DOGE": 12
                    }
                    
                    asset_index = symbol_to_index.get(signal.symbol)
                    if asset_index is None:
                        logger.error(f"   Unknown asset index for {signal.symbol}")
                        return False
                    
                    logger.info(f"   Asset index: {asset_index} for {signal.symbol}")
                    
                    # Use Exchange object's built-in order method
                    logger.info(f"   PLACING ORDER WITH EXCHANGE OBJECT:")
                    logger.info(f"      Asset: {signal.symbol}")
                    logger.info(f"      Size: {formatted_size}")
                    logger.info(f"      Price: ${market_price:.2f}")
                    logger.info(f"      IOC: Immediate execution")
                    
                    # Round price to proper tick size for each symbol
                    if signal.symbol == "BTC":
                        # BTC uses $1.00 tick size for prices above $100K
                        rounded_price = round(float(market_price))
                    elif signal.symbol == "ETH":
                        # ETH uses $0.10 tick size  
                        rounded_price = round(float(market_price), 1)
                    else:
                        # Other symbols use $0.01 tick size
                        rounded_price = round(float(market_price), 2)
                    
                    logger.info(f"   Tick-adjusted price: ${rounded_price:.2f}")
                    
                    order_result = self.exchange.order(
                        signal.symbol,                              # coin (symbol name)
                        is_buy,                                     # is_buy  
                        float(formatted_size),                      # size (as float)
                        float(rounded_price),                       # limit_px (tick-size adjusted)
                        {"limit": {"tif": "Ioc"}}                   # order_type (IOC = immediate execution)
                    )
                    
                    logger.info(f"   Exchange order result: {order_result}")
                    
                    # Check if order was successful  
                    order_success = self._verify_order_success(order_result)
                    
                except Exception as order_error:
                    logger.error(f"   ORDER PLACEMENT ERROR: {order_error}")
                    return False
                
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
    
    def _format_trade_size(self, size: float, symbol: str = "BTC") -> str:
        """Format trade size according to Hyperliquid lot size requirements"""
        # Hyperliquid szDecimals values (from API documentation)
        sz_decimals = {
            "BTC": 5,  # Minimum lot size: 0.00001
            "ETH": 4,  # Minimum lot size: 0.0001  
            "SOL": 2,  # Minimum lot size: 0.01 (typical for altcoins)
            "AVAX": 2, # Minimum lot size: 0.01
            "DOGE": 0  # Minimum lot size: 1.0 (whole numbers only)
        }
        
        decimals = sz_decimals.get(symbol, 3)  # Default to 3 if unknown
        min_lot_size = 10 ** (-decimals)
        
        # Round to appropriate decimal places
        rounded_size = round(size, decimals)
        
        # Ensure minimum lot size
        if rounded_size < min_lot_size:
            rounded_size = min_lot_size
            
        # Format with exact decimal places
        return f"{rounded_size:.{decimals}f}"
    
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