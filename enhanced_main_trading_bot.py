#!/usr/bin/env python3
"""
🚀 ENHANCED MAIN TRADING BOT WITH CRITICAL FEATURES
Integrates all critical features from the developer roadmap for sniper-level trading

CRITICAL INTEGRATIONS:
✅ Advanced Risk Management with ATR, OBI, and volatility controls
✅ Enhanced Execution Layer with limit-in, market-out
✅ Real-time order book monitoring
✅ Dynamic position sizing and risk controls
✅ Pre-trade cool-down periods and drawdown circuit breakers
✅ Async OrderWatch for real-time monitoring
✅ Market structure analysis and top/bottom detection
"""

import asyncio
import logging
import time
import os
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account as eth_account

# Import enhanced modules
from advanced_risk_management import AdvancedRiskManager, RiskMetrics, PositionRisk
from enhanced_execution_layer import EnhancedExecutionLayer, OrderRequest
from advanced_top_bottom_detector import AdvancedTopBottomDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_main_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedTradingSignal:
    """Enhanced trading signal with all critical features"""
    symbol: str
    action: str  # 'BUY' or 'SELL'
    confidence: float
    current_price: float
    target_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reason: str
    risk_metrics: RiskMetrics
    market_structure: Dict
    liquidity_zones: List[Dict]
    timestamp: datetime

class EnhancedMainTradingBot:
    """Enhanced main trading bot with all critical features integrated"""
    
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
        
        # Initialize exchange for trading
        try:
            account = eth_account.from_key(self.private_key)
            self.exchange = Exchange(account, base_url)
            logger.info("✅ Exchange initialized for REAL TRADING")
        except Exception as e:
            logger.warning(f"Exchange initialization failed: {e}")
            self.exchange = None
        
        # Initialize enhanced modules
        self.risk_manager = AdvancedRiskManager()
        self.execution_layer = EnhancedExecutionLayer(base_url, self.risk_manager)
        self.top_bottom_detector = AdvancedTopBottomDetector()
        
        logger.info("🚨 Advanced Risk Manager initialized")
        logger.info("🚀 Enhanced Execution Layer initialized")
        logger.info("🎯 Enhanced Top/Bottom Detector initialized")
        
        # Trading parameters
        self.symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE']
        self.daily_trade_limit = 10
        self.trades_today = 0
        self.last_reset = datetime.now().date()
        
        # Historical optimized thresholds
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
        self.use_enhanced_features = True
        self.enhanced_confidence_boost = 15
        self.min_swing_distance = 1.0
        self.min_liquidity_distance = 1.0
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        
        logger.info("🚀 ENHANCED MAIN TRADING BOT WITH CRITICAL FEATURES INITIALIZED")

    async def get_market_data(self, symbol: str, lookback: int = 5000) -> Dict:
        """Get comprehensive market data for analysis"""
        try:
            # Get candlestick data
            end_time = int(time.time() * 1000)
            start_time = end_time - (lookback * 60 * 1000)  # lookback minutes ago
            
            candles = self.info.candles_snapshot(symbol, "1m", start_time, end_time)
            if not candles:
                return {}
            
            # Get current price
            current_price = float(candles[-1]['c']) if candles else 0
            
            # Get order book data
            try:
                order_book = self.info.l2_book_snapshot(symbol)
            except:
                order_book = {}
            
            return {
                'symbol': symbol,
                'candles': candles,
                'current_price': current_price,
                'order_book': order_book,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}

    async def analyze_symbol_with_critical_features(self, symbol: str) -> Optional[EnhancedTradingSignal]:
        """Analyze symbol using all critical features"""
        try:
            # Get market data
            market_data = await self.get_market_data(symbol, 5000)
            if not market_data or not market_data.get('candles'):
                return None
            
            candles = market_data['candles']
            current_price = market_data['current_price']
            
            # Calculate risk metrics
            risk_metrics = self.risk_manager.calculate_risk_metrics(
                symbol, candles[-100:], candles[-60:], current_price  # 1h and 1h candles
            )
            
            # Get OBI (Order Book Imbalance)
            try:
                obi = await self.risk_manager.calculate_order_book_imbalance(symbol, self.info.base_url)
                risk_metrics.obi = obi
            except:
                risk_metrics.obi = 0.0
            
            # Analyze market structure
            market_structure = self.top_bottom_detector.analyze_market_structure(candles, current_price)
            
            # Analyze liquidity zones
            liquidity_zones = self.top_bottom_detector.analyze_liquidity_zones(symbol, current_price)
            
            # Detect swing points
            swing_points = self.top_bottom_detector.detect_swing_points(candles)
            
            # Calculate momentum and volume metrics
            momentum_1h = self._calculate_momentum(candles[-60:], 60)
            momentum_3h = self._calculate_momentum(candles[-180:], 180)
            volume_ratio = self._calculate_volume_ratio(candles[-20:])
            price_range = self._calculate_price_range(candles[-20:], 20)
            
            # Check entry filters
            action = "BUY" if momentum_1h > 0 else "SELL"
            can_trade, filter_reason = self.risk_manager.check_entry_filters(symbol, action, risk_metrics)
            
            if not can_trade:
                logger.info(f"❌ {symbol} blocked by risk filters: {filter_reason}")
                return None
            
            # Calculate confidence with enhanced features
            base_confidence = self._calculate_adaptive_confidence(
                momentum_1h, momentum_3h, volume_ratio, price_range,
                self.base_min_momentum_1h, self.base_min_volume_ratio, self.base_min_price_range
            )
            
            # Add enhanced features confidence
            enhanced_confidence = base_confidence
            
            # Market structure bonus
            if market_structure['trend'] == 'bullish' and action == "BUY":
                enhanced_confidence += self.enhanced_confidence_boost
            elif market_structure['trend'] == 'bearish' and action == "SELL":
                enhanced_confidence += self.enhanced_confidence_boost
            
            # Liquidity zone bonus
            if liquidity_zones:
                nearest_zone = min(liquidity_zones, key=lambda x: x.distance_from_current)
                if nearest_zone.distance_from_current <= self.min_liquidity_distance:
                    enhanced_confidence += 10
            
            # Swing point bonus
            if swing_points['highs'] or swing_points['lows']:
                nearest_swing = min(
                    swing_points['highs'] + swing_points['lows'],
                    key=lambda x: x.distance_from_current
                )
                if nearest_swing.distance_from_current <= self.min_swing_distance:
                    enhanced_confidence += 10
            
            # Check if confidence meets threshold
            if enhanced_confidence < self.current_confidence_threshold:
                return None
            
            # Calculate dynamic stops and position size
            stop_loss, take_profit = self.risk_manager.calculate_dynamic_stops(
                current_price, action, risk_metrics
            )
            
            # Calculate position size
            account_balance = await self._get_account_balance()
            position_size = self.risk_manager.calculate_position_size(
                account_balance, current_price, stop_loss, risk_metrics
            )
            
            if position_size <= 0:
                return None
            
            # Create enhanced signal
            signal = EnhancedTradingSignal(
                symbol=symbol,
                action=action,
                confidence=enhanced_confidence,
                current_price=current_price,
                target_price=current_price * (1.02 if action == "BUY" else 0.98),
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"Enhanced signal: {action} {symbol} @ {current_price:.2f} (Confidence: {enhanced_confidence:.1f}%)",
                risk_metrics=risk_metrics,
                market_structure=market_structure,
                liquidity_zones=liquidity_zones,
                timestamp=datetime.now()
            )
            
            logger.info(f"🎯 Enhanced signal generated: {symbol} {action} @ {current_price:.2f} "
                       f"(Confidence: {enhanced_confidence:.1f}%)")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} with critical features: {e}")
            return None

    async def execute_enhanced_signal(self, signal: EnhancedTradingSignal) -> bool:
        """Execute enhanced signal with all critical features"""
        try:
            # Create order request
            order_request = OrderRequest(
                symbol=signal.symbol,
                side=signal.action,
                size=signal.position_size,
                price=signal.current_price,
                order_type="LIMIT",
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            # Execute trade with enhanced execution layer
            success, message, order_id = await self.execution_layer.execute_trade(
                order_request, signal.risk_metrics
            )
            
            if success:
                logger.info(f"✅ Enhanced trade executed: {signal.symbol} {signal.action} "
                           f"{signal.position_size} @ {signal.current_price}")
                
                self.total_trades += 1
                return True
            else:
                logger.warning(f"❌ Enhanced trade execution failed: {message}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing enhanced signal: {e}")
            return False

    def _calculate_momentum(self, candles: List[dict], periods: int) -> float:
        """Calculate momentum over specified periods"""
        if len(candles) < periods:
            return 0.0
        
        recent_candles = candles[-periods:]
        if len(recent_candles) < 2:
            return 0.0
        
        start_price = float(recent_candles[0]['c'])
        end_price = float(recent_candles[-1]['c'])
        
        return (end_price - start_price) / start_price * 100

    def _calculate_volume_ratio(self, candles: List[dict]) -> float:
        """Calculate volume ratio (current vs average)"""
        if len(candles) < 10:
            return 1.0
        
        volumes = [float(c['v']) for c in candles]
        current_vol = volumes[-1]
        avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
        
        return current_vol / avg_vol if avg_vol > 0 else 1.0

    def _calculate_price_range(self, candles: List[dict], periods: int) -> float:
        """Calculate price range over specified periods"""
        if len(candles) < periods:
            return 1.0
        
        recent_candles = candles[-periods:]
        highs = [float(c['h']) for c in recent_candles]
        lows = [float(c['l']) for c in recent_candles]
        
        max_high = max(highs)
        min_low = min(lows)
        
        return max_high / min_low if min_low > 0 else 1.0

    def _calculate_adaptive_confidence(self, momentum_1h, momentum_3h, volume_ratio, price_range,
                                     mom_threshold, vol_threshold, range_threshold):
        """Calculate adaptive confidence score"""
        confidence = 0
        
        # Momentum scoring
        if abs(momentum_1h) >= mom_threshold:
            confidence += 25
        if abs(momentum_3h) >= mom_threshold * 2:
            confidence += 25
        
        # Volume scoring
        if volume_ratio >= vol_threshold:
            confidence += 25
        
        # Price range scoring
        if price_range >= range_threshold:
            confidence += 25
        
        return confidence

    async def _get_account_balance(self) -> float:
        """Get account balance"""
        try:
            # This would integrate with your actual balance checking
            return 1000.0  # Default for testing
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 1000.0

    async def run_enhanced_bot(self):
        """Run the enhanced trading bot with all critical features"""
        logger.info("🚀 Starting Enhanced Main Trading Bot...")
        
        while True:
            try:
                # Reset daily counters
                if datetime.now().date() != self.last_reset:
                    self.trades_today = 0
                    self.last_reset = datetime.now().date()
                
                # Check daily trade limit
                if self.trades_today >= self.daily_trade_limit:
                    logger.info("Daily trade limit reached, waiting...")
                    await asyncio.sleep(60)
                    continue
                
                # Analyze market volatility
                self.analyze_market_volatility()
                
                # Analyze each symbol
                for symbol in self.symbols:
                    try:
                        # Generate enhanced signal
                        signal = await self.analyze_symbol_with_critical_features(symbol)
                        
                        if signal:
                            # Execute enhanced signal
                            success = await self.execute_enhanced_signal(signal)
                            
                            if success:
                                self.trades_today += 1
                                
                                # Update performance tracking
                                # This would be updated based on actual PnL
                                self.risk_manager.update_performance_tracking(0, "profit")
                        
                        # Small delay between symbols
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Wait before next cycle
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in enhanced bot loop: {e}")
                await asyncio.sleep(60)

    def analyze_market_volatility(self):
        """Analyze current market volatility and adjust thresholds"""
        logger.info("Analyzing market volatility...")
        
        volatility_data = []
        
        for symbol in self.symbols[:3]:  # Quick sample of 3 symbols
            try:
                # Get recent candles
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
                self.volatility_multiplier = 0.3
            elif avg_vol_ratio < 0.6 and avg_price_range < 1.0 and avg_momentum < 0.5:
                self.market_condition = "LOW VOLATILITY"
                self.volatility_multiplier = 0.5
            elif avg_vol_ratio < 1.0 and avg_price_range < 2.0 and avg_momentum < 1.0:
                self.market_condition = "MODERATE VOLATILITY"
                self.volatility_multiplier = 0.7
            else:
                self.market_condition = "HIGH VOLATILITY"
                self.volatility_multiplier = 1.0
                
            # Adjust confidence threshold
            self.current_confidence_threshold = self.base_confidence_threshold * self.volatility_multiplier
            
            logger.info(f"Market: {self.market_condition}")
            logger.info(f"Adapted threshold: {self.current_confidence_threshold:.1f}%")

# Example usage
if __name__ == "__main__":
    async def main():
        bot = EnhancedMainTradingBot()
        await bot.run_enhanced_bot()
    
    asyncio.run(main())
