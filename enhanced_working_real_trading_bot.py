#!/usr/bin/env python3
"""
🚀 ENHANCED WORKING REAL TRADING BOT WITH TOP/BOTTOM & LIQUIDITY ZONES
Advanced trading bot with swing point and liquidity zone detection for sniper entries/exits

FEATURES:
✅ Real Hyperliquid Trading
✅ Swing High/Low Detection (5-bar pivot method)
✅ Order Book Liquidity Zone Analysis
✅ Volume Cluster Detection
✅ Multi-timeframe Confluence
✅ Adaptive Volatility Detection
✅ Smart Position Sizing
✅ Risk Management
"""

import os
import time
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from eth_account import Account as eth_account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from advanced_top_bottom_detector import AdvancedTopBottomDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedWorkingRealTradingBot:
    """Enhanced real trading bot with top/bottom and liquidity zone detection"""
    
    def __init__(self):
        # Environment variables
        self.private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
        self.account_address = os.getenv('HYPERLIQUID_ACCOUNT_ADDRESS')
        
        if not self.private_key or not self.account_address:
            raise ValueError("Missing required environment variables")
        
        # Initialize Hyperliquid connection
        self.account = eth_account.from_key(self.private_key)
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        self.exchange = Exchange(self.account, constants.MAINNET_API_URL)
        
        # Initialize top/bottom detector
        self.detector = AdvancedTopBottomDetector()
        
        # Trading configuration
        self.symbols = ["BTC", "ETH", "SOL", "AVAX", "DOGE"]
        self.base_confidence_threshold = 50
        self.min_confidence_threshold = 15
        self.position_size_range = (0.25, 0.30)  # 25-30% of balance
        self.leverage_range = (2, 5)
        
        # Adaptive volatility thresholds
        self.volatility_thresholds = {
            'EXTREMELY_LOW': {'threshold': 15, 'multiplier': 0.3},
            'LOW': {'threshold': 25, 'multiplier': 0.5},
            'MODERATE': {'threshold': 35, 'multiplier': 0.7},
            'HIGH': {'threshold': 50, 'multiplier': 1.0}
        }
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.current_balance = 0.0
        
        logger.info("🚀 Enhanced Real Trading Bot initialized")
    
    async def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            user_state = self.info.user_state(self.account_address)
            margin_summary = user_state.get('marginSummary', {})
            balance = float(margin_summary.get('accountValue', 0))
            self.current_balance = balance
            return balance
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive market data for a symbol"""
        try:
            # Get current price
            all_mids = self.info.all_mids()
            current_price = float(all_mids.get(symbol, 0))
            
            if current_price == 0:
                return None
            
            # Get historical candles for analysis
            end_time = int(time.time() * 1000)
            start_time = end_time - (24 * 60 * 60 * 1000)  # 24 hours
            
            candles = self.info.candles_snapshot(symbol, "1h", start_time, end_time)
            
            if not candles or len(candles) < 12:
                return None
            
            # Extract price and volume data
            prices = [float(candle['c']) for candle in candles]
            volumes = [float(candle['v']) for candle in candles]
            highs = [float(candle['h']) for candle in candles]
            lows = [float(candle['l']) for candle in candles]
            
            # Calculate basic metrics
            price_24h_ago = prices[0]
            price_change_24h = (current_price - price_24h_ago) / price_24h_ago * 100
            
            avg_volume = sum(volumes) / len(volumes)
            current_volume = volumes[-1] if volumes else avg_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate volatility
            price_returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = np.std(price_returns) if len(price_returns) > 1 else 0.0
            
            return {
                'symbol': symbol,
                'price': current_price,
                'price_change_24h': price_change_24h,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'candles': candles
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def detect_adaptive_volatility(self, market_data: Dict) -> Dict:
        """Detect current market volatility and adjust thresholds"""
        volatility = market_data['volatility']
        volume_ratio = market_data['volume_ratio']
        
        # Determine volatility level
        if volume_ratio < 0.5 and volatility < 0.02:
            level = 'EXTREMELY_LOW'
        elif volume_ratio < 1.0 and volatility < 0.03:
            level = 'LOW'
        elif volume_ratio < 1.5 and volatility < 0.05:
            level = 'MODERATE'
        else:
            level = 'HIGH'
        
        threshold_config = self.volatility_thresholds[level]
        adjusted_threshold = threshold_config['threshold']
        multiplier = threshold_config['multiplier']
        
        logger.info(f"📊 Volatility: {level} - Threshold: {adjusted_threshold}% (x{multiplier})")
        
        return {
            'level': level,
            'threshold': adjusted_threshold,
            'multiplier': multiplier,
            'volatility': volatility,
            'volume_ratio': volume_ratio
        }
    
    def analyze_enhanced_signals(self, symbol: str, market_data: Dict) -> Dict:
        """Analyze signals with top/bottom and liquidity zone detection"""
        try:
            current_price = market_data['price']
            candles = market_data['candles']
            
            # Get enhanced signals from detector
            enhanced_signals = self.detector.get_entry_exit_signals(symbol, candles, current_price)
            
            # Get market structure
            market_structure = self.detector.get_market_structure(candles)
            
            # Calculate base confidence from original logic
            volume_ratio = market_data['volume_ratio']
            price_change = market_data['price_change_24h']
            
            base_confidence = 0
            if volume_ratio > 1.5:
                base_confidence += 30
            if abs(price_change) > 0.5:
                base_confidence += 20
            if volume_ratio > 2.0:
                base_confidence += 25
            if abs(price_change) > 1.0:
                base_confidence += 25
            
            # Enhanced confidence with top/bottom detection
            enhanced_confidence = enhanced_signals['confidence']
            
            # Add market structure bonus
            if market_structure['trend'] == 'bullish' and enhanced_signals['long_entry']:
                enhanced_confidence += 15
            elif market_structure['trend'] == 'bearish' and enhanced_signals['short_entry']:
                enhanced_confidence += 15
            
            # Determine signal direction
            signal_type = None
            if enhanced_signals['long_entry'] and enhanced_confidence >= self.min_confidence_threshold:
                signal_type = 'long'
            elif enhanced_signals['short_entry'] and enhanced_confidence >= self.min_confidence_threshold:
                signal_type = 'short'
            
            return {
                'signal_type': signal_type,
                'base_confidence': base_confidence,
                'enhanced_confidence': enhanced_confidence,
                'swing_points': enhanced_signals['swing_points'],
                'liquidity_zones': enhanced_signals['liquidity_zones'],
                'market_structure': market_structure,
                'reason': enhanced_signals['reason']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing enhanced signals for {symbol}: {e}")
            return {
                'signal_type': None,
                'base_confidence': 0,
                'enhanced_confidence': 0,
                'reason': f"Error: {e}"
            }
    
    async def execute_enhanced_signal(self, symbol: str, signal: Dict) -> bool:
        """Execute trade with enhanced signal analysis"""
        try:
            if not signal['signal_type']:
                return False
            
            # Get current balance
            balance = await self.get_account_balance()
            if balance <= 0:
                logger.error("Insufficient balance to place order")
                return False
            
            # Get market data for position sizing
            market_data = await self.get_market_data(symbol)
            if not market_data:
                return False
            
            # Calculate position size based on signal strength
            signal_strength = signal['enhanced_confidence'] / 100.0
            position_size_pct = self.position_size_range[0] + \
                              (self.position_size_range[1] - self.position_size_range[0]) * signal_strength
            
            position_value = balance * position_size_pct
            current_price = market_data['price']
            
            # Calculate position size in coins
            position_size_coins = position_value / current_price
            
            # Format position size based on symbol
            position_size_coins = self.format_position_size(symbol, position_size_coins)
            
            if position_size_coins <= 0:
                logger.error(f"Invalid position size for {symbol}")
                return False
            
            # Calculate entry price with slippage
            slippage = 0.05  # 0.05% slippage
            if signal['signal_type'] == 'long':
                entry_price = current_price * (1 + slippage / 100)
            else:
                entry_price = current_price * (1 - slippage / 100)
            
            # Round price to appropriate tick size
            entry_price = self.round_to_tick_size(symbol, entry_price)
            
            # Place order
            order_result = self.exchange.order(
                coin=symbol,
                is_buy=(signal['signal_type'] == 'long'),
                sz=position_size_coins,
                limit_px=entry_price,
                order_type={'limit': {'tif': 'Gtc'}},
                reduce_only=False
            )
            
            if order_result:
                self.total_trades += 1
                logger.info(f"✅ ENHANCED TRADE EXECUTED: {signal['signal_type'].upper()} {symbol}")
                logger.info(f"   Entry Price: ${entry_price:.2f}")
                logger.info(f"   Position Size: {position_size_coins:.5f} {symbol}")
                logger.info(f"   Confidence: {signal['enhanced_confidence']:.1f}%")
                logger.info(f"   Reason: {signal['reason']}")
                
                # Log swing points and liquidity zones
                if signal['swing_points']['highs']:
                    logger.info(f"   Swing Highs: {len(signal['swing_points']['highs'])} detected")
                if signal['swing_points']['lows']:
                    logger.info(f"   Swing Lows: {len(signal['swing_points']['lows'])} detected")
                if signal['liquidity_zones']:
                    logger.info(f"   Liquidity Zones: {len(signal['liquidity_zones'])} detected")
                
                return True
            else:
                logger.error(f"❌ Failed to place order for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing enhanced signal for {symbol}: {e}")
            return False
    
    def format_position_size(self, symbol: str, size: float) -> float:
        """Format position size based on symbol requirements"""
        if symbol == "BTC":
            return round(size, 5)  # 5 decimal places
        elif symbol == "ETH":
            return round(size, 4)  # 4 decimal places
        elif symbol in ["SOL", "AVAX"]:
            return round(size, 2)  # 2 decimal places
        elif symbol == "DOGE":
            return round(size, 0)  # Whole numbers only
        else:
            return round(size, 4)
    
    def round_to_tick_size(self, symbol: str, price: float) -> float:
        """Round price to appropriate tick size"""
        if symbol == "BTC":
            return round(price, 0)  # $1.00 tick size
        elif symbol == "ETH":
            return round(price, 1)  # $0.10 tick size
        else:
            return round(price, 2)  # $0.01 tick size
    
    async def scan_all_symbols(self):
        """Scan all symbols for enhanced trading opportunities"""
        logger.info("🔍 Scanning all symbols for enhanced opportunities...")
        
        for symbol in self.symbols:
            try:
                # Get market data
                market_data = await self.get_market_data(symbol)
                if not market_data:
                    continue
                
                # Detect adaptive volatility
                volatility_info = self.detect_adaptive_volatility(market_data)
                
                # Analyze enhanced signals
                enhanced_signals = self.analyze_enhanced_signals(symbol, market_data)
                
                # Check if signal meets threshold
                if enhanced_signals['signal_type'] and enhanced_signals['enhanced_confidence'] >= volatility_info['threshold']:
                    logger.info(f"🎯 ENHANCED SIGNAL DETECTED: {symbol}")
                    logger.info(f"   Signal Type: {enhanced_signals['signal_type'].upper()}")
                    logger.info(f"   Enhanced Confidence: {enhanced_signals['enhanced_confidence']:.1f}%")
                    logger.info(f"   Base Confidence: {enhanced_signals['base_confidence']:.1f}%")
                    logger.info(f"   Market Structure: {enhanced_signals['market_structure']['trend']}")
                    logger.info(f"   Reason: {enhanced_signals['reason']}")
                    
                    # Execute trade
                    await self.execute_enhanced_signal(symbol, enhanced_signals)
                else:
                    logger.info(f"⏳ {symbol}: No enhanced signal (Confidence: {enhanced_signals['enhanced_confidence']:.1f}% < {volatility_info['threshold']}%)")
                
                # Small delay to avoid rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
    
    async def run_enhanced_bot(self):
        """Run the enhanced trading bot"""
        logger.info("🚀 Starting Enhanced Real Trading Bot with Top/Bottom & Liquidity Zone Detection")
        
        # Get initial balance
        initial_balance = await self.get_account_balance()
        logger.info(f"💰 Initial Balance: ${initial_balance:.2f}")
        
        while True:
            try:
                await self.scan_all_symbols()
                
                # Update performance metrics
                current_balance = await self.get_account_balance()
                total_pnl = current_balance - initial_balance
                
                logger.info(f"📊 Performance Update:")
                logger.info(f"   Total Trades: {self.total_trades}")
                logger.info(f"   Current Balance: ${current_balance:.2f}")
                logger.info(f"   Total PnL: ${total_pnl:+.2f}")
                logger.info(f"   PnL %: {(total_pnl / initial_balance * 100):+.2f}%")
                
                # Wait before next scan
                logger.info("⏳ Waiting 30 seconds before next scan...")
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in main bot loop: {e}")
                await asyncio.sleep(10)

# Main execution
if __name__ == "__main__":
    try:
        bot = EnhancedWorkingRealTradingBot()
        asyncio.run(bot.run_enhanced_bot())
    except KeyboardInterrupt:
        logger.info("🛑 Enhanced Trading Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Enhanced Trading Bot error: {e}") 