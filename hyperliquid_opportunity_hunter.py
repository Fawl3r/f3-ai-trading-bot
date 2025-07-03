#!/usr/bin/env python3
"""
HYPERLIQUID OPPORTUNITY HUNTER AI - PRODUCTION VERSION
Real-time parabolic detection with dynamic capital allocation
LIVE TRADING WITH REAL MONEY - USE WITH CAUTION
"""

import asyncio
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Hyperliquid imports
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import eth_account

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperliquid_opportunity_hunter.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    symbol: str
    price: float
    volume_24h: float
    price_change_24h: float
    timestamp: datetime

@dataclass
class TradingSignal:
    symbol: str
    action: str  # 'BUY' or 'SELL'
    confidence: float
    target_price: float
    stop_loss: float
    reason: str
    timestamp: datetime

class HyperliquidOpportunityHunter:
    def __init__(self):
        """Initialize the Hyperliquid trading bot"""
        
        # Load environment variables with fallbacks
        self.private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY', '') or os.getenv('HL_PRIVATE_KEY', '')
        self.account_address = os.getenv('HYPERLIQUID_ACCOUNT_ADDRESS', '') or os.getenv('HL_ACCOUNT_ADDRESS', '')
        self.testnet = os.getenv('HYPERLIQUID_TESTNET', 'True').lower() == 'true'
        
        if not self.private_key:
            raise ValueError("HYPERLIQUID_PRIVATE_KEY not found in environment variables")
        
        # Trading configuration  
        # Note: testnet setting comes from environment variable above
        self.max_position_size = 0.02  # 2% of balance per trade
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.take_profit_pct = 0.06  # 6% take profit
        self.max_daily_trades = 10
        self.trading_pairs = ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX']
        
        # AI Detection parameters
        self.min_volume_spike = 2.0  # 200% volume increase
        self.min_price_momentum = 0.015  # 1.5% minimum price move
        self.max_price_momentum = 0.08  # 8% maximum price move (avoid FOMO)
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # State tracking
        self.active_positions = {}
        self.daily_trades = 0
        self.last_trade_time = {}
        self.balance = 0.0
        
        # Notification settings
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL', '')
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # Initialize Hyperliquid clients
        self.init_hyperliquid_clients()
        
        logger.info("HYPERLIQUID OPPORTUNITY HUNTER AI INITIALIZED")
        if self.testnet:
            logger.warning("[TESTNET] TESTNET MODE ENABLED - No real money at risk")
        else:
            logger.warning("[LIVE] LIVE TRADING MODE - REAL MONEY AT RISK!")

    def init_hyperliquid_clients(self):
        """Initialize Hyperliquid Info and Exchange clients"""
        try:
            # Choose API URL based on testnet setting
            if self.testnet:
                base_url = constants.TESTNET_API_URL
                logger.info("Using Hyperliquid TESTNET")
            else:
                base_url = constants.MAINNET_API_URL
                logger.info("Using Hyperliquid MAINNET")
            
            # Initialize Info client for market data
            self.info = Info(base_url, skip_ws=False)
            
            # Initialize Exchange client for trading
            account = eth_account.Account.from_key(self.private_key)
            self.exchange = Exchange(account, base_url)
            
            # If account_address not provided, use the account address from private key
            if not self.account_address:
                self.account_address = account.address
                logger.info(f"Using account address from private key: {self.account_address}")
            
            logger.info("Hyperliquid clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hyperliquid clients: {e}")
            raise

    async def get_account_balance(self) -> float:
        """Get account balance"""
        try:
            user_state = self.info.user_state(self.account_address)
            
            if user_state and 'marginSummary' in user_state:
                balance = float(user_state['marginSummary']['accountValue'])
                logger.info(f"Account balance: ${balance:.2f}")
                return balance
            else:
                logger.warning("Could not retrieve account balance")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0

    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for a symbol"""
        try:
            # Get current price data
            all_mids = self.info.all_mids()
            
            if symbol not in all_mids:
                logger.warning(f"Symbol {symbol} not found in market data")
                return None
            
            current_price = float(all_mids[symbol])
            
            # Get 24h stats
            meta = self.info.meta()
            volume_24h = 0.0
            price_change_24h = 0.0
            
            if meta and 'universe' in meta:
                for asset in meta['universe']:
                    if asset['name'] == symbol:
                        if 'dayNtlVlm' in asset:
                            volume_24h = float(asset['dayNtlVlm'])
                        if 'prevDayPx' in asset:
                            prev_price = float(asset['prevDayPx'])
                            if prev_price > 0:
                                price_change_24h = (current_price - prev_price) / prev_price * 100
                        break
            
            return MarketData(
                symbol=symbol,
                price=current_price,
                volume_24h=volume_24h,
                price_change_24h=price_change_24h,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI for price series"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if not enough data
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    async def detect_opportunity(self, symbol: str) -> Optional[TradingSignal]:
        """AI-powered opportunity detection"""
        try:
            # Get current market data
            market_data = await self.get_market_data(symbol)
            if not market_data:
                return None
            
            # Get historical candlestick data for analysis
            end_time = int(time.time() * 1000)
            start_time = end_time - (4 * 60 * 60 * 1000)  # 4 hours of data
            
            candles = self.info.candles_snapshot(symbol, "1m", start_time, end_time)
            
            if not candles or len(candles) < 50:
                logger.warning(f"Insufficient historical data for {symbol}")
                return None
            
            # Extract price data
            prices = [float(candle['c']) for candle in candles]
            volumes = [float(candle['v']) for candle in candles]
            highs = [float(candle['h']) for candle in candles]
            lows = [float(candle['l']) for candle in candles]
            
            current_price = prices[-1]
            current_volume = volumes[-1]
            
            # Calculate technical indicators
            rsi = self.calculate_rsi(prices)
            
            # Volume analysis
            avg_volume = np.mean(volumes[-20:])  # 20-period average
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price momentum analysis
            price_5m_ago = prices[-5] if len(prices) >= 5 else current_price
            price_momentum = (current_price - price_5m_ago) / price_5m_ago * 100
            
            # Volatility analysis
            price_range = (max(highs[-10:]) - min(lows[-10:])) / current_price * 100
            
            # AI Detection Logic
            confidence = 0.0
            action = None
            reason = ""
            
            # BULLISH SIGNALS
            if (volume_spike >= self.min_volume_spike and 
                price_momentum >= self.min_price_momentum and 
                price_momentum <= self.max_price_momentum and
                rsi < self.rsi_overbought):
                
                confidence = min(90.0, 
                    30 + (volume_spike * 10) + 
                    (price_momentum * 5) + 
                    ((70 - rsi) * 0.5)
                )
                action = "BUY"
                reason = f"Volume spike {volume_spike:.1f}x, momentum {price_momentum:.2f}%, RSI {rsi:.1f}"
            
            # BEARISH SIGNALS (for shorting)
            elif (volume_spike >= self.min_volume_spike and 
                  price_momentum <= -self.min_price_momentum and 
                  price_momentum >= -self.max_price_momentum and
                  rsi > self.rsi_oversold):
                
                confidence = min(90.0, 
                    30 + (volume_spike * 10) + 
                    (abs(price_momentum) * 5) + 
                    ((rsi - 30) * 0.5)
                )
                action = "SELL"
                reason = f"Volume spike {volume_spike:.1f}x, momentum {price_momentum:.2f}%, RSI {rsi:.1f}"
            
            # Only return signal if confidence is high enough
            if confidence >= 75.0 and action:
                target_price = current_price * (1.06 if action == "BUY" else 0.94)
                stop_loss = current_price * (0.97 if action == "BUY" else 1.03)
                
                return TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    reason=reason,
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting opportunity for {symbol}: {e}")
            return None

    async def place_order(self, signal: TradingSignal) -> bool:
        """Place order based on trading signal"""
        try:
            if self.daily_trades >= self.max_daily_trades:
                logger.warning("Daily trade limit reached")
                return False
            
            # Calculate position size
            balance = await self.get_account_balance()
            if balance <= 0:
                logger.error("No available balance")
                return False
            
            position_value = balance * self.max_position_size
            current_price = signal.target_price
            
            # Calculate size based on USD value
            size = position_value / current_price
            
            # Determine if this is a long or short position
            is_buy = signal.action == "BUY"
            sz = size if is_buy else -size  # Negative size for short
            
            logger.info(f"Placing {signal.action} order for {signal.symbol}")
            logger.info(f"Size: {abs(sz):.4f}, Price: ${current_price:.4f}")
            logger.info(f"Target: ${signal.target_price:.4f}, Stop: ${signal.stop_loss:.4f}")
            
            # Place market order
            order_result = self.exchange.market_order(signal.symbol, is_buy, sz)
            
            if order_result and order_result.get('status') == 'ok':
                logger.info(f"Order placed successfully for {signal.symbol}")
                
                # Track the position
                self.active_positions[signal.symbol] = {
                    'action': signal.action,
                    'size': sz,
                    'entry_price': current_price,
                    'target_price': signal.target_price,
                    'stop_loss': signal.stop_loss,
                    'timestamp': datetime.now(),
                    'confidence': signal.confidence
                }
                
                self.daily_trades += 1
                self.last_trade_time[signal.symbol] = time.time()
                
                # Send notification
                message = f"[HYPERLIQUID BOT] {signal.action} {signal.symbol}\n"
                message += f"Price: ${current_price:.4f}\n"
                message += f"Size: {abs(sz):.4f}\n"
                message += f"Confidence: {signal.confidence:.1f}%\n"
                message += f"Reason: {signal.reason}"
                
                await self.send_notification(message, urgent=True)
                
                return True
            else:
                logger.error(f"Failed to place order: {order_result}")
                return False
                
        except Exception as e:
            logger.error(f"Error placing order for {signal.symbol}: {e}")
            return False

    async def check_positions(self):
        """Check and manage active positions"""
        try:
            if not self.active_positions:
                return
            
            # Get current positions from Hyperliquid
            user_state = self.info.user_state(self.account_address)
            
            if not user_state or 'assetPositions' not in user_state:
                return
            
            current_positions = {}
            for pos in user_state['assetPositions']:
                if abs(float(pos['position']['szi'])) > 0:
                    current_positions[pos['position']['coin']] = {
                        'size': float(pos['position']['szi']),
                        'unrealized_pnl': float(pos['position']['unrealizedPnl'])
                    }
            
            # Check each tracked position
            for symbol, position_data in list(self.active_positions.items()):
                try:
                    market_data = await self.get_market_data(symbol)
                    if not market_data:
                        continue
                    
                    current_price = market_data.price
                    entry_price = position_data['entry_price']
                    is_long = position_data['action'] == "BUY"
                    
                    # Calculate P&L percentage
                    if is_long:
                        pnl_pct = (current_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - current_price) / entry_price * 100
                    
                    should_close = False
                    close_reason = ""
                    
                    # Check take profit
                    if is_long and current_price >= position_data['target_price']:
                        should_close = True
                        close_reason = f"Take profit hit at ${current_price:.4f}"
                    elif not is_long and current_price <= position_data['target_price']:
                        should_close = True
                        close_reason = f"Take profit hit at ${current_price:.4f}"
                    
                    # Check stop loss
                    elif is_long and current_price <= position_data['stop_loss']:
                        should_close = True
                        close_reason = f"Stop loss hit at ${current_price:.4f}"
                    elif not is_long and current_price >= position_data['stop_loss']:
                        should_close = True
                        close_reason = f"Stop loss hit at ${current_price:.4f}"
                    
                    # Close position if needed
                    if should_close:
                        await self.close_position(symbol, position_data, close_reason, pnl_pct)
                    
                    # Log position status
                    logger.info(f"Position {symbol}: {pnl_pct:+.2f}% P&L")
                    
                except Exception as e:
                    logger.error(f"Error checking position {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error checking positions: {e}")

    async def close_position(self, symbol: str, position_data: dict, reason: str, pnl_pct: float):
        """Close a position"""
        try:
            is_long = position_data['action'] == "BUY"
            
            # Place closing order (opposite direction)
            size = abs(position_data['size'])
            close_order = self.exchange.market_order(symbol, not is_long, size)
            
            if close_order and close_order.get('status') == 'ok':
                logger.info(f"Position closed for {symbol}: {reason}")
                
                # Send notification
                message = f"[HYPERLIQUID BOT] Position CLOSED {symbol}\n"
                message += f"Reason: {reason}\n"
                message += f"P&L: {pnl_pct:+.2f}%\n"
                message += f"Entry: ${position_data['entry_price']:.4f}\n"
                message += f"Exit: Current market price"
                
                await self.send_notification(message)
                
                # Remove from tracking
                del self.active_positions[symbol]
                
            else:
                logger.error(f"Failed to close position for {symbol}")
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    async def send_notification(self, message: str, urgent: bool = False):
        """Send notifications via Discord/Telegram"""
        try:
            # Discord webhook
            if self.discord_webhook:
                import requests
                webhook_data = {
                    "content": f"HYPERLIQUID BOT\n{message}",
                    "username": "HyperliquidHunter"
                }
                if urgent:
                    webhook_data["content"] = f"🚨 URGENT 🚨\n{webhook_data['content']}"
                
                response = requests.post(self.discord_webhook, json=webhook_data)
                if response.status_code == 204:
                    logger.info("Discord notification sent")
                else:
                    logger.warning(f"Discord notification failed: {response.status_code}")
            
            # Telegram
            if self.telegram_bot_token and self.telegram_chat_id:
                import requests
                telegram_url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
                telegram_data = {
                    "chat_id": self.telegram_chat_id,
                    "text": f"HYPERLIQUID BOT\n{message}",
                    "parse_mode": "HTML"
                }
                
                response = requests.post(telegram_url, json=telegram_data)
                if response.status_code == 200:
                    logger.info("Telegram notification sent")
                else:
                    logger.warning(f"Telegram notification failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting Hyperliquid Opportunity Hunter AI...")
        
        # Reset daily trades at start
        today = datetime.now().date()
        last_reset_date = today
        
        while True:
            try:
                current_date = datetime.now().date()
                if current_date != last_reset_date:
                    self.daily_trades = 0
                    last_reset_date = current_date
                    logger.info("Daily trade counter reset")
                
                # Get account balance
                self.balance = await self.get_account_balance()
                
                if self.balance <= 0:
                    logger.warning("No available balance, waiting...")
                    await asyncio.sleep(60)
                    continue
                
                # Check existing positions
                await self.check_positions()
                
                # Scan for new opportunities
                for symbol in self.trading_pairs:
                    try:
                        # Skip if we already have a position in this symbol
                        if symbol in self.active_positions:
                            continue
                        
                        # Rate limiting - don't trade same symbol too frequently
                        if symbol in self.last_trade_time:
                            time_since_last = time.time() - self.last_trade_time[symbol]
                            if time_since_last < 300:  # 5 minutes minimum between trades
                                continue
                        
                        # Detect opportunity
                        signal = await self.detect_opportunity(symbol)
                        
                        if signal and signal.confidence >= 75.0:
                            logger.info(f"OPPORTUNITY DETECTED: {signal.symbol} - {signal.action}")
                            logger.info(f"Confidence: {signal.confidence:.1f}%")
                            logger.info(f"Reason: {signal.reason}")
                            
                            # Place order
                            success = await self.place_order(signal)
                            if success:
                                logger.info(f"Successfully placed order for {signal.symbol}")
                            
                            # Slight delay after placing order
                            await asyncio.sleep(2)
                        
                        # Small delay between symbol scans
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Log status
                logger.info(f"Scan complete. Balance: ${self.balance:.2f}, "
                          f"Positions: {len(self.active_positions)}, "
                          f"Daily trades: {self.daily_trades}/{self.max_daily_trades}")
                
                # Wait before next scan
                await asyncio.sleep(30)  # 30-second scan interval
                
            except KeyboardInterrupt:
                logger.info("Trading loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait longer on errors

async def main():
    """Main function"""
    try:
        bot = HyperliquidOpportunityHunter()
        await bot.run_trading_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return

if __name__ == "__main__":
    asyncio.run(main()) 