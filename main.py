import asyncio
import json
import logging
import pandas as pd
import signal
import sys
import threading
import time
import websocket
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config import (
    WS_PUBLIC_URL, INSTRUMENT_ID, TIMEFRAME, DATA_BUFFER_SIZE,
    RECONNECT_DELAY, HEARTBEAT_INTERVAL
)
from trader import TradingEngine
from okx_client import OKXClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OKXDataFeed:
    def __init__(self, on_candle_callback):
        self.ws = None
        self.on_candle_callback = on_candle_callback
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
    def connect(self):
        """Connect to OKX WebSocket"""
        try:
            logger.info("Connecting to OKX WebSocket...")
            websocket.enableTrace(False)
            self.ws = websocket.WebSocketApp(
                WS_PUBLIC_URL,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Run in a separate thread
            threading.Thread(target=self.ws.run_forever, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self._reconnect()
    
    def _on_open(self, ws):
        """WebSocket connection opened"""
        logger.info("WebSocket connection established")
        self.is_connected = True
        self.reconnect_attempts = 0
        
        # Subscribe to candle data
        subscribe_message = {
            "op": "subscribe",
            "args": [{
                "channel": f"candle{TIMEFRAME}",
                "instId": INSTRUMENT_ID
            }]
        }
        
        ws.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to {INSTRUMENT_ID} {TIMEFRAME} candles")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle subscription confirmation
            if 'event' in data and data['event'] == 'subscribe':
                logger.info(f"Subscription confirmed: {data}")
                return
            
            # Handle candle data
            if 'data' in data and data.get('arg', {}).get('channel', '').startswith('candle'):
                for candle_data in data['data']:
                    self._process_candle(candle_data)
                    
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _process_candle(self, candle_data):
        """Process individual candle data"""
        try:
            # OKX candle format: [timestamp, open, high, low, close, volume, volume_currency, volume_quantity, confirm]
            timestamp = int(candle_data[0])
            open_price = float(candle_data[1])
            high_price = float(candle_data[2])
            low_price = float(candle_data[3])
            close_price = float(candle_data[4])
            volume = float(candle_data[5])
            
            candle = {
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp / 1000),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
            # Call the callback function
            if self.on_candle_callback:
                self.on_candle_callback(candle)
                
        except Exception as e:
            logger.error(f"Error processing candle data: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        self.is_connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        logger.warning(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        self._reconnect()
    
    def _reconnect(self):
        """Attempt to reconnect to WebSocket"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Attempting to reconnect... (attempt {self.reconnect_attempts})")
            time.sleep(RECONNECT_DELAY)
            self.connect()
        else:
            logger.error("Max reconnection attempts reached. Stopping bot.")
            sys.exit(1)
    
    def disconnect(self):
        """Disconnect from WebSocket"""
        if self.ws:
            self.ws.close()
            self.is_connected = False

class TradingBot:
    def __init__(self):
        self.data_feed = None
        self.trading_engine = None
        self.candle_buffer = pd.DataFrame()
        self.is_running = False
        self.last_heartbeat = datetime.now()
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_signals_generated = 0
        self.total_signals_executed = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing OKX SOL-USD Perpetual Trading Bot")
            logger.info("=" * 60)
            
            # Initialize trading engine
            self.trading_engine = TradingEngine()
            if not self.trading_engine.initialize():
                logger.error("Failed to initialize trading engine")
                return False
            
            # Initialize data feed
            self.data_feed = OKXDataFeed(self._on_new_candle)
            
            # Load historical data to populate buffer
            self._load_initial_data()
            
            logger.info("Bot initialization completed successfully")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            return False
    
    def _load_initial_data(self):
        """Load initial historical candle data"""
        try:
            logger.info("Loading initial historical data...")
            okx_client = OKXClient()
            
            # Get historical candles
            response = okx_client.get_candlesticks(
                inst_id=INSTRUMENT_ID,
                bar=TIMEFRAME,
                limit=200  # Get 200 historical candles
            )
            
            if response.get('code') == '0' and response.get('data'):
                candles_data = []
                
                for candle in reversed(response['data']):  # OKX returns newest first
                    candles_data.append({
                        'timestamp': int(candle[0]),
                        'datetime': datetime.fromtimestamp(int(candle[0]) / 1000),
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5])
                    })
                
                self.candle_buffer = pd.DataFrame(candles_data)
                logger.info(f"Loaded {len(self.candle_buffer)} historical candles")
                
            else:
                logger.warning("Failed to load historical data, starting with empty buffer")
                
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
    
    def _on_new_candle(self, candle: Dict):
        """Handle new candle data"""
        try:
            # Add new candle to buffer
            new_row = pd.DataFrame([candle])
            self.candle_buffer = pd.concat([self.candle_buffer, new_row], ignore_index=True)
            
            # Keep buffer size manageable
            if len(self.candle_buffer) > DATA_BUFFER_SIZE:
                self.candle_buffer = self.candle_buffer.tail(DATA_BUFFER_SIZE).reset_index(drop=True)
            
            # Process trading logic
            self._process_trading_logic(candle)
            
        except Exception as e:
            logger.error(f"Error handling new candle: {e}")
    
    def _process_trading_logic(self, current_candle: Dict):
        """Process trading logic with new candle data"""
        try:
            if len(self.candle_buffer) < 50:  # Need enough data for indicators
                return
            
            current_price = current_candle['close']
            
            # Update existing position
            if self.trading_engine.current_position:
                self.trading_engine.update_position(current_price, self.candle_buffer)
            
            # Generate new trading signal
            signal = self.trading_engine.strategy.generate_signal(self.candle_buffer)
            
            if signal:
                self.total_signals_generated += 1
                logger.info(f"Signal Generated: {signal.signal_type.upper()} at ${current_price:.4f}")
                logger.info(f"Confidence: {signal.confidence:.2%}")
                logger.info(f"Reason: {signal.reason}")
                
                # Execute signal
                if self.trading_engine.execute_signal(signal, current_price):
                    self.total_signals_executed += 1
                    logger.info("Signal executed successfully")
                else:
                    logger.warning("Signal execution failed")
                    
        except Exception as e:
            logger.error(f"Error processing trading logic: {e}")
    
    def start(self):
        """Start the trading bot"""
        try:
            if not self.initialize():
                logger.error("Bot initialization failed")
                return False
            
            logger.info("Starting trading bot...")
            self.is_running = True
            
            # Connect to data feed
            self.data_feed.connect()
            
            # Start monitoring loop
            self._monitoring_loop()
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring and heartbeat loop"""
        logger.info("Monitoring loop started")
        
        while self.is_running:
            try:
                # Send heartbeat every interval
                if (datetime.now() - self.last_heartbeat).seconds >= HEARTBEAT_INTERVAL:
                    self._send_heartbeat()
                    self.last_heartbeat = datetime.now()
                
                # Check for daily reset
                self._check_daily_reset()
                
                # Sleep for a short period
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, stopping bot...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before continuing
    
    def _send_heartbeat(self):
        """Send heartbeat and performance update"""
        try:
            stats = self.trading_engine.heartbeat()
            
            # Log performance summary
            uptime = datetime.now() - self.start_time
            execution_rate = (self.total_signals_executed / max(self.total_signals_generated, 1)) * 100
            
            logger.info("=" * 60)
            logger.info("PERFORMANCE SUMMARY")
            logger.info(f"Uptime: {uptime}")
            logger.info(f"Signals Generated: {self.total_signals_generated}")
            logger.info(f"Signals Executed: {self.total_signals_executed}")
            logger.info(f"Execution Rate: {execution_rate:.1f}%")
            logger.info(f"Data Feed Connected: {self.data_feed.is_connected}")
            logger.info(f"Buffer Size: {len(self.candle_buffer)} candles")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
    
    def _check_daily_reset(self):
        """Check if daily statistics should be reset"""
        try:
            # Reset at midnight UTC
            now = datetime.now()
            if now.hour == 0 and now.minute == 0 and now.second < 5:
                logger.info("Daily reset triggered")
                self.trading_engine.reset_daily_stats()
                
        except Exception as e:
            logger.error(f"Error checking daily reset: {e}")
    
    def stop(self):
        """Stop the trading bot"""
        try:
            logger.info("Stopping trading bot...")
            self.is_running = False
            
            # Disconnect data feed
            if self.data_feed:
                self.data_feed.disconnect()
            
            # Emergency stop trading
            if self.trading_engine:
                self.trading_engine.emergency_stop()
            
            # Final performance report
            self._generate_final_report()
            
            logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
    
    def _generate_final_report(self):
        """Generate final performance report"""
        try:
            logger.info("=" * 60)
            logger.info("FINAL PERFORMANCE REPORT")
            logger.info("=" * 60)
            
            if self.trading_engine:
                stats = self.trading_engine.get_performance_stats()
                
                logger.info(f"Total Runtime: {datetime.now() - self.start_time}")
                logger.info(f"Total Trades: {stats['total_trades']}")
                logger.info(f"Winning Trades: {stats['winning_trades']}")
                logger.info(f"Win Rate: {stats['win_rate']:.2f}%")
                logger.info(f"Total P&L: ${stats['total_pnl']:.2f}")
                logger.info(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
                logger.info(f"Signals Generated: {self.total_signals_generated}")
                logger.info(f"Signals Executed: {self.total_signals_executed}")
                
                if stats['current_position']:
                    pos = stats['current_position']
                    logger.info(f"Final Position: {pos['side']} ${pos['unrealized_pnl']:.2f}")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")

def main():
    """Main entry point"""
    try:
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║                   OKX SOL-USD PERPETUAL BOT                 ║
        ║                                                              ║
        ║  High-Frequency Trading Bot with Advanced Technical Analysis ║
        ║  Features: CMF, OBV, Divergence, Range Trading, Parabolic   ║
        ║  Target Accuracy: 80-90%+ with Risk Management              ║
        ║                                                              ║
        ║  ⚠️  TRADING CARRIES RISK OF LOSS                            ║
        ║  💡 START WITH SIMULATION MODE                               ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        bot = TradingBot()
        bot.start()
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 