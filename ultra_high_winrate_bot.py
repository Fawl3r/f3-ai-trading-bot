#!/usr/bin/env python3
"""
ULTRA HIGH WIN RATE AI Reversal Trading Bot
Target: 75-85% Win Rate with Ultra-Strict Conditions
"""

import asyncio
import websocket
import json
import pandas as pd
import numpy as np
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import threading
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Enhanced notifications
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False

try:
    from plyer import notification
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    entry_time: datetime
    direction: str
    entry_price: float
    target_profit: float
    position_size: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl_amount: Optional[float] = None
    pnl_percentage: Optional[float] = None
    confidence: float = 0.0
    quality_score: float = 0.0
    reversal_signals: List[str] = None

class UltraTradeNotifier:
    """Ultra-enhanced trade notification system"""
    
    @staticmethod
    def play_sound(sound_type: str = "entry"):
        if not SOUND_AVAILABLE:
            return
        try:
            if sound_type == "ultra_entry":
                # Special ultra-entry sound sequence
                for freq in [1600, 1800, 2000]:
                    winsound.Beep(freq, 150)
                    time.sleep(0.05)
            elif sound_type == "jackpot":
                # Jackpot sound for big wins
                for freq in [1000, 1200, 1400, 1600, 1800, 2000]:
                    winsound.Beep(freq, 100)
                    time.sleep(0.02)
            elif sound_type == "profit":
                winsound.Beep(1200, 500)
            elif sound_type == "loss":
                winsound.Beep(300, 800)
        except:
            pass
    
    @staticmethod
    def desktop_notification(title: str, message: str, timeout: int = 10):
        if not NOTIFICATION_AVAILABLE:
            return
        try:
            notification.notify(
                title=title,
                message=message,
                timeout=timeout,
                app_name="Ultra High Win Rate Bot"
            )
        except:
            pass

class UltraAIDetector:
    """Ultra-enhanced AI system with strictest signal detection"""
    
    def __init__(self):
        self.price_history = []
        self.features_history = []
        
    def extract_ultra_features(self, data: pd.DataFrame) -> Dict:
        """Extract ultra-comprehensive features for highest accuracy"""
        if len(data) < 150:  # Need even more data for ultra-reliable signals
            return {}
        
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            features = {}
            
            # Ultra-precise price analysis
            features['current_price'] = close[-1]
            
            # Multiple timeframe analysis
            for period in [3, 5, 10, 15, 30, 60]:
                if len(close) > period:
                    features[f'change_{period}m'] = (close[-1] - close[-period]) / close[-period] * 100
            
            # Ultra-strict range analysis (multiple timeframes)
            range_positions = []
            for period in [15, 30, 60, 100]:
                if len(high) >= period:
                    recent_high = np.max(high[-period:])
                    recent_low = np.min(low[-period:])
                    range_size = recent_high - recent_low
                    
                    if range_size > 0:
                        range_pos = (close[-1] - recent_low) / range_size * 100
                        range_positions.append(range_pos)
            
            features['range_position'] = np.mean(range_positions) if range_positions else 50
            features['range_consensus'] = len([r for r in range_positions if r > 90 or r < 10])  # Ultra-extreme
            features['ultra_extreme'] = all(r > 92 or r < 8 for r in range_positions) if range_positions else False
            
            # Ultra-precise moving averages
            ma_periods = [5, 10, 20, 50, 100]
            ma_values = []
            
            for period in ma_periods:
                if len(close) >= period:
                    ma = np.mean(close[-period:])
                    ma_values.append(ma)
                    features[f'sma_{period}'] = ma
                    features[f'price_vs_sma_{period}'] = (close[-1] - ma) / ma * 100
            
            # Ultra-trend analysis
            if len(ma_values) >= 5:
                # Perfect MA alignment
                ascending = all(ma_values[i] <= ma_values[i+1] for i in range(len(ma_values)-1))
                descending = all(ma_values[i] >= ma_values[i+1] for i in range(len(ma_values)-1))
                features['perfect_alignment'] = ascending or descending
                features['trend_strength'] = 100 if ascending else -100 if descending else 0
            
            # Ultra-precise RSI analysis
            features['rsi_7'] = self._calculate_rsi(close, 7)
            features['rsi_14'] = self._calculate_rsi(close, 14)
            features['rsi_21'] = self._calculate_rsi(close, 21)
            features['rsi_consensus'] = abs(features['rsi_14'] - features['rsi_21'])
            features['rsi_ultra_extreme'] = (features['rsi_14'] > 85 or features['rsi_14'] < 15)
            
            # Ultra-volume analysis
            if len(volume) >= 50:
                vol_sma_20 = np.mean(volume[-20:])
                vol_sma_50 = np.mean(volume[-50:])
                features['volume_ratio'] = volume[-1] / vol_sma_20
                features['volume_trend'] = (vol_sma_20 - vol_sma_50) / vol_sma_50 * 100
                features['ultra_volume'] = volume[-1] > vol_sma_20 * 2.5  # Ultra-high volume
            
            # Ultra-momentum indicators
            if len(close) >= 30:
                momentum_5 = (close[-1] - close[-6]) / close[-6] * 100
                momentum_10 = (close[-1] - close[-11]) / close[-11] * 100
                momentum_20 = (close[-1] - close[-21]) / close[-21] * 100
                
                features['momentum_divergence'] = abs(momentum_5 - momentum_20) > 5
                features['momentum_exhaustion'] = (abs(momentum_5) > 8 and abs(momentum_10) < 3)
            
            # Ultra-support/resistance
            features['support_strength'] = self._calculate_ultra_support(low, close[-1])
            features['resistance_strength'] = self._calculate_ultra_resistance(high, close[-1])
            
            return features
            
        except Exception as e:
            logger.error(f"Ultra feature extraction error: {e}")
            return {}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_ultra_support(self, lows: np.ndarray, current_price: float) -> float:
        """Calculate ultra-precise support strength"""
        if len(lows) < 50:
            return 0
        
        # Find support levels in last 50 candles
        support_levels = []
        for i in range(len(lows) - 10):
            window = lows[i:i+10]
            if lows[i+5] == np.min(window):
                support_levels.append(lows[i+5])
        
        if not support_levels:
            return 0
        
        # Find closest support
        distances = [abs(current_price - level) / current_price for level in support_levels]
        closest_distance = min(distances) if distances else 1.0
        
        return max(0, 100 * (1 - closest_distance * 20))  # Stronger if closer
    
    def _calculate_ultra_resistance(self, highs: np.ndarray, current_price: float) -> float:
        """Calculate ultra-precise resistance strength"""
        if len(highs) < 50:
            return 0
        
        # Find resistance levels in last 50 candles
        resistance_levels = []
        for i in range(len(highs) - 10):
            window = highs[i:i+10]
            if highs[i+5] == np.max(window):
                resistance_levels.append(highs[i+5])
        
        if not resistance_levels:
            return 0
        
        # Find closest resistance
        distances = [abs(current_price - level) / current_price for level in resistance_levels]
        closest_distance = min(distances) if distances else 1.0
        
        return max(0, 100 * (1 - closest_distance * 20))  # Stronger if closer
    
    def detect_ultra_signals(self, features: Dict) -> Dict:
        """Ultra-strict signal detection for maximum win rate"""
        if not features:
            return {'confidence': 0, 'direction': None, 'signals': [], 'quality_score': 0}
        
        signals = []
        confidence = 0
        direction = None
        quality_score = 0
        
        range_pos = features.get('range_position', 50)
        rsi_14 = features.get('rsi_14', 50)
        
        # ULTRA-STRICT CONDITIONS FOR SHORT
        if range_pos > 92:  # Ultra-extreme high
            short_signals = []
            short_confidence = 0
            
            # 1. Ultra-extreme RSI
            if rsi_14 > 85:
                short_signals.append("Ultra-Extreme RSI (>85)")
                short_confidence += 25
            
            # 2. Perfect range consensus
            if features.get('range_consensus', 0) >= 3:
                short_signals.append("Perfect Range Consensus")
                short_confidence += 20
            
            # 3. Ultra-volume confirmation
            if features.get('ultra_volume', False):
                short_signals.append("Ultra-High Volume")
                short_confidence += 15
            
            # 4. Momentum exhaustion
            if features.get('momentum_exhaustion', False):
                short_signals.append("Momentum Exhaustion")
                short_confidence += 15
            
            # 5. Strong resistance
            if features.get('resistance_strength', 0) > 70:
                short_signals.append("Strong Resistance")
                short_confidence += 15
            
            # 6. Perfect MA alignment against trend
            if features.get('perfect_alignment', False) and features.get('trend_strength', 0) > 50:
                short_signals.append("Perfect Counter-Trend Setup")
                short_confidence += 10
            
            if short_confidence >= 75 and len(short_signals) >= 4:  # Ultra-strict
                direction = 'short'
                confidence = short_confidence
                signals = short_signals
        
        # ULTRA-STRICT CONDITIONS FOR LONG
        elif range_pos < 8:  # Ultra-extreme low
            long_signals = []
            long_confidence = 0
            
            # 1. Ultra-extreme RSI
            if rsi_14 < 15:
                long_signals.append("Ultra-Extreme RSI (<15)")
                long_confidence += 25
            
            # 2. Perfect range consensus
            if features.get('range_consensus', 0) >= 3:
                long_signals.append("Perfect Range Consensus")
                long_confidence += 20
            
            # 3. Ultra-volume confirmation
            if features.get('ultra_volume', False):
                long_signals.append("Ultra-High Volume")
                long_confidence += 15
            
            # 4. Momentum exhaustion
            if features.get('momentum_exhaustion', False):
                long_signals.append("Momentum Exhaustion")
                long_confidence += 15
            
            # 5. Strong support
            if features.get('support_strength', 0) > 70:
                long_signals.append("Strong Support")
                long_confidence += 15
            
            # 6. Perfect MA alignment against trend
            if features.get('perfect_alignment', False) and features.get('trend_strength', 0) < -50:
                long_signals.append("Perfect Counter-Trend Setup")
                long_confidence += 10
            
            if long_confidence >= 75 and len(long_signals) >= 4:  # Ultra-strict
                direction = 'long'
                confidence = long_confidence
                signals = long_signals
        
        # Ultra-quality scoring
        if direction:
            quality_factors = 0
            
            # Range extremity
            if features.get('ultra_extreme', False):
                quality_factors += 25
            
            # RSI extremity
            if features.get('rsi_ultra_extreme', False):
                quality_factors += 20
            
            # Volume confirmation
            if features.get('ultra_volume', False):
                quality_factors += 20
            
            # Support/Resistance strength
            support_res = max(features.get('support_strength', 0), features.get('resistance_strength', 0))
            if support_res > 80:
                quality_factors += 20
            
            # Perfect alignment
            if features.get('perfect_alignment', False):
                quality_factors += 15
            
            quality_score = min(100, quality_factors)
        
        return {
            'confidence': confidence,
            'direction': direction,
            'signals': signals,
            'quality_score': quality_score,
            'is_ultra_extreme': features.get('ultra_extreme', False),
            'range_position': range_pos
        }

class UltraHighWinRateBot:
    """Ultra High Win Rate AI Reversal Bot - Target: 75-85% Win Rate"""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # ULTRA-STRICT configuration for maximum win rate
        self.config = {
            "symbol": "SOL-USDT-SWAP",
            "base_position_size": 80.0,  # Smaller base size
            "max_position_size": 180.0,  # Smaller max size
            "profit_targets": {
                "conservative": 100,     # Lower targets for higher hit rate
                "aggressive": 140,
                "maximum": 180
            },
            "leverage": 5,               # Lower leverage for safety
            "min_confidence": 92,        # Ultra-high confidence required
            "min_quality_score": 75,     # Ultra-high quality required
            "max_daily_trades": 1,       # Only 1 trade per day maximum
            "risk_per_trade": 0.06,      # Lower risk per trade
            "stop_loss_pct": 2.0,        # Tighter stop loss
            "max_hold_hours": 6,         # Shorter hold time
            "min_data_points": 150,      # More data required
        }
        
        # Ultra AI system
        self.ai_detector = UltraAIDetector()
        
        # Trading state
        self.is_running = False
        self.current_position = None
        self.daily_trades = 0
        self.last_trade_day = None
        self.market_data = []
        self.trades = []
        
        # Performance tracking
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0
        
        # Market data
        self.last_price = 0.0
        self.last_update = None
        
        print("🎯 ULTRA HIGH WIN RATE AI REVERSAL BOT")
        print("🏆 TARGET: 75-85% WIN RATE WITH ULTRA-STRICT CONDITIONS")
        print("=" * 70)
    
    def start_trading(self):
        """Start the ultra high win rate bot"""
        print(f"🚀 Starting Ultra Bot with ${self.current_balance:.2f}")
        print(f"💰 Ultra-conservative sizing: ${self.config['base_position_size']}-${self.config['max_position_size']}")
        print(f"🎯 Ultra-conservative targets: ${self.config['profit_targets']['conservative']}-${self.config['profit_targets']['maximum']}")
        print(f"🧠 Ultra-strict AI: {self.config['min_confidence']}% confidence + {self.config['min_quality_score']}% quality")
        print(f"🛡️ Ultra-safe risk: {self.config['risk_per_trade']*100}% per trade, {self.config['max_daily_trades']} trade/day max")
        print(f"⏰ Ultra-short holds: Max {self.config['max_hold_hours']} hours")
        print("🔔 Only the HIGHEST QUALITY setups will be taken!")
        
        # Start monitoring
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        
        self.is_running = True
        
        try:
            asyncio.run(self._connect_websocket())
        except KeyboardInterrupt:
            print("\n⏹️ Ultra Bot stopped")
            self.stop_trading()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            try:
                self._display_status()
                time.sleep(15)  # Slower refresh for ultra-conservative approach
            except:
                pass
    
    def _display_status(self):
        """Display current ultra bot status"""
        if not self.market_data:
            print("⏳ Waiting for market data...")
            return
        
        print("\033[2J\033[H", end="")  # Clear screen
        
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        print("🏆 ULTRA HIGH WIN RATE BOT - LIVE STATUS")
        print("=" * 80)
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📈 Live Price: ${self.last_price:.4f}")
        print(f"💰 Balance: ${self.current_balance:.2f} ({total_return:+.1f}%)")
        
        if self.current_position:
            hold_time = (datetime.now() - self.current_position.entry_time).total_seconds() / 3600
            if self.current_position.direction == 'long':
                unrealized_pct = (self.last_price - self.current_position.entry_price) / self.current_position.entry_price * 100
            else:
                unrealized_pct = (self.current_position.entry_price - self.last_price) / self.current_position.entry_price * 100
            
            unrealized_amount = self.current_position.position_size * (unrealized_pct / 100) * self.config['leverage']
            target_progress = (unrealized_amount / self.current_position.target_profit) * 100
            
            print(f"\n🔵 ULTRA POSITION ACTIVE:")
            print(f"   Direction: {self.current_position.direction.upper()}")
            print(f"   Entry: ${self.current_position.entry_price:.4f} | Current: ${self.last_price:.4f}")
            print(f"   Size: ${self.current_position.position_size:.2f} | Target: ${self.current_position.target_profit}")
            print(f"   Unrealized P&L: ${unrealized_amount:+.2f} ({unrealized_pct:+.2f}%)")
            print(f"   Target Progress: {target_progress:.1f}%")
            print(f"   Hold Time: {hold_time:.1f}h | Confidence: {self.current_position.confidence:.1f}% | Quality: {self.current_position.quality_score:.1f}%")
        else:
            print(f"\n⚫ ULTRA SCANNING: Waiting for PERFECT setups only")
            print(f"   Data Points: {len(self.market_data)}/{self.config['min_data_points']} required")
            print(f"   Requirements: {self.config['min_confidence']}% confidence + {self.config['min_quality_score']}% quality")
        
        print(f"\n📊 ULTRA PERFORMANCE:")
        print(f"   Trades: {self.total_trades} | Win Rate: {win_rate:.1f}% 🎯TARGET: 75-85%")
        print(f"   Daily: {self.daily_trades}/{self.config['max_daily_trades']} (Ultra-Conservative)")
        print(f"   Total Profit: ${self.total_profit:+.2f}")
        print(f"   🏆 ULTRA-STRICT: Only highest quality setups taken")
        print("=" * 80)
    
    async def _connect_websocket(self):
        """Connect to OKX WebSocket for live data"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._process_data(data)
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
        
        def on_open(ws):
            print("🔗 Connected to OKX WebSocket (Ultra Mode)")
            subscribe = {
                "op": "subscribe",
                "args": [
                    {"channel": "tickers", "instId": self.config['symbol']},
                    {"channel": "candle1m", "instId": self.config['symbol']}
                ]
            }
            ws.send(json.dumps(subscribe))
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
        
        def run_websocket():
            import websocket
            websocket.enableTrace(False)
            ws = websocket.WebSocketApp(
                "wss://ws.okx.com:8443/ws/v5/public",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
        
        # Start WebSocket in background thread
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
        # Keep main thread alive
        while self.is_running:
            await asyncio.sleep(1)
    
    def _process_data(self, data):
        """Process incoming market data"""
        try:
            if 'data' not in data:
                return
            
            for item in data['data']:
                if data.get('arg', {}).get('channel') == 'candle1m':
                    # Process 1-minute candle
                    candle = {
                        'timestamp': int(item[0]),
                        'open': float(item[1]),
                        'high': float(item[2]),
                        'low': float(item[3]),
                        'close': float(item[4]),
                        'volume': float(item[5])
                    }
                    
                    self.market_data.append(candle)
                    if len(self.market_data) > 400:  # Keep even more data for ultra-analysis
                        self.market_data = self.market_data[-400:]
                    
                    self.last_price = candle['close']
                    self.last_update = datetime.now()
                    
                    # Ultra-conservative analysis
                    self._analyze_market_for_ultra_entry()
                    self._check_ultra_exit_conditions()
                
                elif data.get('arg', {}).get('channel') == 'tickers':
                    # Update current price
                    self.last_price = float(item['last'])
                    
        except Exception as e:
            logger.error(f"Data processing error: {e}")
    
    def _analyze_market_for_ultra_entry(self):
        """Ultra-conservative market analysis for entry"""
        if (self.current_position or 
            len(self.market_data) < self.config['min_data_points'] or
            self.daily_trades >= self.config['max_daily_trades']):
            return
        
        try:
            # Reset daily trades if new day
            current_day = datetime.now().date()
            if self.last_trade_day != current_day:
                self.daily_trades = 0
                self.last_trade_day = current_day
            
            # Ultra-comprehensive analysis
            df = pd.DataFrame(self.market_data)
            features = self.ai_detector.extract_ultra_features(df)
            
            if not features:
                return
            
            signals = self.ai_detector.detect_ultra_signals(features)
            
            # Ultra-strict entry check
            if self._should_enter_ultra_trade(signals, features):
                self._execute_ultra_entry(signals, features)
                
        except Exception as e:
            logger.error(f"Ultra market analysis error: {e}")
    
    def _should_enter_ultra_trade(self, signals: Dict, features: Dict) -> bool:
        """ULTRA-STRICT entry conditions for maximum win rate"""
        # Must have ultra-high confidence
        if signals['confidence'] < self.config['min_confidence']:
            return False
        
        # Must have ultra-high quality score
        if signals.get('quality_score', 0) < self.config['min_quality_score']:
            return False
        
        # Must be ultra-extreme range position
        if not signals.get('is_ultra_extreme', False):
            return False
        
        # Must have clear direction
        if not signals['direction']:
            return False
        
        # Must have at least 5 ultra-strong signals
        if len(signals.get('signals', [])) < 5:
            return False
        
        # ULTRA-STRICT confirmations (need ALL 5)
        ultra_confirmations = 0
        
        # 1. Ultra-extreme RSI
        rsi_14 = features.get('rsi_14', 50)
        if ((signals['direction'] == 'short' and rsi_14 > 85) or 
            (signals['direction'] == 'long' and rsi_14 < 15)):
            ultra_confirmations += 1
        
        # 2. Ultra-high volume
        if features.get('ultra_volume', False):
            ultra_confirmations += 1
        
        # 3. Perfect range consensus
        if features.get('range_consensus', 0) >= 3:
            ultra_confirmations += 1
        
        # 4. Strong support/resistance
        if ((signals['direction'] == 'long' and features.get('support_strength', 0) > 80) or
            (signals['direction'] == 'short' and features.get('resistance_strength', 0) > 80)):
            ultra_confirmations += 1
        
        # 5. Perfect MA alignment
        if features.get('perfect_alignment', False):
            ultra_confirmations += 1
        
        # Need ALL 5 confirmations for ultra-high win rate
        return ultra_confirmations >= 5
    
    def _execute_ultra_entry(self, signals: Dict, features: Dict):
        """Execute ultra-conservative entry"""
        try:
            current_price = features['current_price']
            
            # Ultra-conservative position sizing
            confidence_factor = min(signals['confidence'] / 100, 0.7)  # Cap at 70%
            quality_factor = min(signals.get('quality_score', 50) / 100, 0.4)  # Cap at 40%
            
            base_size = self.config['base_position_size']
            max_size = self.config['max_position_size']
            
            size_multiplier = (confidence_factor + quality_factor) / 2
            position_size = base_size + (max_size - base_size) * size_multiplier
            position_size = min(position_size, self.current_balance * self.config['risk_per_trade'])
            
            # Ultra-conservative profit targets
            if signals['confidence'] >= 98 and signals.get('quality_score', 0) >= 90:
                target_profit = self.config['profit_targets']['maximum']
            elif signals['confidence'] >= 95 and signals.get('quality_score', 0) >= 80:
                target_profit = self.config['profit_targets']['aggressive']
            else:
                target_profit = self.config['profit_targets']['conservative']
            
            # Create ultra position
            self.current_position = Trade(
                entry_time=datetime.now(),
                direction=signals['direction'],
                entry_price=current_price,
                target_profit=target_profit,
                position_size=position_size,
                confidence=signals['confidence'],
                quality_score=signals.get('quality_score', 0),
                reversal_signals=signals['signals']
            )
            
            self.daily_trades += 1
            self.total_trades += 1
            
            # Ultra notifications
            self._notify_ultra_entry(self.current_position, signals)
            
            logger.info(f"🏆 ULTRA ENTRY: {signals['direction'].upper()} @ ${current_price:.4f} | "
                       f"Size: ${position_size:.2f} | Target: ${target_profit} | "
                       f"Confidence: {signals['confidence']:.1f}% | Quality: {signals.get('quality_score', 0):.1f}%")
            
        except Exception as e:
            logger.error(f"Ultra entry execution error: {e}")
    
    def _check_ultra_exit_conditions(self):
        """Ultra-precise exit conditions"""
        if not self.current_position:
            return
        
        try:
            position = self.current_position
            current_price = self.last_price
            
            # Calculate current P&L
            if position.direction == 'long':
                pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
            else:
                pnl_pct = (position.entry_price - current_price) / position.entry_price * 100
            
            pnl_amount = position.position_size * (pnl_pct / 100) * self.config['leverage']
            hold_time = (datetime.now() - position.entry_time).total_seconds() / 3600
            
            # Target profit reached
            if pnl_amount >= position.target_profit:
                self._close_ultra_position("target_profit", current_price, hold_time, pnl_amount, pnl_pct)
                return
            
            # Quick partial profit at 60% of target (ultra-conservative)
            if pnl_amount >= position.target_profit * 0.6 and hold_time > 1:
                self._close_ultra_position("quick_profit", current_price, hold_time, pnl_amount, pnl_pct)
                return
            
            # Ultra-tight stop loss
            stop_loss_amount = -(position.position_size * self.config['stop_loss_pct'] / 100)
            if pnl_amount <= stop_loss_amount:
                self._close_ultra_position("stop_loss", current_price, hold_time, pnl_amount, pnl_pct)
                return
            
            # Ultra-short maximum hold time
            if hold_time >= self.config['max_hold_hours']:
                self._close_ultra_position("time_exit", current_price, hold_time, pnl_amount, pnl_pct)
                return
            
            # Ultra-early reversal detection (after 20 minutes)
            if hold_time > 0.33 and len(self.market_data) >= 150:
                df = pd.DataFrame(self.market_data[-150:])
                features = self.ai_detector.extract_ultra_features(df)
                reversal_signals = self.ai_detector.detect_ultra_signals(features)
                
                # Exit on medium reversal signal (ultra-conservative)
                if (reversal_signals['confidence'] > 80 and 
                    reversal_signals['direction'] != position.direction and
                    reversal_signals.get('quality_score', 0) > 50):
                    self._close_ultra_position("ultra_reversal", current_price, hold_time, pnl_amount, pnl_pct)
                    return
            
        except Exception as e:
            logger.error(f"Ultra exit check error: {e}")
    
    def _close_ultra_position(self, reason: str, exit_price: float, hold_time: float, pnl_amount: float, pnl_pct: float):
        """Close ultra position with enhanced tracking"""
        if not self.current_position:
            return
        
        position = self.current_position
        
        # Update position
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.exit_reason = reason
        position.pnl_amount = pnl_amount
        position.pnl_percentage = pnl_pct * self.config['leverage']
        
        # Update balance and stats
        self.current_balance += pnl_amount
        self.total_profit += pnl_amount
        
        if pnl_amount > 0:
            self.wins += 1
        else:
            self.losses += 1
        
        self.trades.append(position)
        
        # Ultra notifications
        self._notify_ultra_exit(position)
        
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        logger.info(f"🏆 ULTRA EXIT: {reason} | P&L: ${pnl_amount:+.2f} | WR: {win_rate:.1f}%")
        
        self.current_position = None
    
    def _notify_ultra_entry(self, position: Trade, signals: Dict):
        """Send ultra entry notifications"""
        # Ultra sound
        UltraTradeNotifier.play_sound("ultra_entry")
        
        # Desktop notification
        UltraTradeNotifier.desktop_notification(
            "🏆 ULTRA HIGH WIN RATE ENTRY",
            f"ULTRA {position.direction.upper()} @ ${position.entry_price:.4f}\n"
            f"Ultra AI: {position.confidence:.1f}%\n"
            f"Ultra Quality: {position.quality_score:.1f}%\n"
            f"Target: ${position.target_profit}"
        )
        
        # Console alert
        timestamp = datetime.now().strftime("%H:%M:%S")
        print("\n" + "="*100)
        print(f"🏆 ULTRA HIGH WIN RATE ENTRY - {timestamp}")
        print("="*100)
        print(f"📊 Direction: {position.direction.upper()}")
        print(f"💰 Entry Price: ${position.entry_price:.4f}")
        print(f"🏆 Ultra AI Confidence: {position.confidence:.1f}%")
        print(f"⭐ Ultra Quality Score: {position.quality_score:.1f}%")
        print(f"💵 Position Size: ${position.position_size:.2f}")
        print(f"🎯 Profit Target: ${position.target_profit}")
        print(f"📈 Expected Return: {((position.target_profit / position.position_size) * 100):.0f}%")
        print(f"🧠 Ultra Signals: {', '.join(position.reversal_signals)}")
        print(f"📏 Ultra-Safe Leverage: {self.config['leverage']}x")
        print(f"🏆 ULTRA MODE: Only PERFECT setups taken for 75-85% win rate")
        print("="*100)
    
    def _notify_ultra_exit(self, position: Trade):
        """Send ultra exit notifications"""
        # Sound based on performance
        if position.pnl_amount >= position.target_profit * 0.8:
            UltraTradeNotifier.play_sound("jackpot")
        elif position.pnl_amount > 0:
            UltraTradeNotifier.play_sound("profit")
        else:
            UltraTradeNotifier.play_sound("loss")
        
        # Desktop notification
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        UltraTradeNotifier.desktop_notification(
            "🏆 ULTRA EXIT",
            f"P&L: ${position.pnl_amount:+.2f}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Reason: {position.exit_reason}"
        )
        
        # Console alert
        profit_icon = "🏆" if position.pnl_amount >= position.target_profit * 0.8 else "💚" if position.pnl_amount > 0 else "❌"
        timestamp = datetime.now().strftime("%H:%M:%S")
        hold_time = (position.exit_time - position.entry_time).total_seconds() / 60
        
        print("\n" + "="*100)
        print(f"{profit_icon} ULTRA HIGH WIN RATE EXIT - {timestamp}")
        print("="*100)
        print(f"📊 Direction: {position.direction.upper()}")
        print(f"📈 Entry: ${position.entry_price:.4f} → Exit: ${position.exit_price:.4f}")
        print(f"💰 P&L: ${position.pnl_amount:+.2f} ({position.pnl_percentage:+.1f}%)")
        print(f"🎯 Target: ${position.target_profit} ({'✅ HIT' if position.pnl_amount >= position.target_profit * 0.8 else '❌ MISSED'})")
        print(f"⏱️ Hold Time: {hold_time:.1f} minutes")
        print(f"🎯 Exit Reason: {position.exit_reason}")
        print(f"🏆 Win Rate: {win_rate:.1f}% (TARGET: 75-85%)")
        print(f"💵 New Balance: ${self.current_balance:.2f}")
        print(f"🏆 ULTRA PERFORMANCE: Only highest quality trades")
        print("="*100)
    
    def stop_trading(self):
        """Stop the ultra bot"""
        self.is_running = False
        if self.current_position:
            self._close_ultra_position("manual_stop", self.last_price, 
                                     (datetime.now() - self.current_position.entry_time).total_seconds() / 3600,
                                     0, 0)
        
        self._show_ultra_results()
    
    def _show_ultra_results(self):
        """Show ultra trading results"""
        if self.total_trades == 0:
            print("No trades completed yet.")
            return
        
        win_rate = (self.wins / self.total_trades) * 100
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        print("\n" + "="*80)
        print("🏆 ULTRA HIGH WIN RATE BOT - FINAL RESULTS")
        print("="*80)
        print(f"📊 Total Trades: {self.total_trades}")
        print(f"✅ Wins: {self.wins} | ❌ Losses: {self.losses}")
        print(f"🏆 Win Rate: {win_rate:.1f}% (TARGET: 75-85%)")
        print(f"💰 Total Profit: ${self.total_profit:+.2f}")
        print(f"📈 Total Return: {total_return:+.1f}%")
        print(f"💵 Final Balance: ${self.current_balance:.2f}")
        
        if win_rate >= 75:
            print("🎉 TARGET ACHIEVED! Win rate 75%+ reached!")
        elif win_rate >= 70:
            print("👍 Close to target! Consider even stricter conditions.")
        else:
            print("⚠️ Below target. Ultra-strict conditions working as intended.")
        
        print("="*80)

def main():
    """Main function"""
    print("🏆 ULTRA HIGH WIN RATE AI REVERSAL BOT")
    print("🎯 TARGET: 75-85% WIN RATE")
    print("=" * 60)
    
    try:
        # Check for required dependencies
        required_modules = ['pandas', 'numpy', 'websocket']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"❌ Missing required modules: {', '.join(missing_modules)}")
            print("Please install with: pip install pandas numpy websocket-client")
            return
        
        print("✅ All dependencies found!")
        print("\n🏆 Starting Ultra High Win Rate Bot...")
        
        # Start bot
        bot = UltraHighWinRateBot(initial_balance=1000.0)
        print("🏆 Ultra High Win Rate Bot initialized!")
        print("📊 Use the backtest to validate 75-85% win rate target")
        
    except Exception as e:
        print(f"❌ Error starting ultra bot: {e}")

if __name__ == "__main__":
    main() 