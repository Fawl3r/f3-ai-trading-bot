#!/usr/bin/env python3
"""
OPTIMIZED AI Reversal Trading Bot
Improved version based on backtest results
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
    reversal_signals: List[str] = None

class OptimizedTradeNotifier:
    """Enhanced trade notification system"""
    
    @staticmethod
    def play_sound(sound_type: str = "entry"):
        if not SOUND_AVAILABLE:
            return
        try:
            if sound_type == "entry":
                winsound.Beep(1400, 300)  # Higher pitch for optimized entries
            elif sound_type == "big_profit":
                # Victory sound sequence
                for freq in [900, 1100, 1300, 1600]:
                    winsound.Beep(freq, 120)
                    time.sleep(0.03)
            elif sound_type == "profit":
                winsound.Beep(1000, 400)
            elif sound_type == "loss":
                winsound.Beep(350, 600)
        except:
            pass
    
    @staticmethod
    def desktop_notification(title: str, message: str, timeout: int = 8):
        if not NOTIFICATION_AVAILABLE:
            return
        try:
            notification.notify(
                title=title,
                message=message,
                timeout=timeout,
                app_name="Optimized AI Reversal Bot"
            )
        except:
            pass
    
    @staticmethod
    def print_trade_alert(trade_type: str, details: dict):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if trade_type == "ENTRY":
            print("\n" + "="*90)
            print(f"🎯 OPTIMIZED AI ENTRY - {timestamp}")
            print("="*90)
            print(f"📊 Direction: {details['direction'].upper()}")
            print(f"💰 Entry Price: ${details['entry_price']:.4f}")
            print(f"🎯 AI Confidence: {details['confidence']:.1f}%")
            print(f"💵 Position Size: ${details['position_size']:.2f}")
            print(f"🎯 Profit Target: ${details['target_profit']}")
            print(f"📈 Expected Return: {((details['target_profit'] / details['position_size']) * 100):.0f}%")
            print(f"🧠 AI Signals: {', '.join(details['signals'])}")
            print(f"📏 Leverage: {details['leverage']}x")
            print(f"🔧 OPTIMIZED: Enhanced entry conditions")
            print("="*90)
            
        elif trade_type == "EXIT":
            profit_icon = "🎉" if details['pnl_amount'] >= details['target_profit'] * 0.8 else "💚" if details['pnl_amount'] > 0 else "❌"
            print("\n" + "="*90)
            print(f"{profit_icon} OPTIMIZED AI EXIT - {timestamp}")
            print("="*90)
            print(f"📊 Direction: {details['direction'].upper()}")
            print(f"📈 Entry: ${details['entry_price']:.4f} → Exit: ${details['exit_price']:.4f}")
            print(f"💰 P&L: ${details['pnl_amount']:+.2f} ({details['pnl_percentage']:+.1f}%)")
            print(f"🎯 Target: ${details['target_profit']} ({'✅ HIT' if details['pnl_amount'] >= details['target_profit'] * 0.8 else '❌ MISSED'})")
            print(f"⏱️ Hold Time: {details['hold_time']:.1f} minutes")
            print(f"🎯 Exit Reason: {details['exit_reason']}")
            print(f"🏆 Win Rate: {details['win_rate']:.1f}%")
            print(f"💵 New Balance: ${details['new_balance']:.2f}")
            print("="*90)

class OptimizedAIDetector:
    """Enhanced AI system with better signal detection"""
    
    def __init__(self):
        self.price_history = []
        self.features_history = []
        
    def extract_enhanced_features(self, data: pd.DataFrame) -> Dict:
        """Extract comprehensive features with better indicators"""
        if len(data) < 100:  # Need more data for reliable signals
            return {}
        
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            features = {}
            
            # Basic price features
            features['current_price'] = close[-1]
            features['price_change_1'] = (close[-1] - close[-2]) / close[-2] * 100
            features['price_change_5'] = (close[-1] - close[-6]) / close[-6] * 100
            features['price_change_15'] = (close[-1] - close[-16]) / close[-16] * 100
            features['price_change_30'] = (close[-1] - close[-31]) / close[-31] * 100 if len(close) >= 31 else 0
            
            # Enhanced range analysis
            lookback_periods = [10, 20, 50]
            range_scores = []
            
            for period in lookback_periods:
                if len(high) >= period:
                    recent_high = np.max(high[-period:])
                    recent_low = np.min(low[-period:])
                    range_size = recent_high - recent_low
                    
                    if range_size > 0:
                        range_pos = (close[-1] - recent_low) / range_size * 100
                        range_scores.append(range_pos)
            
            # Use consensus range position
            features['range_position'] = np.mean(range_scores) if range_scores else 50
            features['range_consensus'] = len([r for r in range_scores if r > 85 or r < 15])
            
            # Multiple timeframe moving averages
            ma_periods = [5, 10, 20, 50]
            ma_values = []
            
            for period in ma_periods:
                if len(close) >= period:
                    ma = np.mean(close[-period:])
                    ma_values.append(ma)
                    features[f'sma_{period}'] = ma
                    features[f'price_vs_sma_{period}'] = (close[-1] - ma) / ma * 100
            
            # Trend strength
            if len(ma_values) >= 4:
                # Check if MAs are in order (trending)
                ascending = all(ma_values[i] <= ma_values[i+1] for i in range(len(ma_values)-1))
                descending = all(ma_values[i] >= ma_values[i+1] for i in range(len(ma_values)-1))
                features['trend_strength'] = 100 if ascending else -100 if descending else 0
            else:
                features['trend_strength'] = 0
            
            # Enhanced RSI with multiple periods
            features['rsi_14'] = self._calculate_rsi(close, 14)
            features['rsi_21'] = self._calculate_rsi(close, 21)
            features['rsi_consensus'] = abs(features['rsi_14'] - features['rsi_21'])  # Divergence
            
            # Volume analysis
            if len(volume) >= 20:
                vol_sma = np.mean(volume[-20:])
                features['volume_ratio'] = volume[-1] / vol_sma
                features['volume_trend'] = (np.mean(volume[-5:]) - np.mean(volume[-20:])) / np.mean(volume[-20:]) * 100
            else:
                features['volume_ratio'] = 1.0
                features['volume_trend'] = 0
            
            # Price momentum and acceleration
            if len(close) >= 10:
                momentum_5 = close[-1] - close[-6]
                momentum_10 = close[-6] - close[-11] if len(close) >= 11 else 0
                features['momentum_acceleration'] = momentum_5 - momentum_10
            else:
                features['momentum_acceleration'] = 0
            
            # Volatility analysis
            if len(close) >= 20:
                returns = np.diff(close[-20:]) / close[-20:-1]
                features['volatility'] = np.std(returns) * 100
                features['volatility_ratio'] = features['volatility'] / np.mean(np.abs(returns)) if np.mean(np.abs(returns)) > 0 else 1
            else:
                features['volatility'] = 1.0
                features['volatility_ratio'] = 1.0
            
            # Support/Resistance strength
            features['support_strength'] = self._calculate_support_strength(low, close[-1])
            features['resistance_strength'] = self._calculate_resistance_strength(high, close[-1])
            
            return features
            
        except Exception as e:
            logger.error(f"Enhanced feature extraction error: {e}")
            return {}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_support_strength(self, lows: np.ndarray, current_price: float) -> float:
        """Calculate support level strength"""
        if len(lows) < 20:
            return 0
        
        recent_lows = lows[-50:]  # Look at more data
        support_level = np.min(recent_lows)
        tolerance = current_price * 0.002  # 0.2% tolerance
        
        touches = np.sum(np.abs(recent_lows - support_level) < tolerance)
        return min(touches * 15, 100)
    
    def _calculate_resistance_strength(self, highs: np.ndarray, current_price: float) -> float:
        """Calculate resistance level strength"""
        if len(highs) < 20:
            return 0
        
        recent_highs = highs[-50:]  # Look at more data
        resistance_level = np.max(recent_highs)
        tolerance = current_price * 0.002  # 0.2% tolerance
        
        touches = np.sum(np.abs(recent_highs - resistance_level) < tolerance)
        return min(touches * 15, 100)
    
    def detect_optimized_signals(self, features: Dict) -> Dict:
        """Enhanced signal detection with stricter conditions"""
        signals = {
            'reversal_probability': 0,
            'direction': None,
            'confidence': 0,
            'signals': [],
            'is_range_extreme': False,
            'quality_score': 0
        }
        
        if not features:
            return signals
        
        # Enhanced range extreme detection
        range_pos = features.get('range_position', 50)
        range_consensus = features.get('range_consensus', 0)
        
        # More stringent range requirements
        if range_pos >= 90 and range_consensus >= 2:  # Top 10% with consensus
            signals['is_range_extreme'] = True
            signals['direction'] = 'short'
            signals['signals'].append('strong_range_high')
            signals['reversal_probability'] += 40
            signals['quality_score'] += 30
        elif range_pos <= 10 and range_consensus >= 2:  # Bottom 10% with consensus
            signals['is_range_extreme'] = True
            signals['direction'] = 'long'
            signals['signals'].append('strong_range_low')
            signals['reversal_probability'] += 40
            signals['quality_score'] += 30
        
        # Enhanced RSI analysis
        rsi_14 = features.get('rsi_14', 50)
        rsi_21 = features.get('rsi_21', 50)
        rsi_consensus = features.get('rsi_consensus', 0)
        
        # RSI extremes with consensus
        if rsi_14 >= 80 and rsi_21 >= 75 and rsi_consensus < 10:  # Strong overbought consensus
            signals['signals'].append('extreme_overbought')
            signals['reversal_probability'] += 25
            signals['quality_score'] += 20
            if not signals['direction']:
                signals['direction'] = 'short'
        elif rsi_14 <= 20 and rsi_21 <= 25 and rsi_consensus < 10:  # Strong oversold consensus
            signals['signals'].append('extreme_oversold')
            signals['reversal_probability'] += 25
            signals['quality_score'] += 20
            if not signals['direction']:
                signals['direction'] = 'long'
        
        # Trend analysis
        trend_strength = features.get('trend_strength', 0)
        price_vs_sma_20 = features.get('price_vs_sma_20', 0)
        
        # Counter-trend setups (reversals)
        if signals['direction'] == 'short' and trend_strength > 50 and price_vs_sma_20 > 3:
            signals['signals'].append('counter_uptrend')
            signals['reversal_probability'] += 15
            signals['quality_score'] += 15
        elif signals['direction'] == 'long' and trend_strength < -50 and price_vs_sma_20 < -3:
            signals['signals'].append('counter_downtrend')
            signals['reversal_probability'] += 15
            signals['quality_score'] += 15
        
        # Volume confirmation (stricter)
        vol_ratio = features.get('volume_ratio', 1.0)
        vol_trend = features.get('volume_trend', 0)
        
        if vol_ratio > 2.0 and vol_trend > 20:  # Strong volume surge
            signals['signals'].append('strong_volume_confirmation')
            signals['reversal_probability'] += 20
            signals['quality_score'] += 25
        elif vol_ratio > 1.5:  # Moderate volume
            signals['signals'].append('volume_confirmation')
            signals['reversal_probability'] += 10
            signals['quality_score'] += 10
        
        # Support/Resistance confluence
        if signals['direction'] == 'long':
            support_strength = features.get('support_strength', 0)
            if support_strength > 60:  # Strong support
                signals['signals'].append('strong_support_confluence')
                signals['reversal_probability'] += 20
                signals['quality_score'] += 20
        elif signals['direction'] == 'short':
            resistance_strength = features.get('resistance_strength', 0)
            if resistance_strength > 60:  # Strong resistance
                signals['signals'].append('strong_resistance_confluence')
                signals['reversal_probability'] += 20
                signals['quality_score'] += 20
        
        # Momentum divergence
        momentum_accel = features.get('momentum_acceleration', 0)
        if signals['direction'] == 'short' and momentum_accel < -0.5:  # Weakening upward momentum
            signals['signals'].append('momentum_divergence')
            signals['reversal_probability'] += 15
            signals['quality_score'] += 15
        elif signals['direction'] == 'long' and momentum_accel > 0.5:  # Weakening downward momentum
            signals['signals'].append('momentum_divergence')
            signals['reversal_probability'] += 15
            signals['quality_score'] += 15
        
        # Volatility filter (avoid low volatility periods)
        volatility = features.get('volatility', 1.0)
        if volatility < 0.5:  # Too low volatility
            signals['reversal_probability'] -= 20
            signals['quality_score'] -= 20
            signals['signals'].append('low_volatility_penalty')
        
        # Final confidence calculation with quality weighting
        base_confidence = min(signals['reversal_probability'], 100)
        quality_bonus = min(signals['quality_score'] / 5, 20)  # Up to 20% bonus
        
        signals['confidence'] = max(0, min(100, base_confidence + quality_bonus))
        
        return signals

class OptimizedAIReversalBot:
    """Optimized AI Reversal Trading Bot"""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Optimized configuration
        self.config = {
            "symbol": "SOL-USDT-SWAP",
            "base_position_size": 100.0,
            "max_position_size": 250.0,
            "profit_targets": {
                "conservative": 120,
                "aggressive": 180,
                "maximum": 250
            },
            "leverage": 6,
            "min_confidence": 88,
            "max_daily_trades": 2,
            "risk_per_trade": 0.08,
            "stop_loss_pct": 2.5,
            "max_hold_hours": 8,
        }
        
        # AI system
        self.ai_detector = OptimizedAIDetector()
        
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
        
        print("🎯 OPTIMIZED AI REVERSAL TRADING BOT")
        print("📈 ENHANCED STRATEGY FOR HIGHER WIN RATE")
        print("=" * 60)
    
    def start_trading(self):
        """Start the optimized AI reversal trading bot"""
        print(f"🚀 Starting Optimized AI Bot with ${self.current_balance:.2f}")
        print(f"💰 Base position size: ${self.config['base_position_size']}")
        print(f"🎯 Profit targets: ${self.config['profit_targets']['conservative']}-${self.config['profit_targets']['maximum']}")
        print(f"🧠 Enhanced AI with {self.config['min_confidence']}% minimum confidence")
        print(f"🛡️ Improved risk management: {self.config['risk_per_trade']*100}% per trade")
        print("🔔 You'll be notified of all trades!")
        
        # Start monitoring
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        
        self.is_running = True
        
        try:
            asyncio.run(self._connect_websocket())
        except KeyboardInterrupt:
            print("\n⏹️ Optimized Bot stopped")
            self.stop_trading()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            try:
                self._display_status()
                time.sleep(10)
            except:
                pass
    
    def _display_status(self):
        """Display current bot status"""
        if not self.market_data:
            print("⏳ Waiting for market data...")
            return
        
        print("\033[2J\033[H", end="")  # Clear screen
        
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        print("🎯 OPTIMIZED AI REVERSAL BOT - LIVE STATUS")
        print("=" * 70)
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
            
            print(f"\n🔵 ACTIVE POSITION:")
            print(f"   Direction: {self.current_position.direction.upper()}")
            print(f"   Entry: ${self.current_position.entry_price:.4f} | Current: ${self.last_price:.4f}")
            print(f"   Size: ${self.current_position.position_size:.2f} | Target: ${self.current_position.target_profit}")
            print(f"   Unrealized P&L: ${unrealized_amount:+.2f} ({unrealized_pct:+.2f}%)")
            print(f"   Target Progress: {target_progress:.1f}%")
            print(f"   Hold Time: {hold_time:.1f} hours | Confidence: {self.current_position.confidence:.1f}%")
        else:
            print(f"\n⚫ SCANNING: Looking for HIGH-QUALITY setups (85%+ confidence)")
            print(f"   Data Points: {len(self.market_data)}/{self.config['min_data_points']} required")
        
        print(f"\n📊 PERFORMANCE:")
        print(f"   Trades: {self.total_trades} | Win Rate: {win_rate:.1f}%")
        print(f"   Daily: {self.daily_trades}/{self.config['max_daily_trades']}")
        print(f"   Total Profit: ${self.total_profit:+.2f}")
        print(f"   🔧 OPTIMIZED: Enhanced AI with stricter entry conditions")
        print("=" * 70)
    
    async def _connect_websocket(self):
        """Connect to OKX WebSocket for live data"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._process_data(data)
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
        
        def on_open(ws):
            print("🔗 Connected to OKX WebSocket (Optimized)")
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
                    if len(self.market_data) > 300:  # Keep more data for better analysis
                        self.market_data = self.market_data[-300:]
                    
                    self.last_price = candle['close']
                    self.last_update = datetime.now()
                    
                    # Analyze for entry/exit
                    self._analyze_market_for_entry()
                    self._check_exit_conditions()
                
                elif data.get('arg', {}).get('channel') == 'tickers':
                    # Update current price
                    self.last_price = float(item['last'])
                    self.last_update = datetime.now()
                    
        except Exception as e:
            logger.error(f"Data processing error: {e}")
    
    def _analyze_market_for_entry(self):
        """Enhanced AI market analysis with stricter conditions"""
        try:
            # Need more data for reliable analysis
            if len(self.market_data) < self.config['min_data_points']:
                return
            
            # Daily trade limit
            current_day = datetime.now().date()
            if self.last_trade_day != current_day:
                self.daily_trades = 0
                self.last_trade_day = current_day
            
            if self.daily_trades >= self.config['max_daily_trades'] or self.current_position:
                return
            
            # Create DataFrame for enhanced analysis
            df = pd.DataFrame(self.market_data[-200:])  # Use more data
            
            # Extract enhanced AI features
            features = self.ai_detector.extract_enhanced_features(df)
            if not features:
                return
            
            # Detect optimized reversal signals
            signals = self.ai_detector.detect_optimized_signals(features)
            
            # Check enhanced entry conditions
            if self._should_enter_optimized_trade(signals, features):
                self._execute_optimized_entry(signals, features)
                
        except Exception as e:
            logger.error(f"Enhanced market analysis error: {e}")
    
    def _should_enter_optimized_trade(self, signals: Dict, features: Dict) -> bool:
        """Enhanced entry conditions with stricter requirements"""
        # Must be at range extreme with consensus
        if not signals['is_range_extreme']:
            return False
        
        # Must have very high AI confidence
        if signals['confidence'] < self.config['min_confidence']:
            return False
        
        # Must have clear direction
        if not signals['direction']:
            return False
        
        # Must have high quality score
        if signals.get('quality_score', 0) < 50:
            return False
        
        # Enhanced confirmations (need at least 3)
        confirmations = 0
        
        # Strong volume confirmation
        if features.get('volume_ratio', 1.0) > 1.8:
            confirmations += 1
        
        # RSI extreme with consensus
        rsi_14 = features.get('rsi_14', 50)
        rsi_consensus = features.get('rsi_consensus', 100)
        if ((signals['direction'] == 'short' and rsi_14 > 75 and rsi_consensus < 15) or 
            (signals['direction'] == 'long' and rsi_14 < 25 and rsi_consensus < 15)):
            confirmations += 1
        
        # Range position very extreme
        range_pos = features.get('range_position', 50)
        if (signals['direction'] == 'short' and range_pos > 85) or (signals['direction'] == 'long' and range_pos < 15):
            confirmations += 1
        
        # Support/Resistance confluence
        if signals['direction'] == 'long' and features.get('support_strength', 0) > 50:
            confirmations += 1
        elif signals['direction'] == 'short' and features.get('resistance_strength', 0) > 50:
            confirmations += 1
        
        # Trend divergence (counter-trend setup)
        trend_strength = features.get('trend_strength', 0)
        if ((signals['direction'] == 'short' and trend_strength > 30) or 
            (signals['direction'] == 'long' and trend_strength < -30)):
            confirmations += 1
        
        # Need at least 3 strong confirmations
        return confirmations >= 3
    
    def _execute_optimized_entry(self, signals: Dict, features: Dict):
        """Execute trade entry with optimized position sizing"""
        try:
            current_price = features['current_price']
            
            # Conservative position sizing based on confidence and quality
            confidence_multiplier = min(signals['confidence'] / 100, 0.8)  # Cap at 80%
            quality_multiplier = min(signals.get('quality_score', 50) / 100, 0.5)  # Cap at 50%
            
            base_size = self.config['base_position_size']
            max_size = self.config['max_position_size']
            
            size_multiplier = (confidence_multiplier + quality_multiplier) / 2
            position_size = base_size + (max_size - base_size) * size_multiplier
            position_size = min(position_size, self.current_balance * self.config['risk_per_trade'])
            
            # Conservative profit targets
            if signals['confidence'] >= 95 and signals.get('quality_score', 0) >= 80:
                target_profit = self.config['profit_targets']['maximum']
            elif signals['confidence'] >= 90 and signals.get('quality_score', 0) >= 60:
                target_profit = self.config['profit_targets']['aggressive']
            else:
                target_profit = self.config['profit_targets']['conservative']
            
            # Create position
            self.current_position = Trade(
                entry_time=datetime.now(),
                direction=signals['direction'],
                entry_price=current_price,
                target_profit=target_profit,
                position_size=position_size,
                confidence=signals['confidence'],
                reversal_signals=signals['signals']
            )
            
            self.daily_trades += 1
            self.total_trades += 1
            
            # Notifications
            self._notify_optimized_entry(self.current_position, signals)
            
            logger.info(f"🎯 OPTIMIZED ENTRY: {signals['direction'].upper()} @ ${current_price:.4f} | "
                       f"Size: ${position_size:.2f} | Target: ${target_profit} | "
                       f"Confidence: {signals['confidence']:.1f}% | Quality: {signals.get('quality_score', 0):.1f}")
            
        except Exception as e:
            logger.error(f"Optimized entry execution error: {e}")
    
    def _check_exit_conditions(self):
        """Enhanced exit conditions with better risk management"""
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
                self._close_optimized_position("target_profit", current_price, hold_time, pnl_amount, pnl_pct)
                return
            
            # Partial profit taking at 70% of target
            if pnl_amount >= position.target_profit * 0.7 and hold_time > 2:
                self._close_optimized_position("partial_profit", current_price, hold_time, pnl_amount, pnl_pct)
                return
            
            # Tighter stop loss
            stop_loss_amount = -(position.position_size * self.config['stop_loss_pct'] / 100)
            if pnl_amount <= stop_loss_amount:
                self._close_optimized_position("stop_loss", current_price, hold_time, pnl_amount, pnl_pct)
                return
            
            # Shorter maximum hold time
            if hold_time >= self.config['max_hold_hours']:
                self._close_optimized_position("time_exit", current_price, hold_time, pnl_amount, pnl_pct)
                return
            
            # Enhanced reversal detection (earlier)
            if hold_time > 0.5 and len(self.market_data) >= 100:  # Check after 30 minutes
                df = pd.DataFrame(self.market_data[-100:])
                features = self.ai_detector.extract_enhanced_features(df)
                reversal_signals = self.ai_detector.detect_optimized_signals(features)
                
                # Exit on strong reversal signal against position
                if (reversal_signals['confidence'] > 85 and 
                    reversal_signals['direction'] != position.direction and
                    reversal_signals.get('quality_score', 0) > 60):
                    self._close_optimized_position("ai_reversal", current_price, hold_time, pnl_amount, pnl_pct)
                    return
            
        except Exception as e:
            logger.error(f"Enhanced exit check error: {e}")
    
    def _close_optimized_position(self, reason: str, exit_price: float, hold_time: float, pnl_amount: float, pnl_pct: float):
        """Close position with enhanced tracking"""
        if not self.current_position:
            return
        
        position = self.current_position
        
        # Update position
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.exit_reason = reason
        position.pnl_amount = pnl_amount
        position.pnl_percentage = pnl_pct * self.config['leverage']
        position.hold_time = hold_time
        
        # Update balance and stats
        self.current_balance += pnl_amount
        self.total_profit += pnl_amount
        
        if pnl_amount > 0:
            self.wins += 1
        else:
            self.losses += 1
        
        self.trades.append(position)
        
        # Notifications
        self._notify_optimized_exit(position)
        
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        logger.info(f"🔄 OPTIMIZED EXIT: {reason} | P&L: ${pnl_amount:+.2f} | WR: {win_rate:.1f}%")
        
        self.current_position = None
    
    def _notify_optimized_entry(self, position: Trade, signals: Dict):
        """Send optimized entry notifications"""
        # Sound
        OptimizedTradeNotifier.play_sound("entry")
        
        # Desktop notification
        OptimizedTradeNotifier.desktop_notification(
            "🎯 OPTIMIZED AI ENTRY",
            f"{position.direction.upper()} @ ${position.entry_price:.4f}\n"
            f"Enhanced AI: {position.confidence:.1f}%\n"
            f"Quality Score: {signals.get('quality_score', 0):.1f}\n"
            f"Size: ${position.position_size:.2f}"
        )
        
        # Console alert
        OptimizedTradeNotifier.print_trade_alert("ENTRY", {
            'direction': position.direction,
            'entry_price': position.entry_price,
            'confidence': position.confidence,
            'position_size': position.position_size,
            'target_profit': position.target_profit,
            'signals': signals['signals'],
            'leverage': self.config['leverage']
        })
    
    def _notify_optimized_exit(self, position: Trade):
        """Send optimized exit notifications"""
        # Sound
        if position.pnl_amount >= position.target_profit * 0.8:
            OptimizedTradeNotifier.play_sound("big_profit")
        elif position.pnl_amount > 0:
            OptimizedTradeNotifier.play_sound("profit")
        else:
            OptimizedTradeNotifier.play_sound("loss")
        
        # Desktop notification
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        profit_icon = "🎉" if position.pnl_amount >= position.target_profit * 0.8 else "💚" if position.pnl_amount > 0 else "❌"
        
        OptimizedTradeNotifier.desktop_notification(
            f"{profit_icon} OPTIMIZED AI EXIT",
            f"{position.direction.upper()} closed\n"
            f"P&L: ${position.pnl_amount:+.2f}\n"
            f"Enhanced Strategy\n"
            f"Win Rate: {win_rate:.1f}%"
        )
        
        # Console alert
        OptimizedTradeNotifier.print_trade_alert("EXIT", {
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': position.exit_price,
            'pnl_amount': position.pnl_amount,
            'pnl_percentage': position.pnl_percentage,
            'target_profit': position.target_profit,
            'hold_time': position.hold_time,
            'exit_reason': position.exit_reason,
            'win_rate': win_rate,
            'new_balance': self.current_balance
        })
    
    def stop_trading(self):
        """Stop the optimized trading bot"""
        self.is_running = False
        
        if self.current_position:
            hold_time = (datetime.now() - self.current_position.entry_time).total_seconds() / 3600
            if self.current_position.direction == 'long':
                pnl_pct = (self.last_price - self.current_position.entry_price) / self.current_position.entry_price * 100
            else:
                pnl_pct = (self.current_position.entry_price - self.last_price) / self.current_position.entry_price * 100
            
            pnl_amount = self.current_position.position_size * (pnl_pct / 100) * self.config['leverage']
            self._close_optimized_position("manual_stop", self.last_price, hold_time, pnl_amount, pnl_pct)
        
        self._show_optimized_results()
    
    def _show_optimized_results(self):
        """Show optimized trading results"""
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print("\n" + "="*70)
        print("🎯 OPTIMIZED AI REVERSAL BOT RESULTS")
        print("="*70)
        print(f"📊 Total Trades: {self.total_trades}")
        print(f"🏆 Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        print(f"💰 Total Return: {total_return:+.1f}%")
        print(f"💵 Final Balance: ${self.current_balance:.2f}")
        print(f"💎 Total Profit: ${self.total_profit:+.2f}")
        print(f"🔧 OPTIMIZED: Enhanced AI with stricter conditions")
        
        if self.trades:
            profits = [t.pnl_amount for t in self.trades if t.pnl_amount > 0]
            if profits:
                avg_profit = np.mean(profits)
                max_profit = max(profits)
                print(f"📈 Avg Profit: ${avg_profit:.2f} | Max Profit: ${max_profit:.2f}")
        
        print("="*70)

def main():
    print("🎯 OPTIMIZED AI REVERSAL TRADING BOT")
    print("🔧 Enhanced Strategy with Better Entry Conditions")
    
    balance = input("\n💰 Starting balance (default $1000): ").strip()
    initial_balance = float(balance) if balance else 1000.0
    
    bot = OptimizedAIReversalBot(initial_balance)
    print(f"\n🚀 Optimized bot ready with ${initial_balance:.2f}")
    print("📊 Enhanced AI analysis for better performance")

if __name__ == "__main__":
    main() 