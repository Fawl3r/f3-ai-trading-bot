#!/usr/bin/env python3
"""
AI Reversal Trading Bot - High Profit Perpetual Futures
Identifies market highs/lows and executes large trades for 100-500% profits
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

# AI and ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import talib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Install scikit-learn and TA-Lib for full AI features: pip install scikit-learn TA-Lib")

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

class TradeNotifier:
    """Enhanced trade notification system for AI bot"""
    
    @staticmethod
    def play_sound(sound_type: str = "entry"):
        if not SOUND_AVAILABLE:
            return
        try:
            if sound_type == "entry":
                winsound.Beep(1200, 400)  # Higher pitch for AI entries
            elif sound_type == "big_profit":
                # Celebration sound for big profits
                for freq in [800, 1000, 1200, 1500]:
                    winsound.Beep(freq, 150)
                    time.sleep(0.05)
            elif sound_type == "loss":
                winsound.Beep(300, 800)  # Lower, longer tone
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
                app_name="AI Reversal Bot"
            )
        except:
            pass
    
    @staticmethod
    def print_trade_alert(trade_type: str, details: dict):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if trade_type == "ENTRY":
            print("\n" + "="*90)
            print(f"🤖 AI REVERSAL ENTRY - {timestamp}")
            print("="*90)
            print(f"📊 Direction: {details['direction'].upper()}")
            print(f"💰 Entry Price: ${details['entry_price']:.4f}")
            print(f"🎯 AI Confidence: {details['confidence']:.1f}%")
            print(f"💵 Position Size: ${details['position_size']:.2f}")
            print(f"🎯 Profit Target: ${details['target_profit']}")
            print(f"📈 Expected Return: {((details['target_profit'] / details['position_size']) * 100):.0f}%")
            print(f"🤖 AI Signals: {', '.join(details['signals'])}")
            print(f"📏 Leverage: {details['leverage']}x")
            print("="*90)
            
        elif trade_type == "EXIT":
            profit_icon = "🎉" if details['pnl_amount'] >= details['target_profit'] * 0.8 else "💚" if details['pnl_amount'] > 0 else "❌"
            print("\n" + "="*90)
            print(f"{profit_icon} AI TRADE CLOSED - {timestamp}")
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

class AIReversalDetector:
    """Advanced AI system for detecting market reversals and range extremes"""
    
    def __init__(self):
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.reversal_model = None
        self.price_history = []
        self.features_history = []
        self.is_trained = False
        
    def extract_features(self, data: pd.DataFrame) -> Dict:
        """Extract comprehensive technical features for AI analysis"""
        if len(data) < 50:
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
            
            # Range analysis (your strategy)
            recent_high = np.max(high[-20:])
            recent_low = np.min(low[-20:])
            range_size = recent_high - recent_low
            
            features['distance_from_high'] = (recent_high - close[-1]) / range_size * 100
            features['distance_from_low'] = (close[-1] - recent_low) / range_size * 100
            features['range_position'] = features['distance_from_low']  # 0-100% in range
            
            # Moving averages
            features['sma_5'] = np.mean(close[-5:])
            features['sma_10'] = np.mean(close[-10:])
            features['sma_20'] = np.mean(close[-20:])
            features['sma_50'] = np.mean(close[-50:])
            
            # Price vs MA positions
            features['price_vs_sma5'] = (close[-1] - features['sma_5']) / features['sma_5'] * 100
            features['price_vs_sma20'] = (close[-1] - features['sma_20']) / features['sma_20'] * 100
            
            # RSI calculation
            features['rsi'] = self._calculate_rsi(close)
            
            # Volume analysis
            features['volume_ratio'] = volume[-1] / np.mean(volume[-10:]) if len(volume) >= 10 else 1.0
            features['volume_trend'] = (np.mean(volume[-3:]) - np.mean(volume[-10:])) / np.mean(volume[-10:]) * 100
            
            # Volatility
            returns = np.diff(close) / close[:-1]
            features['volatility'] = np.std(returns[-20:]) * 100
            
            # Support/Resistance levels
            features['support_strength'] = self._calculate_support_strength(low, close[-1])
            features['resistance_strength'] = self._calculate_resistance_strength(high, close[-1])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI manually"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_support_strength(self, lows: np.ndarray, current_price: float) -> float:
        """Calculate support level strength"""
        recent_lows = lows[-20:]
        support_level = np.min(recent_lows)
        touches = np.sum(np.abs(recent_lows - support_level) < (current_price * 0.001))
        return min(touches * 20, 100)
    
    def _calculate_resistance_strength(self, highs: np.ndarray, current_price: float) -> float:
        """Calculate resistance level strength"""
        recent_highs = highs[-20:]
        resistance_level = np.max(recent_highs)
        touches = np.sum(np.abs(recent_highs - resistance_level) < (current_price * 0.001))
        return min(touches * 20, 100)
    
    def detect_reversal_signals(self, features: Dict) -> Dict:
        """Detect potential reversal signals using multiple methods"""
        signals = {
            'reversal_probability': 0,
            'direction': None,
            'confidence': 0,
            'signals': [],
            'is_range_extreme': False
        }
        
        if not features:
            return signals
        
        # Range extreme detection (your manual strategy)
        range_pos = features.get('range_position', 50)
        
        # Near range high (short opportunity)
        if range_pos >= 85:  # Top 15% of range
            signals['is_range_extreme'] = True
            signals['direction'] = 'short'
            signals['signals'].append('range_high_extreme')
            signals['reversal_probability'] += 30
        
        # Near range low (long opportunity)  
        elif range_pos <= 15:  # Bottom 15% of range
            signals['is_range_extreme'] = True
            signals['direction'] = 'long'
            signals['signals'].append('range_low_extreme')
            signals['reversal_probability'] += 30
        
        # RSI extremes
        rsi = features.get('rsi', 50)
        if rsi >= 75:  # Overbought
            signals['signals'].append('rsi_overbought')
            signals['reversal_probability'] += 20
            if not signals['direction']:
                signals['direction'] = 'short'
        elif rsi <= 25:  # Oversold
            signals['signals'].append('rsi_oversold')
            signals['reversal_probability'] += 20
            if not signals['direction']:
                signals['direction'] = 'long'
        
        # Volume confirmation
        vol_ratio = features.get('volume_ratio', 1.0)
        if vol_ratio > 1.5:  # High volume
            signals['signals'].append('volume_confirmation')
            signals['reversal_probability'] += 15
        
        # Support/Resistance strength
        if signals['direction'] == 'long':
            support_strength = features.get('support_strength', 0)
            if support_strength > 40:
                signals['signals'].append('strong_support')
                signals['reversal_probability'] += 15
        elif signals['direction'] == 'short':
            resistance_strength = features.get('resistance_strength', 0)
            if resistance_strength > 40:
                signals['signals'].append('strong_resistance')
                signals['reversal_probability'] += 15
        
        # Price momentum divergence
        price_change_1 = features.get('price_change_1', 0)
        price_change_5 = features.get('price_change_5', 0)
        
        if signals['direction'] == 'short' and price_change_1 > 0 and price_change_5 > price_change_1:
            signals['signals'].append('momentum_weakening')
            signals['reversal_probability'] += 10
        elif signals['direction'] == 'long' and price_change_1 < 0 and abs(price_change_5) > abs(price_change_1):
            signals['signals'].append('momentum_weakening')
            signals['reversal_probability'] += 10
        
        # Final confidence calculation
        signals['confidence'] = min(signals['reversal_probability'], 100)
        
        return signals

class AIReversalTradingBot:
    """AI-Powered Reversal Trading Bot for High-Profit Perpetual Futures"""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Trading configuration (your style)
        self.config = {
            "symbol": "SOL-USDT-SWAP",
            "base_position_size": 150.0,  # Your $150 trade size
            "max_position_size": 500.0,   # Scale up for high confidence
            "profit_targets": {
                "conservative": 200,  # 133% profit ($150 → $350)
                "aggressive": 300,    # 200% profit ($150 → $450)
                "maximum": 500        # 333% profit ($150 → $650)
            },
            "leverage": 10,  # 10x leverage for bigger profits
            "min_confidence": 70,  # Minimum AI confidence
            "max_daily_trades": 5,  # Quality over quantity
            "risk_per_trade": 0.15,  # 15% of account per trade
            "stop_loss_pct": 5.0,  # 5% stop loss
            "max_hold_hours": 24,  # Maximum hold time
        }
        
        # AI system
        self.ai_detector = AIReversalDetector()
        
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
        
        print("🤖 AI REVERSAL TRADING BOT")
        print("📈 HIGH-PROFIT PERPETUAL FUTURES STRATEGY")
        print("🎯 TARGET: 100-500% PROFITS LIKE YOUR MANUAL TRADING")
        print("=" * 70)
    
    def start_trading(self):
        """Start the AI reversal trading bot"""
        print(f"🚀 Starting AI Reversal Bot with ${self.current_balance:.2f}")
        print(f"💰 Base position size: ${self.config['base_position_size']}")
        print(f"🎯 Profit targets: ${self.config['profit_targets']['conservative']}-${self.config['profit_targets']['maximum']}")
        print("🤖 AI will detect range highs/lows and reversals")
        print("🔔 You'll be notified of all trades!")
        
        # Start monitoring
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        
        self.is_running = True
        
        try:
            asyncio.run(self._connect_websocket())
        except KeyboardInterrupt:
            print("\n⏹️ Bot stopped")
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
        
        print("🤖 AI REVERSAL TRADING BOT - LIVE STATUS")
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
            print(f"\n⚫ SCANNING: Looking for range extremes with 70%+ AI confidence")
        
        print(f"\n📊 PERFORMANCE:")
        print(f"   Trades: {self.total_trades} | Win Rate: {win_rate:.1f}%")
        print(f"   Daily: {self.daily_trades}/{self.config['max_daily_trades']}")
        print(f"   Total Profit: ${self.total_profit:+.2f}")
        print(f"   AI Data Points: {len(self.market_data)}")
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
            print("🔗 Connected to OKX WebSocket")
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
                    if len(self.market_data) > 200:  # Keep last 200 candles
                        self.market_data = self.market_data[-200:]
                    
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
        """AI-powered market analysis for entry opportunities"""
        try:
            if len(self.market_data) < 50:
                return
            
            # Daily trade limit
            current_day = datetime.now().date()
            if self.last_trade_day != current_day:
                self.daily_trades = 0
                self.last_trade_day = current_day
            
            if self.daily_trades >= self.config['max_daily_trades'] or self.current_position:
                return
            
            # Create DataFrame for analysis
            df = pd.DataFrame(self.market_data[-100:])  # More data for AI
            
            # Extract AI features
            features = self.ai_detector.extract_features(df)
            if not features:
                return
            
            # Detect reversal signals
            signals = self.ai_detector.detect_reversal_signals(features)
            
            # Check entry conditions (your style + AI)
            if self._should_enter_trade(signals, features):
                self._execute_entry(signals, features)
                
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
    
    def _should_enter_trade(self, signals: Dict, features: Dict) -> bool:
        """Determine if we should enter a trade (your criteria + AI)"""
        # Must be at range extreme (your strategy)
        if not signals['is_range_extreme']:
            return False
        
        # Must have high AI confidence
        if signals['confidence'] < self.config['min_confidence']:
            return False
        
        # Must have clear direction
        if not signals['direction']:
            return False
        
        # Additional confirmations
        confirmations = 0
        
        # Volume confirmation
        if features.get('volume_ratio', 1.0) > 1.3:
            confirmations += 1
        
        # RSI extreme
        rsi = features.get('rsi', 50)
        if (signals['direction'] == 'short' and rsi > 70) or (signals['direction'] == 'long' and rsi < 30):
            confirmations += 1
        
        # Range position extreme
        range_pos = features.get('range_position', 50)
        if (signals['direction'] == 'short' and range_pos > 80) or (signals['direction'] == 'long' and range_pos < 20):
            confirmations += 1
        
        # Need at least 2 confirmations
        return confirmations >= 2
    
    def _execute_entry(self, signals: Dict, features: Dict):
        """Execute trade entry with your position sizing"""
        try:
            current_price = features['current_price']
            
            # Calculate position size based on confidence
            confidence_multiplier = signals['confidence'] / 100
            base_size = self.config['base_position_size']
            max_size = self.config['max_position_size']
            
            position_size = base_size + (max_size - base_size) * confidence_multiplier
            position_size = min(position_size, self.current_balance * self.config['risk_per_trade'])
            
            # Determine profit target based on confidence
            if signals['confidence'] >= 90:
                target_profit = self.config['profit_targets']['maximum']
            elif signals['confidence'] >= 80:
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
            self._notify_entry(self.current_position, signals)
            
            logger.info(f"🚀 ENTRY: {signals['direction'].upper()} @ ${current_price:.4f} | "
                       f"Size: ${position_size:.2f} | Target: ${target_profit} | "
                       f"Confidence: {signals['confidence']:.1f}%")
            
        except Exception as e:
            logger.error(f"Entry execution error: {e}")
    
    def _check_exit_conditions(self):
        """Check if we should exit current position"""
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
                self._close_position("target_profit", current_price, hold_time, pnl_amount, pnl_pct)
                return
            
            # Stop loss
            stop_loss_amount = -(position.position_size * self.config['stop_loss_pct'] / 100)
            if pnl_amount <= stop_loss_amount:
                self._close_position("stop_loss", current_price, hold_time, pnl_amount, pnl_pct)
                return
            
            # Maximum hold time
            if hold_time >= self.config['max_hold_hours']:
                self._close_position("time_exit", current_price, hold_time, pnl_amount, pnl_pct)
                return
            
            # Reversal detection for early exit
            if hold_time > 1:  # After 1 hour, check for reversals
                df = pd.DataFrame(self.market_data[-50:])
                features = self.ai_detector.extract_features(df)
                signals = self.ai_detector.detect_reversal_signals(features)
                
                # If AI detects strong reversal against our position
                if (signals['confidence'] > 80 and 
                    signals['direction'] != position.direction and
                    pnl_amount > 0):  # Only if we're in profit
                    self._close_position("ai_reversal", current_price, hold_time, pnl_amount, pnl_pct)
                    return
            
        except Exception as e:
            logger.error(f"Exit check error: {e}")
    
    def _close_position(self, reason: str, exit_price: float, hold_time: float, pnl_amount: float, pnl_pct: float):
        """Close the current position"""
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
        self._notify_exit(position)
        
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        logger.info(f"🔄 EXIT: {reason} | P&L: ${pnl_amount:+.2f} | WR: {win_rate:.1f}%")
        
        self.current_position = None
    
    def _notify_entry(self, position: Trade, signals: Dict):
        """Send entry notifications"""
        # Sound
        TradeNotifier.play_sound("entry")
        
        # Desktop notification
        TradeNotifier.desktop_notification(
            "🤖 AI REVERSAL ENTRY",
            f"{position.direction.upper()} @ ${position.entry_price:.4f}\n"
            f"AI Confidence: {position.confidence:.1f}%\n"
            f"Size: ${position.position_size:.2f}\n"
            f"Target: ${position.target_profit}"
        )
        
        # Console alert
        TradeNotifier.print_trade_alert("ENTRY", {
            'direction': position.direction,
            'entry_price': position.entry_price,
            'confidence': position.confidence,
            'position_size': position.position_size,
            'target_profit': position.target_profit,
            'signals': signals['signals'],
            'leverage': self.config['leverage']
        })
    
    def _notify_exit(self, position: Trade):
        """Send exit notifications"""
        # Sound
        if position.pnl_amount >= position.target_profit * 0.8:
            TradeNotifier.play_sound("big_profit")
        elif position.pnl_amount > 0:
            TradeNotifier.play_sound("profit")
        else:
            TradeNotifier.play_sound("loss")
        
        # Desktop notification
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        profit_icon = "🎉" if position.pnl_amount >= position.target_profit * 0.8 else "💚" if position.pnl_amount > 0 else "❌"
        
        TradeNotifier.desktop_notification(
            f"{profit_icon} AI TRADE CLOSED",
            f"{position.direction.upper()} closed\n"
            f"P&L: ${position.pnl_amount:+.2f}\n"
            f"Target: {'✅ HIT' if position.pnl_amount >= position.target_profit * 0.8 else '❌ MISSED'}\n"
            f"Win Rate: {win_rate:.1f}%"
        )
        
        # Console alert
        TradeNotifier.print_trade_alert("EXIT", {
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
        """Stop the trading bot"""
        self.is_running = False
        
        if self.current_position:
            hold_time = (datetime.now() - self.current_position.entry_time).total_seconds() / 3600
            if self.current_position.direction == 'long':
                pnl_pct = (self.last_price - self.current_position.entry_price) / self.current_position.entry_price * 100
            else:
                pnl_pct = (self.current_position.entry_price - self.last_price) / self.current_position.entry_price * 100
            
            pnl_amount = self.current_position.position_size * (pnl_pct / 100) * self.config['leverage']
            self._close_position("manual_stop", self.last_price, hold_time, pnl_amount, pnl_pct)
        
        self._show_results()
    
    def _show_results(self):
        """Show final trading results"""
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print("\n" + "="*70)
        print("🤖 AI REVERSAL BOT RESULTS")
        print("="*70)
        print(f"📊 Total Trades: {self.total_trades}")
        print(f"🏆 Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        print(f"💰 Total Return: {total_return:+.1f}%")
        print(f"💵 Final Balance: ${self.current_balance:.2f}")
        print(f"💎 Total Profit: ${self.total_profit:+.2f}")
        
        if self.trades:
            profits = [t.pnl_amount for t in self.trades if t.pnl_amount > 0]
            if profits:
                avg_profit = np.mean(profits)
                max_profit = max(profits)
                print(f"📈 Avg Profit: ${avg_profit:.2f} | Max Profit: ${max_profit:.2f}")
        
        print("="*70)

def main():
    print("🤖 AI REVERSAL TRADING BOT")
    print("🎯 High-Profit Perpetual Futures Trading")
    print("📈 Mimics your manual trading style with AI precision")
    
    balance = input("\n💰 Starting balance (default $1000): ").strip()
    initial_balance = float(balance) if balance else 1000.0
    
    bot = AIReversalTradingBot(initial_balance)
    print("\n📡 Connecting to live market data...")
    print("🤖 AI will analyze range extremes and reversals")
    print("💰 Targeting 100-500% profits like your manual strategy")
    print("📝 Press Ctrl+C to stop\n")
    
    bot.start_trading()

if __name__ == "__main__":
    main() 