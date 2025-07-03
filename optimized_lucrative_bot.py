#!/usr/bin/env python3
"""
Optimized Lucrative $40 Profit Bot
Based on backtesting results with AI-optimized parameters
Features 20-30x leverage, trailing stops, and maximum profitability
"""

import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import requests
warnings.filterwarnings('ignore')

# Cross-platform notifications
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False

try:
    from plyer import notification
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

class Trade:
    def __init__(self, direction, entry_price, position_size, target_profit, confidence, leverage):
        self.direction = direction
        self.entry_price = entry_price
        self.position_size = position_size
        self.target_profit = target_profit
        self.confidence = confidence
        self.leverage = leverage
        self.entry_time = datetime.now()
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl_amount = None
        self.pnl_percentage = None
        self.hold_time = None
        self.best_price = entry_price
        self.trailing_stop_price = None
        self.trailing_activated = False

class OptimizedLucrativeBot:
    """Optimized high-performance bot with 20-30x leverage and trailing stops"""
    
    def __init__(self, initial_balance: float = 500.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # BACKTESTING-OPTIMIZED CONFIGURATION
        self.config = {
            "symbol": "SOL-USDT-SWAP",
            "target_profit": 40.0,           # Target $40 per trade
            
            # AI-OPTIMIZED LEVERAGE SELECTION
            "leverage_min": 20,              # Minimum leverage
            "leverage_max": 30,              # Maximum leverage
            "adaptive_leverage": True,       # Let AI choose optimal leverage
            
            # POSITION SIZING
            "base_position_size": 150.0,     # Base position size
            "max_position_size": 500.0,      # Maximum position size
            "position_size_pct": 25,         # 25% of balance max
            
            # PROFIT TARGETS AND STOPS
            "profit_target_pct": 2.8,        # Optimized profit target
            "trailing_stop_pct": 0.9,        # Optimized trailing stop
            "trailing_activation": 0.6,      # When to activate trailing
            "emergency_stop_pct": 2.5,       # Emergency stop loss
            
            # AI CONFIDENCE AND ENTRY
            "min_confidence": 65,            # Lowered for more trades
            "confidence_boost_threshold": 80, # High confidence boost
            "max_daily_trades": 12,          # Increased trade frequency
            "max_hold_hours": 6,             # Optimized hold time
            
            # TECHNICAL ANALYSIS
            "rsi_oversold": 25,              # RSI oversold level
            "rsi_overbought": 75,            # RSI overbought level
            "volume_threshold": 1.4,         # Volume confirmation
            "momentum_threshold": 0.8,       # Momentum strength
            
            # RISK MANAGEMENT
            "risk_per_trade": 0.12,          # 12% max risk per trade
            "max_consecutive_losses": 3,     # Stop after losses
            "daily_profit_target": 200.0,    # Daily profit target
        }
        
        # Trading state
        self.is_running = False
        self.current_position = None
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.daily_profit = 0
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
        
        print("🚀 OPTIMIZED LUCRATIVE $40 PROFIT BOT")
        print("⚡ 20-30X ADAPTIVE LEVERAGE WITH TRAILING STOPS")
        print("🎯 AI-OPTIMIZED FOR MAXIMUM PROFITABILITY")
        print("=" * 70)
        print("🔧 OPTIMIZED CONFIGURATION:")
        print(f"   💎 Target Profit: ${self.config['target_profit']}")
        print(f"   ⚡ Leverage Range: {self.config['leverage_min']}-{self.config['leverage_max']}x")
        print(f"   💰 Position Size: ${self.config['base_position_size']}-${self.config['max_position_size']}")
        print(f"   🛡️ Trailing Stop: {self.config['trailing_stop_pct']}%")
        print(f"   🎯 Profit Target: {self.config['profit_target_pct']}%")
        print(f"   🤖 Min Confidence: {self.config['min_confidence']}%")
        print(f"   📊 Max Daily Trades: {self.config['max_daily_trades']}")
        print("=" * 70)
    
    def _fetch_market_data(self):
        """Fetch market data from OKX REST API"""
        try:
            # Get current ticker
            ticker_url = "https://www.okx.com/api/v5/market/ticker?instId=SOL-USDT-SWAP"
            ticker_response = requests.get(ticker_url, timeout=10)
            
            if ticker_response.status_code == 200:
                ticker_data = ticker_response.json()
                if ticker_data.get('data'):
                    self.last_price = float(ticker_data['data'][0]['last'])
                    self.last_update = datetime.now()
                    
                    # Get historical candles
                    candles_url = "https://www.okx.com/api/v5/market/candles?instId=SOL-USDT-SWAP&bar=1m&limit=100"
                    candles_response = requests.get(candles_url, timeout=10)
                    
                    if candles_response.status_code == 200:
                        candles_data = candles_response.json()
                        if candles_data.get('data'):
                            self.market_data = []
                            for candle in candles_data['data']:
                                candle_info = {
                                    'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000),
                                    'open': float(candle[1]),
                                    'high': float(candle[2]),
                                    'low': float(candle[3]),
                                    'close': float(candle[4]),
                                    'volume': float(candle[5])
                                }
                                self.market_data.append(candle_info)
                            
                            # Sort by timestamp
                            self.market_data.sort(key=lambda x: x['timestamp'])
                            return True
            
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return False
        
        return False
    
    def _calculate_advanced_indicators(self) -> dict:
        """Calculate advanced technical indicators"""
        if len(self.market_data) < 50:
            return {"confidence": 0, "direction": "hold", "reason": "insufficient_data"}
        
        df = pd.DataFrame(self.market_data)
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Moving averages
        ma_5 = df['close'].rolling(5).mean().iloc[-1]
        ma_20 = df['close'].rolling(20).mean().iloc[-1]
        ma_50 = df['close'].rolling(50).mean().iloc[-1]
        
        # Volume analysis
        volume_ma = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / volume_ma
        
        # Price momentum
        price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100
        price_change_20 = (df['close'].iloc[-1] - df['close'].iloc[-21]) / df['close'].iloc[-21] * 100
        
        # Bollinger Bands
        bb_std = df['close'].rolling(20).std().iloc[-1]
        bb_mean = df['close'].rolling(20).mean().iloc[-1]
        bb_upper = bb_mean + (bb_std * 2)
        bb_lower = bb_mean - (bb_std * 2)
        bb_position = (df['close'].iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
        ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
        macd = ema_12 - ema_26
        
        return {
            'rsi': current_rsi,
            'ma_5': ma_5,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'volume_ratio': volume_ratio,
            'price_change_5': price_change_5,
            'price_change_20': price_change_20,
            'bb_position': bb_position,
            'macd': macd,
            'current_price': df['close'].iloc[-1],
            'high': df['high'].iloc[-1],
            'low': df['low'].iloc[-1]
        }
    
    def _ai_leverage_selection(self, confidence: float, market_volatility: float) -> int:
        """AI-driven leverage selection based on confidence and market conditions"""
        if not self.config['adaptive_leverage']:
            return self.config['leverage_min']
        
        base_leverage = self.config['leverage_min']
        max_leverage = self.config['leverage_max']
        
        # Confidence-based leverage scaling
        confidence_multiplier = confidence / 100
        
        # Volatility adjustment (higher volatility = lower leverage)
        volatility_adjustment = max(0.5, 1 - market_volatility)
        
        # Win rate adjustment
        win_rate = (self.wins / max(self.total_trades, 1)) * 100
        win_rate_multiplier = min(1.5, win_rate / 70) if win_rate > 0 else 0.8
        
        # Calculate optimal leverage
        leverage_multiplier = confidence_multiplier * volatility_adjustment * win_rate_multiplier
        optimal_leverage = int(base_leverage + (max_leverage - base_leverage) * leverage_multiplier)
        
        return max(base_leverage, min(max_leverage, optimal_leverage))
    
    def _analyze_market_opportunity(self) -> dict:
        """Advanced AI analysis for profitable opportunities"""
        indicators = self._calculate_advanced_indicators()
        
        if indicators.get("confidence") == 0:
            return indicators
        
        confidence = 0
        direction = "hold"
        reasons = []
        
        # RSI signals (30% weight)
        if indicators['rsi'] < self.config['rsi_oversold']:
            confidence += 30
            direction = "long"
            reasons.append("strong_oversold")
        elif indicators['rsi'] < 35:
            confidence += 20
            direction = "long"
            reasons.append("oversold")
        elif indicators['rsi'] > self.config['rsi_overbought']:
            confidence += 30
            direction = "short"
            reasons.append("strong_overbought")
        elif indicators['rsi'] > 65:
            confidence += 20
            direction = "short"
            reasons.append("overbought")
        
        # Moving average confluence (25% weight)
        current_price = indicators['current_price']
        if current_price > indicators['ma_5'] > indicators['ma_20'] > indicators['ma_50']:
            if direction == "long" or direction == "hold":
                confidence += 25
                direction = "long"
                reasons.append("bullish_ma_stack")
        elif current_price < indicators['ma_5'] < indicators['ma_20'] < indicators['ma_50']:
            if direction == "short" or direction == "hold":
                confidence += 25
                direction = "short"
                reasons.append("bearish_ma_stack")
        
        # Volume confirmation (20% weight)
        if indicators['volume_ratio'] > self.config['volume_threshold']:
            confidence += 20
            reasons.append("volume_breakout")
        elif indicators['volume_ratio'] > self.config['volume_threshold'] * 0.8:
            confidence += 10
            reasons.append("volume_support")
        
        # Momentum analysis (15% weight)
        if abs(indicators['price_change_5']) > self.config['momentum_threshold']:
            if direction == "long" and indicators['price_change_5'] > 0:
                confidence += 15
                reasons.append("positive_momentum")
            elif direction == "short" and indicators['price_change_5'] < 0:
                confidence += 15
                reasons.append("negative_momentum")
        
        # Bollinger Band signals (10% weight)
        if indicators['bb_position'] < 0.1 and direction == "long":
            confidence += 10
            reasons.append("bb_oversold")
        elif indicators['bb_position'] > 0.9 and direction == "short":
            confidence += 10
            reasons.append("bb_overbought")
        
        # MACD confirmation
        if direction == "long" and indicators['macd'] > 0:
            confidence += 5
            reasons.append("macd_bullish")
        elif direction == "short" and indicators['macd'] < 0:
            confidence += 5
            reasons.append("macd_bearish")
        
        # Market volatility calculation
        volatility = abs(indicators['price_change_5']) / 100
        
        # High confidence bonus
        if confidence >= self.config['confidence_boost_threshold']:
            confidence += 10
            reasons.append("high_confidence_boost")
        
        return {
            "confidence": min(confidence, 95),
            "direction": direction,
            "reasons": reasons,
            "indicators": indicators,
            "volatility": volatility
        }
    
    def _check_entry_conditions(self) -> bool:
        """Check if we should enter a trade"""
        # Reset daily counters
        today = datetime.now().date()
        if self.last_trade_day != today:
            self.daily_trades = 0
            self.daily_profit = 0
            self.last_trade_day = today
        
        # Basic checks
        if self.current_position is not None:
            return False
        
        if self.daily_trades >= self.config['max_daily_trades']:
            return False
        
        if self.consecutive_losses >= self.config['max_consecutive_losses']:
            return False
        
        if self.daily_profit >= self.config['daily_profit_target']:
            return False
        
        if len(self.market_data) < 50:
            return False
        
        # Analyze market
        analysis = self._analyze_market_opportunity()
        
        if analysis['confidence'] < self.config['min_confidence']:
            return False
        
        if analysis['direction'] == "hold":
            return False
        
        # AI leverage selection
        leverage = self._ai_leverage_selection(analysis['confidence'], analysis['volatility'])
        
        # Calculate position size
        base_size = self.config['base_position_size']
        max_size = min(self.config['max_position_size'], 
                      self.current_balance * self.config['position_size_pct'] / 100)
        
        confidence_ratio = analysis['confidence'] / 100
        position_size = base_size + (max_size - base_size) * confidence_ratio
        
        # Risk management
        max_risk = self.current_balance * self.config['risk_per_trade']
        risk_amount = position_size * self.config['trailing_stop_pct'] / 100 * leverage
        
        if risk_amount > max_risk:
            position_size = max_risk / (self.config['trailing_stop_pct'] / 100 * leverage)
        
        # Create trade
        self.current_position = Trade(
            direction=analysis['direction'],
            entry_price=self.last_price,
            position_size=position_size,
            target_profit=self.config['target_profit'],
            confidence=analysis['confidence'],
            leverage=leverage
        )
        
        self.total_trades += 1
        self.daily_trades += 1
        
        # Send entry notification
        self._notify_entry(self.current_position, analysis)
        
        print(f"\n🚀 OPTIMIZED ENTRY: {analysis['direction'].upper()} @ ${self.last_price:.4f}")
        print(f"   💰 Size: ${position_size:.2f} | Target: ${self.config['target_profit']}")
        print(f"   ⚡ Leverage: {leverage}x | Confidence: {analysis['confidence']:.1f}%")
        print(f"   🛡️ Trailing Stop: {self.config['trailing_stop_pct']}%")
        print(f"   📊 Reasons: {', '.join(analysis['reasons'])}")
        
        return True
    
    def _update_trailing_stop(self):
        """Update trailing stop loss"""
        if not self.current_position:
            return
        
        current_price = self.last_price
        direction = self.current_position.direction
        
        # Update best price
        if direction == 'long' and current_price > self.current_position.best_price:
            self.current_position.best_price = current_price
        elif direction == 'short' and current_price < self.current_position.best_price:
            self.current_position.best_price = current_price
        
        # Check if we should activate trailing stop
        if not self.current_position.trailing_activated:
            entry_price = self.current_position.entry_price
            
            if direction == 'long':
                profit_pct = (self.current_position.best_price - entry_price) / entry_price * 100
            else:
                profit_pct = (entry_price - self.current_position.best_price) / entry_price * 100
            
            if profit_pct >= self.config['trailing_activation']:
                self.current_position.trailing_activated = True
                print(f"🛡️ Trailing stop ACTIVATED at {profit_pct:.2f}% profit")
        
        # Update trailing stop price
        if self.current_position.trailing_activated:
            if direction == 'long':
                new_stop = self.current_position.best_price * (1 - self.config['trailing_stop_pct'] / 100)
                if self.current_position.trailing_stop_price is None or new_stop > self.current_position.trailing_stop_price:
                    self.current_position.trailing_stop_price = new_stop
            else:
                new_stop = self.current_position.best_price * (1 + self.config['trailing_stop_pct'] / 100)
                if self.current_position.trailing_stop_price is None or new_stop < self.current_position.trailing_stop_price:
                    self.current_position.trailing_stop_price = new_stop
    
    def _check_exit_conditions(self):
        """Check if we should exit current position"""
        if not self.current_position:
            return
        
        entry_price = self.current_position.entry_price
        current_price = self.last_price
        direction = self.current_position.direction
        leverage = self.current_position.leverage
        
        # Update trailing stop
        self._update_trailing_stop()
        
        # Calculate current P&L
        if direction == 'long':
            price_change_pct = (current_price - entry_price) / entry_price * 100
        else:
            price_change_pct = (entry_price - current_price) / entry_price * 100
        
        pnl_pct = price_change_pct * leverage
        pnl_amount = self.current_position.position_size * (price_change_pct / 100)
        
        # Exit conditions
        should_exit = False
        exit_reason = None
        
        # Take profit target
        target_profit_pct = self.config['profit_target_pct']
        if pnl_amount >= self.config['target_profit'] * 0.9:  # 90% of target
            should_exit = True
            exit_reason = "take_profit"
        
        # Trailing stop
        elif self.current_position.trailing_activated and self.current_position.trailing_stop_price:
            if direction == 'long' and current_price <= self.current_position.trailing_stop_price:
                should_exit = True
                exit_reason = "trailing_stop"
            elif direction == 'short' and current_price >= self.current_position.trailing_stop_price:
                should_exit = True
                exit_reason = "trailing_stop"
        
        # Emergency stop loss
        elif pnl_pct <= -self.config['emergency_stop_pct']:
            should_exit = True
            exit_reason = "emergency_stop"
        
        # Time exit
        elif (datetime.now() - self.current_position.entry_time).total_seconds() > self.config['max_hold_hours'] * 3600:
            should_exit = True
            exit_reason = "time_exit"
        
        # Smart reversal exit
        elif pnl_amount > self.config['target_profit'] * 0.4:  # At least 40% of target
            analysis = self._analyze_market_opportunity()
            if analysis['direction'] != direction and analysis['confidence'] > 75:
                should_exit = True
                exit_reason = "reversal_signal"
        
        if should_exit:
            self._close_position(exit_reason, current_price, pnl_amount, pnl_pct)
    
    def _close_position(self, reason: str, exit_price: float, pnl_amount: float, pnl_pct: float):
        """Close the current position"""
        if not self.current_position:
            return
        
        position = self.current_position
        
        # Update position details
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.exit_reason = reason
        position.pnl_amount = pnl_amount
        position.pnl_percentage = pnl_pct
        position.hold_time = (position.exit_time - position.entry_time).total_seconds() / 60
        
        # Update balance and stats
        self.current_balance += pnl_amount
        self.total_profit += pnl_amount
        self.daily_profit += pnl_amount
        
        if pnl_amount > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
        
        self.trades.append(position)
        
        # Send exit notification
        self._notify_exit(position)
        
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print(f"\n📤 OPTIMIZED EXIT: {reason.upper()} @ ${exit_price:.4f}")
        print(f"   💰 P&L: ${pnl_amount:+.2f} ({pnl_pct:+.2f}%)")
        print(f"   ⚡ Leverage: {position.leverage}x")
        print(f"   🛡️ Trailing: {'✅' if position.trailing_activated else '❌'}")
        print(f"   ⏱️ Hold: {position.hold_time:.1f} min")
        print(f"   🏆 Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        print(f"   💵 Balance: ${self.current_balance:.2f}")
        print(f"   📊 Daily Profit: ${self.daily_profit:+.2f}")
        
        self.current_position = None
    
    def _notify_entry(self, position: Trade, analysis: dict):
        """Send entry notifications"""
        if SOUND_AVAILABLE:
            try:
                winsound.Beep(1200, 400)  # Entry sound
            except:
                pass
        
        if NOTIFICATIONS_AVAILABLE:
            try:
                notification.notify(
                    title="🚀 OPTIMIZED BOT ENTRY",
                    message=f"{position.direction.upper()} @ ${position.entry_price:.4f}\n"
                            f"Leverage: {position.leverage}x\n"
                            f"Target: ${position.target_profit}\n"
                            f"Confidence: {position.confidence:.1f}%",
                    timeout=6
                )
            except:
                pass
    
    def _notify_exit(self, position: Trade):
        """Send exit notifications"""
        if SOUND_AVAILABLE:
            try:
                if position.pnl_amount >= position.target_profit * 0.8:
                    # Big profit - celebration sound
                    for freq in [800, 1000, 1200]:
                        winsound.Beep(freq, 200)
                elif position.pnl_amount > 0:
                    winsound.Beep(1400, 400)  # Profit
                else:
                    winsound.Beep(300, 600)   # Loss
            except:
                pass
        
        if NOTIFICATIONS_AVAILABLE:
            try:
                profit_icon = "🎉" if position.pnl_amount >= position.target_profit * 0.8 else "💚" if position.pnl_amount > 0 else "❌"
                win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
                
                notification.notify(
                    title=f"{profit_icon} OPTIMIZED BOT EXIT",
                    message=f"P&L: ${position.pnl_amount:+.2f}\n"
                            f"Leverage: {position.leverage}x\n"
                            f"Reason: {position.exit_reason}\n"
                            f"Win Rate: {win_rate:.1f}%",
                    timeout=8
                )
            except:
                pass
    
    def _display_status(self):
        """Display current trading status"""
        now = datetime.now()
        data_age = (now - self.last_update).total_seconds() if self.last_update else 999
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"🚀 OPTIMIZED LUCRATIVE BOT - {now.strftime('%H:%M:%S')}")
        print(f"{'='*70}")
        print(f"💵 SOL Price: ${self.last_price:.4f} (Age: {data_age:.1f}s)")
        print(f"💰 Balance: ${self.current_balance:.2f} | Daily P&L: ${self.daily_profit:+.2f}")
        
        if self.current_position:
            position = self.current_position
            entry_price = position.entry_price
            direction = position.direction
            
            if direction == 'long':
                unrealized_pct = (self.last_price - entry_price) / entry_price * 100
            else:
                unrealized_pct = (entry_price - self.last_price) / entry_price * 100
            
            unrealized_amount = position.position_size * (unrealized_pct / 100)
            hold_time = (now - position.entry_time).total_seconds() / 60
            target_progress = (unrealized_amount / self.config['target_profit']) * 100
            
            print(f"\n🔵 ACTIVE POSITION:")
            print(f"   📊 {direction.upper()} @ ${entry_price:.4f} | Size: ${position.position_size:.2f}")
            print(f"   ⚡ Leverage: {position.leverage}x | Confidence: {position.confidence:.1f}%")
            print(f"   💰 Unrealized P&L: ${unrealized_amount:+.2f} ({unrealized_pct:+.2f}%)")
            print(f"   🎯 Target Progress: {target_progress:.1f}%")
            print(f"   🛡️ Trailing: {'✅ ACTIVE' if position.trailing_activated else '⏸️ WAITING'}")
            if position.trailing_stop_price:
                print(f"   🛑 Stop Price: ${position.trailing_stop_price:.4f}")
            print(f"   ⏱️ Hold Time: {hold_time:.1f} min")
        else:
            print(f"\n⚫ SCANNING: Looking for high-confidence opportunities")
            print(f"   🎯 Target: ${self.config['target_profit']} profit per trade")
            print(f"   ⚡ Leverage: {self.config['leverage_min']}-{self.config['leverage_max']}x adaptive")
        
        print(f"\n📊 PERFORMANCE:")
        print(f"   🏆 Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        print(f"   📈 Total Profit: ${self.total_profit:+.2f}")
        print(f"   📊 Daily Trades: {self.daily_trades}/{self.config['max_daily_trades']}")
        print(f"   🔗 Market Data: {len(self.market_data)} candles")
        print(f"   🔥 Consecutive Losses: {self.consecutive_losses}/{self.config['max_consecutive_losses']}")
    
    def start_trading(self):
        """Start the optimized trading bot"""
        print("🚀 Starting Optimized Lucrative Bot...")
        print("🔗 Connecting to OKX market data...")
        
        self.is_running = True
        last_data_fetch = 0
        last_status_display = 0
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Fetch market data every 20 seconds
                if current_time - last_data_fetch > 20:
                    if self._fetch_market_data():
                        print(f"📡 Market data updated - SOL: ${self.last_price:.4f}")
                    last_data_fetch = current_time
                
                if self.last_price > 0:
                    # Check for entry opportunities
                    if not self.current_position:
                        self._check_entry_conditions()
                    
                    # Check for exit conditions
                    if self.current_position:
                        self._check_exit_conditions()
                    
                    # Display status every 45 seconds
                    if current_time - last_status_display > 45:
                        self._display_status()
                        last_status_display = current_time
                
                time.sleep(3)  # Check every 3 seconds
                
        except KeyboardInterrupt:
            print("\n⚠️ Bot stopped by user")
            self.is_running = False
        
        finally:
            self._show_final_results()
    
    def _show_final_results(self):
        """Show final trading results"""
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print("\n" + "="*80)
        print("🚀 OPTIMIZED LUCRATIVE BOT - FINAL RESULTS")
        print("="*80)
        print(f"📊 Total Trades: {self.total_trades}")
        print(f"🏆 Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        print(f"💰 Total Return: {total_return:+.1f}%")
        print(f"💵 Final Balance: ${self.current_balance:.2f}")
        print(f"💎 Total Profit: ${self.total_profit:+.2f}")
        
        if self.trades:
            profitable_trades = [t for t in self.trades if t.pnl_amount > 0]
            if profitable_trades:
                avg_profit = np.mean([t.pnl_amount for t in profitable_trades])
                max_profit = max([t.pnl_amount for t in profitable_trades])
                target_hits = len([t for t in profitable_trades if t.pnl_amount >= self.config['target_profit'] * 0.9])
                
                print(f"📈 Avg Profit: ${avg_profit:.2f} | Max Profit: ${max_profit:.2f}")
                print(f"🎯 Target Hits: {target_hits}/{len(profitable_trades)} ({target_hits/len(profitable_trades)*100:.1f}%)")
            
            # Leverage analysis
            leverage_used = [t.leverage for t in self.trades]
            if leverage_used:
                avg_leverage = np.mean(leverage_used)
                print(f"⚡ Average Leverage Used: {avg_leverage:.1f}x")
            
            # Trailing stop analysis
            trailing_activations = len([t for t in self.trades if t.trailing_activated])
            if trailing_activations > 0:
                print(f"🛡️ Trailing Stop Activations: {trailing_activations}/{self.total_trades} ({trailing_activations/self.total_trades*100:.1f}%)")
        
        print("="*80)

def main():
    """Main function"""
    print("🚀 OPTIMIZED LUCRATIVE $40 PROFIT BOT")
    print("⚡ AI-OPTIMIZED 20-30X LEVERAGE WITH TRAILING STOPS")
    print("=" * 60)
    
    try:
        balance = float(input("💵 Enter starting balance (default $500): ") or "500")
    except ValueError:
        balance = 500.0
    
    bot = OptimizedLucrativeBot(initial_balance=balance)
    
    try:
        bot.start_trading()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped. Thanks for using Optimized Lucrative Bot!")

if __name__ == "__main__":
    main()