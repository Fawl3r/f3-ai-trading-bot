#!/usr/bin/env python3
"""
Lucrative $40 Profit Bot
Single high-performing bot focused on consistent $40 profits
Targets 70%+ win rate with minimal losses
"""

import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import sys
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
    def __init__(self, direction, entry_price, position_size, target_profit, confidence):
        self.direction = direction
        self.entry_price = entry_price
        self.position_size = position_size
        self.target_profit = target_profit
        self.confidence = confidence
        self.entry_time = datetime.now()
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl_amount = None
        self.pnl_percentage = None
        self.hold_time = None

class Lucrative40ProfitBot:
    """High-performance bot targeting consistent $40 profits"""
    
    def __init__(self, initial_balance: float = 500.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # OPTIMIZED $40 PROFIT CONFIGURATION
        self.config = {
            "symbol": "SOL-USDT-SWAP",
            "target_profit": 40.0,           # Target $40 per trade
            "base_position_size": 200.0,     # $200 base position for $40 profit
            "max_position_size": 400.0,      # $400 max for high confidence
            "leverage": 10,                  # 10x leverage for efficient capital use
            "profit_target_pct": 2.0,        # 2% profit target (adjusted by leverage)
            "stop_loss_pct": 1.0,            # 1% stop loss
            "min_confidence": 75,            # 75% minimum AI confidence
            "max_daily_trades": 8,           # Quality over quantity
            "max_hold_hours": 4,             # 4 hour maximum hold
            "risk_per_trade": 0.08,          # 8% of balance maximum risk
        }
        
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
        
        print("💰 LUCRATIVE $40 PROFIT BOT")
        print("🎯 TARGET: $40 PROFIT PER TRADE")
        print("🏆 GOAL: 70%+ WIN RATE")
        print("=" * 60)
        print("🔧 BOT CONFIGURATION:")
        print(f"   💎 Target Profit: ${self.config['target_profit']}")
        print(f"   💵 Position Size: ${self.config['base_position_size']}-${self.config['max_position_size']}")
        print(f"   ⚡ Leverage: {self.config['leverage']}x")
        print(f"   🎯 Profit Target: {self.config['profit_target_pct']}%")
        print(f"   🛑 Stop Loss: {self.config['stop_loss_pct']}%")
        print(f"   🤖 Min Confidence: {self.config['min_confidence']}%")
        print(f"   📊 Max Daily Trades: {self.config['max_daily_trades']}")
        print("=" * 60)
    
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
    
    def _calculate_technical_indicators(self) -> dict:
        """Calculate technical indicators for AI analysis"""
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
        
        return {
            'rsi': current_rsi,
            'ma_5': ma_5,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'volume_ratio': volume_ratio,
            'price_change_5': price_change_5,
            'price_change_20': price_change_20,
            'current_price': df['close'].iloc[-1]
        }
    
    def _analyze_market_opportunity(self) -> dict:
        """Advanced AI analysis for $40 profit opportunities"""
        indicators = self._calculate_technical_indicators()
        
        if indicators.get("confidence") == 0:
            return indicators
        
        confidence = 0
        direction = "hold"
        reasons = []
        
        # RSI-based signals (30% weight)
        if indicators['rsi'] < 30:  # Oversold
            confidence += 25
            direction = "long"
            reasons.append("oversold_rsi")
        elif indicators['rsi'] > 70:  # Overbought
            confidence += 25
            direction = "short"
            reasons.append("overbought_rsi")
        elif 45 <= indicators['rsi'] <= 55:  # Neutral momentum
            confidence += 15
            reasons.append("neutral_momentum")
        
        # Moving average analysis (25% weight)
        current_price = indicators['current_price']
        if current_price > indicators['ma_5'] > indicators['ma_20'] > indicators['ma_50']:
            if direction == "long" or direction == "hold":
                confidence += 20
                direction = "long"
                reasons.append("bullish_ma_alignment")
        elif current_price < indicators['ma_5'] < indicators['ma_20'] < indicators['ma_50']:
            if direction == "short" or direction == "hold":
                confidence += 20
                direction = "short"
                reasons.append("bearish_ma_alignment")
        
        # Volume confirmation (20% weight)
        if indicators['volume_ratio'] > 1.5:
            confidence += 15
            reasons.append("high_volume_confirmation")
        elif indicators['volume_ratio'] > 1.2:
            confidence += 10
            reasons.append("moderate_volume")
        
        # Momentum analysis (25% weight)
        if abs(indicators['price_change_5']) > 1.0:  # Strong recent momentum
            if direction == "long" and indicators['price_change_5'] > 0:
                confidence += 15
                reasons.append("positive_momentum")
            elif direction == "short" and indicators['price_change_5'] < 0:
                confidence += 15
                reasons.append("negative_momentum")
        
        # Trend consistency bonus
        if direction == "long" and indicators['price_change_20'] > 0:
            confidence += 10
            reasons.append("consistent_uptrend")
        elif direction == "short" and indicators['price_change_20'] < 0:
            confidence += 10
            reasons.append("consistent_downtrend")
        
        # Reduce confidence for choppy markets
        if abs(indicators['price_change_5']) < 0.3 and 40 < indicators['rsi'] < 60:
            confidence *= 0.7
            reasons.append("low_volatility_penalty")
        
        return {
            "confidence": min(confidence, 95),  # Cap at 95%
            "direction": direction,
            "reasons": reasons,
            "indicators": indicators
        }
    
    def _check_entry_conditions(self) -> bool:
        """Check if we should enter a trade"""
        # Reset daily trades at start of new day
        today = datetime.now().date()
        if self.last_trade_day != today:
            self.daily_trades = 0
            self.last_trade_day = today
        
        # Basic checks
        if self.current_position is not None:
            return False
        
        if self.daily_trades >= self.config['max_daily_trades']:
            return False
        
        if len(self.market_data) < 50:
            return False
        
        # Analyze market
        analysis = self._analyze_market_opportunity()
        
        if analysis['confidence'] < self.config['min_confidence']:
            return False
        
        if analysis['direction'] == "hold":
            return False
        
        # Calculate position size based on confidence
        base_size = self.config['base_position_size']
        max_size = self.config['max_position_size']
        confidence_ratio = analysis['confidence'] / 100
        position_size = base_size + (max_size - base_size) * confidence_ratio
        
        # Risk management
        max_risk = self.current_balance * self.config['risk_per_trade']
        risk_amount = position_size * self.config['stop_loss_pct'] / 100 * self.config['leverage']
        
        if risk_amount > max_risk:
            position_size = max_risk / (self.config['stop_loss_pct'] / 100 * self.config['leverage'])
        
        # Create trade
        self.current_position = Trade(
            direction=analysis['direction'],
            entry_price=self.last_price,
            position_size=position_size,
            target_profit=self.config['target_profit'],
            confidence=analysis['confidence']
        )
        
        self.total_trades += 1
        self.daily_trades += 1
        
        # Send entry notification
        self._notify_entry(self.current_position, analysis)
        
        print(f"\n🚀 ENTRY: {analysis['direction'].upper()} @ ${self.last_price:.4f}")
        print(f"   💰 Size: ${position_size:.2f} | Target: ${self.config['target_profit']}")
        print(f"   🤖 Confidence: {analysis['confidence']:.1f}%")
        print(f"   📊 Reasons: {', '.join(analysis['reasons'])}")
        
        return True
    
    def _check_exit_conditions(self):
        """Check if we should exit current position"""
        if not self.current_position:
            return
        
        entry_price = self.current_position.entry_price
        current_price = self.last_price
        direction = self.current_position.direction
        leverage = self.config['leverage']
        
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
        
        # Take profit
        if pnl_amount >= self.config['target_profit'] * 0.9:  # 90% of target
            should_exit = True
            exit_reason = "take_profit"
        
        # Stop loss
        elif pnl_pct <= -self.config['stop_loss_pct']:
            should_exit = True
            exit_reason = "stop_loss"
        
        # Time exit
        elif (datetime.now() - self.current_position.entry_time).total_seconds() > self.config['max_hold_hours'] * 3600:
            should_exit = True
            exit_reason = "time_exit"
        
        # Smart exit based on momentum reversal
        elif pnl_amount > self.config['target_profit'] * 0.5:  # At least 50% of target
            analysis = self._analyze_market_opportunity()
            if analysis['direction'] != direction and analysis['confidence'] > 70:
                should_exit = True
                exit_reason = "momentum_reversal"
        
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
        
        if pnl_amount > 0:
            self.wins += 1
        else:
            self.losses += 1
        
        self.trades.append(position)
        
        # Send exit notification
        self._notify_exit(position)
        
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print(f"\n📤 EXIT: {reason.upper()} @ ${exit_price:.4f}")
        print(f"   💰 P&L: ${pnl_amount:+.2f} ({pnl_pct:+.2f}%)")
        print(f"   ⏱️ Hold: {position.hold_time:.1f} min")
        print(f"   🏆 Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        print(f"   💵 Balance: ${self.current_balance:.2f}")
        
        self.current_position = None
    
    def _notify_entry(self, position: Trade, analysis: dict):
        """Send entry notifications"""
        # Sound notification
        if SOUND_AVAILABLE:
            try:
                winsound.Beep(1000, 500)  # 1000Hz for 0.5 seconds
            except:
                pass
        
        # Desktop notification
        if NOTIFICATIONS_AVAILABLE:
            try:
                notification.notify(
                    title="💰 LUCRATIVE BOT ENTRY",
                    message=f"{position.direction.upper()} @ ${position.entry_price:.4f}\n"
                            f"Target: ${position.target_profit}\n"
                            f"Confidence: {position.confidence:.1f}%",
                    timeout=5
                )
            except:
                pass
    
    def _notify_exit(self, position: Trade):
        """Send exit notifications"""
        # Sound notification
        if SOUND_AVAILABLE:
            try:
                if position.pnl_amount >= position.target_profit * 0.8:
                    # Big profit - ascending tones
                    for freq in [800, 900, 1000]:
                        winsound.Beep(freq, 200)
                elif position.pnl_amount > 0:
                    winsound.Beep(1200, 300)  # Profit
                else:
                    winsound.Beep(400, 500)   # Loss
            except:
                pass
        
        # Desktop notification
        if NOTIFICATIONS_AVAILABLE:
            try:
                profit_icon = "🎉" if position.pnl_amount >= position.target_profit * 0.8 else "💚" if position.pnl_amount > 0 else "❌"
                win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
                
                notification.notify(
                    title=f"{profit_icon} LUCRATIVE BOT EXIT",
                    message=f"P&L: ${position.pnl_amount:+.2f}\n"
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
        
        print(f"\n{'='*60}")
        print(f"💰 LUCRATIVE $40 PROFIT BOT - {now.strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"💵 SOL Price: ${self.last_price:.4f} (Age: {data_age:.1f}s)")
        print(f"💰 Balance: ${self.current_balance:.2f} | Target: ${self.config['target_profit']}/trade")
        
        if self.current_position:
            entry_price = self.current_position.entry_price
            direction = self.current_position.direction
            
            if direction == 'long':
                unrealized_pct = (self.last_price - entry_price) / entry_price * 100
            else:
                unrealized_pct = (entry_price - self.last_price) / entry_price * 100
            
            unrealized_amount = self.current_position.position_size * (unrealized_pct / 100)
            hold_time = (now - self.current_position.entry_time).total_seconds() / 60
            target_progress = (unrealized_amount / self.config['target_profit']) * 100
            
            print(f"\n🔵 ACTIVE POSITION:")
            print(f"   📊 {direction.upper()} @ ${entry_price:.4f} | Size: ${self.current_position.position_size:.2f}")
            print(f"   💰 Unrealized P&L: ${unrealized_amount:+.2f} ({unrealized_pct:+.2f}%)")
            print(f"   🎯 Target Progress: {target_progress:.1f}%")
            print(f"   ⏱️ Hold Time: {hold_time:.1f} min | Confidence: {self.current_position.confidence:.1f}%")
        else:
            print(f"\n⚫ SCANNING: Looking for high-confidence $40 profit opportunities")
        
        print(f"\n📊 PERFORMANCE:")
        print(f"   🏆 Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        print(f"   📈 Total Profit: ${self.total_profit:+.2f}")
        print(f"   📊 Daily Trades: {self.daily_trades}/{self.config['max_daily_trades']}")
        print(f"   🔗 Market Data: {len(self.market_data)} candles")
    
    def start_trading(self):
        """Start the lucrative trading bot"""
        print("🚀 Starting Lucrative $40 Profit Bot...")
        print("🔗 Connecting to OKX market data...")
        
        self.is_running = True
        last_data_fetch = 0
        last_status_display = 0
        
        # Main trading loop
        try:
            while self.is_running:
                current_time = time.time()
                
                # Fetch market data every 30 seconds
                if current_time - last_data_fetch > 30:
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
                    
                    # Display status every 60 seconds
                    if current_time - last_status_display > 60:
                        self._display_status()
                        last_status_display = current_time
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\n⚠️ Bot stopped by user")
            self.is_running = False
        
        finally:
            self._show_final_results()
    
    def _show_final_results(self):
        """Show final trading results"""
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print("\n" + "="*70)
        print("💰 LUCRATIVE $40 PROFIT BOT - FINAL RESULTS")
        print("="*70)
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
        
        print("="*70)

def main():
    """Main function"""
    print("💰 LUCRATIVE $40 PROFIT BOT")
    print("=" * 40)
    
    try:
        balance = float(input("💵 Enter starting balance (default $500): ") or "500")
    except ValueError:
        balance = 500.0
    
    bot = Lucrative40ProfitBot(initial_balance=balance)
    
    try:
        bot.start_trading()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped. Thanks for using Lucrative $40 Profit Bot!")

if __name__ == "__main__":
    main()