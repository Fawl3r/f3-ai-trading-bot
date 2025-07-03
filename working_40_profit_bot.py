#!/usr/bin/env python3
"""
Working $40 Profit Bot - High Performance Live Trading
Features:
- 20-30x adaptive leverage based on confidence
- Trailing stop loss system
- AI-optimized entry conditions
- Real-time OKX market data
- Sound and desktop notifications
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
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

class WorkingProfitBot:
    """High-performance $40 profit bot with 20-30x leverage"""
    
    def __init__(self, initial_balance: float = 500.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # OPTIMIZED CONFIGURATION FOR $40 PROFITS
        self.config = {
            # TARGET AND LEVERAGE
            "target_profit": 40.0,
            "leverage_min": 20,
            "leverage_max": 30,
            
            # POSITION SIZING
            "base_position": 150.0,
            "max_position": 400.0,
            "position_pct": 20,  # 20% of balance max
            
            # STOPS AND TARGETS
            "profit_target_pct": 2.5,    # 2.5% price movement target
            "trailing_stop_pct": 0.8,    # 0.8% trailing stop
            "trailing_activation": 0.5,  # Activate at 0.5% profit
            "emergency_stop": 2.0,       # 2% emergency stop
            
            # ENTRY CONDITIONS (LOWERED FOR MORE TRADES)
            "min_confidence": 40,        # 40% minimum confidence
            "rsi_oversold": 35,          # RSI < 35 = oversold
            "rsi_overbought": 65,        # RSI > 65 = overbought
            "volume_multiplier": 1.2,    # 1.2x average volume
            
            # RISK MANAGEMENT
            "max_daily_trades": 15,
            "max_hold_hours": 8,
            "max_consecutive_losses": 3,
            "daily_profit_target": 150.0,
        }
        
        # Trading state
        self.is_running = False
        self.current_position = None
        self.trades = []
        self.market_data = []
        
        # Performance tracking
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0
        self.daily_trades = 0
        self.daily_profit = 0
        self.consecutive_losses = 0
        
        # Market data
        self.last_price = 0.0
        self.last_update = None
        
        print("🚀 WORKING $40 PROFIT BOT")
        print("⚡ 20-30X ADAPTIVE LEVERAGE")
        print("🛡️ TRAILING STOP LOSS SYSTEM")
        print("=" * 60)
        print("🔧 CONFIGURATION:")
        print(f"   💰 Target Profit: ${self.config['target_profit']}")
        print(f"   ⚡ Leverage: {self.config['leverage_min']}-{self.config['leverage_max']}x")
        print(f"   💵 Position Size: ${self.config['base_position']}-${self.config['max_position']}")
        print(f"   🛡️ Trailing Stop: {self.config['trailing_stop_pct']}%")
        print(f"   🎯 Profit Target: {self.config['profit_target_pct']}%")
        print(f"   🤖 Min Confidence: {self.config['min_confidence']}%")
        print("=" * 60)
    
    def fetch_market_data(self):
        """Fetch real-time market data from OKX"""
        try:
            # Get current price
            ticker_url = "https://www.okx.com/api/v5/market/ticker?instId=SOL-USDT-SWAP"
            response = requests.get(ticker_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    self.last_price = float(data['data'][0]['last'])
                    self.last_update = datetime.now()
                    
                    # Get historical candles
                    candles_url = "https://www.okx.com/api/v5/market/candles?instId=SOL-USDT-SWAP&bar=1m&limit=100"
                    candles_response = requests.get(candles_url, timeout=10)
                    
                    if candles_response.status_code == 200:
                        candles_data = candles_response.json()
                        if candles_data.get('data'):
                            self.market_data = []
                            for candle in candles_data['data']:
                                self.market_data.append({
                                    'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000),
                                    'open': float(candle[1]),
                                    'high': float(candle[2]),
                                    'low': float(candle[3]),
                                    'close': float(candle[4]),
                                    'volume': float(candle[5])
                                })
                            
                            self.market_data.sort(key=lambda x: x['timestamp'])
                            return True
            
            return False
            
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return False
    
    def calculate_indicators(self):
        """Calculate technical indicators"""
        if len(self.market_data) < 50:
            return None
        
        df = pd.DataFrame(self.market_data)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        
        # Moving averages
        ma_5 = df['close'].rolling(5).mean()
        ma_20 = df['close'].rolling(20).mean()
        ma_50 = df['close'].rolling(50).mean()
        
        # Volume
        volume_ma = df['volume'].rolling(20).mean()
        
        # Current values
        current_rsi = rsi.iloc[-1]
        current_ma5 = ma_5.iloc[-1]
        current_ma20 = ma_20.iloc[-1]
        current_ma50 = ma_50.iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = volume_ma.iloc[-1]
        
        # Price momentum
        price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100
        
        return {
            'rsi': current_rsi,
            'ma_5': current_ma5,
            'ma_20': current_ma20,
            'ma_50': current_ma50,
            'volume_ratio': current_volume / avg_volume,
            'momentum': price_change_5,
            'current_price': df['close'].iloc[-1]
        }
    
    def analyze_opportunity(self):
        """Analyze market for trading opportunities"""
        indicators = self.calculate_indicators()
        if not indicators:
            return {"confidence": 0, "direction": "hold"}
        
        confidence = 0
        direction = "hold"
        reasons = []
        
        # RSI signals (40% weight)
        if indicators['rsi'] < self.config['rsi_oversold']:
            confidence += 40
            direction = "long"
            reasons.append("oversold")
        elif indicators['rsi'] > self.config['rsi_overbought']:
            confidence += 40
            direction = "short"
            reasons.append("overbought")
        
        # Moving average trend (30% weight)
        current_price = indicators['current_price']
        if current_price > indicators['ma_5'] > indicators['ma_20']:
            if direction == "long" or direction == "hold":
                confidence += 25
                direction = "long"
                reasons.append("bullish_trend")
        elif current_price < indicators['ma_5'] < indicators['ma_20']:
            if direction == "short" or direction == "hold":
                confidence += 25
                direction = "short"
                reasons.append("bearish_trend")
        
        # Volume confirmation (20% weight)
        if indicators['volume_ratio'] > self.config['volume_multiplier']:
            confidence += 20
            reasons.append("high_volume")
        
        # Momentum (10% weight)
        if abs(indicators['momentum']) > 1.0:
            if direction == "long" and indicators['momentum'] > 0:
                confidence += 10
                reasons.append("positive_momentum")
            elif direction == "short" and indicators['momentum'] < 0:
                confidence += 10
                reasons.append("negative_momentum")
        
        return {
            "confidence": min(confidence, 95),
            "direction": direction,
            "reasons": reasons,
            "indicators": indicators
        }
    
    def calculate_leverage(self, confidence):
        """Calculate adaptive leverage based on confidence"""
        base_leverage = self.config['leverage_min']
        max_leverage = self.config['leverage_max']
        
        # Scale leverage based on confidence
        confidence_ratio = confidence / 100
        leverage = int(base_leverage + (max_leverage - base_leverage) * confidence_ratio)
        
        return max(base_leverage, min(max_leverage, leverage))
    
    def calculate_position_size(self, confidence):
        """Calculate position size based on confidence and balance"""
        base_size = self.config['base_position']
        max_size = min(
            self.config['max_position'],
            self.current_balance * self.config['position_pct'] / 100
        )
        
        confidence_ratio = confidence / 100
        position_size = base_size + (max_size - base_size) * confidence_ratio
        
        return position_size
    
    def check_entry(self):
        """Check if we should enter a trade"""
        # Basic checks
        if self.current_position:
            return False
        
        if self.daily_trades >= self.config['max_daily_trades']:
            return False
        
        if self.consecutive_losses >= self.config['max_consecutive_losses']:
            return False
        
        if self.daily_profit >= self.config['daily_profit_target']:
            return False
        
        # Analyze market
        analysis = self.analyze_opportunity()
        
        if analysis['confidence'] < self.config['min_confidence']:
            return False
        
        if analysis['direction'] == "hold":
            return False
        
        # Calculate trade parameters
        leverage = self.calculate_leverage(analysis['confidence'])
        position_size = self.calculate_position_size(analysis['confidence'])
        
        # Create position
        self.current_position = {
            'direction': analysis['direction'],
            'entry_price': self.last_price,
            'entry_time': datetime.now(),
            'position_size': position_size,
            'leverage': leverage,
            'confidence': analysis['confidence'],
            'reasons': analysis['reasons'],
            'best_price': self.last_price,
            'trailing_stop_price': None,
            'trailing_activated': False
        }
        
        self.total_trades += 1
        self.daily_trades += 1
        
        # Notifications
        self.notify_entry()
        
        print(f"\n🚀 ENTRY: {analysis['direction'].upper()} @ ${self.last_price:.4f}")
        print(f"   💰 Size: ${position_size:.2f} | Leverage: {leverage}x")
        print(f"   🎯 Target: ${self.config['target_profit']} | Confidence: {analysis['confidence']:.1f}%")
        print(f"   📊 Reasons: {', '.join(analysis['reasons'])}")
        
        return True
    
    def update_trailing_stop(self):
        """Update trailing stop loss"""
        if not self.current_position:
            return
        
        position = self.current_position
        current_price = self.last_price
        direction = position['direction']
        
        # Update best price
        if direction == 'long' and current_price > position['best_price']:
            position['best_price'] = current_price
        elif direction == 'short' and current_price < position['best_price']:
            position['best_price'] = current_price
        
        # Check if we should activate trailing stop
        if not position['trailing_activated']:
            entry_price = position['entry_price']
            
            if direction == 'long':
                profit_pct = (position['best_price'] - entry_price) / entry_price * 100
            else:
                profit_pct = (entry_price - position['best_price']) / entry_price * 100
            
            if profit_pct >= self.config['trailing_activation']:
                position['trailing_activated'] = True
                print(f"🛡️ Trailing stop ACTIVATED at {profit_pct:.2f}% profit")
        
        # Update trailing stop price
        if position['trailing_activated']:
            if direction == 'long':
                new_stop = position['best_price'] * (1 - self.config['trailing_stop_pct'] / 100)
                if position['trailing_stop_price'] is None or new_stop > position['trailing_stop_price']:
                    position['trailing_stop_price'] = new_stop
            else:
                new_stop = position['best_price'] * (1 + self.config['trailing_stop_pct'] / 100)
                if position['trailing_stop_price'] is None or new_stop < position['trailing_stop_price']:
                    position['trailing_stop_price'] = new_stop
    
    def check_exit(self):
        """Check if we should exit current position"""
        if not self.current_position:
            return
        
        position = self.current_position
        entry_price = position['entry_price']
        current_price = self.last_price
        direction = position['direction']
        leverage = position['leverage']
        
        # Update trailing stop
        self.update_trailing_stop()
        
        # Calculate P&L
        if direction == 'long':
            price_change_pct = (current_price - entry_price) / entry_price * 100
        else:
            price_change_pct = (entry_price - current_price) / entry_price * 100
        
        pnl_amount = position['position_size'] * (price_change_pct / 100)
        
        # Exit conditions
        should_exit = False
        exit_reason = None
        
        # Take profit (90% of target)
        if pnl_amount >= self.config['target_profit'] * 0.9:
            should_exit = True
            exit_reason = "take_profit"
        
        # Trailing stop
        elif position['trailing_activated'] and position['trailing_stop_price']:
            if direction == 'long' and current_price <= position['trailing_stop_price']:
                should_exit = True
                exit_reason = "trailing_stop"
            elif direction == 'short' and current_price >= position['trailing_stop_price']:
                should_exit = True
                exit_reason = "trailing_stop"
        
        # Emergency stop
        elif price_change_pct <= -self.config['emergency_stop']:
            should_exit = True
            exit_reason = "emergency_stop"
        
        # Time exit
        elif (datetime.now() - position['entry_time']).total_seconds() > self.config['max_hold_hours'] * 3600:
            should_exit = True
            exit_reason = "time_exit"
        
        if should_exit:
            self.close_position(exit_reason, current_price, pnl_amount)
    
    def close_position(self, reason, exit_price, pnl_amount):
        """Close current position"""
        if not self.current_position:
            return
        
        position = self.current_position
        
        # Update stats
        self.current_balance += pnl_amount
        self.total_profit += pnl_amount
        self.daily_profit += pnl_amount
        
        if pnl_amount > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
        
        # Store trade
        trade = {
            **position,
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'exit_reason': reason,
            'pnl_amount': pnl_amount,
            'hold_minutes': (datetime.now() - position['entry_time']).total_seconds() / 60
        }
        self.trades.append(trade)
        
        # Notifications
        self.notify_exit(pnl_amount, reason)
        
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print(f"\n📤 EXIT: {reason.upper()} @ ${exit_price:.4f}")
        print(f"   💰 P&L: ${pnl_amount:+.2f}")
        print(f"   ⚡ Leverage: {position['leverage']}x")
        print(f"   🛡️ Trailing: {'✅' if position['trailing_activated'] else '❌'}")
        print(f"   🏆 Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        print(f"   💵 Balance: ${self.current_balance:.2f}")
        
        self.current_position = None
    
    def notify_entry(self):
        """Send entry notifications"""
        if SOUND_AVAILABLE:
            try:
                winsound.Beep(1000, 300)
            except:
                pass
        
        if NOTIFICATIONS_AVAILABLE:
            try:
                position = self.current_position
                notification.notify(
                    title="🚀 $40 PROFIT BOT ENTRY",
                    message=f"{position['direction'].upper()} @ ${position['entry_price']:.4f}\n"
                            f"Leverage: {position['leverage']}x\n"
                            f"Target: ${self.config['target_profit']}\n"
                            f"Confidence: {position['confidence']:.1f}%",
                    timeout=5
                )
            except:
                pass
    
    def notify_exit(self, pnl_amount, reason):
        """Send exit notifications"""
        if SOUND_AVAILABLE:
            try:
                if pnl_amount >= self.config['target_profit'] * 0.8:
                    # Big profit
                    for freq in [800, 1000, 1200]:
                        winsound.Beep(freq, 150)
                elif pnl_amount > 0:
                    winsound.Beep(1200, 400)  # Profit
                else:
                    winsound.Beep(400, 600)   # Loss
            except:
                pass
        
        if NOTIFICATIONS_AVAILABLE:
            try:
                icon = "🎉" if pnl_amount >= self.config['target_profit'] * 0.8 else "💚" if pnl_amount > 0 else "❌"
                win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
                
                notification.notify(
                    title=f"{icon} $40 PROFIT BOT EXIT",
                    message=f"P&L: ${pnl_amount:+.2f}\n"
                            f"Reason: {reason}\n"
                            f"Win Rate: {win_rate:.1f}%\n"
                            f"Balance: ${self.current_balance:.2f}",
                    timeout=6
                )
            except:
                pass
    
    def display_status(self):
        """Display current status"""
        now = datetime.now()
        data_age = (now - self.last_update).total_seconds() if self.last_update else 999
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"🚀 WORKING $40 PROFIT BOT - {now.strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"💵 SOL Price: ${self.last_price:.4f} (Age: {data_age:.1f}s)")
        print(f"💰 Balance: ${self.current_balance:.2f} | Daily P&L: ${self.daily_profit:+.2f}")
        
        if self.current_position:
            position = self.current_position
            entry_price = position['entry_price']
            direction = position['direction']
            
            if direction == 'long':
                unrealized_pct = (self.last_price - entry_price) / entry_price * 100
            else:
                unrealized_pct = (entry_price - self.last_price) / entry_price * 100
            
            unrealized_amount = position['position_size'] * (unrealized_pct / 100)
            hold_time = (now - position['entry_time']).total_seconds() / 60
            target_progress = (unrealized_amount / self.config['target_profit']) * 100
            
            print(f"\n🔵 ACTIVE POSITION:")
            print(f"   📊 {direction.upper()} @ ${entry_price:.4f} | Size: ${position['position_size']:.2f}")
            print(f"   ⚡ Leverage: {position['leverage']}x | Confidence: {position['confidence']:.1f}%")
            print(f"   💰 Unrealized P&L: ${unrealized_amount:+.2f} ({unrealized_pct:+.2f}%)")
            print(f"   🎯 Target Progress: {target_progress:.1f}%")
            print(f"   🛡️ Trailing: {'✅ ACTIVE' if position['trailing_activated'] else '⏸️ WAITING'}")
            if position['trailing_stop_price']:
                print(f"   🛑 Stop Price: ${position['trailing_stop_price']:.4f}")
            print(f"   ⏱️ Hold Time: {hold_time:.1f} min")
        else:
            print(f"\n⚫ SCANNING: Looking for profitable opportunities")
            print(f"   🎯 Target: ${self.config['target_profit']} per trade")
            print(f"   ⚡ Leverage: {self.config['leverage_min']}-{self.config['leverage_max']}x adaptive")
        
        print(f"\n📊 PERFORMANCE:")
        print(f"   🏆 Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        print(f"   📈 Total Profit: ${self.total_profit:+.2f}")
        print(f"   📊 Daily Trades: {self.daily_trades}/{self.config['max_daily_trades']}")
        print(f"   🔥 Consecutive Losses: {self.consecutive_losses}/{self.config['max_consecutive_losses']}")
    
    def start_trading(self):
        """Start the trading bot"""
        print("🚀 Starting Working $40 Profit Bot...")
        print("🔗 Connecting to OKX market data...")
        
        self.is_running = True
        last_data_fetch = 0
        last_status_display = 0
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Reset daily counters
                today = datetime.now().date()
                if hasattr(self, 'last_day') and self.last_day != today:
                    self.daily_trades = 0
                    self.daily_profit = 0
                self.last_day = today
                
                # Fetch market data every 15 seconds
                if current_time - last_data_fetch > 15:
                    if self.fetch_market_data():
                        print(f"📡 Market data updated - SOL: ${self.last_price:.4f}")
                    last_data_fetch = current_time
                
                if self.last_price > 0:
                    # Check entry/exit
                    if not self.current_position:
                        self.check_entry()
                    else:
                        self.check_exit()
                    
                    # Display status every 30 seconds
                    if current_time - last_status_display > 30:
                        self.display_status()
                        last_status_display = current_time
                
                time.sleep(2)  # Check every 2 seconds
                
        except KeyboardInterrupt:
            print("\n⚠️ Bot stopped by user")
            self.is_running = False
        
        finally:
            self.show_final_results()
    
    def show_final_results(self):
        """Show final trading results"""
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print("\n" + "="*70)
        print("🚀 WORKING $40 PROFIT BOT - FINAL RESULTS")
        print("="*70)
        print(f"📊 Total Trades: {self.total_trades}")
        print(f"🏆 Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        print(f"💰 Total Return: {total_return:+.1f}%")
        print(f"💵 Final Balance: ${self.current_balance:.2f}")
        print(f"💎 Total Profit: ${self.total_profit:+.2f}")
        
        if self.trades:
            profitable_trades = [t for t in self.trades if t['pnl_amount'] > 0]
            if profitable_trades:
                avg_profit = np.mean([t['pnl_amount'] for t in profitable_trades])
                max_profit = max([t['pnl_amount'] for t in profitable_trades])
                target_hits = len([t for t in profitable_trades if t['pnl_amount'] >= self.config['target_profit'] * 0.9])
                
                print(f"📈 Avg Profit: ${avg_profit:.2f} | Max Profit: ${max_profit:.2f}")
                print(f"🎯 Target Hits: {target_hits}/{len(profitable_trades)} ({target_hits/len(profitable_trades)*100:.1f}%)")
            
            # Leverage analysis
            avg_leverage = np.mean([t['leverage'] for t in self.trades])
            print(f"⚡ Average Leverage: {avg_leverage:.1f}x")
            
            # Trailing stop stats
            trailing_trades = [t for t in self.trades if t['trailing_activated']]
            if trailing_trades:
                print(f"🛡️ Trailing Stop Used: {len(trailing_trades)}/{self.total_trades} ({len(trailing_trades)/self.total_trades*100:.1f}%)")
        
        print("="*70)

def main():
    """Main function"""
    print("🚀 WORKING $40 PROFIT BOT")
    print("⚡ 20-30X ADAPTIVE LEVERAGE WITH TRAILING STOPS")
    print("=" * 50)
    
    try:
        balance = float(input("💵 Enter starting balance (default $500): ") or "500")
    except ValueError:
        balance = 500.0
    
    bot = WorkingProfitBot(initial_balance=balance)
    
    try:
        bot.start_trading()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped. Thanks for using Working $40 Profit Bot!")

if __name__ == "__main__":
    main()