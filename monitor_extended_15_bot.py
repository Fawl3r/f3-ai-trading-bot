#!/usr/bin/env python3
"""
📊 EXTENDED 15 BOT MONITOR
Real-time monitoring dashboard for the Extended 15 Production Bot

Features:
- Live performance tracking
- Trade history
- Profit/loss monitoring
- Win rate calculations
- Position tracking
"""

import os
import json
import time
import sqlite3
from datetime import datetime, timedelta
from hyperliquid.info import Info
from hyperliquid.utils import constants
import warnings
warnings.filterwarnings('ignore')

class Extended15BotMonitor:
    """Real-time monitoring for Extended 15 Bot"""
    
    def __init__(self):
        print("📊 EXTENDED 15 BOT MONITOR")
        print("🔍 Real-time performance tracking")
        print("=" * 80)
        
        self.setup_config()
        self.setup_hyperliquid()
        self.setup_database()
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'daily_trades': 0,
            'active_positions': 0,
            'win_rate': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
        
        # Expected Extended 15 targets
        self.targets = {
            'win_rate': 70.1,
            'daily_trades': 4.5,
            'annual_trades': 1642,
            'profit_per_trade': 0.82
        }

    def setup_config(self):
        """Load configuration"""
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print("❌ config.json not found")
            raise

    def setup_hyperliquid(self):
        """Setup Hyperliquid connection"""
        try:
            self.info = Info(constants.MAINNET_API_URL if self.config['is_mainnet'] else constants.TESTNET_API_URL)
            print("✅ Hyperliquid connection established")
        except Exception as e:
            print(f"❌ Hyperliquid connection failed: {e}")
            raise

    def setup_database(self):
        """Setup SQLite database for tracking"""
        self.db_path = 'extended_15_metrics.db'
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME,
                    symbol TEXT,
                    signal_type TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    position_size REAL,
                    leverage REAL,
                    pnl REAL,
                    win INTEGER,
                    duration_minutes INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    date TEXT PRIMARY KEY,
                    trades_count INTEGER,
                    total_profit REAL,
                    win_rate REAL,
                    largest_win REAL,
                    largest_loss REAL
                )
            ''')
            
            conn.commit()

    def get_account_balance(self):
        """Get current account balance"""
        try:
            user_state = self.info.user_state(self.config['wallet_address'])
            if user_state and 'marginSummary' in user_state:
                return float(user_state['marginSummary'].get('accountValue', 0))
            return 0.0
        except Exception as e:
            print(f"❌ Error getting balance: {e}")
            return 0.0

    def get_active_positions(self):
        """Get active positions"""
        try:
            user_state = self.info.user_state(self.config['wallet_address'])
            if user_state and 'assetPositions' in user_state:
                positions = []
                for pos in user_state['assetPositions']:
                    if float(pos['position']['szi']) != 0:
                        positions.append({
                            'symbol': pos['position']['coin'],
                            'size': float(pos['position']['szi']),
                            'entry_price': float(pos['position']['entryPx']),
                            'unrealized_pnl': float(pos['position']['unrealizedPnl']),
                            'margin_used': float(pos['position']['marginUsed'])
                        })
                return positions
            return []
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return []

    def calculate_performance_metrics(self):
        """Calculate current performance metrics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get trade statistics
            cursor.execute('''
                SELECT COUNT(*) as total_trades,
                       SUM(win) as winning_trades,
                       SUM(pnl) as total_profit,
                       MAX(pnl) as largest_win,
                       MIN(pnl) as largest_loss
                FROM trades
            ''')
            
            row = cursor.fetchone()
            if row and row[0] > 0:
                self.performance_metrics['total_trades'] = row[0]
                self.performance_metrics['winning_trades'] = row[1] or 0
                self.performance_metrics['total_profit'] = row[2] or 0.0
                self.performance_metrics['largest_win'] = row[3] or 0.0
                self.performance_metrics['largest_loss'] = row[4] or 0.0
                
                # Calculate win rate
                if self.performance_metrics['total_trades'] > 0:
                    self.performance_metrics['win_rate'] = (self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']) * 100
            
            # Get today's trades
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT COUNT(*) FROM trades 
                WHERE DATE(timestamp) = ?
            ''', (today,))
            
            row = cursor.fetchone()
            self.performance_metrics['daily_trades'] = row[0] if row else 0

    def display_live_dashboard(self):
        """Display live dashboard"""
        
        # Get current data
        balance = self.get_account_balance()
        positions = self.get_active_positions()
        self.performance_metrics['active_positions'] = len(positions)
        
        # Calculate metrics
        self.calculate_performance_metrics()
        
        # Clear screen and display dashboard
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🚀" + "=" * 78 + "🚀")
        print("📊 EXTENDED 15 BOT LIVE DASHBOARD")
        print("🚀" + "=" * 78 + "🚀")
        print(f"⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Account Overview
        print("💰 ACCOUNT OVERVIEW")
        print("-" * 50)
        print(f"   💎 Account Balance: ${balance:.2f}")
        print(f"   📊 Active Positions: {self.performance_metrics['active_positions']}")
        print(f"   💰 Total Profit: ${self.performance_metrics['total_profit']:.2f}")
        print(f"   📈 Profit % of Balance: {(self.performance_metrics['total_profit'] / balance * 100):.1f}%")
        print()
        
        # Performance Metrics
        print("📈 PERFORMANCE METRICS")
        print("-" * 50)
        print(f"   🎯 Win Rate: {self.performance_metrics['win_rate']:.1f}% (Target: {self.targets['win_rate']:.1f}%)")
        print(f"   📊 Total Trades: {self.performance_metrics['total_trades']}")
        print(f"   🏆 Winning Trades: {self.performance_metrics['winning_trades']}")
        print(f"   📉 Losing Trades: {self.performance_metrics['total_trades'] - self.performance_metrics['winning_trades']}")
        print(f"   💰 Largest Win: ${self.performance_metrics['largest_win']:.2f}")
        print(f"   📉 Largest Loss: ${self.performance_metrics['largest_loss']:.2f}")
        print()
        
        # Daily Performance
        print("📅 TODAY'S PERFORMANCE")
        print("-" * 50)
        print(f"   📊 Daily Trades: {self.performance_metrics['daily_trades']}")
        print(f"   🎯 Target Daily: {self.targets['daily_trades']:.1f}")
        print(f"   📈 Daily Progress: {(self.performance_metrics['daily_trades'] / self.targets['daily_trades'] * 100):.1f}%")
        print()
        
        # Active Positions
        if positions:
            print("🎲 ACTIVE POSITIONS")
            print("-" * 50)
            for pos in positions:
                pnl_color = "🟢" if pos['unrealized_pnl'] > 0 else "🔴"
                print(f"   {pnl_color} {pos['symbol']}: Size {pos['size']:.4f} | Entry ${pos['entry_price']:.4f} | PnL ${pos['unrealized_pnl']:.2f}")
            print()
        
        # Target Tracking
        print("🎯 TARGET TRACKING")
        print("-" * 50)
        win_rate_status = "✅" if self.performance_metrics['win_rate'] >= self.targets['win_rate'] else "❌"
        daily_trades_status = "✅" if self.performance_metrics['daily_trades'] >= self.targets['daily_trades'] else "⏳"
        
        print(f"   {win_rate_status} Win Rate: {self.performance_metrics['win_rate']:.1f}% / {self.targets['win_rate']:.1f}%")
        print(f"   {daily_trades_status} Daily Trades: {self.performance_metrics['daily_trades']} / {self.targets['daily_trades']:.1f}")
        print()
        
        # Projections
        if self.performance_metrics['total_trades'] > 0:
            avg_profit_per_trade = self.performance_metrics['total_profit'] / self.performance_metrics['total_trades']
            projected_daily_profit = avg_profit_per_trade * self.targets['daily_trades']
            projected_monthly_profit = projected_daily_profit * 30
            
            print("📊 PROFIT PROJECTIONS")
            print("-" * 50)
            print(f"   💰 Avg Profit/Trade: ${avg_profit_per_trade:.2f}")
            print(f"   📈 Projected Daily: ${projected_daily_profit:.2f}")
            print(f"   🚀 Projected Monthly: ${projected_monthly_profit:.2f}")
            print()
        
        print("🔄 Press Ctrl+C to stop monitoring")
        print("=" * 80)

    def run_monitor(self):
        """Run the monitoring loop"""
        
        try:
            while True:
                self.display_live_dashboard()
                time.sleep(10)  # Update every 10 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")

def main():
    """Main function"""
    
    try:
        monitor = Extended15BotMonitor()
        monitor.run_monitor()
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main() 