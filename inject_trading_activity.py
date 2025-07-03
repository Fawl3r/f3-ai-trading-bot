#!/usr/bin/env python3
"""
Script to inject trading activity and market events for demonstration
"""

import requests
import time
import random
from datetime import datetime

def inject_market_events():
    """Inject various market events to generate trading activity"""
    
    print("💥 Injecting Market Events to Generate Trading Activity")
    print("=" * 60)
    
    events = [
        ("market_pump", "🚀 Market Pump (+5%)"),
        ("high_volatility", "⚡ High Volatility Period"),
        ("market_dump", "📉 Market Dump (-5%)"),
        ("low_volatility", "😴 Low Volatility Period"),
        ("market_pump", "🚀 Another Market Pump (+5%)")
    ]
    
    for i, (event_type, description) in enumerate(events):
        print(f"\n[Event {i+1}/5] {description}")
        
        # For demonstration, we'll manually create some trading activity
        # by calling the metrics update directly
        
        # Simulate a trade
        if event_type == "market_pump":
            # Simulate a profitable long trade
            simulate_trade("long", 151.0, 155.5, 45.0)
        elif event_type == "market_dump":
            # Simulate a profitable short trade
            simulate_trade("short", 149.0, 142.0, 35.0)
        
        # Wait between events
        print(f"⏱️  Waiting 10 seconds for market to react...")
        time.sleep(10)
        
        # Check current metrics
        check_current_metrics()
    
    print("\n✅ All market events injected!")
    print("🌐 Check the dashboard at http://127.0.0.1:5000 to see the activity!")

def simulate_trade(side: str, entry_price: float, exit_price: float, pnl: float):
    """Simulate a completed trade by updating the database directly"""
    import sqlite3
    from datetime import datetime
    
    try:
        # Connect to the metrics database
        conn = sqlite3.connect('bot_metrics.db')
        cursor = conn.cursor()
        
        # Insert simulated trade
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update trades table
        cursor.execute("""
            INSERT INTO trades (symbol, side, size, price, pnl, status, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'SOL-USD-SWAP',
            side,
            100.0 / entry_price,  # position size
            exit_price,
            pnl,
            'closed',
            timestamp,
            f'{{"entry_price": {entry_price}, "simulation": true}}'
        ))
        
        # Update performance metrics
        cursor.execute("""
            INSERT INTO performance_metrics (
                total_pnl, win_rate, profit_factor, max_drawdown, 
                total_trades, active_positions, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            pnl,  # total_pnl
            0.8,  # win_rate (80%)
            1.5,  # profit_factor
            0.05, # max_drawdown (5%)
            1,    # total_trades
            random.randint(0, 2),  # active_positions
            timestamp
        ))
        
        conn.commit()
        conn.close()
        
        print(f"  📈 Simulated {side.upper()} trade: ${entry_price:.2f} → ${exit_price:.2f} = +${pnl:.2f}")
        
    except Exception as e:
        print(f"  ❌ Error simulating trade: {e}")

def check_current_metrics():
    """Check current metrics from the API"""
    try:
        response = requests.get("http://127.0.0.1:5000/api/metrics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            real_time = data.get('real_time', {})
            trading = real_time.get('trading', {})
            trend = real_time.get('trend', {})
            
            print(f"  💰 Current P&L: ${trading.get('total_pnl', 0):.2f}")
            print(f"  🔄 Total Trades: {trading.get('total_trades', 0)}")
            print(f"  💹 Current Price: ${trend.get('current_price', 0):.4f}")
            print(f"  📊 Trend: {trend.get('direction', 'unknown')}")
            
    except Exception as e:
        print(f"  ❌ Error checking metrics: {e}")

def generate_realistic_trading_history():
    """Generate a realistic trading history for demonstration"""
    import sqlite3
    from datetime import datetime, timedelta
    import random
    
    print("\n📚 Generating Realistic Trading History...")
    
    try:
        conn = sqlite3.connect('bot_metrics.db')
        cursor = conn.cursor()
        
        # Clear existing performance data
        cursor.execute("DELETE FROM performance_metrics")
        cursor.execute("DELETE FROM trades")
        
        base_time = datetime.now() - timedelta(hours=2)
        cumulative_pnl = 0.0
        trade_count = 0
        winning_trades = 0
        
        # Generate 20 realistic trades over the past 2 hours
        for i in range(20):
            timestamp = base_time + timedelta(minutes=i*6)
            
            # Random trade parameters
            side = random.choice(['long', 'short'])
            entry_price = random.uniform(148.0, 153.0)
            
            # 70% chance of winning trade
            is_winning = random.random() < 0.7
            
            if is_winning:
                if side == 'long':
                    exit_price = entry_price * random.uniform(1.005, 1.025)  # 0.5-2.5% gain
                else:
                    exit_price = entry_price * random.uniform(0.975, 0.995)  # 0.5-2.5% gain
                pnl = random.uniform(10.0, 45.0)
                winning_trades += 1
            else:
                if side == 'long':
                    exit_price = entry_price * random.uniform(0.98, 0.995)   # 0.5-2% loss
                else:
                    exit_price = entry_price * random.uniform(1.005, 1.02)  # 0.5-2% loss
                pnl = -random.uniform(8.0, 25.0)
            
            cumulative_pnl += pnl
            trade_count += 1
            
            # Insert trade
            cursor.execute("""
                INSERT INTO trades (symbol, side, size, price, pnl, status, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'SOL-USD-SWAP',
                side,
                100.0 / entry_price,
                exit_price,
                pnl,
                'closed',
                timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                f'{{"entry_price": {entry_price}, "simulation": true, "confidence": {random.uniform(0.6, 0.9):.2f}}}'
            ))
            
            # Insert performance snapshot
            win_rate = winning_trades / trade_count
            profit_factor = max(1.0, win_rate / (1 - win_rate)) if win_rate < 1.0 else 2.5
            max_drawdown = random.uniform(0.02, 0.08)  # 2-8%
            
            cursor.execute("""
                INSERT INTO performance_metrics (
                    total_pnl, win_rate, profit_factor, max_drawdown, 
                    total_trades, active_positions, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                cumulative_pnl,
                win_rate,
                profit_factor,
                max_drawdown,
                trade_count,
                random.randint(0, 2),
                timestamp.strftime('%Y-%m-%d %H:%M:%S')
            ))
        
        conn.commit()
        conn.close()
        
        print(f"  ✅ Generated {trade_count} trades")
        print(f"  💰 Final P&L: ${cumulative_pnl:.2f}")
        print(f"  🎯 Win Rate: {win_rate:.1%}")
        print(f"  🏆 Winning Trades: {winning_trades}/{trade_count}")
        
    except Exception as e:
        print(f"  ❌ Error generating history: {e}")

def main():
    """Main function"""
    print("🎮 Trading Activity Injection Script")
    print("📅 Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    # Generate realistic trading history first
    generate_realistic_trading_history()
    
    # Then inject some live events
    inject_market_events()
    
    print("\n🌐 Dashboard is now populated with trading data!")
    print("📊 Visit http://127.0.0.1:5000 to see the results!")

if __name__ == "__main__":
    main() 