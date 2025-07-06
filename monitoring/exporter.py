#!/usr/bin/env python3
"""
Prometheus Exporter for Elite Parabolic Trading System
Exports comprehensive metrics for monitoring and alerting
"""

import time
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
from pathlib import Path
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
SYSTEM_INFO = Info('elite_system_info', 'System information')
CURRENT_BALANCE = Gauge('elite_balance_current', 'Current account balance')
INITIAL_BALANCE = Gauge('elite_balance_initial', 'Initial account balance')
TOTAL_RETURN = Gauge('elite_return_total_percent', 'Total return percentage')
DAILY_RETURN = Gauge('elite_return_daily_percent', 'Daily return percentage')

# Trade metrics
TRADES_TOTAL = Counter('elite_trades_total', 'Total trades executed')
TRADES_WINNING = Counter('elite_trades_winning_total', 'Total winning trades')
TRADES_LOSING = Counter('elite_trades_losing_total', 'Total losing trades')
WIN_RATE = Gauge('elite_win_rate_percent', 'Win rate percentage')

# Performance metrics
PROFIT_FACTOR = Gauge('elite_profit_factor', 'Profit factor ratio')
EXPECTANCY = Gauge('elite_expectancy_percent', 'Expectancy per trade percentage')
SHARPE_RATIO = Gauge('elite_sharpe_ratio', 'Sharpe ratio')
AVG_R_MULTIPLE = Gauge('elite_avg_r_multiple', 'Average R multiple')

# Risk metrics
MAX_DRAWDOWN = Gauge('elite_max_drawdown_percent', 'Maximum drawdown percentage')
CURRENT_DRAWDOWN = Gauge('elite_current_drawdown_percent', 'Current drawdown percentage')
DAILY_VAR = Gauge('elite_daily_var_percent', 'Daily Value at Risk')
CONSECUTIVE_LOSSES = Gauge('elite_consecutive_losses', 'Current consecutive losses')

# Trade distribution
TRADE_PNL = Histogram('elite_trade_pnl_dollars', 'Trade P&L distribution in dollars', 
                     buckets=(-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 20, float('inf')))
TRADE_R_MULTIPLE = Histogram('elite_trade_r_multiple', 'Trade R multiple distribution',
                            buckets=(-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, 8, float('inf')))
HOLD_TIME_MINUTES = Histogram('elite_hold_time_minutes', 'Trade hold time in minutes',
                             buckets=(5, 15, 30, 60, 120, 240, 480, 960, float('inf')))

# Signal metrics
SIGNAL_STRENGTH = Histogram('elite_signal_strength', 'Signal strength distribution',
                           buckets=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, float('inf')))
SIGNALS_BY_TYPE = Counter('elite_signals_total', 'Signals by type', ['signal_type'])
TRADES_BY_SYMBOL = Counter('elite_trades_by_symbol_total', 'Trades by symbol', ['symbol'])

# System health
SYSTEM_UPTIME = Gauge('elite_system_uptime_seconds', 'System uptime in seconds')
LAST_TRADE_TIME = Gauge('elite_last_trade_timestamp', 'Last trade timestamp')
DB_CONNECTIONS = Gauge('elite_db_connections', 'Active database connections')
API_ERRORS = Counter('elite_api_errors_total', 'API errors by type', ['error_type'])

# Risk events
RISK_KILLS = Counter('elite_risk_kills_total', 'Risk kill-switch triggers')
STOP_LOSSES = Counter('elite_stop_losses_total', 'Stop loss exits')
TAKE_PROFITS = Counter('elite_take_profits_total', 'Take profit exits')

class ElitePrometheusExporter:
    def __init__(self, db_path: str = "shadow_trades.db", port: int = 8000):
        self.db_path = db_path
        self.port = port
        self.start_time = time.time()
        self.running = False
        self.trade_table_name = None
        self.equity_table_name = None
        self.risk_table_name = None
        
        # Detect table schema
        self._detect_table_schema()
        
        # Initialize system info
        SYSTEM_INFO.info({
            'version': '1.0.0',
            'system': 'Elite Parabolic Trading System',
            'start_time': datetime.now().isoformat(),
            'database': db_path,
            'trade_table': self.trade_table_name or 'none',
            'equity_table': self.equity_table_name or 'none',
            'risk_table': self.risk_table_name or 'none'
        })
        
        logger.info(f"Prometheus exporter initialized - DB: {db_path}, Port: {port}")
        logger.info(f"Detected tables - Trades: {self.trade_table_name}, Equity: {self.equity_table_name}, Risk: {self.risk_table_name}")
    
    def _detect_table_schema(self):
        """Detect the table schema in the database"""
        conn = self.get_db_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Look for trade tables (various naming patterns)
            trade_candidates = [t for t in tables if 'trade' in t.lower()]
            if trade_candidates:
                self.trade_table_name = trade_candidates[0]  # Take the first one
            
            # Look for equity curve tables
            equity_candidates = [t for t in tables if 'equity' in t.lower()]
            if equity_candidates:
                self.equity_table_name = equity_candidates[0]
            
            # Look for risk event tables
            risk_candidates = [t for t in tables if 'risk' in t.lower()]
            if risk_candidates:
                self.risk_table_name = risk_candidates[0]
                
        except Exception as e:
            logger.error(f"Error detecting table schema: {e}")
        finally:
            conn.close()
    
    def get_db_connection(self):
        """Get database connection with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            API_ERRORS.labels(error_type='database').inc()
            return None
    
    def update_trade_metrics(self):
        """Update trade-related metrics"""
        if not self.trade_table_name:
            logger.warning("No trade table found, skipping trade metrics")
            return
            
        conn = self.get_db_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            
            # Get all trades
            cursor.execute(f"SELECT * FROM {self.trade_table_name} ORDER BY timestamp")
            trades = cursor.fetchall()
            
            if not trades:
                return
            
            # Basic counts
            total_trades = len(trades)
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            # Update counters (only increment new trades)
            current_total = TRADES_TOTAL._value._value
            if total_trades > current_total:
                TRADES_TOTAL.inc(total_trades - current_total)
            
            # Win rate
            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            WIN_RATE.set(win_rate)
            
            # Profit factor
            total_wins = sum(t['pnl'] for t in winning_trades)
            total_losses = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            PROFIT_FACTOR.set(profit_factor)
            
            # Expectancy
            avg_pnl = sum(t['pnl'] for t in trades) / total_trades
            expectancy = avg_pnl / 50.0 * 100  # Assuming $50 initial balance
            EXPECTANCY.set(expectancy)
            
            # R multiples
            r_multiples = [t['r_multiple'] for t in trades]
            avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
            AVG_R_MULTIPLE.set(avg_r)
            
            # Update histograms for recent trades
            for trade in trades[-10:]:  # Last 10 trades to avoid duplicate observations
                TRADE_PNL.observe(trade['pnl'])
                TRADE_R_MULTIPLE.observe(trade['r_multiple'])
                HOLD_TIME_MINUTES.observe(trade['hold_time_minutes'])
                SIGNAL_STRENGTH.observe(trade['signal_strength'])
                
                # Update counters by labels
                SIGNALS_BY_TYPE.labels(signal_type=trade['signal_type']).inc()
                TRADES_BY_SYMBOL.labels(symbol=trade['symbol']).inc()
                
                # Exit reason counters
                if trade['exit_reason'] == 'Stop Loss':
                    STOP_LOSSES.inc()
                elif trade['exit_reason'] == 'Take Profit':
                    TAKE_PROFITS.inc()
            
            # Last trade time
            if trades:
                last_trade = trades[-1]
                last_time = datetime.fromisoformat(last_trade['timestamp']).timestamp()
                LAST_TRADE_TIME.set(last_time)
            
            # Consecutive losses
            consecutive = 0
            for trade in reversed(trades):
                if trade['pnl'] <= 0:
                    consecutive += 1
                else:
                    break
            CONSECUTIVE_LOSSES.set(consecutive)
            
            # Calculate current balance from trades
            current_balance = 50.0  # Initial balance
            for trade in trades:
                current_balance += trade['pnl']
            CURRENT_BALANCE.set(current_balance)
            
            # Calculate drawdown
            peak_balance = 50.0
            max_drawdown = 0.0
            running_balance = 50.0
            
            for trade in trades:
                running_balance += trade['pnl']
                if running_balance > peak_balance:
                    peak_balance = running_balance
                
                current_dd = (peak_balance - running_balance) / peak_balance * 100
                if current_dd > max_drawdown:
                    max_drawdown = current_dd
            
            MAX_DRAWDOWN.set(max_drawdown)
            current_dd = (peak_balance - current_balance) / peak_balance * 100
            CURRENT_DRAWDOWN.set(current_dd)
            
            # Total return
            total_return = (current_balance - 50.0) / 50.0 * 100
            TOTAL_RETURN.set(total_return)
            
        except Exception as e:
            logger.error(f"Error updating trade metrics: {e}")
            API_ERRORS.labels(error_type='metrics_update').inc()
        finally:
            conn.close()
    
    def update_balance_metrics(self):
        """Update balance and return metrics"""
        if not self.equity_table_name:
            logger.debug("No equity table found, balance metrics calculated from trades")
            return
            
        conn = self.get_db_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            
            # Get latest equity curve point
            cursor.execute(f"""
                SELECT * FROM {self.equity_table_name} 
                ORDER BY timestamp DESC LIMIT 1
            """)
            latest = cursor.fetchone()
            
            if latest:
                CURRENT_BALANCE.set(latest['balance'])
                CURRENT_DRAWDOWN.set(latest['drawdown'])
                
                # Daily return
                cursor.execute(f"""
                    SELECT * FROM {self.equity_table_name} 
                    WHERE date(timestamp) = date('now') 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                today = cursor.fetchone()
                if today:
                    DAILY_RETURN.set(today['daily_pnl'] / 50.0 * 100)
                
                # Total return
                total_return = (latest['balance'] - 50.0) / 50.0 * 100
                TOTAL_RETURN.set(total_return)
                
                # Max drawdown from equity curve
                cursor.execute(f"SELECT MAX(drawdown) as max_dd FROM {self.equity_table_name}")
                max_dd = cursor.fetchone()
                if max_dd and max_dd['max_dd']:
                    MAX_DRAWDOWN.set(max_dd['max_dd'])
                    
        except Exception as e:
            logger.error(f"Error updating balance metrics: {e}")
            API_ERRORS.labels(error_type='balance_update').inc()
        finally:
            conn.close()
    
    def update_risk_metrics(self):
        """Update risk-related metrics"""
        if not self.risk_table_name:
            logger.debug("No risk table found, skipping risk metrics")
            return
            
        conn = self.get_db_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            
            # Count risk events
            cursor.execute(f"SELECT COUNT(*) as count FROM {self.risk_table_name}")
            risk_count = cursor.fetchone()
            if risk_count:
                RISK_KILLS.inc(risk_count['count'])
                
            # Get recent risk events
            cursor.execute(f"""
                SELECT * FROM {self.risk_table_name} 
                ORDER BY timestamp DESC LIMIT 10
            """)
            events = cursor.fetchall()
            
            for event in events:
                API_ERRORS.labels(error_type=event['event_type']).inc()
                
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
            API_ERRORS.labels(error_type='risk_update').inc()
        finally:
            conn.close()
    
    def update_system_metrics(self):
        """Update system health metrics"""
        uptime = time.time() - self.start_time
        SYSTEM_UPTIME.set(uptime)
        
        # DB connections (always 1 for SQLite)
        DB_CONNECTIONS.set(1)
    
    def collect_metrics(self):
        """Collect all metrics"""
        try:
            self.update_trade_metrics()
            self.update_balance_metrics()
            self.update_risk_metrics()
            self.update_system_metrics()
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            API_ERRORS.labels(error_type='collection').inc()
    
    def start_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            logger.info(f"Metrics available at: http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def run(self):
        """Main run loop"""
        self.running = True
        self.start_server()
        
        logger.info("Starting Elite Prometheus Exporter...")
        
        while self.running:
            try:
                self.collect_metrics()
                time.sleep(10)  # Update every 10 seconds
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)
        
        logger.info("Prometheus exporter stopped")
    
    def stop(self):
        """Stop the exporter"""
        self.running = False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Elite Prometheus Exporter')
    parser.add_argument('--db', default='shadow_trades.db', help='Database path')
    parser.add_argument('--port', type=int, default=8000, help='Metrics server port')
    
    args = parser.parse_args()
    
    exporter = ElitePrometheusExporter(db_path=args.db, port=args.port)
    
    try:
        exporter.run()
    except KeyboardInterrupt:
        logger.info("Exporter interrupted by user")
    except Exception as e:
        logger.error(f"Exporter failed: {e}")
    finally:
        exporter.stop()

if __name__ == "__main__":
    main() 