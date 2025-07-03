"""
Centralized Metrics Collection System
Collects and aggregates metrics from all bot components
"""

import json
import time
import threading
import psutil
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import sqlite3
import os

class MetricsCollector:
    """Centralized metrics collection and storage"""
    
    def __init__(self, db_path: str = "bot_metrics.db"):
        self.db_path = db_path
        self.metrics_buffer = deque(maxlen=10000)
        self.real_time_metrics = {}
        self.system_metrics = {}
        self.trading_metrics = {}
        self.backtest_metrics = {}
        self.strategy_metrics = {}
        self.trend_metrics = {}
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._collector_thread = None
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    side TEXT,
                    size REAL,
                    price REAL,
                    pnl REAL,
                    status TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_pnl REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    total_trades INTEGER,
                    active_positions INTEGER
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_latency REAL,
                    api_status TEXT,
                    websocket_status TEXT
                )
            ''')
    
    def start_collection(self):
        """Start metrics collection in background thread"""
        if self._running:
            return
            
        self._running = True
        self._collector_thread = threading.Thread(target=self._collection_loop)
        self._collector_thread.daemon = True
        self._collector_thread.start()
    
    def stop_collection(self):
        """Stop metrics collection"""
        self._running = False
        if self._collector_thread:
            self._collector_thread.join()
    
    def _collection_loop(self):
        """Main collection loop running in background"""
        while self._running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Store metrics to database
                self._store_metrics()
                
                time.sleep(1)  # Collect every second
                
            except Exception as e:
                print(f"Error in metrics collection: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Try different disk paths for Windows compatibility
            disk_usage = 0.0
            disk_free = 0.0
            try:
                disk = psutil.disk_usage('/')  # Try root first
                disk_usage = float(disk.percent) if disk.percent is not None else 0.0
                disk_free = float(disk.free / (1024**3)) if disk.free is not None else 0.0
            except:
                try:
                    disk = psutil.disk_usage('.')  # Try current directory
                    disk_usage = float(disk.percent) if disk.percent is not None else 0.0
                    disk_free = float(disk.free / (1024**3)) if disk.free is not None else 0.0
                except:
                    disk_usage = 0.0
                    disk_free = 0.0
            
            with self._lock:
                self.system_metrics.update({
                    'timestamp': datetime.now(),
                    'cpu_usage': float(cpu_percent) if cpu_percent is not None else 0.0,
                    'memory_usage': float(memory.percent) if memory.percent is not None else 0.0,
                    'memory_available': float(memory.available / (1024**3)) if memory.available is not None else 0.0,  # GB
                    'disk_usage': disk_usage,
                    'disk_free': disk_free,
                })
                
        except Exception as e:
            # Use str() to avoid format errors
            print("Error collecting system metrics:", str(e))
    
    def update_trading_metrics(self, metrics: Dict):
        """Update trading performance metrics"""
        with self._lock:
            self.trading_metrics.update({
                'timestamp': datetime.now(),
                **metrics
            })
    
    def update_strategy_metrics(self, metrics: Dict):
        """Update strategy-specific metrics"""
        with self._lock:
            self.strategy_metrics.update({
                'timestamp': datetime.now(),
                **metrics
            })
    
    def update_trend_metrics(self, metrics: Dict):
        """Update trend analysis metrics"""
        with self._lock:
            self.trend_metrics.update({
                'timestamp': datetime.now(),
                **metrics
            })
    
    def update_backtest_metrics(self, metrics: Dict):
        """Update backtesting metrics"""
        with self._lock:
            self.backtest_metrics.update({
                'timestamp': datetime.now(),
                **metrics
            })
    
    def log_trade(self, trade_data: Dict):
        """Log individual trade"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO trades (symbol, side, size, price, pnl, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('symbol'),
                trade_data.get('side'),
                trade_data.get('size'),
                trade_data.get('price'),
                trade_data.get('pnl'),
                trade_data.get('status'),
                json.dumps(trade_data.get('metadata', {}))
            ))
    
    def _store_metrics(self):
        """Store current metrics to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                timestamp = datetime.now()
                
                # Store system metrics
                if self.system_metrics:
                    try:
                        cpu_val = self.system_metrics.get('cpu_usage', 0)
                        mem_val = self.system_metrics.get('memory_usage', 0)
                        disk_val = self.system_metrics.get('disk_usage', 0)
                        
                        # Ensure all values are valid floats
                        def safe_float(val):
                            try:
                                result = float(val) if val is not None else 0.0
                                # Handle NaN and infinity
                                if result != result or result == float('inf') or result == float('-inf'):
                                    return 0.0
                                return result
                            except (ValueError, TypeError):
                                return 0.0
                        
                        cpu_val = safe_float(cpu_val)
                        mem_val = safe_float(mem_val)
                        disk_val = safe_float(disk_val)
                        
                        conn.execute('''
                            INSERT INTO system_health (cpu_usage, memory_usage, disk_usage, network_latency, api_status, websocket_status)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (cpu_val, mem_val, disk_val, 0.0, 'Unknown', 'Unknown'))
                    except Exception as e:
                        print("Error storing system metrics:", str(e))
                
                # Store performance metrics
                if self.trading_metrics:
                    try:
                        def safe_float(val):
                            try:
                                result = float(val) if val is not None else 0.0
                                if result != result or result == float('inf') or result == float('-inf'):
                                    return 0.0
                                return result
                            except (ValueError, TypeError):
                                return 0.0
                        
                        conn.execute('''
                            INSERT INTO performance (total_pnl, win_rate, profit_factor, 
                                                   max_drawdown, sharpe_ratio, total_trades, active_positions)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            safe_float(self.trading_metrics.get('total_pnl', 0)),
                            safe_float(self.trading_metrics.get('win_rate', 0)),
                            safe_float(self.trading_metrics.get('profit_factor', 0)),
                            safe_float(self.trading_metrics.get('max_drawdown', 0)),
                            safe_float(self.trading_metrics.get('sharpe_ratio', 0)),
                            int(self.trading_metrics.get('total_trades', 0)),
                            int(self.trading_metrics.get('active_positions', 0))
                        ))
                    except Exception as e:
                        print("Error storing performance metrics:", str(e))
                
        except Exception as e:
            print("Error storing metrics:", str(e))
    
    def get_real_time_metrics(self) -> Dict:
        """Get current real-time metrics"""
        with self._lock:
            # Helper function to serialize datetime objects and handle nested data
            def serialize_data(data):
                if isinstance(data, datetime):
                    return data.isoformat()
                elif isinstance(data, dict):
                    return {key: serialize_data(value) for key, value in data.items()}
                elif isinstance(data, list):
                    return [serialize_data(item) for item in data]
                else:
                    return data
            
            return {
                'system': serialize_data(self.system_metrics.copy()),
                'trading': serialize_data(self.trading_metrics.copy()),
                'strategy': serialize_data(self.strategy_metrics.copy()),
                'trend': serialize_data(self.trend_metrics.copy()),
                'backtest': serialize_data(self.backtest_metrics.copy()),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_historical_metrics(self, metric_type: str, hours: int = 24) -> List[Dict]:
        """Get historical metrics from database"""
        since = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            if metric_type == 'performance':
                cursor = conn.execute('''
                    SELECT * FROM performance 
                    WHERE timestamp > ? 
                    ORDER BY timestamp DESC
                ''', (since,))
                
            elif metric_type == 'system':
                cursor = conn.execute('''
                    SELECT * FROM system_health 
                    WHERE timestamp > ? 
                    ORDER BY timestamp DESC
                ''', (since,))
                
            elif metric_type == 'trades':
                cursor = conn.execute('''
                    SELECT * FROM trades 
                    WHERE timestamp > ? 
                    ORDER BY timestamp DESC
                ''', (since,))
            else:
                return []
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        with sqlite3.connect(self.db_path) as conn:
            # Get latest performance data
            latest_perf = conn.execute('''
                SELECT * FROM performance 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''').fetchone()
            
            # Get trade statistics
            trade_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades
                WHERE timestamp > datetime('now', '-24 hours')
            ''').fetchone()
            
            # Get daily PnL
            daily_pnl = conn.execute('''
                SELECT SUM(pnl) as daily_pnl
                FROM trades
                WHERE DATE(timestamp) = DATE('now')
            ''').fetchone()
            
            return {
                'latest_performance': dict(zip([
                    'id', 'timestamp', 'total_pnl', 'win_rate', 'profit_factor',
                    'max_drawdown', 'sharpe_ratio', 'total_trades', 'active_positions'
                ], latest_perf)) if latest_perf else {},
                'trade_statistics': dict(zip([
                    'total_trades', 'avg_pnl', 'winning_trades', 'best_trade', 'worst_trade'
                ], trade_stats)) if trade_stats else {},
                'daily_pnl': daily_pnl[0] if daily_pnl and daily_pnl[0] else 0
            }
    
    def get_system_health(self) -> Dict:
        """Get current system health status"""
        with self._lock:
            cpu_status = "Good"
            if self.system_metrics.get('cpu_usage', 0) > 80:
                cpu_status = "High"
            elif self.system_metrics.get('cpu_usage', 0) > 60:
                cpu_status = "Medium"
            
            memory_status = "Good"
            if self.system_metrics.get('memory_usage', 0) > 80:
                memory_status = "High"
            elif self.system_metrics.get('memory_usage', 0) > 60:
                memory_status = "Medium"
            
            return {
                'overall_status': 'Healthy' if cpu_status == "Good" and memory_status == "Good" else 'Warning',
                'cpu_status': cpu_status,
                'memory_status': memory_status,
                'metrics': self.system_metrics.copy()
            }
    
    def get_trading_dashboard_data(self) -> Dict:
        """Get comprehensive data for trading dashboard"""
        performance = self.get_performance_summary()
        system_health = self.get_system_health()
        real_time = self.get_real_time_metrics()
        
        return {
            'performance': performance,
            'system_health': system_health,
            'real_time': real_time,
            'charts': {
                'performance_history': self.get_historical_metrics('performance', 24),
                'system_history': self.get_historical_metrics('system', 24),
                'recent_trades': self.get_historical_metrics('trades', 6)
            }
        }
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file"""
        data = self.get_trading_dashboard_data()
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Export performance history to CSV
            df = pd.DataFrame(data['charts']['performance_history'])
            df.to_csv(filepath, index=False)
        
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from database"""
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff,))
            conn.execute('DELETE FROM trades WHERE timestamp < ?', (cutoff,))
            conn.execute('DELETE FROM performance WHERE timestamp < ?', (cutoff,))
            conn.execute('DELETE FROM system_health WHERE timestamp < ?', (cutoff,))
    
    def get_latest_trading_metrics(self) -> Dict:
        """Get latest trading metrics"""
        with self._lock:
            return self.trading_metrics.copy()
    
    def get_trend_summary(self) -> Dict:
        """Get trend analysis summary"""
        with self._lock:
            trend_data = self.trend_metrics.copy()
            
            # Convert direction enum to string if needed
            direction = trend_data.get('direction', 'unknown')
            if hasattr(direction, 'value'):
                direction = direction.value
            
            return {
                'current_direction': direction,
                'confidence': trend_data.get('confidence', 0),
                'strength': trend_data.get('strength', 'unknown'),
                'current_price': trend_data.get('current_price', 0),
                'price_change_24h': trend_data.get('price_change_24h', 0)
            } 