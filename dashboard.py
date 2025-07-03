"""
Comprehensive Dashboard Controller
Manages all dashboard functionality and real-time updates
"""

import json
import time
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from metrics_collector import MetricsCollector
from trend_analyzer import TrendAnalyzer
from advanced_backtest import AdvancedBacktester
from parameter_optimizer import ParameterOptimizer

class DashboardController:
    """Main dashboard controller managing all bot monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector, trend_analyzer: TrendAnalyzer):
        self.metrics_collector = metrics_collector
        self.trend_analyzer = trend_analyzer
        self.backtester = AdvancedBacktester()
        self.optimizer = ParameterOptimizer()
        
        # Dashboard state
        self.dashboard_state = {
            'bot_status': 'stopped',
            'last_update': None,
            'alerts': [],
            'performance_summary': {},
            'trend_analysis': {},
            'system_health': {},
            'recent_trades': [],
            'optimization_status': {},
            'backtest_results': {}
        }
        
        # Update thread
        self._update_thread = None
        self._running = False
    
    def start_monitoring(self):
        """Start dashboard monitoring"""
        if self._running:
            return
            
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
    
    def stop_monitoring(self):
        """Stop dashboard monitoring"""
        self._running = False
        if self._update_thread:
            self._update_thread.join()
    
    def _update_loop(self):
        """Main update loop for dashboard"""
        while self._running:
            try:
                self._update_dashboard_state()
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                print(f"Dashboard update error: {e}")
                time.sleep(10)
    
    def _update_dashboard_state(self):
        """Update all dashboard state"""
        try:
            # Update performance summary
            self.dashboard_state['performance_summary'] = self._get_performance_summary()
            
            # Update trend analysis
            self.dashboard_state['trend_analysis'] = self._get_trend_summary()
            
            # Update system health
            self.dashboard_state['system_health'] = self._get_system_health()
            
            # Update recent trades
            self.dashboard_state['recent_trades'] = self._get_recent_trades()
            
            # Check for alerts
            self._check_alerts()
            
            # Update timestamp
            self.dashboard_state['last_update'] = datetime.now().isoformat()
            
        except Exception as e:
            print(f"Error updating dashboard state: {e}")
    
    def _get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            metrics = self.metrics_collector.get_performance_summary()
            
            # Calculate additional metrics
            if metrics.get('latest_performance'):
                perf = metrics['latest_performance']
                
                # Calculate daily/weekly performance
                daily_trades = self.metrics_collector.get_historical_metrics('trades', 24)
                weekly_trades = self.metrics_collector.get_historical_metrics('trades', 168)
                
                daily_pnl = sum(trade.get('pnl', 0) for trade in daily_trades)
                weekly_pnl = sum(trade.get('pnl', 0) for trade in weekly_trades)
                
                return {
                    'total_pnl': perf.get('total_pnl', 0),
                    'win_rate': perf.get('win_rate', 0),
                    'profit_factor': perf.get('profit_factor', 0),
                    'max_drawdown': perf.get('max_drawdown', 0),
                    'sharpe_ratio': perf.get('sharpe_ratio', 0),
                    'total_trades': perf.get('total_trades', 0),
                    'active_positions': perf.get('active_positions', 0),
                    'daily_pnl': daily_pnl,
                    'weekly_pnl': weekly_pnl,
                    'avg_trade_duration': self._calculate_avg_trade_duration(),
                    'best_performing_hours': self._get_best_performing_hours(),
                    'risk_metrics': self._calculate_risk_metrics()
                }
            
            return {}
            
        except Exception as e:
            print(f"Error getting performance summary: {e}")
            return {}
    
    def _get_trend_summary(self) -> Dict:
        """Get trend analysis summary"""
        try:
            trend_metrics = self.metrics_collector.trend_metrics
            
            # Always return a dictionary with default values
            default_trend = {
                'current_direction': 'UNKNOWN',
                'strength': 'UNKNOWN',
                'confidence': 0.0,
                'timeframe_alignment': {},
                'momentum_indicators': {},
                'volume_analysis': {},
                'trend_duration': self._calculate_trend_duration(),
                'trend_changes_today': self._count_trend_changes_today(),
                'support_resistance': self._get_support_resistance_levels()
            }
            
            if trend_metrics:
                # Update with actual values if available
                default_trend.update({
                    'current_direction': trend_metrics.get('direction', 'UNKNOWN'),
                    'strength': trend_metrics.get('strength', 'UNKNOWN'),
                    'confidence': float(trend_metrics.get('confidence', 0.0)),
                    'timeframe_alignment': trend_metrics.get('timeframe_alignment', {}),
                    'momentum_indicators': trend_metrics.get('momentum', {}),
                    'volume_analysis': trend_metrics.get('volume_confirmation', {})
                })
            
            return default_trend
            
        except Exception as e:
            print(f"Error getting trend summary: {e}")
            # Return default values even on error
            return {
                'current_direction': 'UNKNOWN',
                'strength': 'UNKNOWN',
                'confidence': 0.0,
                'timeframe_alignment': {},
                'momentum_indicators': {},
                'volume_analysis': {},
                'trend_duration': 0,
                'trend_changes_today': 0,
                'support_resistance': {}
            }
    
    def _get_system_health(self) -> Dict:
        """Get system health summary"""
        try:
            health = self.metrics_collector.get_system_health()
            
            # Add additional health metrics
            return {
                **health,
                'uptime': self._calculate_uptime(),
                'api_latency': self._measure_api_latency(),
                'websocket_status': self._check_websocket_status(),
                'database_size': self._get_database_size(),
                'error_rate': self._calculate_error_rate(),
                'memory_trend': self._get_memory_trend()
            }
            
        except Exception as e:
            print(f"Error getting system health: {e}")
            return {}
    
    def _get_recent_trades(self) -> List[Dict]:
        """Get recent trades with analysis"""
        try:
            trades = self.metrics_collector.get_historical_metrics('trades', 6)
            
            # Enhance trades with analysis
            enhanced_trades = []
            for trade in trades[:20]:  # Last 20 trades
                enhanced_trade = {
                    **trade,
                    'duration': self._calculate_trade_duration(trade),
                    'price_movement': self._calculate_price_movement(trade),
                    'market_condition': self._get_market_condition_at_time(trade.get('timestamp'))
                }
                enhanced_trades.append(enhanced_trade)
            
            return enhanced_trades
            
        except Exception as e:
            print(f"Error getting recent trades: {e}")
            return []
    
    def _check_alerts(self):
        """Check for system alerts"""
        alerts = []
        
        try:
            # Performance alerts
            perf = self.dashboard_state.get('performance_summary', {})
            if perf.get('max_drawdown', 0) > 0.05:  # 5% drawdown
                alerts.append({
                    'type': 'warning',
                    'message': f"High drawdown detected: {perf['max_drawdown']:.2%}",
                    'timestamp': datetime.now().isoformat(),
                    'category': 'performance'
                })
            
            # Only check win rate if we have trades
            if perf.get('total_trades', 0) > 0 and perf.get('win_rate', 0) < 0.6:  # Below 60% win rate
                alerts.append({
                    'type': 'warning',
                    'message': f"Low win rate: {perf.get('win_rate', 0):.1%}",
                    'timestamp': datetime.now().isoformat(),
                    'category': 'performance'
                })
            
            # System alerts
            system = self.dashboard_state.get('system_health', {})
            if system.get('cpu_usage', 0) > 80:
                alerts.append({
                    'type': 'warning',
                    'message': f"High CPU usage: {system['cpu_usage']:.1f}%",
                    'timestamp': datetime.now().isoformat(),
                    'category': 'system'
                })
            
            if system.get('memory_usage', 0) > 80:
                alerts.append({
                    'type': 'warning',
                    'message': f"High memory usage: {system['memory_usage']:.1f}%",
                    'timestamp': datetime.now().isoformat(),
                    'category': 'system'
                })
            
            # Trend alerts
            trend = self.dashboard_state.get('trend_analysis', {})
            if trend.get('confidence', 0) < 50:
                alerts.append({
                    'type': 'info',
                    'message': f"Low trend confidence: {trend['confidence']:.1f}%",
                    'timestamp': datetime.now().isoformat(),
                    'category': 'trend'
                })
            
            # Keep only recent alerts (last 24 hours)
            cutoff = datetime.now() - timedelta(hours=24)
            self.dashboard_state['alerts'] = [
                alert for alert in alerts 
                if datetime.fromisoformat(alert['timestamp']) > cutoff
            ]
            
        except Exception as e:
            print("Error checking alerts:", str(e))
    
    def run_backtest(self, days: int = 7) -> Dict:
        """Run backtest and return results"""
        try:
            # Fetch data
            df = self.backtester.fetch_historical_data(
                symbol="SOL-USD-SWAP",
                timeframe="1m",
                days=days
            )
            
            if len(df) < 100:
                return {'error': 'Insufficient data for backtest'}
            
            # Run backtest
            results = self.backtester.run_backtest(df)
            
            # Update dashboard state
            self.dashboard_state['backtest_results'] = {
                'last_run': datetime.now().isoformat(),
                'period_days': days,
                'results': results
            }
            
            # Update metrics collector
            self.metrics_collector.update_backtest_metrics(results)
            
            return results
            
        except Exception as e:
            print(f"Error running backtest: {e}")
            return {'error': str(e)}
    
    def run_optimization(self, target_accuracy: float = 0.85) -> Dict:
        """Run parameter optimization"""
        try:
            # Fetch data for optimization
            df = self.backtester.fetch_historical_data(
                symbol="SOL-USD-SWAP",
                timeframe="1m",
                days=7
            )
            
            if len(df) < 1000:
                return {'error': 'Insufficient data for optimization'}
            
            # Run optimization
            results = self.optimizer.optimize_for_accuracy(
                df=df,
                target_accuracy=target_accuracy,
                method='differential_evolution'
            )
            
            # Update dashboard state
            self.dashboard_state['optimization_status'] = {
                'last_run': datetime.now().isoformat(),
                'target_accuracy': target_accuracy,
                'results': results
            }
            
            # Update metrics collector
            self.metrics_collector.update_backtest_metrics({
                'optimization_date': datetime.now().isoformat(),
                'optimization_results': results
            })
            
            return results
            
        except Exception as e:
            print(f"Error running optimization: {e}")
            return {'error': str(e)}
    
    def get_dashboard_data(self) -> Dict:
        """Get complete dashboard data"""
        return self.dashboard_state.copy()
    
    def get_performance_charts_data(self) -> Dict:
        """Get data for performance charts"""
        try:
            performance_history = self.metrics_collector.get_historical_metrics('performance', 24)
            trades_history = self.metrics_collector.get_historical_metrics('trades', 24)
            
            return {
                'performance_timeline': self._format_performance_timeline(performance_history),
                'pnl_distribution': self._format_pnl_distribution(trades_history),
                'hourly_performance': self._format_hourly_performance(trades_history),
                'drawdown_chart': self._format_drawdown_chart(performance_history),
                'win_rate_trend': self._format_win_rate_trend(performance_history)
            }
            
        except Exception as e:
            print(f"Error getting chart data: {e}")
            return {}
    
    # Helper methods for calculations
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in minutes"""
        # Placeholder - would need trade open/close timestamps
        return 45.0
    
    def _get_best_performing_hours(self) -> List[int]:
        """Get hours of day with best performance"""
        trades = self.metrics_collector.get_historical_metrics('trades', 168)  # 7 days
        
        hourly_pnl = {}
        for trade in trades:
            try:
                hour = datetime.fromisoformat(trade['timestamp']).hour
                hourly_pnl[hour] = hourly_pnl.get(hour, 0) + trade.get('pnl', 0)
            except:
                continue
        
        # Return top 3 hours
        sorted_hours = sorted(hourly_pnl.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, pnl in sorted_hours[:3]]
    
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk metrics"""
        trades = self.metrics_collector.get_historical_metrics('trades', 168)
        
        if not trades:
            return {}
        
        pnls = [trade.get('pnl', 0) for trade in trades]
        
        return {
            'var_95': float(pd.Series(pnls).quantile(0.05)),  # Value at Risk
            'expected_shortfall': float(pd.Series(pnls)[pd.Series(pnls) <= pd.Series(pnls).quantile(0.05)].mean()),
            'volatility': float(pd.Series(pnls).std()),
            'skewness': float(pd.Series(pnls).skew()) if len(pnls) > 2 else 0
        }
    
    def _calculate_trend_duration(self) -> int:
        """Calculate current trend duration in minutes"""
        # Placeholder - would track trend changes
        return 120
    
    def _count_trend_changes_today(self) -> int:
        """Count trend direction changes today"""
        # Placeholder - would track trend history
        return 3
    
    def _get_support_resistance_levels(self) -> Dict:
        """Get current support and resistance levels"""
        # Placeholder - would calculate from price data
        return {
            'support': 180.50,
            'resistance': 185.75,
            'strength': 'moderate'
        }
    
    def _calculate_uptime(self) -> str:
        """Calculate bot uptime"""
        # Placeholder - would track start time
        return "2h 45m"
    
    def _measure_api_latency(self) -> float:
        """Measure API response latency"""
        # Placeholder - would ping API
        return 45.2
    
    def _check_websocket_status(self) -> str:
        """Check WebSocket connection status"""
        # Placeholder - would check actual connection
        return "connected"
    
    def _get_database_size(self) -> float:
        """Get database size in MB"""
        try:
            import os
            if os.path.exists(self.metrics_collector.db_path):
                return os.path.getsize(self.metrics_collector.db_path) / (1024 * 1024)
        except:
            pass
        return 0.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage"""
        # Placeholder - would track errors
        return 0.5
    
    def _get_memory_trend(self) -> str:
        """Get memory usage trend"""
        history = self.metrics_collector.get_historical_metrics('system', 1)
        if len(history) >= 2:
            current = history[0].get('memory_usage', 0)
            previous = history[1].get('memory_usage', 0)
            if current > previous:
                return "increasing"
            elif current < previous:
                return "decreasing"
        return "stable"
    
    def _calculate_trade_duration(self, trade: Dict) -> int:
        """Calculate trade duration in minutes"""
        # Placeholder - would need close timestamp
        return 30
    
    def _calculate_price_movement(self, trade: Dict) -> float:
        """Calculate price movement during trade"""
        # Placeholder - would need market data
        return 0.25
    
    def _get_market_condition_at_time(self, timestamp: str) -> str:
        """Get market condition at specific time"""
        # Placeholder - would analyze market data
        return "trending"
    
    def _format_performance_timeline(self, data: List[Dict]) -> List[Dict]:
        """Format performance data for timeline chart"""
        return [
            {
                'timestamp': item.get('timestamp'),
                'total_pnl': item.get('total_pnl', 0),
                'win_rate': item.get('win_rate', 0)
            }
            for item in data
        ]
    
    def _format_pnl_distribution(self, trades: List[Dict]) -> List[float]:
        """Format PnL distribution for histogram"""
        return [trade.get('pnl', 0) for trade in trades]
    
    def _format_hourly_performance(self, trades: List[Dict]) -> Dict:
        """Format hourly performance data"""
        hourly_data = {}
        for trade in trades:
            try:
                hour = datetime.fromisoformat(trade['timestamp']).hour
                hourly_data[hour] = hourly_data.get(hour, 0) + trade.get('pnl', 0)
            except:
                continue
        return hourly_data
    
    def _format_drawdown_chart(self, data: List[Dict]) -> List[Dict]:
        """Format drawdown data for chart"""
        return [
            {
                'timestamp': item.get('timestamp'),
                'drawdown': item.get('max_drawdown', 0)
            }
            for item in data
        ]
    
    def _format_win_rate_trend(self, data: List[Dict]) -> List[Dict]:
        """Format win rate trend data"""
        return [
            {
                'timestamp': item.get('timestamp'),
                'win_rate': item.get('win_rate', 0)
            }
            for item in data
        ] 