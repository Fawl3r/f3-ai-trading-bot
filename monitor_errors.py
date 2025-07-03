#!/usr/bin/env python3
"""
Error monitoring script for OKX Trading Bot
Monitors the bot output and identifies any remaining errors
"""

import subprocess
import time
import requests
import psutil
from datetime import datetime

class BotMonitor:
    def __init__(self):
        self.dashboard_url = "http://127.0.0.1:5000"
        self.error_patterns = [
            "Error collecting system metrics:",
            "Error checking alerts:",
            "Error fetching data:",
            "Traceback",
            "SyntaxError",
            "TypeError",
            "ValueError",
            "KeyError"
        ]
        
    def check_dashboard_health(self):
        """Check if dashboard is responding"""
        try:
            response = requests.get(self.dashboard_url, timeout=5)
            return {
                'status': 'OK' if response.status_code == 200 else 'ERROR',
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'response_time': None
            }
    
    def check_api_endpoints(self):
        """Check specific API endpoints"""
        endpoints = [
            '/api/metrics',
            '/api/performance',
            '/api/system',
            '/api/charts/performance',
            '/api/charts/system',
            '/api/charts/trades'
        ]
        
        results = {}
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.dashboard_url}{endpoint}", timeout=5)
                results[endpoint] = {
                    'status': 'OK' if response.status_code == 200 else 'ERROR',
                    'status_code': response.status_code
                }
            except Exception as e:
                results[endpoint] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        return results
    
    def check_system_resources(self):
        """Check system resource usage"""
        try:
            # Find Python processes
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                if proc.info['name'] == 'python.exe':
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024
                    })
            
            return {
                'system_cpu': psutil.cpu_percent(),
                'system_memory': psutil.virtual_memory().percent,
                'python_processes': python_processes
            }
        except Exception as e:
            return {'error': str(e)}
    
    def check_database_connectivity(self):
        """Check database operations"""
        try:
            from metrics_collector import MetricsCollector
            collector = MetricsCollector()
            
            # Test basic operations
            collector._collect_system_metrics()
            metrics = collector.get_real_time_metrics()
            
            return {
                'status': 'OK',
                'metrics_available': len(metrics) > 0,
                'system_metrics': bool(metrics.get('system')),
                'trading_metrics': bool(metrics.get('trading'))
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def run_comprehensive_test(self):
        """Run all health checks"""
        print("🔍 OKX Trading Bot - Comprehensive Health Check")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. Dashboard Health
        print("📊 Dashboard Health Check:")
        dashboard_health = self.check_dashboard_health()
        if dashboard_health['status'] == 'OK':
            print(f"  ✅ Dashboard responding (Status: {dashboard_health['status_code']}, "
                  f"Response time: {dashboard_health['response_time']:.3f}s)")
        else:
            print(f"  ❌ Dashboard error: {dashboard_health.get('error', 'Unknown error')}")
        print()
        
        # 2. API Endpoints
        print("🔌 API Endpoints Check:")
        api_results = self.check_api_endpoints()
        for endpoint, result in api_results.items():
            status_icon = "✅" if result['status'] == 'OK' else "❌"
            print(f"  {status_icon} {endpoint}: {result['status']}")
        print()
        
        # 3. System Resources
        print("🖥️ System Resources:")
        resources = self.check_system_resources()
        if 'error' not in resources:
            print(f"  📊 System CPU: {resources['system_cpu']:.1f}%")
            print(f"  💾 System Memory: {resources['system_memory']:.1f}%")
            print(f"  🐍 Python Processes: {len(resources['python_processes'])}")
            for proc in resources['python_processes']:
                print(f"    - PID {proc['pid']}: CPU {proc['cpu_percent']:.1f}%, "
                      f"Memory {proc['memory_mb']:.1f}MB")
        else:
            print(f"  ❌ Error checking resources: {resources['error']}")
        print()
        
        # 4. Database Connectivity
        print("🗄️ Database Operations:")
        db_health = self.check_database_connectivity()
        if db_health['status'] == 'OK':
            print(f"  ✅ Database operations working")
            print(f"  📈 System metrics: {'✅' if db_health['system_metrics'] else '❌'}")
            print(f"  💹 Trading metrics: {'✅' if db_health['trading_metrics'] else '❌'}")
        else:
            print(f"  ❌ Database error: {db_health['error']}")
        print()
        
        # 5. Overall Status
        all_checks = [
            dashboard_health['status'] == 'OK',
            all(r['status'] == 'OK' for r in api_results.values()),
            'error' not in resources,
            db_health['status'] == 'OK'
        ]
        
        overall_status = "✅ ALL SYSTEMS OPERATIONAL" if all(all_checks) else "⚠️ SOME ISSUES DETECTED"
        print("🎯 Overall Status:")
        print(f"  {overall_status}")
        
        return all(all_checks)

def main():
    monitor = BotMonitor()
    
    try:
        # Run comprehensive test
        all_good = monitor.run_comprehensive_test()
        
        if all_good:
            print("\n🚀 Bot is running perfectly with no errors detected!")
            print("📊 Dashboard: http://127.0.0.1:5000")
        else:
            print("\n⚠️  Some issues were detected. Check the details above.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

if __name__ == "__main__":
    main() 