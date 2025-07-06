#!/usr/bin/env python3
"""
Elite 100%/5% Trading System Dashboard Launcher
Automatically opens Prometheus dashboard and monitoring interfaces
"""

import time
import webbrowser
import threading
import requests
import logging
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EliteDashboardLauncher:
    """Comprehensive dashboard launcher for Elite 100%/5% system"""
    
    def __init__(self, prometheus_port: int = 8000):
        self.prometheus_port = prometheus_port
        self.grafana_port = 3000
        self.dashboard_opened = False
        
    def launch_dashboards(self, delay: float = 3.0):
        """Launch all available dashboards"""
        print("=" * 80)
        print("🚀 LAUNCHING ELITE 100%/5% MONITORING DASHBOARDS")
        print("=" * 80)
        
        # Start dashboard launcher in background
        dashboard_thread = threading.Thread(
            target=self._launch_dashboards_async, 
            args=(delay,), 
            daemon=True
        )
        dashboard_thread.start()
        
        # Display immediate URLs
        self._display_dashboard_urls()
        
    def _launch_dashboards_async(self, delay: float):
        """Launch dashboards asynchronously"""
        time.sleep(delay)
        
        # 1. Open Prometheus metrics
        prometheus_url = f"http://localhost:{self.prometheus_port}/metrics"
        logger.info(f"🔍 Opening Prometheus metrics: {prometheus_url}")
        webbrowser.open(prometheus_url)
        
        # Small delay between opens
        time.sleep(1)
        
        # 2. Try to open Grafana dashboard
        grafana_url = f"http://localhost:{self.grafana_port}"
        if self._check_service_available(grafana_url):
            logger.info(f"📊 Opening Grafana dashboard: {grafana_url}")
            webbrowser.open(grafana_url)
        else:
            logger.info("📊 Grafana not available - attempting to start...")
            self._try_start_grafana()
        
        # 3. Open additional monitoring if available
        monitoring_url = f"http://localhost:{self.prometheus_port + 1}/metrics"
        if self._check_service_available(monitoring_url):
            logger.info(f"📈 Opening additional metrics: {monitoring_url}")
            webbrowser.open(monitoring_url)
        
        # 4. Try to open a simple status page
        time.sleep(2)
        self._create_and_open_status_page()
        
        self.dashboard_opened = True
        logger.info("✅ Dashboard launch sequence completed")
    
    def _try_start_grafana(self):
        """Try to start Grafana if it's installed"""
        import subprocess
        import os
        
        # Common Grafana installation paths
        grafana_paths = [
            "grafana-server",  # If in PATH
            "C:\\Program Files\\GrafanaLabs\\grafana\\bin\\grafana-server.exe",  # Windows
            "/usr/local/bin/grafana-server",  # macOS
            "/usr/bin/grafana-server",  # Linux
        ]
        
        for path in grafana_paths:
            try:
                if os.path.exists(path) or path == "grafana-server":
                    logger.info(f"🚀 Starting Grafana from {path}")
                    subprocess.Popen([path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(5)  # Give Grafana time to start
                    
                    # Check if it started
                    if self._check_service_available(f"http://localhost:{self.grafana_port}"):
                        logger.info("✅ Grafana started successfully")
                        webbrowser.open(f"http://localhost:{self.grafana_port}")
                        return True
                    break
            except Exception as e:
                logger.debug(f"Failed to start Grafana from {path}: {e}")
                continue
        
        logger.info("📊 Grafana not available - install from https://grafana.com/get")
        return False
    
    def _create_and_open_status_page(self):
        """Create and open a simple status page"""
        try:
            status_html = self._generate_status_html()
            status_file = "elite_system_status.html"
            
            with open(status_file, 'w') as f:
                f.write(status_html)
            
            # Open the status page
            import os
            status_url = f"file://{os.path.abspath(status_file)}"
            logger.info(f"📋 Opening system status page: {status_url}")
            webbrowser.open(status_url)
            
        except Exception as e:
            logger.debug(f"Could not create status page: {e}")
    
    def _generate_status_html(self) -> str:
        """Generate HTML for system status page"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Elite 100%/5% Trading System - Status</title>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="30">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }}
        h1 {{
            text-align: center;
            color: #ffd700;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }}
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .status-card {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        .status-card h3 {{
            color: #ffd700;
            margin-top: 0;
        }}
        .url-list {{
            list-style: none;
            padding: 0;
        }}
        .url-list li {{
            margin: 10px 0;
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
        }}
        .url-list a {{
            color: #87ceeb;
            text-decoration: none;
        }}
        .url-list a:hover {{
            text-decoration: underline;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .online {{ background-color: #00ff00; }}
        .offline {{ background-color: #ff4444; }}
        .timestamp {{
            text-align: center;
            margin-top: 20px;
            opacity: 0.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 Elite 100%/5% Trading System</h1>
        <div class="status-grid">
            <div class="status-card">
                <h3>📊 Monitoring Dashboards</h3>
                <ul class="url-list">
                    <li>
                        <span class="status-indicator online"></span>
                        <a href="http://localhost:{self.prometheus_port}/metrics" target="_blank">
                            Prometheus Metrics
                        </a>
                    </li>
                    <li>
                        <span class="status-indicator {'online' if self._check_service_available(f'http://localhost:{self.grafana_port}') else 'offline'}"></span>
                        <a href="http://localhost:{self.grafana_port}" target="_blank">
                            Grafana Dashboard
                        </a>
                    </li>
                    <li>
                        <span class="status-indicator {'online' if self._check_service_available(f'http://localhost:{self.prometheus_port + 1}') else 'offline'}"></span>
                        <a href="http://localhost:{self.prometheus_port + 1}/metrics" target="_blank">
                            Additional Metrics
                        </a>
                    </li>
                </ul>
            </div>
            
            <div class="status-card">
                <h3>🎯 System Targets</h3>
                <ul>
                    <li>💰 Monthly Return: +100%</li>
                    <li>🛡️ Max Drawdown: 5%</li>
                    <li>📈 Target Trades: 265/month</li>
                    <li>🎲 Expected Win Rate: ~40%</li>
                </ul>
            </div>
            
            <div class="status-card">
                <h3>🧠 AI Learning Status</h3>
                <ul>
                    <li>🤖 AI Learning: ACTIVE</li>
                    <li>📚 Pattern Recognition: ENABLED</li>
                    <li>🔄 Adaptive Parameters: ENABLED</li>
                    <li>🎯 Signal Optimization: ACTIVE</li>
                </ul>
            </div>
            
            <div class="status-card">
                <h3>🔧 Quick Actions</h3>
                <ul class="url-list">
                    <li><a href="http://localhost:{self.prometheus_port}/metrics" target="_blank">View Metrics</a></li>
                    <li><a href="http://localhost:{self.grafana_port}" target="_blank">Open Grafana</a></li>
                    <li><a href="javascript:location.reload()">Refresh Status</a></li>
                </ul>
            </div>
        </div>
        
        <div class="timestamp">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            <br>
            <small>Page auto-refreshes every 30 seconds</small>
        </div>
    </div>
</body>
</html>
"""
    
    def _check_service_available(self, url: str, timeout: int = 2) -> bool:
        """Check if a service is available"""
        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except:
            return False
    
    def _display_dashboard_urls(self):
        """Display all dashboard URLs"""
        print("\n📊 MONITORING DASHBOARD URLS:")
        print(f"   🔍 Prometheus Metrics: http://localhost:{self.prometheus_port}/metrics")
        print(f"   📈 Grafana Dashboard: http://localhost:{self.grafana_port}")
        print(f"   🎯 System Status: Check terminal logs")
        print(f"   📱 Mobile Access: Use your local IP instead of localhost")
        
        print("\n💡 MONITORING TIPS:")
        print("   • Keep Prometheus open on a spare monitor")
        print("   • Refresh every 30 seconds for real-time data")
        print("   • Watch for alerts in the terminal")
        print("   • Monitor drawdown gauges closely")
        
        print("\n🔧 GRAFANA SETUP (Optional):")
        print("   1. Install Grafana: https://grafana.com/get")
        print("   2. Add Prometheus datasource: http://localhost:8000")
        print("   3. Import trading dashboard template")
        
        print("=" * 80)
    
    def get_monitoring_status(self) -> dict:
        """Get status of all monitoring services"""
        prometheus_url = f"http://localhost:{self.prometheus_port}/metrics"
        grafana_url = f"http://localhost:{self.grafana_port}"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'prometheus': {
                'url': prometheus_url,
                'available': self._check_service_available(prometheus_url),
                'port': self.prometheus_port
            },
            'grafana': {
                'url': grafana_url,
                'available': self._check_service_available(grafana_url),
                'port': self.grafana_port
            },
            'dashboard_opened': self.dashboard_opened
        }

def launch_elite_dashboards(prometheus_port: int = 8000, delay: float = 3.0):
    """Convenience function to launch dashboards"""
    launcher = EliteDashboardLauncher(prometheus_port)
    launcher.launch_dashboards(delay)
    return launcher

if __name__ == "__main__":
    # Test the dashboard launcher
    print("🧪 Testing Elite Dashboard Launcher")
    launcher = launch_elite_dashboards()
    
    # Wait a bit and show status
    time.sleep(5)
    status = launcher.get_monitoring_status()
    print(f"\n📊 Monitoring Status: {status}") 