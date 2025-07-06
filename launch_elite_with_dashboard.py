#!/usr/bin/env python3
"""
Elite 100%/5% Trading System - Complete Launch Script
Starts the system with full dashboard integration and monitoring
"""

import os
import sys
import time
import asyncio
import subprocess
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

def print_launch_banner():
    """Print launch banner"""
    print("=" * 80)
    print("🏆 ELITE 100%/5% TRADING SYSTEM - COMPLETE LAUNCH")
    print("💰 Target: +100% Monthly Returns | 🛡️ Max DD: 5%")
    print("=" * 80)
    print(f"⏰ Launch time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 Starting complete system with monitoring...")
    print("=" * 80)

def check_prerequisites():
    """Check system prerequisites"""
    print("\n🔍 CHECKING PREREQUISITES:")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print("   ✅ Python version OK")
    else:
        print("   ❌ Python 3.8+ required")
        return False
    
    # Check required files
    required_files = [
        'elite_100_5_trading_system.py',
        'elite_dashboard_launcher.py',
        'risk_manager_enhanced.py',
        'deployment_config_100_5.yaml'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file} found")
        else:
            print(f"   ❌ {file} missing")
            return False
    
    # Check .env file
    if Path('.env').exists():
        print("   ✅ .env configuration found")
    else:
        print("   ⚠️  .env file not found (using defaults)")
    
    return True

def start_prometheus_exporter():
    """Start Prometheus exporter in background"""
    print("\n📊 STARTING PROMETHEUS EXPORTER:")
    
    # Check if database exists
    db_files = ['shadow_trades_tuned.db', 'shadow_trades_tuned_full.db', 'shadow_trades.db']
    db_file = None
    
    for db in db_files:
        if Path(db).exists():
            db_file = db
            break
    
    if db_file:
        print(f"   📁 Using database: {db_file}")
        
        # Start exporter
        try:
            cmd = [sys.executable, 'monitoring/exporter.py', '--db', db_file, '--port', '8001']
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("   ✅ Prometheus exporter started on port 8001")
            time.sleep(2)  # Give it time to start
            return process
        except Exception as e:
            print(f"   ⚠️  Could not start exporter: {e}")
            return None
    else:
        print("   ⚠️  No database found for metrics export")
        return None

def open_monitoring_dashboards():
    """Open monitoring dashboards"""
    print("\n🖥️  OPENING MONITORING DASHBOARDS:")
    
    def open_dashboards():
        time.sleep(3)  # Wait for services to start
        
        # Open Prometheus metrics
        prometheus_url = "http://localhost:8000/metrics"
        print(f"   🔍 Opening Prometheus: {prometheus_url}")
        webbrowser.open(prometheus_url)
        
        time.sleep(1)
        
        # Open exporter metrics if available
        exporter_url = "http://localhost:8001/metrics"
        print(f"   📊 Opening Exporter: {exporter_url}")
        webbrowser.open(exporter_url)
        
        # Try to open Grafana if available
        try:
            import requests
            grafana_url = "http://localhost:3000"
            response = requests.get(grafana_url, timeout=2)
            if response.status_code == 200:
                print(f"   📈 Opening Grafana: {grafana_url}")
                webbrowser.open(grafana_url)
        except:
            pass
    
    # Start dashboard opener in background
    dashboard_thread = threading.Thread(target=open_dashboards, daemon=True)
    dashboard_thread.start()
    
    print("   ✅ Dashboard opener started")

def display_monitoring_info():
    """Display monitoring information"""
    print("\n📊 MONITORING DASHBOARD URLS:")
    print("   🔍 Prometheus Metrics: http://localhost:8000/metrics")
    print("   📊 Exporter Metrics: http://localhost:8001/metrics")
    print("   📈 Grafana Dashboard: http://localhost:3000")
    print("   🎯 System Status: Check terminal logs")
    
    print("\n💡 MONITORING TIPS:")
    print("   • Keep Prometheus open on spare monitor")
    print("   • Refresh every 30 seconds for real-time data")
    print("   • Watch for alerts in terminal")
    print("   • Monitor drawdown gauges closely")
    
    print("\n🚨 EMERGENCY CONTROLS:")
    print("   • Ctrl+C: Graceful shutdown")
    print("   • ./emergency_revert.sh: Emergency stop")
    print("   • Check logs for halt conditions")

async def launch_elite_system():
    """Launch the Elite 100%/5% system"""
    print("\n🚀 LAUNCHING ELITE 100%/5% TRADING SYSTEM:")
    
    try:
        # Import and run the main system
        from elite_100_5_trading_system import main as elite_main
        
        print("   ✅ System modules loaded")
        print("   🎯 Starting trading cycle...")
        print("=" * 80)
        
        # Run the main system
        await elite_main()
        
    except ImportError as e:
        print(f"   ❌ Failed to import system: {e}")
        print("   💡 Check dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"   ❌ System error: {e}")
        raise

async def main():
    """Main launch function"""
    try:
        print_launch_banner()
        
        # Check prerequisites
        if not check_prerequisites():
            print("\n❌ PREREQUISITES FAILED")
            print("💡 Fix the issues above and try again")
            return
        
        # Start Prometheus exporter
        exporter_process = start_prometheus_exporter()
        
        # Open monitoring dashboards
        open_monitoring_dashboards()
        
        # Display monitoring info
        display_monitoring_info()
        
        # Launch the Elite system
        await launch_elite_system()
        
    except KeyboardInterrupt:
        print("\n🛑 SHUTDOWN REQUESTED")
        print("⏳ Closing positions and saving state...")
    except Exception as e:
        print(f"\n💥 LAUNCH FAILED: {e}")
        raise
    finally:
        print("\n👋 Elite 100%/5% System Shutdown Complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Launch cancelled by user")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1) 