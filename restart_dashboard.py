#!/usr/bin/env python3
"""
Simple Dashboard Restart Script
Quick fix for dashboard issues
"""

import subprocess
import sys
import time
import os

def restart_dashboard():
    print("🔄 RESTARTING HYPERLIQUID DASHBOARD")
    print("=" * 50)
    
    try:
        # Kill any existing streamlit processes
        print("🛑 Stopping existing dashboard...")
        os.system("taskkill /f /im streamlit.exe 2>nul")
        time.sleep(2)
        
        print("🚀 Starting fresh dashboard...")
        print("📍 URL: http://localhost:8502")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 50)
        
        # Start the dashboard
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'hyperliquid_dashboard.py', 
            '--server.port=8502',
            '--server.headless=false'
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try manually: streamlit run hyperliquid_dashboard.py --server.port=8502")

if __name__ == "__main__":
    restart_dashboard() 