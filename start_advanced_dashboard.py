#!/usr/bin/env python3
"""
Advanced Dashboard Launcher
Easy startup script for the advanced trading dashboard
"""

import os
import sys
import subprocess
from datetime import datetime

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install missing requirements"""
    print("📦 Installing required packages...")
    
    requirements = [
        'streamlit>=1.28.0',
        'plotly>=5.15.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0'
    ]
    
    for req in requirements:
        print(f"   Installing {req}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', req], 
                      capture_output=True, text=True)
    
    print("✅ All packages installed!")

def main():
    """Launch the advanced dashboard"""
    
    print("🏆" * 80)
    print("📊 ADVANCED TRADING DASHBOARD LAUNCHER")
    print("🚀 Real-time Analytics & Comprehensive Trading Metrics")
    print("🏆" * 80)
    
    print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check requirements
    missing = check_requirements()
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        install_choice = input("📦 Install missing packages? (y/n): ").lower()
        
        if install_choice == 'y':
            install_requirements()
        else:
            print("❌ Cannot start dashboard without required packages")
            input("Press Enter to exit...")
            return
    
    print("🔧 Initializing advanced dashboard...")
    
    try:
        # Check if advanced_dashboard.py exists
        if not os.path.exists('advanced_dashboard.py'):
            print("❌ advanced_dashboard.py not found!")
            print("💡 Make sure the file is in the current directory")
            input("Press Enter to exit...")
            return
        
        print("✅ Dashboard file found")
        print("🚀 Starting Streamlit dashboard...")
        print("\n" + "="*80)
        print("🌐 DASHBOARD WILL OPEN IN YOUR WEB BROWSER")
        print("📍 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C in this window to stop the dashboard")
        print("="*80)
        
        # Launch Streamlit dashboard
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'advanced_dashboard.py', 
               '--server.port=8501', '--server.headless=false']
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   • Make sure advanced_dashboard.py exists")
        print("   • Check that all required packages are installed")
        print("   • Try running: pip install streamlit plotly pandas numpy")
        print("   • Manual start: streamlit run advanced_dashboard.py")
    
    print(f"\n⏰ Session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    input("🔥 Press Enter to exit...")

if __name__ == "__main__":
    main() 