#!/usr/bin/env python3
"""
Clean restart script for OKX Trading Bot
Stops all current processes and restarts with fixes
"""

import subprocess
import time
import sys
import os

def kill_python_processes():
    """Kill existing Python processes"""
    try:
        # Kill Python processes on Windows
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                      capture_output=True, text=True)
        subprocess.run(['taskkill', '/F', '/IM', 'pythonw.exe'], 
                      capture_output=True, text=True)
        print("✅ Stopped existing Python processes")
        time.sleep(2)
    except Exception as e:
        print(f"Note: {e}")

def cleanup_database():
    """Clean up database if needed"""
    try:
        if os.path.exists('bot_metrics.db'):
            os.remove('bot_metrics.db')
            print("✅ Cleaned up old database")
    except Exception as e:
        print(f"Note: Could not clean database - {e}")

def main():
    print("🔧 OKX Trading Bot - Clean Restart")
    print("=" * 50)
    
    # Step 1: Kill existing processes
    print("⏹️  Stopping existing processes...")
    kill_python_processes()
    
    # Step 2: Clean database (optional)
    response = input("Clean database? (y/N): ").lower()
    if response == 'y':
        cleanup_database()
    
    # Step 3: Restart bot
    print("🚀 Starting bot in 3 seconds...")
    time.sleep(3)
    
    try:
        # Start the dashboard
        subprocess.run([sys.executable, 'start_dashboard.py'], input="1\n", text=True)
    except KeyboardInterrupt:
        print("\n⏹️  Restart cancelled")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")

if __name__ == "__main__":
    main() 