#!/usr/bin/env python3
"""View Live Bot Output"""

import subprocess
import sys

def main():
    print("🤖 Starting Hyperliquid Trading Bot with Live Output...")
    print("📊 Dashboard: http://localhost:8503")
    print("🔍 Watching for confidence levels > 0.0%...")
    print("=" * 60)
    
    try:
        process = subprocess.Popen([
            sys.executable, "hyperliquid_opportunity_hunter.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
           universal_newlines=True, bufsize=1)
        
        print("🚀 Bot started! Press Ctrl+C to stop...")
        print("=" * 60)
        
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                if 'confidence' in line.lower():
                    print(f"🔥 {line}")
                elif 'analyzing' in line.lower():
                    print(f"📊 {line}")
                elif 'error' in line.lower():
                    print(f"❌ {line}")
                else:
                    print(f"   {line}")
                    
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        if 'process' in locals():
            process.terminate()

if __name__ == "__main__":
    main() 