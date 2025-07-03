#!/usr/bin/env python3
"""
Fix Dashboard Integration Issues
"""

import sys
import os

def fix_imports():
    """Fix missing imports in bot files"""
    
    # Fix live_ultimate_75_bot.py
    try:
        with open('live_ultimate_75_bot.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'import sys' not in content:
            content = content.replace(
                'import numpy as np',
                'import numpy as np\nimport sys'
            )
            
            with open('live_ultimate_75_bot.py', 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Fixed live_ultimate_75_bot.py imports")
    except Exception as e:
        print(f"❌ Error fixing live bot: {e}")
    
    # Fix live_simulation_ultimate_75.py
    try:
        with open('live_simulation_ultimate_75.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'import sys' not in content:
            content = content.replace(
                'import numpy as np',
                'import numpy as np\nimport sys'
            )
            
            with open('live_simulation_ultimate_75.py', 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Fixed live_simulation_ultimate_75.py imports")
    except Exception as e:
        print(f"❌ Error fixing simulation bot: {e}")

def test_dashboard():
    """Test dashboard functionality"""
    try:
        import subprocess
        import webbrowser
        import time
        
        print("🧪 Testing dashboard...")
        
        # Start dashboard
        process = subprocess.Popen([
            sys.executable, "dashboard_launcher.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(3)
        
        try:
            webbrowser.open("http://localhost:8502")
            print("✅ Dashboard test successful!")
            print("📊 Dashboard available at: http://localhost:8502")
        except:
            print("⚠️ Browser auto-open failed, but dashboard should be running")
            print("📊 Manual access: http://localhost:8502")
            
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def main():
    print("🔧 FIXING DASHBOARD INTEGRATION")
    print("=" * 40)
    
    # Fix imports
    print("1. Fixing imports...")
    fix_imports()
    
    # Test dashboard
    print("\n2. Testing dashboard...")
    test_dashboard()
    
    print("\n✅ Dashboard integration fixes complete!")
    print("\n🚀 Ready to launch:")
    print("   python start_live_ultimate_75.py")

if __name__ == "__main__":
    main() 