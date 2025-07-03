#!/usr/bin/env python3
"""
🚀 HYPERLIQUID TRADE TEST LAUNCHER
Simple launcher for testing trade execution on Hyperliquid
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'hyperliquid-python-sdk',
        'python-dotenv',
        'eth-account'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'hyperliquid-python-sdk':
                import hyperliquid
            elif package == 'python-dotenv':
                import dotenv
            elif package == 'eth-account':
                import eth_account
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing packages"""
    print(f"📦 Installing missing packages: {', '.join(packages)}")
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def check_env_file():
    """Check if .env file exists and is configured"""
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("💡 Copy hyperliquid_env_example.txt to .env and configure it")
        return False
    
    # Check if .env has the required keys
    with open('.env', 'r') as f:
        env_content = f.read()
    
    if 'your_private_key_here' in env_content:
        print("❌ .env file not configured!")
        print("💡 Replace 'your_private_key_here' with your actual private key")
        return False
    
    if 'HYPERLIQUID_PRIVATE_KEY=' not in env_content:
        print("❌ HYPERLIQUID_PRIVATE_KEY not found in .env")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🚀 HYPERLIQUID TRADE TEST LAUNCHER")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    
    print("✅ Python version OK")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"📦 Missing packages: {', '.join(missing)}")
        
        install_choice = input("Install missing packages? (y/n): ").lower()
        if install_choice == 'y':
            if not install_dependencies(missing):
                print("❌ Failed to install dependencies")
                return
        else:
            print("❌ Cannot proceed without dependencies")
            return
    
    print("✅ All dependencies available")
    
    # Check .env file
    if not check_env_file():
        return
    
    print("✅ Environment configured")
    
    # Show test information
    print("\n" + "=" * 50)
    print("🎯 TRADE EXECUTION TEST INFO")
    print("=" * 50)
    
    # Check if testnet or mainnet
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
        trade_size = os.getenv('TEST_TRADE_SIZE_USD', '10.0')
        
        print(f"🔗 Network: {'TESTNET' if testnet else 'MAINNET'}")
        print(f"💰 Trade size: ${trade_size}")
        
        if not testnet:
            print("🚨 WARNING: You're about to test on MAINNET with REAL money!")
        
    except Exception as e:
        print(f"⚠️  Could not read .env: {e}")
    
    print("\n📋 What this test will do:")
    print("1. Check your Hyperliquid connection")
    print("2. Verify your account balance")
    print("3. Open a small LONG position")
    print("4. Close the LONG position")
    print("5. Open a small SHORT position")
    print("6. Close the SHORT position")
    print("7. Verify trades in history")
    print("8. Clean up any remaining positions")
    
    # Get user confirmation
    print("\n" + "=" * 50)
    proceed = input("🚀 Ready to run the test? (y/n): ").lower()
    
    if proceed != 'y':
        print("❌ Test cancelled")
        return
    
    # Run the actual test
    print("\n🚀 STARTING TRADE EXECUTION TEST...")
    print("=" * 50)
    
    try:
        from hyperliquid_trade_execution_test import main as run_test
        results = run_test()
        
        if results:
            print("\n✅ Test completed successfully!")
        else:
            print("\n❌ Test failed - check output above")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("💡 Make sure your .env file is configured correctly")

if __name__ == "__main__":
    main() 