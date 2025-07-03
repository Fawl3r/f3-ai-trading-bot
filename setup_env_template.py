#!/usr/bin/env python3
"""
.ENV SETUP HELPER
Shows the correct format for your .env file
"""

def show_env_template():
    print("=" * 70)
    print("YOUR .ENV FILE SHOULD LOOK EXACTLY LIKE THIS:")
    print("=" * 70)
    print()
    
    template = """# OKX API Credentials (KEEP THESE SECURE!)
# Get these from: https://www.okx.com/account/my-api
# Enable ONLY 'Trade' and 'Read' permissions (NOT Withdraw!)

OKX_API_KEY=your_actual_api_key_here
OKX_SECRET_KEY=your_actual_secret_key_here  
OKX_PASSPHRASE=your_actual_passphrase_here

# Optional: Override sandbox mode
# SANDBOX_MODE=true"""
    
    print(template)
    print()
    print("=" * 70)
    print("IMPORTANT NOTES:")
    print("=" * 70)
    print("1. NO SPACES around the = sign")
    print("2. NO QUOTES around the values") 
    print("3. Replace 'your_actual_...' with real credentials")
    print("4. Keep this file PRIVATE and SECURE")
    print("5. NEVER share your .env file with anyone")
    print()
    print("EXAMPLE with fake credentials:")
    print("OKX_API_KEY=a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6")
    print("OKX_SECRET_KEY=A1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R8S9T0")
    print("OKX_PASSPHRASE=MySecretPass123")
    print()

def check_current_env():
    """Check if .env file exists and has correct format"""
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    
    load_dotenv()
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        return False
    
    api_key = os.getenv('OKX_API_KEY', '')
    secret_key = os.getenv('OKX_SECRET_KEY', '')
    passphrase = os.getenv('OKX_PASSPHRASE', '')
    
    print("=" * 50)
    print("CHECKING YOUR .ENV FILE:")
    print("=" * 50)
    
    if api_key and api_key != 'your_actual_api_key_here':
        print(f"✅ OKX_API_KEY: Found (length: {len(api_key)})")
    else:
        print("❌ OKX_API_KEY: Missing or placeholder")
        
    if secret_key and secret_key != 'your_actual_secret_key_here':
        print(f"✅ OKX_SECRET_KEY: Found (length: {len(secret_key)})")
    else:
        print("❌ OKX_SECRET_KEY: Missing or placeholder")
        
    if passphrase and passphrase != 'your_actual_passphrase_here':
        print(f"✅ OKX_PASSPHRASE: Found (length: {len(passphrase)})")
    else:
        print("❌ OKX_PASSPHRASE: Missing or placeholder")
    
    all_good = (api_key and api_key != 'your_actual_api_key_here' and
                secret_key and secret_key != 'your_actual_secret_key_here' and
                passphrase and passphrase != 'your_actual_passphrase_here')
    
    if all_good:
        print("\n🎉 ALL CREDENTIALS FOUND! Ready for trading!")
        return True
    else:
        print("\n⚠️  Please fix the missing credentials in your .env file")
        return False

def main():
    print("🔐 OKX API CREDENTIALS SETUP HELPER")
    show_env_template()
    
    print("\nChecking your current .env file...")
    try:
        if check_current_env():
            print("\n🚀 You're ready to start trading!")
            print("Run: python start_live_trading.py")
        else:
            print("\n🔧 Please update your .env file with the correct format above")
    except Exception as e:
        print(f"\n❌ Error checking .env file: {e}")
        print("Make sure python-dotenv is installed: pip install python-dotenv")

if __name__ == "__main__":
    main() 