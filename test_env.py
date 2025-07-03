#!/usr/bin/env python3
"""
Simple .env test script
"""

try:
    from dotenv import load_dotenv
    import os
    
    print("Loading .env file...")
    load_dotenv()
    
    api_key = os.getenv('OKX_API_KEY', 'NOT_FOUND')
    secret_key = os.getenv('OKX_SECRET_KEY', '') or os.getenv('OKX_API_SECRET', 'NOT_FOUND')
    passphrase = os.getenv('OKX_PASSPHRASE', '') or os.getenv('OKX_API_PASSPHRASE', 'NOT_FOUND')
    
    print(f"API Key: {api_key[:10]}..." if len(api_key) > 10 else f"API Key: {api_key}")
    print(f"Secret Key: {secret_key[:10]}..." if len(secret_key) > 10 else f"Secret Key: {secret_key}")
    print(f"Passphrase: {passphrase[:5]}..." if len(passphrase) > 5 else f"Passphrase: {passphrase}")
    
    if api_key != 'NOT_FOUND' and secret_key != 'NOT_FOUND' and passphrase != 'NOT_FOUND':
        print("\n✅ All credentials found!")
    else:
        print("\n❌ Some credentials missing")
        
except Exception as e:
    print(f"Error: {e}")
    print("Make sure python-dotenv is installed: pip install python-dotenv") 