#!/usr/bin/env python3
"""
Check all OKX environment variables
"""

from dotenv import load_dotenv
import os

print("Loading .env file...")
load_dotenv()

print("\nAll environment variables starting with 'OKX':")
print("=" * 50)

found_any = False
for key, value in os.environ.items():
    if key.startswith('OKX'):
        found_any = True
        # Show first few characters for security
        display_value = value[:10] + "..." if len(value) > 10 else value
        print(f"{key} = {display_value}")

if not found_any:
    print("No OKX environment variables found!")
    
print("\nThe bot expects these exact variable names:")
print("OKX_API_KEY")
print("OKX_SECRET_KEY") 
print("OKX_PASSPHRASE")
print("\nMake sure your .env file uses EXACTLY these names!") 