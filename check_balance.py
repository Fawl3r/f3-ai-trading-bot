#!/usr/bin/env python3
"""
Quick balance checker for Hyperliquid account
"""

from hyperliquid.info import Info
from hyperliquid.utils import constants

def check_balance():
    print('🔍 Checking Hyperliquid account balance...')
    
    # Initialize Info client for testnet
    info = Info(constants.TESTNET_API_URL, skip_ws=True)
    
    try:
        # Corrected wallet address that matches the private key
        wallet_address = '0xAD64671bd7f593B5Eb89CE42553299DAF3c03d1f'
        user_state = info.user_state(wallet_address)
        
        if user_state and 'marginSummary' in user_state:
            balance = float(user_state['marginSummary']['accountValue'])
            withdrawable = float(user_state.get('withdrawable', 0))
            total_pos = float(user_state['marginSummary']['totalNtlPos'])
            margin_used = float(user_state['marginSummary']['totalMarginUsed'])
            
            print(f'💰 Account Value: ${balance:.2f}')
            print(f'💵 Withdrawable: ${withdrawable:.2f}')
            print(f'📈 Total Position Size: ${total_pos:.2f}')
            print(f'🔒 Margin Used: ${margin_used:.2f}')
            
            asset_positions = user_state.get('assetPositions', [])
            if len(asset_positions) > 0:
                print(f'📊 Asset Positions: {len(asset_positions)} positions')
                for i, pos in enumerate(asset_positions[:3]):
                    print(f'   Position {i+1}: {pos}')
            else:
                print('📊 Asset Positions: None')
                
            print(f'🕒 Last Update: {user_state.get("time", "Unknown")}')
            
        else:
            print('❌ Could not retrieve account information')
            
    except Exception as e:
        print(f'❌ Error checking balance: {e}')

if __name__ == "__main__":
    check_balance() 