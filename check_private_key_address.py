#!/usr/bin/env python3
"""
Check which wallet address corresponds to the private key
"""

import eth_account
from hyperliquid.info import Info
from hyperliquid.utils import constants

def check_private_key_address():
    # Get the actual address from your private key
    private_key = '0xda301ce66aff83263dbc996e3514b521d79979bfade1bcdbc6115fc73e9b803d'
    account = eth_account.Account.from_key(private_key)
    actual_address = account.address
    
    print(f'🔑 Your private key corresponds to: {actual_address}')
    print(f'📍 Provided wallet address was: 0x80bb5c7e714a3280658910b70d03646b45c292e1')
    
    if actual_address.lower() == '0x80bb5c7e714a3280658910b70d03646b45c292e1'.lower():
        print('✅ Addresses MATCH! Perfect!')
    else:
        print('⚠️  Addresses DIFFERENT - using private key address')
    
    # Check balance for the correct address
    info = Info(constants.TESTNET_API_URL, skip_ws=True)
    user_state = info.user_state(actual_address)
    
    if user_state and 'marginSummary' in user_state:
        balance = float(user_state['marginSummary']['accountValue'])
        withdrawable = float(user_state.get('withdrawable', 0))
        total_pos = float(user_state['marginSummary']['totalNtlPos'])
        
        print(f'💰 Account Value: ${balance:.2f}')
        print(f'💵 Withdrawable: ${withdrawable:.2f}')
        print(f'📈 Total Position Size: ${total_pos:.2f}')
        
        asset_positions = user_state.get('assetPositions', [])
        if len(asset_positions) > 0:
            print(f'📊 Asset Positions: {len(asset_positions)} positions')
        else:
            print('📊 Asset Positions: None')
    else:
        print('📊 Account data not found')

if __name__ == "__main__":
    check_private_key_address() 