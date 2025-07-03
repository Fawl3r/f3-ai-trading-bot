#!/usr/bin/env python3
"""
Check Hyperliquid PERPETUAL account balance specifically
"""

from hyperliquid.info import Info
from hyperliquid.utils import constants

def check_perp_balance():
    print('🔍 Checking Hyperliquid PERPETUAL account balance...')
    
    # Initialize Info client for testnet
    info = Info(constants.TESTNET_API_URL, skip_ws=True)
    
    try:
        # Your wallet address that matches the private key
        wallet_address = '0xAD64671bd7f593B5Eb89CE42553299DAF3c03d1f'
        
        # Get perpetual account state
        user_state = info.user_state(wallet_address)
        
        if user_state:
            print(f'🏦 Wallet Address: {wallet_address}')
            print('=' * 60)
            
            # Check margin summary (perpetual account)
            if 'marginSummary' in user_state:
                margin = user_state['marginSummary']
                account_value = float(margin.get('accountValue', 0))
                total_margin_used = float(margin.get('totalMarginUsed', 0))
                total_ntl_pos = float(margin.get('totalNtlPos', 0))
                
                print('📊 PERPETUAL ACCOUNT:')
                print(f'💰 Account Value: ${account_value:.2f}')
                print(f'🔒 Margin Used: ${total_margin_used:.2f}')
                print(f'📈 Total Position Size: ${total_ntl_pos:.2f}')
                
            # Check withdrawable amount
            withdrawable = float(user_state.get('withdrawable', 0))
            print(f'💵 Withdrawable: ${withdrawable:.2f}')
            
            # Check cross margin summary
            if 'crossMarginSummary' in user_state:
                cross = user_state['crossMarginSummary']
                cross_value = float(cross.get('accountValue', 0))
                cross_margin = float(cross.get('totalMarginUsed', 0))
                
                print('🔄 CROSS MARGIN:')
                print(f'💰 Cross Account Value: ${cross_value:.2f}')
                print(f'🔒 Cross Margin Used: ${cross_margin:.2f}')
            
            # Check asset positions
            asset_positions = user_state.get('assetPositions', [])
            print(f'📊 Asset Positions: {len(asset_positions)}')
            
            if asset_positions:
                for i, pos in enumerate(asset_positions):
                    print(f'   Position {i+1}: {pos}')
            else:
                print('   No asset positions found')
                
            # Check if account has any USDC for trading
            print('=' * 60)
            if account_value > 0:
                print(f'✅ PERP ACCOUNT FUNDED: ${account_value:.2f} available for trading')
            else:
                print('❌ PERP ACCOUNT EMPTY: Need to deposit USDC for trading')
                print('💡 To fund account:')
                print('   - Testnet: Get free USDC at testnet.hyperliquid.xyz')
                print('   - Mainnet: Deposit USDC at app.hyperliquid.xyz')
                
        else:
            print('❌ Could not retrieve account information')
            
    except Exception as e:
        print(f'❌ Error checking perp balance: {e}')

if __name__ == "__main__":
    check_perp_balance() 