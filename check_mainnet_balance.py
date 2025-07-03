#!/usr/bin/env python3
"""
Check Hyperliquid MAINNET PERPETUAL account balance (REAL MONEY)
"""

from hyperliquid.info import Info
from hyperliquid.utils import constants

def check_mainnet_balance():
    print('🔍 Checking Hyperliquid MAINNET account balance...')
    print('⚠️  MAINNET = REAL MONEY')
    
    # Initialize Info client for MAINNET (real money)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    try:
        # Your wallet address that matches the private key
        wallet_address = '0xAD64671bd7f593B5Eb89CE42553299DAF3c03d1f'
        
        # Get perpetual account state from MAINNET
        user_state = info.user_state(wallet_address)
        
        if user_state:
            print(f'🏦 MAINNET Wallet: {wallet_address}')
            print('=' * 60)
            
            # Check margin summary (perpetual account)
            if 'marginSummary' in user_state:
                margin = user_state['marginSummary']
                account_value = float(margin.get('accountValue', 0))
                total_margin_used = float(margin.get('totalMarginUsed', 0))
                total_ntl_pos = float(margin.get('totalNtlPos', 0))
                
                print('📊 MAINNET PERPETUAL ACCOUNT:')
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
                    coin = pos.get('coin', 'Unknown')
                    total = pos.get('total', {})
                    size = float(total.get('size', 0))
                    unrealized_pnl = float(total.get('unrealizedPnl', 0))
                    print(f'   Position {i+1}: {coin} | Size: {size} | PnL: ${unrealized_pnl:.2f}')
            else:
                print('   No asset positions found')
                
            # Check if account has any USDC for trading
            print('=' * 60)
            if account_value > 0:
                print(f'✅ MAINNET FUNDED: ${account_value:.2f} REAL MONEY available')
                print(f'🚀 Ready for LIVE TRADING!')
            else:
                print('❌ MAINNET EMPTY: No real money deposited')
                print('💡 To fund mainnet account:')
                print('   - Go to app.hyperliquid.xyz')
                print('   - Deposit USDC to start live trading')
                print('   - Recommended: Start with $20-50')
                
        else:
            print('❌ Could not retrieve mainnet account information')
            
    except Exception as e:
        print(f'❌ Error checking mainnet balance: {e}')

if __name__ == "__main__":
    check_mainnet_balance() 