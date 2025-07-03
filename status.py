import json
from hyperliquid.info import Info
from hyperliquid.utils import constants
from datetime import datetime

print('📊 EXTENDED 15 BOT STATUS CHECK')
print('=' * 60)

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Check account
info = Info(constants.MAINNET_API_URL)
user_state = info.user_state(config['wallet_address'])
balance = float(user_state['marginSummary']['accountValue'])

print(f'💰 Account Balance: ${balance:.2f}')
print(f'📊 Wallet: {config["wallet_address"]}')
print(f'🌐 Network: MAINNET')
print(f'🕐 Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# Check positions
positions = []
if 'assetPositions' in user_state:
    for pos in user_state['assetPositions']:
        if float(pos['position']['szi']) != 0:
            positions.append({
                'symbol': pos['position']['coin'],
                'size': float(pos['position']['szi']),
                'pnl': float(pos['position']['unrealizedPnl'])
            })

print(f'🎲 Active Positions: {len(positions)}')
if positions:
    total_pnl = sum(pos['pnl'] for pos in positions)
    print(f'💰 Total PnL: ${total_pnl:.2f}')
    for pos in positions:
        emoji = '🟢' if pos['pnl'] > 0 else '🔴'
        print(f'   {emoji} {pos["symbol"]}: ${pos["pnl"]:.2f}')

print('✅ Extended 15 Bot Status Check Complete') 