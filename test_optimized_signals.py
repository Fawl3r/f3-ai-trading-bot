#!/usr/bin/env python3
"""Quick test of optimized signal detection"""

import asyncio
from optimized_hyperliquid_bot import OptimizedHyperliquidBot

async def quick_test():
    print('🔍 Quick test of optimized signal detection...')
    print('📊 Using 50% confidence threshold (82.4% historical win rate)')
    
    bot = OptimizedHyperliquidBot()
    
    signals_found = 0
    
    for symbol in ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE']:
        print(f'\n🔍 Testing {symbol}...')
        signal = await bot.detect_opportunity(symbol)
        if signal:
            print(f'🎯 SIGNAL FOUND!')
            print(f'   Symbol: {symbol}')
            print(f'   Action: {signal.action}')
            print(f'   Confidence: {signal.confidence:.1f}%')
            print(f'   Target: ${signal.target_price:.2f}')
            print(f'   Stop Loss: ${signal.stop_loss:.2f}')
            print(f'   Reason: {signal.reason}')
            signals_found += 1
        else:
            print(f'   ❌ No signal for {symbol}')
    
    print(f'\n📊 RESULTS: {signals_found}/{len(["BTC", "ETH", "SOL", "AVAX", "DOGE"])} signals found')
    if signals_found > 0:
        print('✅ Optimized bot is generating signals!')
    else:
        print('❌ No signals - may need further optimization')

if __name__ == "__main__":
    asyncio.run(quick_test()) 