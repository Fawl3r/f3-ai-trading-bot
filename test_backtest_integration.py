#!/usr/bin/env python3
"""
🧪 QUICK BACKTEST INTEGRATION TEST
Tests the integration between enhanced main bot and comprehensive backtest
"""

import asyncio
import logging
from comprehensive_5000_candle_backtest import ComprehensiveBacktest
from enhanced_main_trading_bot import EnhancedMainTradingBot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_integration():
    """Test the integration between enhanced bot and backtest"""
    logger.info("🧪 Testing backtest integration...")
    
    try:
        # Test backtest initialization
        backtest = ComprehensiveBacktest()
        logger.info("✅ Backtest system initialized")
        
        # Test enhanced bot initialization
        bot = EnhancedMainTradingBot()
        logger.info("✅ Enhanced main bot initialized")
        
        # Test single backtest run
        logger.info("🔄 Running single backtest test...")
        metrics = await backtest.run_single_backtest("SOL", "1m", "BULL_TREND")
        
        if metrics.total_trades > 0:
            logger.info(f"✅ Single backtest completed successfully")
            logger.info(f"📊 Results: {metrics.total_trades} trades, PnL=${metrics.net_pnl:.2f}")
        else:
            logger.info("⚠️ No trades generated in test (this may be normal)")
        
        logger.info("🎉 Integration test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_integration()) 