#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE BACKTEST LAUNCHER
Launches the professional-grade backtesting system with all critical metrics

This script will:
✅ Test all symbols (SOL, BTC, ETH, AVAX, DOGE)
✅ Test all timeframes (1m, 5m, 1h)
✅ Test all market regimes (Bull, Bear, Range, High Volatility)
✅ Generate comprehensive reports with all metrics
✅ Save results to CSV and JSON files
"""

import asyncio
import logging
import time
import os
from datetime import datetime
from comprehensive_5000_candle_backtest import ComprehensiveBacktest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_launcher.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Main backtest launcher function"""
    start_time = datetime.now()
    logger.info("🚀 STARTING COMPREHENSIVE BACKTEST LAUNCHER")
    logger.info(f"⏰ Start time: {start_time}")
    
    try:
        # Initialize backtest system
        backtest = ComprehensiveBacktest()
        
        # Run comprehensive backtest
        logger.info("🔄 Launching comprehensive backtest suite...")
        await backtest.run_comprehensive_backtest()
        
        # Calculate total runtime
        end_time = datetime.now()
        runtime = end_time - start_time
        logger.info(f"✅ BACKTEST COMPLETED")
        logger.info(f"⏰ End time: {end_time}")
        logger.info(f"⏱️ Total runtime: {runtime}")
        
        # Check for results files
        if os.path.exists('comprehensive_backtest_results.csv'):
            logger.info("📊 Results saved to: comprehensive_backtest_results.csv")
        if os.path.exists('backtest_summary.json'):
            logger.info("📋 Summary saved to: backtest_summary.json")
        
        logger.info("🎉 BACKTEST LAUNCHER COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"❌ Error in backtest launcher: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 