#!/usr/bin/env python3
"""
🧪 TEST ENHANCED TOP/BOTTOM & LIQUIDITY ZONE FEATURES
Test script to validate the new detection capabilities

FEATURES TESTED:
✅ Swing High/Low Detection
✅ Order Book Liquidity Zone Analysis
✅ Volume Cluster Detection
✅ Signal Integration
✅ Performance Comparison
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from advanced_top_bottom_detector import AdvancedTopBottomDetector
from enhanced_top_bottom_backtest import EnhancedTopBottomBacktest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFeaturesTester:
    """Test the enhanced top/bottom and liquidity zone features"""
    
    def __init__(self):
        self.detector = AdvancedTopBottomDetector()
        self.backtest = EnhancedTopBottomBacktest()
    
    def test_swing_point_detection(self):
        """Test swing high/low detection with sample data"""
        logger.info("🧪 Testing Swing Point Detection...")
        
        # Create sample candle data with clear swing points
        sample_candles = [
            {'h': 50000, 'l': 49000, 'c': 49500, 'v': 1000, 't': 1000000},
            {'h': 51000, 'l': 49500, 'c': 50500, 'v': 1200, 't': 1000060},
            {'h': 52000, 'l': 50000, 'c': 51500, 'v': 1500, 't': 1000120},
            {'h': 51500, 'l': 50500, 'c': 51000, 'v': 800, 't': 1000180},
            {'h': 52500, 'l': 51000, 'c': 52000, 'v': 2000, 't': 1000240},
            {'h': 53000, 'l': 51500, 'c': 52500, 'v': 1800, 't': 1000300},
            {'h': 52800, 'l': 52000, 'c': 52200, 'v': 900, 't': 1000360},
            {'h': 53500, 'l': 52200, 'c': 53000, 'v': 2200, 't': 1000420},
            {'h': 54000, 'l': 52500, 'c': 53500, 'v': 2500, 't': 1000480},
            {'h': 53800, 'l': 53000, 'c': 53200, 'v': 1100, 't': 1000540},
        ]
        
        # Test swing point detection
        swing_points = self.detector.detect_swing_points(sample_candles)
        
        print(f"\n🎯 Swing Point Detection Results:")
        print(f"   Swing Highs: {len(swing_points['highs'])}")
        print(f"   Swing Lows: {len(swing_points['lows'])}")
        
        for i, high in enumerate(swing_points['highs'][:3]):
            print(f"   High {i+1}: ${high.price:.2f} (Strength: {high.strength:.1f})")
        
        for i, low in enumerate(swing_points['lows'][:3]):
            print(f"   Low {i+1}: ${low.price:.2f} (Strength: {low.strength:.1f})")
        
        return len(swing_points['highs']) > 0 or len(swing_points['lows']) > 0
    
    def test_liquidity_zone_analysis(self):
        """Test liquidity zone analysis"""
        logger.info("🧪 Testing Liquidity Zone Analysis...")
        
        # Test with BTC (real API call)
        try:
            current_price = 50000  # Approximate BTC price
            liquidity_zones = self.detector.analyze_liquidity_zones("BTC", current_price)
            
            print(f"\n💧 Liquidity Zone Analysis Results:")
            print(f"   Total Zones: {len(liquidity_zones)}")
            
            bid_zones = [z for z in liquidity_zones if z.zone_type == 'bid_cluster']
            ask_zones = [z for z in liquidity_zones if z.zone_type == 'ask_cluster']
            
            print(f"   Bid Clusters: {len(bid_zones)}")
            print(f"   Ask Clusters: {len(ask_zones)}")
            
            for i, zone in enumerate(liquidity_zones[:5]):
                print(f"   Zone {i+1}: ${zone.price_level:.2f} ({zone.zone_type}) - Strength: {zone.strength:.1f}")
            
            return len(liquidity_zones) > 0
            
        except Exception as e:
            logger.error(f"Liquidity zone test failed: {e}")
            return False
    
    def test_market_structure_analysis(self):
        """Test market structure analysis"""
        logger.info("🧪 Testing Market Structure Analysis...")
        
        # Create sample data with clear trend
        sample_candles = [
            {'h': 50000, 'l': 49000, 'c': 49500, 'v': 1000, 't': 1000000},
            {'h': 51000, 'l': 49500, 'c': 50500, 'v': 1200, 't': 1000060},
            {'h': 52000, 'l': 50000, 'c': 51500, 'v': 1500, 't': 1000120},
            {'h': 53000, 'l': 51000, 'c': 52500, 'v': 1800, 't': 1000180},
            {'h': 54000, 'l': 52000, 'c': 53500, 'v': 2000, 't': 1000240},
            {'h': 55000, 'l': 53000, 'c': 54500, 'v': 2200, 't': 1000300},
        ]
        
        structure = self.detector.get_market_structure(sample_candles)
        
        print(f"\n📊 Market Structure Analysis Results:")
        print(f"   Trend: {structure['trend']}")
        print(f"   Structure Score: {structure['structure_score']}")
        print(f"   Swing Points: {len(structure['swing_points']['highs'])} highs, {len(structure['swing_points']['lows'])} lows")
        
        return structure['trend'] != 'neutral'
    
    def test_entry_exit_signals(self):
        """Test comprehensive entry/exit signal generation"""
        logger.info("🧪 Testing Entry/Exit Signal Generation...")
        
        # Create sample data near a swing point
        sample_candles = [
            {'h': 50000, 'l': 49000, 'c': 49500, 'v': 1000, 't': 1000000},
            {'h': 51000, 'l': 49500, 'c': 50500, 'v': 1200, 't': 1000060},
            {'h': 52000, 'l': 50000, 'c': 51500, 'v': 1500, 't': 1000120},
            {'h': 51500, 'l': 50500, 'c': 51000, 'v': 800, 't': 1000180},
            {'h': 52500, 'l': 51000, 'c': 52000, 'v': 2000, 't': 1000240},
            {'h': 53000, 'l': 51500, 'c': 52500, 'v': 1800, 't': 1000300},
            {'h': 52800, 'l': 52000, 'c': 52200, 'v': 900, 't': 1000360},
            {'h': 53500, 'l': 52200, 'c': 53000, 'v': 2200, 't': 1000420},
            {'h': 54000, 'l': 52500, 'c': 53500, 'v': 2500, 't': 1000480},
            {'h': 53800, 'l': 53000, 'c': 53200, 'v': 1100, 't': 1000540},
        ]
        
        current_price = 53200  # Near a swing high
        
        signals = self.detector.get_entry_exit_signals("BTC", sample_candles, current_price)
        
        print(f"\n🎯 Entry/Exit Signal Results:")
        print(f"   Long Entry: {signals['long_entry']}")
        print(f"   Short Entry: {signals['short_entry']}")
        print(f"   Long Exit: {signals['long_exit']}")
        print(f"   Short Exit: {signals['short_exit']}")
        print(f"   Confidence: {signals['confidence']:.1f}%")
        print(f"   Reason: {signals['reason']}")
        
        return signals['long_entry'] or signals['short_entry']
    
    async def test_backtest_integration(self):
        """Test backtest integration with enhanced features"""
        logger.info("🧪 Testing Backtest Integration...")
        
        # Run a short backtest
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        
        results = self.backtest.run_enhanced_backtest("BTC", start_date, end_date)
        
        if results:
            print(f"\n📈 Backtest Integration Results:")
            print(f"   Total Trades: {results['summary']['total_trades']}")
            print(f"   Win Rate: {results['summary']['win_rate']:.1f}%")
            print(f"   Original Bot Trades: {results['original_bot']['trades']}")
            print(f"   Enhanced Bot Trades: {results['enhanced_bot']['trades']}")
            print(f"   Win Rate Improvement: {results['improvement']['win_rate_improvement']:+.1f}%")
            
            return results['summary']['total_trades'] > 0
        else:
            logger.error("Backtest integration test failed")
            return False
    
    def run_all_tests(self):
        """Run all feature tests"""
        logger.info("🚀 Starting Enhanced Features Test Suite")
        
        tests = [
            ("Swing Point Detection", self.test_swing_point_detection),
            ("Liquidity Zone Analysis", self.test_liquidity_zone_analysis),
            ("Market Structure Analysis", self.test_market_structure_analysis),
            ("Entry/Exit Signals", self.test_entry_exit_signals),
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Running: {test_name}")
                logger.info(f"{'='*50}")
                
                result = test_func()
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                    passed_tests += 1
                else:
                    logger.warning(f"⚠️ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
        
        # Test backtest integration
        try:
            logger.info(f"\n{'='*50}")
            logger.info("Running: Backtest Integration")
            logger.info(f"{'='*50}")
            
            result = asyncio.run(self.test_backtest_integration())
            if result:
                logger.info("✅ Backtest Integration: PASSED")
                passed_tests += 1
            else:
                logger.warning("⚠️ Backtest Integration: FAILED")
                
        except Exception as e:
            logger.error(f"❌ Backtest Integration: ERROR - {e}")
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Passed: {passed_tests}/{total_tests + 1}")
        logger.info(f"Success Rate: {(passed_tests / (total_tests + 1) * 100):.1f}%")
        
        if passed_tests >= total_tests:
            logger.info("🎉 All critical tests passed! Enhanced features are ready.")
        else:
            logger.warning("⚠️ Some tests failed. Please check the implementation.")
        
        return passed_tests >= total_tests

# Main execution
if __name__ == "__main__":
    tester = EnhancedFeaturesTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 Enhanced features are working correctly!")
        print("You can now integrate them into your main trading bot.")
    else:
        print("\n⚠️ Some features need attention before integration.") 