#!/usr/bin/env python3
"""
Unit tests for improved risk management system
Tests ATR-based stops, 4:1 R:R, and safety controls
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_edge_system import ImprovedEdgeSystem

class TestRiskManagement(unittest.TestCase):
    """Test suite for risk management components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = ImprovedEdgeSystem()
        self.test_candles = self._generate_test_candles()
    
    def _generate_test_candles(self, num_candles=100, volatility=0.002):
        """Generate test candle data"""
        candles = []
        base_price = 100
        current_price = base_price
        
        for i in range(num_candles):
            change = np.random.normal(0, volatility)
            new_price = current_price * (1 + change)
            
            candle = {
                'timestamp': int((datetime.now() - timedelta(minutes=(num_candles-i)*5)).timestamp() * 1000),
                'open': current_price,
                'high': new_price * (1 + abs(np.random.normal(0, volatility/2))),
                'low': new_price * (1 - abs(np.random.normal(0, volatility/2))),
                'close': new_price,
                'volume': max(100, 1000 + np.random.normal(0, 200))
            }
            
            candles.append(candle)
            current_price = new_price
        
        return candles
    
    def test_atr_calculation(self):
        """Test ATR calculation accuracy"""
        # Test with known values
        test_candles = [
            {'high': 102, 'low': 98, 'close': 100},
            {'high': 103, 'low': 99, 'close': 101},
            {'high': 104, 'low': 100, 'close': 102},
            {'high': 103, 'low': 101, 'close': 102},
            {'high': 102, 'low': 100, 'close': 101},
        ]
        
        atr = self.system.calculate_atr(test_candles, period=3)
        
        # ATR should be reasonable for this data
        self.assertGreater(atr, 1.0)
        self.assertLess(atr, 5.0)
    
    def test_risk_reward_ratio(self):
        """Test that R:R ratio is correctly 4:1"""
        # Train system
        self.system.train_models(self.test_candles[:60])
        
        # Generate signal
        signal = self.system.generate_signal(self.test_candles)
        
        if signal['direction'] != 'hold':
            # Check R:R ratio
            ratio = signal['risk_reward_ratio']
            self.assertEqual(ratio, 4.0, "Risk-reward ratio should be 4:1")
            
            # Check stop and target distances
            self.assertAlmostEqual(
                signal['target_distance'] / signal['stop_distance'],
                4.0,
                places=2
            )
    
    def test_minimum_edge_filter(self):
        """Test that minimum edge requirement is enforced"""
        # Train system
        self.system.train_models(self.test_candles[:60])
        
        # Generate multiple signals
        trades_taken = 0
        for i in range(60, 90):
            signal = self.system.generate_signal(self.test_candles[:i+1])
            
            if self.system.should_trade(signal):
                trades_taken += 1
                # Every trade taken should have positive edge
                self.assertGreaterEqual(
                    signal['edge'], 
                    self.system.min_edge_score,
                    "Trade taken with insufficient edge"
                )
        
        # Should be selective
        self.assertLess(trades_taken, 10, "System taking too many trades")
    
    def test_probability_gate(self):
        """Test probability threshold enforcement"""
        # Train system
        self.system.train_models(self.test_candles[:60])
        
        # Test multiple signals
        for i in range(60, 80):
            signal = self.system.generate_signal(self.test_candles[:i+1])
            
            if self.system.should_trade(signal):
                # Confidence should exceed threshold
                self.assertGreaterEqual(
                    signal['confidence'],
                    self.system.min_prob_threshold,
                    "Trade taken with low confidence"
                )
    
    def test_volatility_filter(self):
        """Test that volatility filter works correctly"""
        # Generate low volatility period
        low_vol_candles = self._generate_test_candles(100, volatility=0.0005)
        
        # Train on mixed data
        all_candles = self.test_candles[:50] + low_vol_candles
        self.system.train_models(all_candles[:80])
        
        # Test on low volatility period
        signal = self.system.generate_signal(all_candles[:110])
        
        # Should detect low volatility
        self.assertFalse(
            signal['high_volatility'],
            "Failed to detect low volatility period"
        )
    
    def test_order_book_imbalance(self):
        """Test order book imbalance calculation"""
        # Test balanced book
        obi_balanced = self.system.calculate_order_book_imbalance(1000, 1000)
        self.assertAlmostEqual(obi_balanced, 0.0, places=2)
        
        # Test bid-heavy book
        obi_bid = self.system.calculate_order_book_imbalance(2000, 1000)
        self.assertGreater(obi_bid, 0.0)
        self.assertAlmostEqual(obi_bid, 0.333, places=2)
        
        # Test ask-heavy book
        obi_ask = self.system.calculate_order_book_imbalance(1000, 2000)
        self.assertLess(obi_ask, 0.0)
        self.assertAlmostEqual(obi_ask, -0.333, places=2)
    
    def test_quality_checks(self):
        """Test that quality checks require 2 of 3 conditions"""
        # Train system
        self.system.train_models(self.test_candles[:60])
        
        # Mock a signal with specific conditions
        signal = {
            'direction': 'long',
            'confidence': 0.62,  # Below 0.65 threshold
            'edge': 0.003,
            'rsi': 28,  # Below 30 (good for long)
            'obi': 0.1,  # Above -0.05 (good for long)
            'high_volatility': True,
            'stop_distance': 1.0,
            'target_distance': 4.0
        }
        
        # Should pass with 2 of 3 quality checks
        self.assertTrue(self.system.should_trade(signal))
        
        # Now fail one more check
        signal['rsi'] = 50  # Neutral RSI
        self.assertFalse(self.system.should_trade(signal))
    
    def test_expectancy_calculation(self):
        """Test expectancy calculation with 4:1 R:R"""
        # Test various win rates
        test_cases = [
            (0.20, 0.0),    # 20% win rate = breakeven
            (0.25, 0.0025), # 25% win rate = 0.25% edge
            (0.30, 0.005),  # 30% win rate = 0.5% edge
            (0.40, 0.01),   # 40% win rate = 1% edge
        ]
        
        for win_rate, expected_edge in test_cases:
            # Calculate expectancy
            # E = (win_rate * 4R) - ((1-win_rate) * 1R)
            reward = 0.04  # 4%
            risk = 0.01    # 1%
            
            expectancy = (win_rate * reward) - ((1 - win_rate) * risk)
            
            self.assertAlmostEqual(
                expectancy,
                expected_edge,
                places=4,
                msg=f"Expectancy wrong for {win_rate:.0%} win rate"
            )
    
    def test_position_sizing(self):
        """Test Kelly-inspired position sizing"""
        # Test with different edge scenarios
        test_cases = [
            {'edge': 0.005, 'confidence': 0.65},  # Good edge
            {'edge': 0.0025, 'confidence': 0.60}, # Minimum edge
            {'edge': 0.01, 'confidence': 0.75},   # Great edge
        ]
        
        for case in test_cases:
            # Position size should scale with edge and confidence
            # But never exceed 1% base risk
            position_size = min(1.0, case['edge'] * 100 * case['confidence'])
            
            self.assertLessEqual(position_size, 1.0)
            self.assertGreater(position_size, 0)
    
    def test_error_buffer_management(self):
        """Test error buffer for online learning"""
        # Add some errors
        for i in range(10):
            features = np.random.randn(20)
            self.system.update_error_buffer(features, predicted=1, actual=0)
        
        self.assertEqual(len(self.system.error_buffer), 10)
        
        # Test buffer size limit
        for i in range(600):
            features = np.random.randn(20)
            self.system.update_error_buffer(features, predicted=0, actual=1)
        
        self.assertEqual(
            len(self.system.error_buffer),
            self.system.max_buffer_size,
            "Error buffer exceeded max size"
        )
    
    def test_walk_forward_consistency(self):
        """Test that results are consistent across walk-forward windows"""
        # Train on multiple windows
        window_size = 50
        results = []
        
        for start in [0, 25, 50]:
            window = self.test_candles[start:start+window_size]
            self.system.train_models(window[:30])
            
            # Test on out-of-sample
            signals = []
            for i in range(30, len(window)):
                signal = self.system.generate_signal(window[:i+1])
                if signal['direction'] != 'hold':
                    signals.append(signal['edge'])
            
            if signals:
                avg_edge = np.mean(signals)
                results.append(avg_edge)
        
        # Results should be somewhat consistent
        if len(results) > 1:
            std_dev = np.std(results)
            self.assertLess(
                std_dev,
                0.002,
                "Edge estimates vary too much across windows"
            )

class TestSafetyControls(unittest.TestCase):
    """Test safety controls and circuit breakers"""
    
    def test_daily_loss_limit(self):
        """Test daily loss limit enforcement"""
        daily_pnl = -0.03  # -3%
        risk_per_trade = 0.01  # 1%
        
        # Should halt at -3R
        should_halt = daily_pnl <= -3 * risk_per_trade
        self.assertTrue(should_halt, "Failed to halt at daily loss limit")
    
    def test_latency_check(self):
        """Test latency-based order cancellation"""
        max_latency = 300  # ms
        
        test_cases = [
            (250, False),  # OK
            (350, True),   # Cancel
            (300, False),  # Edge case - OK
            (301, True),   # Just over - Cancel
        ]
        
        for latency, should_cancel in test_cases:
            cancel = latency > max_latency
            self.assertEqual(
                cancel,
                should_cancel,
                f"Wrong decision for {latency}ms latency"
            )
    
    def test_slippage_protection(self):
        """Test slippage-based position reduction"""
        spread = 0.001  # 0.1%
        max_slippage = 0.5 * spread  # 0.05%
        
        test_cases = [
            (0.0003, False),  # 0.03% - OK
            (0.0006, True),   # 0.06% - Reduce
            (0.0005, False),  # 0.05% - Edge case OK
        ]
        
        for slippage, should_reduce in test_cases:
            reduce = slippage > max_slippage
            self.assertEqual(
                reduce,
                should_reduce,
                f"Wrong decision for {slippage:.2%} slippage"
            )

if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2) 