#!/usr/bin/env python3
"""
Elite AI Learning System
Advanced AI that learns from every trade and continuously improves performance
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EliteAILearningSystem:
    """
    Advanced AI Learning System for Elite 100%/5% Trading System
    
    Features:
    - Learns from every trade outcome
    - Adapts signal weights based on performance
    - Identifies profitable patterns
    - Blacklists losing patterns
    - Continuously optimizes parameters
    """
    
    def __init__(self, db_path: str = "elite_ai_learning.db"):
        self.db_path = db_path
        
        # AI Learning Parameters
        self.learning_rate = 0.02
        self.min_sample_size = 10
        self.confidence_threshold = 0.45
        
        # Signal Weights (start with equal weights)
        self.signal_weights = {
            'parabolic_burst': 0.25,
            'fade_signal': 0.25,
            'breakout_signal': 0.25,
            'volume_confirmation': 0.25
        }
        
        # Pattern Recognition
        self.profitable_patterns = {}
        self.losing_patterns = {}
        self.pattern_blacklist = set()
        
        # Performance Tracking
        self.trade_outcomes = []
        self.learning_iterations = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Adaptive Parameters
        self.adaptive_params = {
            'risk_multiplier': 1.0,
            'confidence_threshold': 0.45,
            'signal_strength_threshold': 2.0,
            'volume_threshold': 1.5
        }
        
        # Initialize database
        self._init_database()
        
        # Load existing learning data
        self._load_learning_data()
        
        logger.info("Elite AI Learning System initialized")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Total historical trades: {len(self.trade_outcomes)}")
    
    def _init_database(self):
        """Initialize AI learning database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trade outcomes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_trade_outcomes (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                signal_type TEXT,
                signal_strength REAL,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                r_multiple REAL,
                outcome TEXT,
                confidence REAL,
                timestamp TEXT,
                market_conditions TEXT
            )
        ''')
        
        # Learning data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_learning_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                signal_weights TEXT,
                adaptive_params TEXT,
                performance_metrics TEXT,
                learning_iteration INTEGER
            )
        ''')
        
        # Pattern analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                pattern_data TEXT,
                success_rate REAL,
                trade_count INTEGER,
                avg_pnl REAL,
                status TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_learning_data(self):
        """Load existing learning data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load trade outcomes
        cursor.execute("SELECT * FROM ai_trade_outcomes ORDER BY timestamp DESC LIMIT 1000")
        self.trade_outcomes = [dict(zip([col[0] for col in cursor.description], row)) 
                              for row in cursor.fetchall()]
        
        # Load latest learning iteration
        cursor.execute("SELECT * FROM ai_learning_data ORDER BY learning_iteration DESC LIMIT 1")
        latest_learning = cursor.fetchone()
        
        if latest_learning:
            self.signal_weights = json.loads(latest_learning[2])
            self.adaptive_params = json.loads(latest_learning[3])
            self.learning_iterations = latest_learning[5]
        
        # Load patterns
        cursor.execute("SELECT * FROM ai_patterns WHERE status = 'active'")
        for row in cursor.fetchall():
            pattern_data = json.loads(row[2])
            if row[1] == 'profitable':
                self.profitable_patterns[row[0]] = pattern_data
            elif row[1] == 'losing':
                self.losing_patterns[row[0]] = pattern_data
                if pattern_data.get('blacklisted', False):
                    self.pattern_blacklist.add(row[0])
        
        conn.close()
        
        # Update performance stats
        self.total_trades = len(self.trade_outcomes)
        self.winning_trades = sum(1 for trade in self.trade_outcomes if trade['outcome'] == 'win')
        
        logger.info(f"Loaded {len(self.trade_outcomes)} trade outcomes")
        logger.info(f"Loaded {len(self.profitable_patterns)} profitable patterns")
        logger.info(f"Loaded {len(self.losing_patterns)} losing patterns")
    
    def learn_from_trade(self, trade_data: Dict):
        """
        Learn from a completed trade
        
        Args:
            trade_data: Dictionary containing trade information
        """
        trade_outcome = {
            'trade_id': trade_data['trade_id'],
            'symbol': trade_data['symbol'],
            'signal_type': trade_data['signal_type'],
            'signal_strength': trade_data['signal_strength'],
            'entry_price': trade_data['entry_price'],
            'exit_price': trade_data['exit_price'],
            'pnl': trade_data['pnl'],
            'r_multiple': trade_data['r_multiple'],
            'outcome': 'win' if trade_data['pnl'] > 0 else 'loss',
            'confidence': trade_data.get('confidence', 0.5),
            'timestamp': datetime.now().isoformat(),
            'market_conditions': json.dumps(trade_data.get('market_conditions', {}))
        }
        
        # Add to outcomes
        self.trade_outcomes.append(trade_outcome)
        
        # Update counters
        self.total_trades += 1
        if trade_outcome['outcome'] == 'win':
            self.winning_trades += 1
        
        # Save to database
        self._save_trade_outcome(trade_outcome)
        
        # Learn from the outcome
        self._update_signal_weights(trade_outcome)
        self._analyze_patterns(trade_outcome)
        self._adapt_parameters(trade_outcome)
        
        # Increment learning iteration
        self.learning_iterations += 1
        
        # Save learning state every 10 trades
        if self.learning_iterations % 10 == 0:
            self._save_learning_state()
        
        logger.info(f"AI learned from trade {trade_data['trade_id']}: {trade_outcome['outcome']} "
                   f"({trade_data['pnl']:.2f})")
        
        # Return learning insights
        return self._generate_learning_insights(trade_outcome)
    
    def _update_signal_weights(self, trade_outcome: Dict):
        """Update signal weights based on trade outcome"""
        signal_type = trade_outcome['signal_type']
        outcome = trade_outcome['outcome']
        r_multiple = trade_outcome['r_multiple']
        
        if signal_type in self.signal_weights:
            if outcome == 'win':
                # Boost weight for profitable signals
                adjustment = self.learning_rate * abs(r_multiple)
                self.signal_weights[signal_type] += adjustment
            else:
                # Reduce weight for losing signals
                adjustment = self.learning_rate * abs(r_multiple)
                self.signal_weights[signal_type] -= adjustment
            
            # Keep weights positive and normalized
            self.signal_weights[signal_type] = max(0.1, self.signal_weights[signal_type])
        
        # Normalize weights
        total_weight = sum(self.signal_weights.values())
        for signal in self.signal_weights:
            self.signal_weights[signal] /= total_weight
    
    def _analyze_patterns(self, trade_outcome: Dict):
        """Analyze and learn from trading patterns"""
        # Create pattern fingerprint
        pattern_data = {
            'signal_type': trade_outcome['signal_type'],
            'signal_strength': round(trade_outcome['signal_strength'], 1),
            'symbol': trade_outcome['symbol'],
            'outcome': trade_outcome['outcome'],
            'pnl': trade_outcome['pnl'],
            'r_multiple': trade_outcome['r_multiple']
        }
        
        # Generate pattern ID
        pattern_id = f"{pattern_data['signal_type']}_{pattern_data['symbol']}_{pattern_data['signal_strength']}"
        
        # Update pattern tracking
        if trade_outcome['outcome'] == 'win':
            if pattern_id not in self.profitable_patterns:
                self.profitable_patterns[pattern_id] = {
                    'pattern_data': pattern_data,
                    'success_count': 0,
                    'total_count': 0,
                    'total_pnl': 0.0,
                    'avg_r_multiple': 0.0
                }
            
            pattern = self.profitable_patterns[pattern_id]
            pattern['success_count'] += 1
            pattern['total_count'] += 1
            pattern['total_pnl'] += trade_outcome['pnl']
            pattern['avg_r_multiple'] = pattern['total_pnl'] / pattern['total_count']
        
        else:  # Loss
            if pattern_id not in self.losing_patterns:
                self.losing_patterns[pattern_id] = {
                    'pattern_data': pattern_data,
                    'loss_count': 0,
                    'total_count': 0,
                    'total_pnl': 0.0,
                    'avg_r_multiple': 0.0
                }
            
            pattern = self.losing_patterns[pattern_id]
            pattern['loss_count'] += 1
            pattern['total_count'] += 1
            pattern['total_pnl'] += trade_outcome['pnl']
            pattern['avg_r_multiple'] = pattern['total_pnl'] / pattern['total_count']
            
            # Blacklist if consistently losing
            if pattern['total_count'] >= 5 and pattern['avg_r_multiple'] < -0.5:
                self.pattern_blacklist.add(pattern_id)
                logger.warning(f"Pattern {pattern_id} blacklisted due to consistent losses")
    
    def _adapt_parameters(self, trade_outcome: Dict):
        """Adapt trading parameters based on performance"""
        recent_trades = self.trade_outcomes[-20:] if len(self.trade_outcomes) >= 20 else self.trade_outcomes
        
        if len(recent_trades) >= 10:
            recent_win_rate = sum(1 for t in recent_trades if t['outcome'] == 'win') / len(recent_trades)
            recent_avg_r = sum(t['r_multiple'] for t in recent_trades) / len(recent_trades)
            
            # Adapt confidence threshold
            if recent_win_rate < 0.4:  # Too many losses
                self.adaptive_params['confidence_threshold'] += 0.01
                self.adaptive_params['signal_strength_threshold'] += 0.1
            elif recent_win_rate > 0.6:  # High win rate
                self.adaptive_params['confidence_threshold'] = max(0.35, 
                                                                 self.adaptive_params['confidence_threshold'] - 0.005)
            
            # Adapt risk multiplier
            if recent_avg_r > 0.5:  # Good performance
                self.adaptive_params['risk_multiplier'] = min(1.2, 
                                                            self.adaptive_params['risk_multiplier'] + 0.05)
            elif recent_avg_r < -0.3:  # Poor performance
                self.adaptive_params['risk_multiplier'] = max(0.8, 
                                                            self.adaptive_params['risk_multiplier'] - 0.05)
    
    def _generate_learning_insights(self, trade_outcome: Dict) -> Dict:
        """Generate insights from the learning process"""
        insights = {
            'trade_outcome': trade_outcome['outcome'],
            'learning_iteration': self.learning_iterations,
            'current_win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'signal_weights': self.signal_weights.copy(),
            'adaptive_params': self.adaptive_params.copy(),
            'patterns_learned': len(self.profitable_patterns) + len(self.losing_patterns),
            'blacklisted_patterns': len(self.pattern_blacklist)
        }
        
        return insights
    
    def get_signal_recommendation(self, signal_data: Dict) -> Dict:
        """Get AI recommendation for a trading signal"""
        signal_type = signal_data['signal_type']
        signal_strength = signal_data['signal_strength']
        symbol = signal_data['symbol']
        
        # Check if pattern is blacklisted
        pattern_id = f"{signal_type}_{symbol}_{round(signal_strength, 1)}"
        if pattern_id in self.pattern_blacklist:
            return {
                'recommendation': 'REJECT',
                'reason': 'Pattern blacklisted due to consistent losses',
                'confidence': 0.0
            }
        
        # Calculate confidence based on learned patterns
        base_confidence = self.signal_weights.get(signal_type, 0.25)
        
        # Boost confidence for profitable patterns
        if pattern_id in self.profitable_patterns:
            pattern = self.profitable_patterns[pattern_id]
            if pattern['avg_r_multiple'] > 0.3:
                base_confidence *= 1.5
        
        # Apply adaptive thresholds
        if (base_confidence >= self.adaptive_params['confidence_threshold'] and 
            signal_strength >= self.adaptive_params['signal_strength_threshold']):
            
            recommendation = 'ACCEPT'
            adjusted_confidence = min(0.95, base_confidence * self.adaptive_params['risk_multiplier'])
        else:
            recommendation = 'REJECT'
            adjusted_confidence = base_confidence
        
        return {
            'recommendation': recommendation,
            'confidence': adjusted_confidence,
            'signal_weight': self.signal_weights.get(signal_type, 0.25),
            'adaptive_threshold': self.adaptive_params['confidence_threshold'],
            'risk_multiplier': self.adaptive_params['risk_multiplier']
        }
    
    def _save_trade_outcome(self, trade_outcome: Dict):
        """Save trade outcome to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO ai_trade_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_outcome['trade_id'],
            trade_outcome['symbol'],
            trade_outcome['signal_type'],
            trade_outcome['signal_strength'],
            trade_outcome['entry_price'],
            trade_outcome['exit_price'],
            trade_outcome['pnl'],
            trade_outcome['r_multiple'],
            trade_outcome['outcome'],
            trade_outcome['confidence'],
            trade_outcome['timestamp'],
            trade_outcome['market_conditions']
        ))
        
        conn.commit()
        conn.close()
    
    def _save_learning_state(self):
        """Save current learning state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        performance_metrics = {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'patterns_learned': len(self.profitable_patterns) + len(self.losing_patterns),
            'blacklisted_patterns': len(self.pattern_blacklist)
        }
        
        cursor.execute('''
            INSERT INTO ai_learning_data (timestamp, signal_weights, adaptive_params, performance_metrics, learning_iteration)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(self.signal_weights),
            json.dumps(self.adaptive_params),
            json.dumps(performance_metrics),
            self.learning_iterations
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"AI learning state saved (iteration {self.learning_iterations})")
    
    def get_learning_status(self) -> Dict:
        """Get comprehensive learning status"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate * 100,
            'learning_iterations': self.learning_iterations,
            'signal_weights': self.signal_weights,
            'adaptive_params': self.adaptive_params,
            'profitable_patterns': len(self.profitable_patterns),
            'losing_patterns': len(self.losing_patterns),
            'blacklisted_patterns': len(self.pattern_blacklist),
            'learning_rate': self.learning_rate,
            'ai_enabled': True
        }
    
    def reset_learning(self):
        """Reset AI learning data (use with caution)"""
        self.signal_weights = {
            'parabolic_burst': 0.25,
            'fade_signal': 0.25,
            'breakout_signal': 0.25,
            'volume_confirmation': 0.25
        }
        
        self.adaptive_params = {
            'risk_multiplier': 1.0,
            'confidence_threshold': 0.45,
            'signal_strength_threshold': 2.0,
            'volume_threshold': 1.5
        }
        
        self.profitable_patterns = {}
        self.losing_patterns = {}
        self.pattern_blacklist = set()
        self.trade_outcomes = []
        self.learning_iterations = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.warning("AI learning data reset - starting fresh!")

if __name__ == "__main__":
    # Test the AI learning system
    ai = EliteAILearningSystem()
    
    # Simulate some trades
    test_trades = [
        {
            'trade_id': 'test_1',
            'symbol': 'BTC',
            'signal_type': 'parabolic_burst',
            'signal_strength': 3.2,
            'entry_price': 45000,
            'exit_price': 46800,
            'pnl': 180,
            'r_multiple': 1.8,
            'confidence': 0.75
        },
        {
            'trade_id': 'test_2',
            'symbol': 'ETH',
            'signal_type': 'fade_signal',
            'signal_strength': 2.8,
            'entry_price': 3200,
            'exit_price': 3120,
            'pnl': -80,
            'r_multiple': -0.8,
            'confidence': 0.65
        }
    ]
    
    print("🧠 Testing AI Learning System")
    print("=" * 50)
    
    for trade in test_trades:
        insights = ai.learn_from_trade(trade)
        print(f"Trade {trade['trade_id']}: {insights['trade_outcome']}")
        print(f"Learning iteration: {insights['learning_iteration']}")
        print(f"Current win rate: {insights['current_win_rate']:.1%}")
        print()
    
    # Test signal recommendation
    test_signal = {
        'signal_type': 'parabolic_burst',
        'signal_strength': 3.0,
        'symbol': 'BTC'
    }
    
    recommendation = ai.get_signal_recommendation(test_signal)
    print(f"Signal recommendation: {recommendation}")
    
    # Show learning status
    status = ai.get_learning_status()
    print(f"\nAI Learning Status:")
    print(f"Total trades: {status['total_trades']}")
    print(f"Win rate: {status['win_rate']:.1f}%")
    print(f"Learning iterations: {status['learning_iterations']}")
    print(f"Patterns learned: {status['profitable_patterns'] + status['losing_patterns']}") 