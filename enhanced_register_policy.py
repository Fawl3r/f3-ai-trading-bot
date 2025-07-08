#!/usr/bin/env python3
"""
Enhanced Policy Registration with Thompson Sampling Bandit
Automatically allocates traffic and tracks performance
"""

import sqlite3
import numpy as np
import pandas as pd
import torch
import logging
import argparse
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThompsonSamplingBandit:
    """Thompson Sampling bandit for policy selection"""
    
    def __init__(self, db_path='models/policy_bandit.db'):
        self.db_path = db_path
        self.init_database()
        
        logger.info(f"🎰 Thompson Sampling Bandit initialized: {db_path}")
    
    def init_database(self):
        """Initialize SQLite database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Policies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS policies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    baseline_win_rate REAL DEFAULT 0.0,
                    baseline_sharpe REAL DEFAULT 0.0,
                    baseline_max_dd REAL DEFAULT 0.0
                )
            ''')
            
            # Bandit arms table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bandit_arms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    policy_id INTEGER,
                    alpha REAL DEFAULT 1.0,
                    beta REAL DEFAULT 1.0,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0.0,
                    traffic_allocation REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (policy_id) REFERENCES policies (id)
                )
            ''')
            
            # Performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    policy_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    trade_outcome BOOLEAN,
                    pnl REAL,
                    trade_duration INTEGER,
                    symbol TEXT,
                    position_size REAL,
                    win_rate_rolling REAL,
                    sharpe_rolling REAL,
                    drawdown_current REAL,
                    FOREIGN KEY (policy_id) REFERENCES policies (id)
                )
            ''')
            
            # Traffic allocation log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS traffic_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    policy_id INTEGER,
                    allocation_percent REAL,
                    reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (policy_id) REFERENCES policies (id)
                )
            ''')
            
            conn.commit()
            logger.info("✅ Database tables initialized")
    
    def register_policy(self, name: str, model_path: str, policy_type: str = 'supervised', 
                       config: Dict = None, initial_allocation: float = 0.10) -> int:
        """Register new policy with Thompson Sampling"""
        
        logger.info(f"📝 Registering policy: {name}")
        logger.info(f"📁 Model: {model_path}")
        logger.info(f"🎯 Type: {policy_type}")
        logger.info(f"📊 Initial allocation: {initial_allocation:.1%}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model to verify and extract config
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            model_config = checkpoint.get('config', {})
            model_hash = checkpoint.get('model_hash', 'unknown')
            
            # Merge configs
            if config is None:
                config = {}
            config.update(model_config)
            config['model_hash'] = model_hash
            
            logger.info(f"🔑 Model hash: {model_hash}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load model config: {e}")
            config = config or {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert policy
            cursor.execute('''
                INSERT INTO policies (name, type, model_path, config)
                VALUES (?, ?, ?, ?)
            ''', (name, policy_type, model_path, json.dumps(config)))
            
            policy_id = cursor.lastrowid
            
            # Initialize bandit arm with optimistic priors
            cursor.execute('''
                INSERT INTO bandit_arms (policy_id, alpha, beta, traffic_allocation)
                VALUES (?, ?, ?, ?)
            ''', (policy_id, 2.0, 1.0, initial_allocation))  # Optimistic prior
            
            # Log initial allocation
            cursor.execute('''
                INSERT INTO traffic_log (policy_id, allocation_percent, reason)
                VALUES (?, ?, ?)
            ''', (policy_id, initial_allocation, 'Initial registration'))
            
            conn.commit()
        
        # Rebalance traffic allocations
        self.rebalance_traffic()
        
        logger.info(f"✅ Policy registered with ID: {policy_id}")
        return policy_id
    
    def update_performance(self, policy_id: int, trade_outcome: bool, pnl: float, 
                          symbol: str = None, position_size: float = None):
        """Update policy performance and bandit parameters"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Update bandit arm
            cursor.execute('''
                UPDATE bandit_arms 
                SET total_trades = total_trades + 1,
                    wins = wins + ?,
                    total_pnl = total_pnl + ?,
                    alpha = alpha + ?,
                    beta = beta + ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE policy_id = ?
            ''', (1 if trade_outcome else 0, pnl, 
                 1 if trade_outcome else 0, 0 if trade_outcome else 1, policy_id))
            
            # Calculate rolling metrics
            cursor.execute('''
                SELECT trade_outcome, pnl FROM performance_log 
                WHERE policy_id = ? 
                ORDER BY timestamp DESC LIMIT 50
            ''', (policy_id,))
            
            recent_trades = cursor.fetchall()
            
            if recent_trades:
                outcomes = [t[0] for t in recent_trades]
                pnls = [t[1] for t in recent_trades]
                
                win_rate = np.mean(outcomes)
                sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
                
                # Simple drawdown calculation
                cumulative_pnl = np.cumsum(pnls[::-1])  # Reverse for chronological order
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = np.min(cumulative_pnl - running_max)
            else:
                win_rate = 0.5
                sharpe = 0.0
                drawdown = 0.0
            
            # Log performance
            cursor.execute('''
                INSERT INTO performance_log 
                (policy_id, trade_outcome, pnl, symbol, position_size, 
                 win_rate_rolling, sharpe_rolling, drawdown_current)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (policy_id, trade_outcome, pnl, symbol, position_size,
                 win_rate, sharpe, drawdown))
            
            conn.commit()
        
        # Trigger rebalancing if significant performance change
        if len(recent_trades) % 20 == 0:  # Every 20 trades
            self.rebalance_traffic()
    
    def select_policy(self) -> int:
        """Select policy using Thompson Sampling"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get active bandit arms
            cursor.execute('''
                SELECT ba.policy_id, ba.alpha, ba.beta, p.name
                FROM bandit_arms ba
                JOIN policies p ON ba.policy_id = p.id
                WHERE p.is_active = 1
            ''')
            
            arms = cursor.fetchall()
        
        if not arms:
            logger.warning("⚠️  No active policies found")
            return None
        
        # Thompson Sampling
        samples = []
        for policy_id, alpha, beta, name in arms:
            sample = np.random.beta(alpha, beta)
            samples.append((policy_id, sample, name))
        
        # Select best sample
        selected = max(samples, key=lambda x: x[1])
        policy_id, sample_value, name = selected
        
        logger.debug(f"🎯 Selected policy: {name} (ID: {policy_id}, Sample: {sample_value:.3f})")
        
        return policy_id
    
    def rebalance_traffic(self):
        """Rebalance traffic allocation based on Thompson Sampling"""
        
        logger.info("⚖️  Rebalancing traffic allocation...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all active policies with their bandit parameters
            cursor.execute('''
                SELECT ba.policy_id, ba.alpha, ba.beta, ba.total_trades, p.name
                FROM bandit_arms ba
                JOIN policies p ON ba.policy_id = p.id
                WHERE p.is_active = 1
            ''')
            
            arms = cursor.fetchall()
        
        if len(arms) <= 1:
            logger.info("📊 Only one active policy - no rebalancing needed")
            return
        
        # Calculate new allocations
        new_allocations = {}
        
        # Minimum allocation for new policies (first 100 trades)
        min_allocation = 0.05
        
        # Sample multiple times and take average for stable allocation
        n_samples = 1000
        policy_scores = {arm[0]: [] for arm in arms}
        
        for _ in range(n_samples):
            for policy_id, alpha, beta, total_trades, name in arms:
                sample = np.random.beta(alpha, beta)
                policy_scores[policy_id].append(sample)
        
        # Calculate mean scores
        mean_scores = {pid: np.mean(scores) for pid, scores in policy_scores.items()}
        total_score = sum(mean_scores.values())
        
        # Calculate allocations with minimum guarantees
        for policy_id, alpha, beta, total_trades, name in arms:
            if total_trades < 100:  # New policy protection
                allocation = max(min_allocation, mean_scores[policy_id] / total_score)
            else:
                allocation = mean_scores[policy_id] / total_score
            
            new_allocations[policy_id] = allocation
        
        # Normalize to ensure sum = 1
        total_allocation = sum(new_allocations.values())
        new_allocations = {pid: alloc / total_allocation 
                          for pid, alloc in new_allocations.items()}
        
        # Update database
        for policy_id, allocation in new_allocations.items():
            cursor.execute('''
                UPDATE bandit_arms 
                SET traffic_allocation = ?, last_updated = CURRENT_TIMESTAMP
                WHERE policy_id = ?
            ''', (allocation, policy_id))
            
            # Log allocation change
            cursor.execute('''
                INSERT INTO traffic_log (policy_id, allocation_percent, reason)
                VALUES (?, ?, ?)
            ''', (policy_id, allocation, 'Automatic rebalancing'))
        
        conn.commit()
        
        # Log results
        cursor.execute('''
            SELECT p.name, ba.traffic_allocation, ba.alpha, ba.beta, ba.total_trades
            FROM bandit_arms ba
            JOIN policies p ON ba.policy_id = p.id
            WHERE p.is_active = 1
            ORDER BY ba.traffic_allocation DESC
        ''')
        
        allocations = cursor.fetchall()
        
        logger.info("📊 New traffic allocations:")
        for name, allocation, alpha, beta, trades in allocations:
            win_rate = alpha / (alpha + beta)
            logger.info(f"  {name}: {allocation:.1%} (WR: {win_rate:.1%}, Trades: {trades})")
    
    def get_policy_stats(self) -> pd.DataFrame:
        """Get comprehensive policy statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT 
                    p.name,
                    p.type,
                    p.created_at,
                    ba.alpha,
                    ba.beta,
                    ba.total_trades,
                    ba.wins,
                    ba.total_pnl,
                    ba.traffic_allocation,
                    ROUND(ba.alpha / (ba.alpha + ba.beta), 3) as win_rate,
                    ROUND(ba.total_pnl / NULLIF(ba.total_trades, 0), 2) as avg_pnl_per_trade
                FROM policies p
                LEFT JOIN bandit_arms ba ON p.id = ba.policy_id
                WHERE p.is_active = 1
                ORDER BY ba.traffic_allocation DESC
            '''
            
            df = pd.read_sql_query(query, conn)
        
        return df
    
    def deactivate_policy(self, policy_id: int, reason: str = "Manual deactivation"):
        """Deactivate a policy"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE policies SET is_active = 0 WHERE id = ?
            ''', (policy_id,))
            
            cursor.execute('''
                UPDATE bandit_arms SET traffic_allocation = 0 WHERE policy_id = ?
            ''', (policy_id,))
            
            cursor.execute('''
                INSERT INTO traffic_log (policy_id, allocation_percent, reason)
                VALUES (?, ?, ?)
            ''', (policy_id, 0.0, reason))
            
            conn.commit()
        
        logger.info(f"❌ Policy {policy_id} deactivated: {reason}")
        self.rebalance_traffic()

def simulate_policy_performance(bandit: ThompsonSamplingBandit, policy_id: int, 
                               n_trades: int = 50, base_win_rate: float = 0.65):
    """Simulate policy performance for testing"""
    
    logger.info(f"🎲 Simulating {n_trades} trades for policy {policy_id}")
    
    for i in range(n_trades):
        # Simulate trade outcome
        trade_outcome = np.random.random() < base_win_rate
        
        # Simulate PnL
        if trade_outcome:
            pnl = np.random.normal(100, 50)  # Winning trade
        else:
            pnl = np.random.normal(-80, 30)  # Losing trade
        
        # Update performance
        bandit.update_performance(
            policy_id=policy_id,
            trade_outcome=trade_outcome,
            pnl=pnl,
            symbol=np.random.choice(['SOL', 'BTC', 'ETH']),
            position_size=np.random.uniform(0.1, 1.0)
        )
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Simulated {i + 1}/{n_trades} trades")

def main():
    """Main registration and testing function"""
    
    parser = argparse.ArgumentParser(description='Enhanced Policy Registration')
    parser.add_argument('--path', type=str, help='Path to model file')
    parser.add_argument('--name', type=str, help='Policy name')
    parser.add_argument('--type', type=str, default='supervised', 
                       choices=['supervised', 'reinforcement', 'ensemble'],
                       help='Policy type')
    parser.add_argument('--simulate', action='store_true', help='Run simulation')
    parser.add_argument('--stats', action='store_true', help='Show policy stats')
    parser.add_argument('--select', action='store_true', help='Test policy selection')
    
    args = parser.parse_args()
    
    # Initialize bandit
    bandit = ThompsonSamplingBandit()
    
    if args.path:
        # Register new policy
        if not args.name:
            # Generate name from model hash
            try:
                checkpoint = torch.load(args.path, map_location='cpu')
                model_hash = checkpoint.get('model_hash', 'unknown')
                timestamp = checkpoint.get('timestamp', datetime.now().strftime('%Y%m%d'))
                args.name = f"{args.type}_{timestamp}_{model_hash}"
            except:
                args.name = f"{args.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        policy_id = bandit.register_policy(
            name=args.name,
            model_path=args.path,
            policy_type=args.type
        )
        
        logger.info(f"✅ Policy registered: {args.name} (ID: {policy_id})")
        
        # Simulate some performance if requested
        if args.simulate:
            base_win_rate = 0.65 if args.type == 'supervised' else 0.70
            simulate_policy_performance(bandit, policy_id, n_trades=50, 
                                       base_win_rate=base_win_rate)
    
    if args.stats:
        # Show policy statistics
        stats_df = bandit.get_policy_stats()
        print("\n📊 Policy Performance Statistics:")
        print("=" * 80)
        print(stats_df.to_string(index=False))
        
        # Show traffic allocation breakdown
        total_allocation = stats_df['traffic_allocation'].sum()
        print(f"\n📈 Total Traffic Allocation: {total_allocation:.1%}")
        
        if len(stats_df) > 1:
            print("\n🎯 Thompson Sampling Status:")
            for _, row in stats_df.iterrows():
                confidence = row['alpha'] / (row['alpha'] + row['beta'])
                uncertainty = 1 / (row['alpha'] + row['beta'])
                print(f"  {row['name']}: {row['traffic_allocation']:.1%} "
                     f"(Confidence: {confidence:.3f}, Uncertainty: {uncertainty:.3f})")
    
    if args.select:
        # Test policy selection
        logger.info("🎲 Testing policy selection...")
        selections = {}
        
        for _ in range(100):
            selected_id = bandit.select_policy()
            if selected_id:
                selections[selected_id] = selections.get(selected_id, 0) + 1
        
        print("\n🎯 Selection Distribution (100 samples):")
        stats_df = bandit.get_policy_stats()
        for policy_id, count in selections.items():
            policy_name = stats_df[stats_df.index == policy_id - 1]['name'].iloc[0]
            print(f"  {policy_name}: {count}% of selections")

if __name__ == "__main__":
    main() 