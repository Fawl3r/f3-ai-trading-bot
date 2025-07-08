#!/usr/bin/env python3
"""
Enhanced Policy Registration System
Handles supervised (LightGBM, TimesNet) and reinforcement learning (PPO) models
with proper traffic allocation and metadata tracking
"""

import sqlite3
import json
import logging
import argparse
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyRegistry:
    """Enhanced policy registry with support for multiple model types"""
    
    def __init__(self, db_path='models/policy_bandit.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced policies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS policies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id TEXT UNIQUE NOT NULL,
                model_type TEXT NOT NULL,  -- 'supervised', 'reinforcement', 'ensemble'
                model_subtype TEXT,        -- 'lightgbm', 'timesnet', 'ppo', 'meta'
                model_path TEXT NOT NULL,
                symbols TEXT,              -- JSON array of symbols
                config TEXT,               -- JSON configuration
                metadata TEXT,             -- JSON metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'  -- 'active', 'inactive', 'testing'
            )
        ''')
        
        # Enhanced bandit_arms table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bandit_arms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id TEXT NOT NULL,
                arm_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                traffic_allocation REAL DEFAULT 0.05,  -- Start with 5%
                alpha REAL DEFAULT 1.0,
                beta REAL DEFAULT 1.0,
                total_trades INTEGER DEFAULT 0,
                successful_trades INTEGER DEFAULT 0,
                pf_150 REAL DEFAULT 0.0,
                dd_150 REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                FOREIGN KEY (policy_id) REFERENCES policies (policy_id)
            )
        ''')
        
        # Performance tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id TEXT NOT NULL,
                trade_id TEXT,
                symbol TEXT,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                pnl REAL,
                return_pct REAL,
                drawdown_pct REAL,
                confidence REAL,
                model_prediction TEXT,  -- JSON prediction details
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (policy_id) REFERENCES policies (policy_id)
            )
        ''')
        
        # Model correlation tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id_1 TEXT NOT NULL,
                policy_id_2 TEXT NOT NULL,
                correlation_score REAL,
                signal_overlap REAL,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (policy_id_1) REFERENCES policies (policy_id),
                FOREIGN KEY (policy_id_2) REFERENCES policies (policy_id)
            )
        ''')
        
        # Risk controls table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_controls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id TEXT NOT NULL,
                control_type TEXT NOT NULL,  -- 'traffic_cap', 'dd_limit', 'correlation_limit'
                control_value REAL,
                threshold REAL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (policy_id) REFERENCES policies (policy_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Enhanced database schema initialized")
    
    def register_policy(self, model_path: str, model_type: str, 
                       model_subtype: str = None, symbols: List[str] = None,
                       config: Dict = None, metadata: Dict = None,
                       initial_traffic: float = 0.05) -> str:
        """Register a new policy with enhanced metadata"""
        
        # Generate policy ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_hash = hashlib.sha256(f"{model_path}_{timestamp}".encode()).hexdigest()[:8]
        policy_id = f"{model_type}_{timestamp}_{model_hash}"
        
        # Default values
        if symbols is None:
            symbols = ['SOL', 'BTC', 'ETH']
        if config is None:
            config = {}
        if metadata is None:
            metadata = {}
        
        # Add registration metadata
        metadata.update({
            'registered_at': datetime.now().isoformat(),
            'model_file_size': os.path.getsize(model_path) if os.path.exists(model_path) else 0,
            'model_hash': model_hash
        })
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert policy
            cursor.execute('''
                INSERT INTO policies (policy_id, model_type, model_subtype, model_path, 
                                    symbols, config, metadata, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                policy_id,
                model_type,
                model_subtype,
                model_path,
                json.dumps(symbols),
                json.dumps(config),
                json.dumps(metadata),
                'active'
            ))
            
            # Create bandit arm
            arm_name = f"{model_subtype}_{model_hash}" if model_subtype else f"{model_type}_{model_hash}"
            
            cursor.execute('''
                INSERT INTO bandit_arms (policy_id, arm_name, model_type, traffic_allocation,
                                       alpha, beta, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                policy_id,
                arm_name,
                model_type,
                initial_traffic,
                1.0,  # Prior alpha
                1.0,  # Prior beta
                'active'
            ))
            
            # Add risk controls
            self._add_default_risk_controls(cursor, policy_id, model_type)
            
            # Rebalance traffic allocation
            self._rebalance_traffic_allocation(cursor, model_type)
            
            conn.commit()
            
            logger.info(f"✅ Policy registered: {policy_id}")
            logger.info(f"📊 Model type: {model_type} ({model_subtype})")
            logger.info(f"🎯 Initial traffic: {initial_traffic:.1%}")
            logger.info(f"📁 Model path: {model_path}")
            
            return policy_id
            
        except sqlite3.IntegrityError as e:
            logger.error(f"❌ Policy registration failed: {e}")
            return None
        finally:
            conn.close()
    
    def _add_default_risk_controls(self, cursor, policy_id: str, model_type: str):
        """Add default risk controls for new policies"""
        
        # Traffic cap based on model type
        if model_type == 'supervised':
            traffic_cap = 0.25  # 25% max for supervised models
        elif model_type == 'reinforcement':
            traffic_cap = 0.15  # 15% max for RL models (more risky)
        elif model_type == 'ensemble':
            traffic_cap = 0.35  # 35% max for ensemble models
        else:
            traffic_cap = 0.20  # Default 20%
        
        # Add traffic cap control
        cursor.execute('''
            INSERT INTO risk_controls (policy_id, control_type, control_value, threshold)
            VALUES (?, ?, ?, ?)
        ''', (policy_id, 'traffic_cap', traffic_cap, 500))  # 500 trades threshold
        
        # Add drawdown limit
        dd_limit = 0.03 if model_type == 'reinforcement' else 0.05  # Stricter for RL
        cursor.execute('''
            INSERT INTO risk_controls (policy_id, control_type, control_value, threshold)
            VALUES (?, ?, ?, ?)
        ''', (policy_id, 'dd_limit', dd_limit, 150))  # 150 trades threshold
        
        # Add correlation limit
        cursor.execute('''
            INSERT INTO risk_controls (policy_id, control_type, control_value, threshold)
            VALUES (?, ?, ?, ?)
        ''', (policy_id, 'correlation_limit', 0.85, 100))  # Max 85% correlation
    
    def _rebalance_traffic_allocation(self, cursor, new_model_type: str):
        """Rebalance traffic allocation when new model is added"""
        
        # Get all active arms
        cursor.execute('''
            SELECT policy_id, arm_name, model_type, traffic_allocation, total_trades
            FROM bandit_arms 
            WHERE status = 'active'
            ORDER BY policy_id
        ''')
        
        arms = cursor.fetchall()
        
        if len(arms) <= 1:
            return  # No rebalancing needed
        
        # Calculate current total allocation
        total_allocation = sum(arm[3] for arm in arms)
        
        # If over 100%, scale down proportionally
        if total_allocation > 1.0:
            scale_factor = 0.95 / total_allocation  # Leave 5% buffer
            
            for arm in arms:
                policy_id, arm_name, model_type, allocation, trades = arm
                new_allocation = allocation * scale_factor
                
                cursor.execute('''
                    UPDATE bandit_arms 
                    SET traffic_allocation = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE policy_id = ?
                ''', (new_allocation, policy_id))
            
            logger.info(f"🔄 Traffic rebalanced: {len(arms)} arms, scale factor: {scale_factor:.3f}")
    
    def get_active_policies(self) -> List[Dict]:
        """Get all active policies with current performance"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.policy_id, p.model_type, p.model_subtype, p.model_path,
                   p.symbols, p.config, p.metadata, p.created_at,
                   ba.arm_name, ba.traffic_allocation, ba.alpha, ba.beta,
                   ba.total_trades, ba.successful_trades, ba.pf_150, ba.dd_150
            FROM policies p
            JOIN bandit_arms ba ON p.policy_id = ba.policy_id
            WHERE p.status = 'active' AND ba.status = 'active'
            ORDER BY ba.traffic_allocation DESC
        ''')
        
        policies = []
        for row in cursor.fetchall():
            policy = {
                'policy_id': row[0],
                'model_type': row[1],
                'model_subtype': row[2],
                'model_path': row[3],
                'symbols': json.loads(row[4]),
                'config': json.loads(row[5]),
                'metadata': json.loads(row[6]),
                'created_at': row[7],
                'arm_name': row[8],
                'traffic_allocation': row[9],
                'alpha': row[10],
                'beta': row[11],
                'total_trades': row[12],
                'successful_trades': row[13],
                'pf_150': row[14],
                'dd_150': row[15],
                'win_rate': row[13] / max(row[12], 1),
                'thompson_score': np.random.beta(row[10], row[11])
            }
            policies.append(policy)
        
        conn.close()
        return policies
    
    def update_performance(self, policy_id: str, trade_data: Dict):
        """Update policy performance with trade data"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Log trade
        cursor.execute('''
            INSERT INTO performance_log (policy_id, trade_id, symbol, entry_time, 
                                       exit_time, pnl, return_pct, drawdown_pct, 
                                       confidence, model_prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            policy_id,
            trade_data.get('trade_id'),
            trade_data.get('symbol'),
            trade_data.get('entry_time'),
            trade_data.get('exit_time'),
            trade_data.get('pnl', 0),
            trade_data.get('return_pct', 0),
            trade_data.get('drawdown_pct', 0),
            trade_data.get('confidence', 0),
            json.dumps(trade_data.get('prediction', {}))
        ))
        
        # Update bandit arm
        is_successful = trade_data.get('pnl', 0) > 0
        
        cursor.execute('''
            UPDATE bandit_arms 
            SET total_trades = total_trades + 1,
                successful_trades = successful_trades + ?,
                alpha = alpha + ?,
                beta = beta + ?,
                last_updated = CURRENT_TIMESTAMP
            WHERE policy_id = ?
        ''', (
            1 if is_successful else 0,
            1 if is_successful else 0,
            0 if is_successful else 1,
            policy_id
        ))
        
        # Update 150-trade metrics if we have enough data
        cursor.execute('''
            SELECT COUNT(*), AVG(pnl), AVG(ABS(drawdown_pct))
            FROM performance_log 
            WHERE policy_id = ? 
            ORDER BY created_at DESC 
            LIMIT 150
        ''', (policy_id,))
        
        result = cursor.fetchone()
        if result and result[0] >= 150:
            trade_count, avg_pnl, avg_dd = result
            
            # Calculate profit factor (simplified)
            cursor.execute('''
                SELECT SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END),
                       SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END)
                FROM performance_log 
                WHERE policy_id = ? 
                ORDER BY created_at DESC 
                LIMIT 150
            ''', (policy_id,))
            
            profits, losses = cursor.fetchone()
            pf_150 = profits / max(losses, 0.01) if losses > 0 else 0
            
            cursor.execute('''
                UPDATE bandit_arms 
                SET pf_150 = ?, dd_150 = ?
                WHERE policy_id = ?
            ''', (pf_150, avg_dd * 100, policy_id))
        
        conn.commit()
        conn.close()
    
    def check_promotion_criteria(self, policy_id: str) -> Dict:
        """Check if policy meets promotion criteria"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ba.total_trades, ba.pf_150, ba.dd_150, ba.traffic_allocation,
                   rc.control_value as traffic_cap
            FROM bandit_arms ba
            LEFT JOIN risk_controls rc ON ba.policy_id = rc.policy_id 
                                        AND rc.control_type = 'traffic_cap'
            WHERE ba.policy_id = ? AND ba.status = 'active'
        ''', (policy_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return {'eligible': False, 'reason': 'Policy not found'}
        
        trades, pf_150, dd_150, current_traffic, traffic_cap = result
        
        # Check promotion criteria
        criteria = {
            'min_trades': trades >= 150,
            'min_pf': pf_150 >= 2.0,
            'max_dd': dd_150 <= 3.0,
            'under_cap': current_traffic < (traffic_cap or 0.25)
        }
        
        eligible = all(criteria.values())
        
        return {
            'eligible': eligible,
            'criteria': criteria,
            'current_metrics': {
                'trades': trades,
                'pf_150': pf_150,
                'dd_150': dd_150,
                'traffic': current_traffic,
                'cap': traffic_cap
            },
            'promotion_factor': 1.5 if eligible else 1.0
        }
    
    def promote_policy(self, policy_id: str) -> bool:
        """Promote policy by increasing traffic allocation"""
        
        promotion_check = self.check_promotion_criteria(policy_id)
        
        if not promotion_check['eligible']:
            logger.warning(f"❌ Policy {policy_id} not eligible for promotion")
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Increase traffic allocation
        factor = promotion_check['promotion_factor']
        
        cursor.execute('''
            UPDATE bandit_arms 
            SET traffic_allocation = traffic_allocation * ?,
                last_updated = CURRENT_TIMESTAMP
            WHERE policy_id = ?
        ''', (factor, policy_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"🚀 Policy {policy_id} promoted: traffic ×{factor}")
        return True
    
    def throttle_policy(self, policy_id: str, reason: str = "performance") -> bool:
        """Throttle policy by reducing traffic to 0%"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE bandit_arms 
            SET traffic_allocation = 0.0,
                status = 'throttled',
                last_updated = CURRENT_TIMESTAMP
            WHERE policy_id = ?
        ''', (policy_id,))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"⚠️ Policy {policy_id} throttled: {reason}")
        return True
    
    def get_traffic_summary(self) -> Dict:
        """Get summary of current traffic allocation"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.model_type, p.model_subtype, ba.arm_name, 
                   ba.traffic_allocation, ba.total_trades, ba.pf_150, ba.dd_150
            FROM policies p
            JOIN bandit_arms ba ON p.policy_id = ba.policy_id
            WHERE p.status = 'active'
            ORDER BY ba.traffic_allocation DESC
        ''')
        
        arms = cursor.fetchall()
        conn.close()
        
        summary = {
            'total_arms': len(arms),
            'total_allocation': sum(arm[3] for arm in arms),
            'by_type': {},
            'arms': []
        }
        
        for arm in arms:
            model_type, subtype, arm_name, allocation, trades, pf, dd = arm
            
            # Group by type
            if model_type not in summary['by_type']:
                summary['by_type'][model_type] = {
                    'count': 0,
                    'allocation': 0,
                    'subtypes': {}
                }
            
            summary['by_type'][model_type]['count'] += 1
            summary['by_type'][model_type]['allocation'] += allocation
            
            if subtype:
                if subtype not in summary['by_type'][model_type]['subtypes']:
                    summary['by_type'][model_type]['subtypes'][subtype] = {
                        'count': 0,
                        'allocation': 0
                    }
                summary['by_type'][model_type]['subtypes'][subtype]['count'] += 1
                summary['by_type'][model_type]['subtypes'][subtype]['allocation'] += allocation
            
            # Individual arm info
            summary['arms'].append({
                'name': arm_name,
                'type': f"{model_type}_{subtype}" if subtype else model_type,
                'allocation': allocation,
                'trades': trades,
                'pf_150': pf,
                'dd_150': dd
            })
        
        return summary

def main():
    """Main function for policy registration"""
    
    parser = argparse.ArgumentParser(description='Register trading policy')
    parser.add_argument('--path', required=True, help='Path to model file')
    parser.add_argument('--type', required=True, 
                       choices=['supervised', 'reinforcement', 'ensemble'],
                       help='Model type')
    parser.add_argument('--subtype', help='Model subtype (lightgbm, timesnet, ppo, meta)')
    parser.add_argument('--symbols', default='SOL,BTC,ETH', help='Comma-separated symbols')
    parser.add_argument('--traffic', type=float, default=0.05, help='Initial traffic allocation')
    parser.add_argument('--config', help='JSON configuration string')
    parser.add_argument('--list', action='store_true', help='List active policies')
    parser.add_argument('--summary', action='store_true', help='Show traffic summary')
    
    args = parser.parse_args()
    
    registry = PolicyRegistry()
    
    if args.list:
        # List active policies
        policies = registry.get_active_policies()
        
        print("\n📊 Active Policies:")
        print("-" * 80)
        for policy in policies:
            print(f"🎯 {policy['policy_id']}")
            print(f"   Type: {policy['model_type']} ({policy['model_subtype']})")
            print(f"   Traffic: {policy['traffic_allocation']:.1%}")
            print(f"   Trades: {policy['total_trades']}")
            print(f"   PF: {policy['pf_150']:.2f}, DD: {policy['dd_150']:.1f}%")
            print()
        
        return
    
    if args.summary:
        # Show traffic summary
        summary = registry.get_traffic_summary()
        
        print("\n📊 Traffic Allocation Summary:")
        print("-" * 50)
        print(f"Total Arms: {summary['total_arms']}")
        print(f"Total Allocation: {summary['total_allocation']:.1%}")
        print()
        
        for model_type, info in summary['by_type'].items():
            print(f"🎯 {model_type.upper()}: {info['allocation']:.1%} ({info['count']} arms)")
            for subtype, sub_info in info['subtypes'].items():
                print(f"   └─ {subtype}: {sub_info['allocation']:.1%} ({sub_info['count']} arms)")
        
        return
    
    # Register new policy
    if not os.path.exists(args.path):
        logger.error(f"❌ Model file not found: {args.path}")
        return
    
    symbols = args.symbols.split(',')
    config = json.loads(args.config) if args.config else {}
    
    policy_id = registry.register_policy(
        model_path=args.path,
        model_type=args.type,
        model_subtype=args.subtype,
        symbols=symbols,
        config=config,
        initial_traffic=args.traffic
    )
    
    if policy_id:
        print(f"\n✅ Policy registered successfully!")
        print(f"🆔 Policy ID: {policy_id}")
        print(f"📊 Type: {args.type} ({args.subtype})")
        print(f"🎯 Initial Traffic: {args.traffic:.1%}")
        
        # Show updated summary
        summary = registry.get_traffic_summary()
        print(f"\n📈 Total Allocation: {summary['total_allocation']:.1%}")
    else:
        print("❌ Policy registration failed!")

if __name__ == "__main__":
    main() 