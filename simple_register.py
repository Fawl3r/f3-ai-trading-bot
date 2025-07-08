#!/usr/bin/env python3
"""
Simple Policy Registration for Expansion Models
Works with existing database schema
"""

import sqlite3
import json
import logging
import argparse
import os
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_expansion_models():
    """Register all expansion models with existing schema"""
    
    db_path = 'models/policy_bandit.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Models to register
    models_to_register = [
        {
            'name': 'lightgbm_tsa_mae',
            'type': 'supervised',
            'model_path': 'models/lgbm_SOL_20250707_191855_0a65ca5b.pkl',
            'config': {
                'model_type': 'lightgbm',
                'encoder_path': 'models/encoder_20250707_153740_b59c66da.pt',
                'symbols': ['SOL', 'BTC', 'ETH'],
                'features': 79,
                'accuracy': 0.3522
            }
        },
        {
            'name': 'timesnet_longrange',
            'type': 'supervised', 
            'model_path': 'models/timesnet_SOL_20250707_204629_93387ccf.pt',
            'config': {
                'model_type': 'timesnet',
                'window_size': 512,
                'd_model': 64,
                'heads': 4,
                'layers': 3,
                'symbols': ['SOL', 'BTC', 'ETH'],
                'accuracy': 0.8688
            }
        },
        {
            'name': 'ppo_strict_enhanced',
            'type': 'reinforcement',
            'model_path': 'models/ppo_strict_20250707_161252.pt',
            'config': {
                'model_type': 'ppo',
                'max_drawdown': 0.002,
                'max_pyramid_units': 3,
                'win_rate': 0.353,
                'symbols': ['SOL', 'BTC', 'ETH']
            }
        }
    ]
    
    registered_policies = []
    
    for model_info in models_to_register:
        if not os.path.exists(model_info['model_path']):
            logger.warning(f"Model file not found: {model_info['model_path']}")
            continue
        
        try:
            # Insert into policies table
            cursor.execute('''
                INSERT INTO policies (name, type, model_path, config, is_active,
                                    baseline_win_rate, baseline_sharpe, baseline_max_dd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_info['name'],
                model_info['type'],
                model_info['model_path'],
                json.dumps(model_info['config']),
                True,
                model_info['config'].get('accuracy', 0.5),
                model_info['config'].get('sharpe', 1.5),
                model_info['config'].get('max_drawdown', 0.03)
            ))
            
            policy_id = cursor.lastrowid
            
            # Create corresponding bandit arm
            cursor.execute('''
                INSERT INTO bandit_arms (policy_id, alpha, beta, traffic_allocation)
                VALUES (?, ?, ?, ?)
            ''', (
                policy_id,
                1.0,  # Prior alpha
                1.0,  # Prior beta
                0.05  # 5% initial traffic
            ))
            
            registered_policies.append({
                'id': policy_id,
                'name': model_info['name'],
                'type': model_info['type']
            })
            
            logger.info(f"✅ Registered: {model_info['name']} (ID: {policy_id})")
            
        except Exception as e:
            logger.error(f"❌ Failed to register {model_info['name']}: {e}")
    
    # Rebalance traffic allocation
    cursor.execute('SELECT COUNT(*) FROM bandit_arms WHERE traffic_allocation > 0')
    active_arms = cursor.fetchone()[0]
    
    if active_arms > 0:
        # Scale down existing allocations to make room
        scale_factor = 0.85 / active_arms  # Leave 15% buffer
        
        cursor.execute('''
            UPDATE bandit_arms 
            SET traffic_allocation = traffic_allocation * ?
            WHERE traffic_allocation > 0
        ''', (scale_factor,))
        
        logger.info(f"🔄 Rebalanced {active_arms} arms with scale factor {scale_factor:.3f}")
    
    conn.commit()
    conn.close()
    
    return registered_policies

def show_traffic_summary():
    """Show current traffic allocation summary"""
    
    db_path = 'models/policy_bandit.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT p.name, p.type, ba.traffic_allocation, ba.alpha, ba.beta,
               (ba.alpha / (ba.alpha + ba.beta)) as win_rate_estimate
        FROM policies p
        JOIN bandit_arms ba ON p.id = ba.policy_id
        WHERE p.is_active = 1
        ORDER BY ba.traffic_allocation DESC
    ''')
    
    policies = cursor.fetchall()
    conn.close()
    
    print("\n📊 Current Policy Traffic Allocation:")
    print("-" * 80)
    
    total_allocation = 0
    for policy in policies:
        name, ptype, allocation, alpha, beta, win_rate = policy
        total_allocation += allocation
        
        print(f"🎯 {name}")
        print(f"   Type: {ptype}")
        print(f"   Traffic: {allocation:.1%}")
        print(f"   Thompson Score: α={alpha:.1f}, β={beta:.1f}")
        print(f"   Estimated Win Rate: {win_rate:.1%}")
        print()
    
    print(f"📈 Total Allocation: {total_allocation:.1%}")
    print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description='Register expansion models')
    parser.add_argument('--register', action='store_true', help='Register all models')
    parser.add_argument('--summary', action='store_true', help='Show traffic summary')
    
    args = parser.parse_args()
    
    if args.summary:
        show_traffic_summary()
        return
    
    if args.register:
        logger.info("🚀 Registering expansion models...")
        registered = register_expansion_models()
        
        logger.info(f"✅ Successfully registered {len(registered)} models:")
        for policy in registered:
            logger.info(f"   - {policy['name']} ({policy['type']})")
        
        logger.info("\n📊 Updated traffic allocation:")
        show_traffic_summary()
    else:
        print("Usage: python simple_register.py --register")
        print("       python simple_register.py --summary")

if __name__ == "__main__":
    main() 