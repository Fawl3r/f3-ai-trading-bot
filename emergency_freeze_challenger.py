#!/usr/bin/env python3
"""
🚨 EMERGENCY: Freeze Challenger Policy
Immediately sets traffic allocation to 0% and reverts to production
"""

import sqlite3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def emergency_freeze_challenger():
    """Execute emergency freeze of challenger policy"""
    
    db_path = "models/policy_bandit.db"
    challenger_sha = "reinforcement_20250707_154526_eee02a74"
    
    logger.critical(f"🚨 EMERGENCY FREEZE EXECUTING")
    logger.critical(f"Policy: {challenger_sha}")
    logger.critical(f"Reason: Drawdown 12.4% > 4% threshold")
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Step 1: Set challenger traffic to 0%
            cursor.execute('''
                UPDATE bandit_arms 
                SET traffic_allocation = 0.0,
                    last_updated = CURRENT_TIMESTAMP
                WHERE policy_id IN (
                    SELECT id FROM policies 
                    WHERE name LIKE '%eee02a74%'
                )
            ''')
            
            # Step 2: Deactivate challenger policy
            cursor.execute('''
                UPDATE policies 
                SET is_active = 0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE name LIKE '%eee02a74%'
            ''')
            
            # Step 3: Set production to 100% traffic
            cursor.execute('''
                UPDATE bandit_arms 
                SET traffic_allocation = 1.0,
                    last_updated = CURRENT_TIMESTAMP
                WHERE policy_id = 1
            ''')
            
            # Step 4: Log emergency action
            cursor.execute('''
                INSERT INTO traffic_log (policy_id, allocation_percent, reason)
                SELECT id, 0.0, 'EMERGENCY FREEZE: DD 12.4% > 4% threshold'
                FROM policies WHERE name LIKE '%eee02a74%'
            ''')
            
            conn.commit()
            
            # Verify freeze
            cursor.execute('''
                SELECT p.name, ba.traffic_allocation, p.is_active
                FROM policies p
                JOIN bandit_arms ba ON p.id = ba.policy_id
                WHERE p.name LIKE '%eee02a74%'
            ''')
            
            result = cursor.fetchone()
            if result:
                name, traffic, active = result
                logger.critical(f"✅ FREEZE CONFIRMED")
                logger.critical(f"   Policy: {name}")
                logger.critical(f"   Traffic: {traffic:.1%}")
                logger.critical(f"   Active: {bool(active)}")
            
            # Check production status
            cursor.execute('''
                SELECT p.name, ba.traffic_allocation
                FROM policies p
                JOIN bandit_arms ba ON p.id = ba.policy_id
                WHERE ba.policy_id = 1
            ''')
            
            prod_result = cursor.fetchone()
            if prod_result:
                prod_name, prod_traffic = prod_result
                logger.critical(f"✅ PRODUCTION RESTORED")
                logger.critical(f"   Policy: {prod_name or 'Production'}")
                logger.critical(f"   Traffic: {prod_traffic:.1%}")
            
            return True
            
    except Exception as e:
        logger.critical(f"❌ EMERGENCY FREEZE FAILED: {e}")
        return False

def verify_freeze_status():
    """Verify that the freeze was successful"""
    
    db_path = "models/policy_bandit.db"
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.name, ba.traffic_allocation, p.is_active,
                   ba.total_trades, ba.total_pnl
            FROM policies p
            JOIN bandit_arms ba ON p.id = ba.policy_id
            ORDER BY ba.traffic_allocation DESC
        ''')
        
        policies = cursor.fetchall()
        
        logger.info("="*60)
        logger.info("🚨 POST-FREEZE POLICY STATUS")
        logger.info("="*60)
        
        for name, traffic, active, trades, pnl in policies:
            status = "ACTIVE" if active else "FROZEN"
            logger.info(f"{name or 'Production'}: {traffic:.1%} traffic, {status}")
            logger.info(f"  Trades: {trades}, PnL: ${pnl:.2f}")
        
        logger.info("="*60)

if __name__ == "__main__":
    logger.critical("🚨 EXECUTING EMERGENCY CHALLENGER FREEZE")
    
    success = emergency_freeze_challenger()
    
    if success:
        logger.critical("✅ EMERGENCY FREEZE COMPLETED")
        verify_freeze_status()
        
        print("\n" + "="*60)
        print("🚨 EMERGENCY ACTIONS COMPLETED")
        print("="*60)
        print("✅ Challenger traffic: 0%")
        print("✅ Challenger deactivated")
        print("✅ Production restored: 100%")
        print("📋 Next: Diagnose losing trades and retrain PPO")
        print("="*60)
        
    else:
        logger.critical("❌ EMERGENCY FREEZE FAILED - MANUAL INTERVENTION REQUIRED") 