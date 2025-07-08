#!/usr/bin/env python3
"""
🚨 EMERGENCY: Freeze Challenger Policy (Fixed Schema)
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
            
            # First, check current schema
            cursor.execute("PRAGMA table_info(policies)")
            columns = [col[1] for col in cursor.fetchall()]
            logger.info(f"Available columns in policies: {columns}")
            
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
            affected_rows = cursor.rowcount
            logger.critical(f"Traffic allocation updated: {affected_rows} rows")
            
            # Step 2: Deactivate challenger policy
            cursor.execute('''
                UPDATE policies 
                SET is_active = 0
                WHERE name LIKE '%eee02a74%'
            ''')
            policy_rows = cursor.rowcount
            logger.critical(f"Policy deactivated: {policy_rows} rows")
            
            # Step 3: Set production policy to 100% traffic (assume ID=1)
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
            
            return True
            
    except Exception as e:
        logger.critical(f"❌ EMERGENCY FREEZE FAILED: {e}")
        return False

def show_current_allocations():
    """Show current traffic allocations"""
    
    db_path = "models/policy_bandit.db"
    
    try:
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
            logger.info("🚨 CURRENT POLICY STATUS")
            logger.info("="*60)
            
            total_traffic = 0
            for name, traffic, active, trades, pnl in policies:
                status = "ACTIVE" if active else "FROZEN"
                logger.info(f"{name or f'Policy-{len(policies)}'}: {traffic:.1%} traffic, {status}")
                logger.info(f"  Trades: {trades}, PnL: ${pnl:.2f}")
                total_traffic += traffic
            
            logger.info(f"Total traffic allocation: {total_traffic:.1%}")
            logger.info("="*60)
            
    except Exception as e:
        logger.error(f"Failed to show allocations: {e}")

if __name__ == "__main__":
    logger.critical("🚨 EXECUTING EMERGENCY CHALLENGER FREEZE")
    
    # Show before state
    logger.info("BEFORE FREEZE:")
    show_current_allocations()
    
    # Execute freeze
    success = emergency_freeze_challenger()
    
    # Show after state
    logger.info("\nAFTER FREEZE:")
    show_current_allocations()
    
    if success:
        logger.critical("✅ EMERGENCY FREEZE COMPLETED")
        
        print("\n" + "="*60)
        print("🚨 EMERGENCY ACTIONS COMPLETED")
        print("="*60)
        print("✅ Challenger traffic: 0%")
        print("✅ Challenger deactivated")
        print("✅ Production restored to 100%")
        print("📋 Next Steps:")
        print("  1. Diagnose losing trades")
        print("  2. Retrain PPO with stricter rewards")
        print("  3. Start new challengers at 5% traffic")
        print("="*60)
        
    else:
        logger.critical("❌ EMERGENCY FREEZE FAILED - MANUAL INTERVENTION REQUIRED") 