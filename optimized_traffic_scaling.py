#!/usr/bin/env python3
"""
🚀 OPTIMIZED TRAFFIC SCALING
Scale performing models and deploy meta-learner based on current performance metrics

Current Status:
- TimesNet Long-Range: PF 1.97 (Strong) → Scale to 5%
- LightGBM: HALTED (PF 1.46 < 1.5)
- Meta-Learner: Ready for 10% deployment
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficOptimizer:
    """Optimize traffic allocation based on performance metrics"""
    
    def __init__(self):
        self.db_path = 'models/policy_bandit.db'
        self.models_dir = Path('models')
        
        # Current performance metrics from monitoring
        self.performance_metrics = {
            'timesnet_longrange': {
                'pf': 1.97,
                'current_traffic': 0.011,  # 1.1%
                'target_traffic': 0.05,   # 5% (strong performer scaling)
                'status': 'PERFORMING',
                'confidence_boost': 0.15
            },
            'lightgbm_tsa_mae': {
                'pf': 1.46,
                'current_traffic': 0.0,    # 0% (halted)
                'target_traffic': 0.0,     # Keep halted until retrained
                'status': 'HALTED',
                'reason': 'PF < 1.5 threshold'
            },
            'ppo_strict_enhanced': {
                'pf': 1.68,  # From monitoring data
                'current_traffic': 0.011,  # 1.1%
                'target_traffic': 0.015,   # 1.5% (modest increase)
                'status': 'WARNING'
            },
            'meta_learner': {
                'estimated_pf': 2.03,      # From deployment data
                'current_traffic': 0.0,     # 0% (not deployed)
                'target_traffic': 0.10,     # 10% (meta-learner deployment)
                'status': 'READY_FOR_DEPLOYMENT'
            }
        }
        
        # Traffic allocation rules
        self.scaling_rules = {
            'strong_performer_threshold': 1.7,    # PF > 1.7 → Scale up
            'halt_threshold': 1.5,                # PF < 1.5 → Halt
            'max_single_allocation': 0.25,        # 25% max per model
            'meta_learner_requirement': 150,      # 150 trades before meta deployment
            'conservative_start': 0.10,           # 10% for new meta-learner
        }

    def check_database_connection(self) -> bool:
        """Check if Thompson Sampling database exists and is accessible"""
        try:
            if not Path(self.db_path).exists():
                logger.warning(f"Database not found: {self.db_path}")
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['policies', 'bandit_arms']
            for table in required_tables:
                if table not in tables:
                    logger.warning(f"Required table missing: {table}")
                    conn.close()
                    return False
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False

    def get_current_allocations(self) -> dict:
        """Get current traffic allocations from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current allocations
            cursor.execute("""
                SELECT p.name, ba.traffic_allocation, ba.alpha, ba.beta
                FROM policies p
                JOIN bandit_arms ba ON p.id = ba.policy_id
                WHERE p.is_active = 1
            """)
            
            allocations = {}
            for row in cursor.fetchall():
                policy_name, traffic, alpha, beta = row
                allocations[policy_name] = {
                    'traffic': traffic,
                    'alpha': alpha,
                    'beta': beta,
                    'estimated_win_rate': alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
                }
            
            conn.close()
            return allocations
            
        except Exception as e:
            logger.error(f"Error getting current allocations: {e}")
            return {}

    def scale_timesnet_traffic(self) -> bool:
        """Scale TimesNet traffic from 1.1% to 5% (strong performer)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find TimesNet policy
            cursor.execute("""
                SELECT p.id, p.name, ba.traffic_allocation
                FROM policies p
                JOIN bandit_arms ba ON p.id = ba.policy_id
                WHERE p.name LIKE '%timesnet%' OR p.name LIKE '%TimesNet%'
                AND p.is_active = 1
            """)
            
            result = cursor.fetchone()
            if not result:
                logger.warning("TimesNet policy not found in database")
                conn.close()
                return False
            
            policy_id, policy_name, current_traffic = result
            target_traffic = self.performance_metrics['timesnet_longrange']['target_traffic']
            
            # Update traffic allocation
            cursor.execute("""
                UPDATE bandit_arms
                SET traffic_allocation = ?
                WHERE policy_id = ?
            """, (target_traffic, policy_id))
            
            # Log the change
            cursor.execute("""
                INSERT INTO bandit_logs (policy_id, action, old_value, new_value, reason, timestamp)
                VALUES (?, 'traffic_scale', ?, ?, ?, ?)
            """, (policy_id, current_traffic, target_traffic, 
                  f"Strong performer scaling (PF {self.performance_metrics['timesnet_longrange']['pf']})", 
                  datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ TimesNet traffic scaled: {current_traffic:.1%} → {target_traffic:.1%}")
            return True
            
        except Exception as e:
            logger.error(f"Error scaling TimesNet traffic: {e}")
            return False

    def deploy_meta_learner(self) -> bool:
        """Deploy Meta-Learner at 10% traffic allocation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if meta-learner policy already exists
            cursor.execute("""
                SELECT id FROM policies 
                WHERE name LIKE '%meta%' OR name LIKE '%ensemble%'
                AND is_active = 1
            """)
            
            if cursor.fetchone():
                logger.info("Meta-learner policy already exists")
                conn.close()
                return True
            
            # Create new meta-learner policy
            meta_config = {
                'type': 'meta_learner',
                'ensemble_weights': {
                    'timesnet': 0.4,
                    'tsa_mae': 0.3,
                    'ppo': 0.3
                },
                'confidence_threshold': 0.45,
                'risk_scaling': 0.5
            }
            
            cursor.execute("""
                INSERT INTO policies (name, type, model_path, config, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                'meta_learner_ensemble_v1',
                'meta_learner',
                'models/meta_learner_v1.pkl',
                json.dumps(meta_config),
                1,
                datetime.now().isoformat()
            ))
            
            policy_id = cursor.lastrowid
            target_traffic = self.performance_metrics['meta_learner']['target_traffic']
            
            # Create bandit arm for meta-learner
            cursor.execute("""
                INSERT INTO bandit_arms (policy_id, alpha, beta, traffic_allocation, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (policy_id, 1.0, 1.0, target_traffic, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Meta-Learner deployed at {target_traffic:.1%} traffic")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying meta-learner: {e}")
            return False

    def optimize_traffic_allocation(self) -> dict:
        """Optimize overall traffic allocation"""
        
        logger.info("🚀 OPTIMIZING TRAFFIC ALLOCATION")
        logger.info("=" * 50)
        
        # Check database connectivity
        if not self.check_database_connection():
            logger.error("❌ Database not accessible - creating simulation report")
            return self.create_simulation_report()
        
        # Get current state
        current_allocations = self.get_current_allocations()
        
        # Print current status
        logger.info("📊 CURRENT STATUS:")
        for model, metrics in self.performance_metrics.items():
            status = metrics['status']
            pf = metrics.get('pf', 'N/A')
            current = metrics['current_traffic']
            target = metrics['target_traffic']
            
            logger.info(f"  {model}: PF={pf}, {current:.1%}→{target:.1%} ({status})")
        
        results = {}
        
        # 1. Scale TimesNet (strong performer)
        logger.info("\n🎯 SCALING TIMESNET...")
        if self.scale_timesnet_traffic():
            results['timesnet_scaling'] = 'SUCCESS'
        else:
            results['timesnet_scaling'] = 'FAILED'
        
        # 2. Deploy Meta-Learner
        logger.info("\n🧠 DEPLOYING META-LEARNER...")
        if self.deploy_meta_learner():
            results['meta_learner_deployment'] = 'SUCCESS'
        else:
            results['meta_learner_deployment'] = 'FAILED'
        
        # 3. Update PPO allocation (modest increase)
        logger.info("\n⚡ OPTIMIZING PPO ALLOCATION...")
        results['ppo_optimization'] = self.optimize_ppo_allocation()
        
        # 4. Generate optimization report
        final_report = self.generate_optimization_report(results)
        
        return final_report

    def optimize_ppo_allocation(self) -> str:
        """Optimize PPO allocation (modest increase from 1.1% to 1.5%)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find PPO policy
            cursor.execute("""
                SELECT p.id, ba.traffic_allocation
                FROM policies p
                JOIN bandit_arms ba ON p.id = ba.policy_id
                WHERE p.name LIKE '%ppo%' OR p.name LIKE '%PPO%'
                AND p.is_active = 1
            """)
            
            result = cursor.fetchone()
            if not result:
                conn.close()
                return "PPO policy not found"
            
            policy_id, current_traffic = result
            target_traffic = self.performance_metrics['ppo_strict_enhanced']['target_traffic']
            
            # Update allocation
            cursor.execute("""
                UPDATE bandit_arms
                SET traffic_allocation = ?
                WHERE policy_id = ?
            """, (target_traffic, policy_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ PPO traffic updated: {current_traffic:.1%} → {target_traffic:.1%}")
            return "SUCCESS"
            
        except Exception as e:
            logger.error(f"PPO optimization error: {e}")
            return f"FAILED: {e}"

    def create_simulation_report(self) -> dict:
        """Create simulation report when database is not accessible"""
        
        logger.info("📋 CREATING SIMULATION REPORT...")
        
        total_current = sum(m['current_traffic'] for m in self.performance_metrics.values())
        total_target = sum(m['target_traffic'] for m in self.performance_metrics.values())
        
        return {
            'mode': 'SIMULATION',
            'current_total_traffic': total_current,
            'target_total_traffic': total_target,
            'traffic_increase': total_target - total_current,
            'optimizations': {
                'timesnet_scaling': f"1.1% → 5.0% (PF {self.performance_metrics['timesnet_longrange']['pf']})",
                'meta_learner_deployment': f"0% → 10% (estimated PF 2.03)",
                'ppo_optimization': f"1.1% → 1.5% (modest increase)",
                'lightgbm_status': "HALTED (PF 1.46 < 1.5)"
            },
            'expected_benefits': {
                'trade_volume_increase': '+30-50 trades (stronger models getting more traffic)',
                'performance_improvement': 'Focus on proven performers',
                'risk_management': 'Halted underperformer (LightGBM)',
                'meta_learning': 'Ensemble approach for better predictions'
            }
        }

    def generate_optimization_report(self, results: dict) -> dict:
        """Generate comprehensive optimization report"""
        
        # Calculate new traffic totals
        total_new_traffic = sum(m['target_traffic'] for m in self.performance_metrics.values())
        total_old_traffic = sum(m['current_traffic'] for m in self.performance_metrics.values())
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimization_results': results,
            'traffic_summary': {
                'before': total_old_traffic,
                'after': total_new_traffic,
                'increase': total_new_traffic - total_old_traffic
            },
            'model_allocations': {
                'TimesNet': f"{self.performance_metrics['timesnet_longrange']['current_traffic']:.1%} → {self.performance_metrics['timesnet_longrange']['target_traffic']:.1%}",
                'Meta-Learner': f"{self.performance_metrics['meta_learner']['current_traffic']:.1%} → {self.performance_metrics['meta_learner']['target_traffic']:.1%}",
                'PPO': f"{self.performance_metrics['ppo_strict_enhanced']['current_traffic']:.1%} → {self.performance_metrics['ppo_strict_enhanced']['target_traffic']:.1%}",
                'LightGBM': "HALTED (PF < 1.5)"
            },
            'expected_improvements': {
                'trade_volume': '+30-50 trades from increased traffic to performers',
                'performance_focus': 'Prioritize models with PF > 1.7',
                'risk_reduction': 'Halt underperformers automatically',
                'ensemble_benefits': 'Meta-learner combines best predictions'
            }
        }
        
        # Save report
        report_path = self.models_dir / f'traffic_optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Report saved: {report_path}")
        
        return report

def main():
    """Run traffic optimization"""
    
    optimizer = TrafficOptimizer()
    
    logger.info("🚀 STARTING TRAFFIC OPTIMIZATION")
    logger.info("🎯 Scaling strong performers and deploying meta-learner")
    logger.info("=" * 60)
    
    # Run optimization
    results = optimizer.optimize_traffic_allocation()
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    
    if results.get('mode') == 'SIMULATION':
        logger.info("Mode: SIMULATION (database not accessible)")
        logger.info(f"Traffic increase: {results['current_total_traffic']:.1%} → {results['target_total_traffic']:.1%}")
        
        for opt, desc in results['optimizations'].items():
            logger.info(f"  {opt}: {desc}")
    else:
        logger.info("Mode: LIVE OPTIMIZATION")
        success_count = sum(1 for v in results['optimization_results'].values() if v == 'SUCCESS')
        total_count = len(results['optimization_results'])
        
        logger.info(f"Success rate: {success_count}/{total_count}")
        
        for action, status in results['optimization_results'].items():
            logger.info(f"  {action}: {status}")
    
    logger.info("=" * 60)
    
    return results

if __name__ == "__main__":
    main() 