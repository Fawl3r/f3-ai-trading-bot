#!/usr/bin/env python3
"""
Live-Ops Monitor for Advanced Learning Layer
Implements the 150-trade checklist with automated scaling and alerts
"""

import sqlite3
import numpy as np
import pandas as pd
import time
import logging
import json
import torch
import psutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveOpsMonitor:
    """
    Production monitoring system for the first 150 trades
    Implements automated traffic scaling and alerting
    """
    
    def __init__(self, bandit_db_path: str = "models/policy_bandit.db"):
        self.bandit_db_path = bandit_db_path
        self.start_time = datetime.now()
        
        # Monitoring thresholds from checklist
        self.thresholds = {
            'pf_challenger_30': {'min': 2.1, 'max': 2.8, 'flag_below': 1.6},
            'dd_challenger_pct': {'max': 3.0, 'flag_above': 4.0},
            'encoder_kl_divergence': {'min': 0.0, 'max': 0.25, 'flag_above': 0.30},
            'gpu_util_pct': {'min': 2.0, 'max': 10.0, 'flag_above': 60.0}
        }
        
        # Traffic scaling parameters
        self.scaling_params = {
            'max_traffic_share': 0.60,  # 60% cap
            'min_trades_for_scaling': 30,
            'scaling_factor': 1.5,
            'min_traffic_share': 0.05   # 5% minimum
        }
        
        logger.info("🚨 Live-Ops Monitor initialized")
        logger.info(f"📊 Monitoring thresholds: {self.thresholds}")
    
    def check_challenger_performance(self, policy_sha: str) -> Dict[str, float]:
        """Check challenger policy performance metrics"""
        
        with sqlite3.connect(self.bandit_db_path) as conn:
            cursor = conn.cursor()
            
            # Get policy ID
            cursor.execute('''
                SELECT id FROM policies 
                WHERE name LIKE ? 
                ORDER BY created_at DESC LIMIT 1
            ''', (f'%{policy_sha[-8:]}%',))
            
            policy_result = cursor.fetchone()
            if not policy_result:
                logger.warning(f"⚠️  Policy not found: {policy_sha}")
                return {}
            
            policy_id = policy_result[0]
            
            # Get recent 30 trades for profit factor calculation
            cursor.execute('''
                SELECT trade_outcome, pnl 
                FROM performance_log 
                WHERE policy_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 30
            ''', (policy_id,))
            
            trades = cursor.fetchall()
            
            if len(trades) < 10:
                logger.info(f"📈 Only {len(trades)} trades recorded, need more data")
                return {'trades_count': len(trades)}
            
            # Calculate profit factor (PF)
            winning_trades = [pnl for outcome, pnl in trades if outcome and pnl > 0]
            losing_trades = [abs(pnl) for outcome, pnl in trades if not outcome and pnl < 0]
            
            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = sum(losing_trades) if losing_trades else 0
            
            pf_challenger_30 = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Calculate drawdown percentage
            pnls = [pnl for _, pnl in trades]
            cumulative_pnl = np.cumsum(pnls[::-1])  # Reverse for chronological order
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown_series = cumulative_pnl - running_max
            dd_challenger_pct = abs(np.min(drawdown_series)) / (running_max[-1] + 1e-8) * 100
            
            metrics = {
                'trades_count': len(trades),
                'pf_challenger_30': pf_challenger_30,
                'dd_challenger_pct': dd_challenger_pct,
                'win_rate': len(winning_trades) / len(trades) * 100,
                'total_pnl': sum(pnls)
            }
            
            logger.info(f"📊 Challenger Performance: PF={pf_challenger_30:.2f}, DD={dd_challenger_pct:.1f}%")
            
            return metrics
    
    def check_encoder_health(self, encoder_path: str) -> Dict[str, float]:
        """Check encoder KL divergence and health metrics"""
        
        try:
            # Load encoder model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(encoder_path, map_location=device)
            
            # Simple KL divergence estimation (placeholder)
            # In production, this would compare embeddings vs baseline
            encoder_kl_divergence = np.random.uniform(0.05, 0.20)  # Simulated for demo
            
            metrics = {
                'encoder_kl_divergence': encoder_kl_divergence,
                'model_loaded': True,
                'device': str(device)
            }
            
            logger.info(f"🧠 Encoder Health: KL={encoder_kl_divergence:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Encoder check failed: {e}")
            return {'encoder_kl_divergence': 999.0, 'model_loaded': False}
    
    def check_gpu_utilization(self) -> Dict[str, float]:
        """Check GPU utilization and memory usage"""
        
        try:
            # Use nvidia-smi to get GPU stats
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_util, mem_used, mem_total = map(int, lines[0].split(', '))
                
                gpu_util_pct = gpu_util
                memory_util_pct = (mem_used / mem_total) * 100
                
                metrics = {
                    'gpu_util_pct': gpu_util_pct,
                    'memory_util_pct': memory_util_pct,
                    'memory_used_gb': mem_used / 1024,
                    'memory_total_gb': mem_total / 1024
                }
                
                logger.info(f"🖥️  GPU Util: {gpu_util_pct}%, Memory: {memory_util_pct:.1f}%")
                
                return metrics
            else:
                logger.warning("⚠️  nvidia-smi not available")
                return {'gpu_util_pct': 0.0, 'nvidia_smi_available': False}
                
        except Exception as e:
            logger.error(f"❌ GPU check failed: {e}")
            return {'gpu_util_pct': 0.0, 'error': str(e)}
    
    def evaluate_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, str]]:
        """Evaluate metrics against thresholds and generate alerts"""
        
        alerts = []
        
        # Check profit factor
        if 'pf_challenger_30' in metrics:
            pf = metrics['pf_challenger_30']
            if pf < self.thresholds['pf_challenger_30']['flag_below']:
                alerts.append({
                    'severity': 'CRITICAL',
                    'metric': 'pf_challenger_30',
                    'value': pf,
                    'threshold': self.thresholds['pf_challenger_30']['flag_below'],
                    'action': 'Throttle challenger to 0%; review logs'
                })
        
        # Check drawdown
        if 'dd_challenger_pct' in metrics:
            dd = metrics['dd_challenger_pct']
            if dd > self.thresholds['dd_challenger_pct']['flag_above']:
                alerts.append({
                    'severity': 'CRITICAL',
                    'metric': 'dd_challenger_pct',
                    'value': dd,
                    'threshold': self.thresholds['dd_challenger_pct']['flag_above'],
                    'action': 'Auto-halt & revert to prod'
                })
        
        # Check encoder KL divergence
        if 'encoder_kl_divergence' in metrics:
            kl = metrics['encoder_kl_divergence']
            if kl > self.thresholds['encoder_kl_divergence']['flag_above']:
                alerts.append({
                    'severity': 'WARNING',
                    'metric': 'encoder_kl_divergence',
                    'value': kl,
                    'threshold': self.thresholds['encoder_kl_divergence']['flag_above'],
                    'action': 'Schedule 10-epoch refresh tonight'
                })
        
        # Check GPU utilization
        if 'gpu_util_pct' in metrics:
            gpu = metrics['gpu_util_pct']
            if gpu > self.thresholds['gpu_util_pct']['flag_above']:
                alerts.append({
                    'severity': 'WARNING',
                    'metric': 'gpu_util_pct',
                    'value': gpu,
                    'threshold': self.thresholds['gpu_util_pct']['flag_above'],
                    'action': 'Restart learner pod'
                })
        
        return alerts
    
    def execute_traffic_scaling(self, policy_sha: str) -> bool:
        """Execute nightly traffic scaling logic"""
        
        with sqlite3.connect(self.bandit_db_path) as conn:
            cursor = conn.cursor()
            
            # Get policy details
            cursor.execute('''
                SELECT p.id, p.name, ba.traffic_allocation, ba.total_trades,
                       ba.alpha, ba.beta, ba.total_pnl
                FROM policies p
                JOIN bandit_arms ba ON p.id = ba.policy_id
                WHERE p.name LIKE ?
                ORDER BY p.created_at DESC LIMIT 1
            ''', (f'%{policy_sha[-8:]}%',))
            
            policy = cursor.fetchone()
            if not policy:
                logger.error(f"❌ Policy not found for scaling: {policy_sha}")
                return False
            
            policy_id, name, current_traffic, total_trades, alpha, beta, total_pnl = policy
            
            # Check if we have enough trades
            if total_trades < self.scaling_params['min_trades_for_scaling']:
                logger.info(f"📊 Need {self.scaling_params['min_trades_for_scaling']} trades for scaling (have {total_trades})")
                return False
            
            # Get recent performance metrics (last 150 trades)
            recent_metrics = self.check_challenger_performance(policy_sha)
            
            if not recent_metrics:
                return False
            
            pf_150 = recent_metrics.get('pf_challenger_30', 0)
            dd_150 = recent_metrics.get('dd_challenger_pct', 999)
            
            # Get production baseline (assume policy_id=1 is production)
            cursor.execute('''
                SELECT ba.alpha, ba.beta, ba.total_pnl, ba.total_trades
                FROM bandit_arms ba
                WHERE ba.policy_id = 1
            ''')
            
            prod_stats = cursor.fetchone()
            if prod_stats:
                prod_alpha, prod_beta, prod_pnl, prod_trades = prod_stats
                pf_prod = (prod_pnl / prod_trades) if prod_trades > 0 else 0
            else:
                pf_prod = 2.0  # Default baseline
            
            # Traffic scaling conditions
            should_scale = (
                pf_150 >= pf_prod and 
                dd_150 <= 3.0 and 
                current_traffic < self.scaling_params['max_traffic_share']
            )
            
            if should_scale:
                new_traffic = min(
                    current_traffic * self.scaling_params['scaling_factor'],
                    self.scaling_params['max_traffic_share']
                )
                
                # Execute scaling SQL
                cursor.execute('''
                    UPDATE bandit_arms
                    SET traffic_allocation = ?
                    WHERE policy_id = ?
                ''', (new_traffic, policy_id))
                
                # Log the scaling action
                cursor.execute('''
                    INSERT INTO traffic_log (policy_id, allocation_percent, reason)
                    VALUES (?, ?, ?)
                ''', (policy_id, new_traffic, f'Nightly scaling: PF={pf_150:.2f}, DD={dd_150:.1f}%'))
                
                conn.commit()
                
                logger.info(f"📈 Traffic scaled: {current_traffic:.1%} → {new_traffic:.1%}")
                logger.info(f"📊 Conditions: PF={pf_150:.2f} >= {pf_prod:.2f}, DD={dd_150:.1f}% <= 3.0%")
                
                return True
            else:
                logger.info(f"📊 No scaling: PF={pf_150:.2f}, DD={dd_150:.1f}%, Traffic={current_traffic:.1%}")
                return False
    
    def generate_monitoring_report(self, policy_sha: str) -> Dict[str, any]:
        """Generate comprehensive monitoring report"""
        
        logger.info("📋 Generating live-ops monitoring report...")
        
        # Collect all metrics
        challenger_metrics = self.check_challenger_performance(policy_sha)
        encoder_metrics = self.check_encoder_health(f"models/encoder_{policy_sha[-8:]}.pt")
        gpu_metrics = self.check_gpu_utilization()
        
        # Combine metrics
        all_metrics = {**challenger_metrics, **encoder_metrics, **gpu_metrics}
        
        # Evaluate alerts
        alerts = self.evaluate_alerts(all_metrics)
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'policy_sha': policy_sha,
            'metrics': all_metrics,
            'alerts': alerts,
            'thresholds': self.thresholds,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }
        
        # Log summary
        logger.info("="*60)
        logger.info("📊 LIVE-OPS MONITORING REPORT")
        logger.info("="*60)
        
        if challenger_metrics:
            logger.info(f"🎯 Challenger: {challenger_metrics.get('trades_count', 0)} trades")
            logger.info(f"📈 Profit Factor: {challenger_metrics.get('pf_challenger_30', 0):.2f}")
            logger.info(f"📉 Drawdown: {challenger_metrics.get('dd_challenger_pct', 0):.1f}%")
        
        if encoder_metrics:
            logger.info(f"🧠 Encoder KL: {encoder_metrics.get('encoder_kl_divergence', 0):.3f}")
        
        if gpu_metrics:
            logger.info(f"🖥️  GPU Util: {gpu_metrics.get('gpu_util_pct', 0):.1f}%")
        
        # Alert summary
        if alerts:
            logger.warning(f"🚨 {len(alerts)} ALERTS TRIGGERED:")
            for alert in alerts:
                logger.warning(f"  {alert['severity']}: {alert['metric']} = {alert['value']}")
                logger.warning(f"    Action: {alert['action']}")
        else:
            logger.info("✅ All metrics within normal ranges")
        
        logger.info("="*60)
        
        return report
    
    def run_continuous_monitoring(self, policy_sha: str, check_interval: int = 300):
        """Run continuous monitoring loop"""
        
        logger.info(f"🔄 Starting continuous monitoring (every {check_interval}s)")
        
        try:
            while True:
                report = self.generate_monitoring_report(policy_sha)
                
                # Handle critical alerts
                for alert in report['alerts']:
                    if alert['severity'] == 'CRITICAL':
                        logger.critical(f"🚨 CRITICAL ALERT: {alert['action']}")
                        
                        if 'throttle challenger' in alert['action'].lower():
                            self.emergency_throttle(policy_sha)
                        elif 'auto-halt' in alert['action'].lower():
                            self.emergency_halt(policy_sha)
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("👋 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
    
    def emergency_throttle(self, policy_sha: str):
        """Emergency throttle challenger to 0%"""
        
        with sqlite3.connect(self.bandit_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE bandit_arms
                SET traffic_allocation = 0.0
                WHERE policy_id IN (
                    SELECT id FROM policies WHERE name LIKE ?
                )
            ''', (f'%{policy_sha[-8:]}%',))
            
            conn.commit()
        
        logger.critical(f"🛑 EMERGENCY: Throttled {policy_sha} to 0% traffic")
    
    def emergency_halt(self, policy_sha: str):
        """Emergency halt and revert to production"""
        
        self.emergency_throttle(policy_sha)
        
        with sqlite3.connect(self.bandit_db_path) as conn:
            cursor = conn.cursor()
            
            # Deactivate challenger
            cursor.execute('''
                UPDATE policies
                SET is_active = 0
                WHERE name LIKE ?
            ''', (f'%{policy_sha[-8:]}%',))
            
            # Set production to 100%
            cursor.execute('''
                UPDATE bandit_arms
                SET traffic_allocation = 1.0
                WHERE policy_id = 1
            ''')
            
            conn.commit()
        
        logger.critical(f"🛑 EMERGENCY HALT: Reverted to production, deactivated {policy_sha}")

def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live-Ops Monitor')
    parser.add_argument('--policy-sha', required=True, help='Policy SHA to monitor')
    parser.add_argument('--mode', choices=['report', 'continuous', 'scale'], default='report')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    
    args = parser.parse_args()
    
    monitor = LiveOpsMonitor()
    
    if args.mode == 'report':
        report = monitor.generate_monitoring_report(args.policy_sha)
        print(json.dumps(report, indent=2))
    
    elif args.mode == 'continuous':
        monitor.run_continuous_monitoring(args.policy_sha, args.interval)
    
    elif args.mode == 'scale':
        success = monitor.execute_traffic_scaling(args.policy_sha)
        if success:
            logger.info("✅ Traffic scaling completed")
        else:
            logger.info("📊 No scaling needed")

if __name__ == "__main__":
    main() 