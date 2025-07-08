#!/usr/bin/env python3
"""
Weekly Maintenance Routine for Advanced Learning Layer
Automates snapshots, retraining, optimization, and cleanup
"""

import os
import sqlite3
import shutil
import subprocess
import logging
import json
import boto3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeeklyMaintenance:
    """
    Automated weekly maintenance for the Advanced Learning Layer
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 db_path: str = "models/policy_bandit.db",
                 s3_bucket: str = "elite-policies"):
        self.models_dir = Path(models_dir)
        self.db_path = db_path
        self.s3_bucket = s3_bucket
        self.snapshot_date = datetime.now().strftime("%Y-%m-%d")
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3')
            logger.info("✅ S3 client initialized")
        except Exception as e:
            logger.warning(f"⚠️  S3 client failed: {e}")
            self.s3_client = None
    
    def snapshot_models_to_s3(self) -> bool:
        """Snapshot encoder SHA + TabNet weights to S3"""
        
        if not self.s3_client:
            logger.warning("⚠️  S3 not available, creating local snapshots only")
            return self._create_local_snapshots()
        
        try:
            snapshot_prefix = f"snapshots/{self.snapshot_date}/"
            
            # Find latest encoder models
            encoder_files = list(self.models_dir.glob("encoder_*.pt"))
            tabnet_files = list(self.models_dir.glob("tabnet_*.pt"))
            ppo_files = list(self.models_dir.glob("ppo_*.pt"))
            
            uploaded_files = []
            
            # Upload encoder models
            for encoder_file in encoder_files[-5:]:  # Keep last 5 encoders
                s3_key = f"{snapshot_prefix}encoders/{encoder_file.name}"
                self.s3_client.upload_file(str(encoder_file), self.s3_bucket, s3_key)
                uploaded_files.append(s3_key)
                logger.info(f"📤 Uploaded: {s3_key}")
            
            # Upload TabNet models
            for tabnet_file in tabnet_files[-3:]:  # Keep last 3 TabNet models
                s3_key = f"{snapshot_prefix}tabnet/{tabnet_file.name}"
                self.s3_client.upload_file(str(tabnet_file), self.s3_bucket, s3_key)
                uploaded_files.append(s3_key)
                logger.info(f"📤 Uploaded: {s3_key}")
            
            # Upload PPO models
            for ppo_file in ppo_files[-3:]:  # Keep last 3 PPO models
                s3_key = f"{snapshot_prefix}ppo/{ppo_file.name}"
                self.s3_client.upload_file(str(ppo_file), self.s3_bucket, s3_key)
                uploaded_files.append(s3_key)
                logger.info(f"📤 Uploaded: {s3_key}")
            
            # Upload policy database
            if os.path.exists(self.db_path):
                s3_key = f"{snapshot_prefix}database/policy_bandit_{self.snapshot_date}.db"
                self.s3_client.upload_file(self.db_path, self.s3_bucket, s3_key)
                uploaded_files.append(s3_key)
                logger.info(f"📤 Uploaded: {s3_key}")
            
            # Create manifest file
            manifest = {
                'snapshot_date': self.snapshot_date,
                'uploaded_files': uploaded_files,
                'total_files': len(uploaded_files),
                'created_at': datetime.now().isoformat()
            }
            
            manifest_path = f"/tmp/manifest_{self.snapshot_date}.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            s3_key = f"{snapshot_prefix}manifest.json"
            self.s3_client.upload_file(manifest_path, self.s3_bucket, s3_key)
            
            logger.info(f"✅ Snapshot complete: {len(uploaded_files)} files uploaded to S3")
            return True
            
        except Exception as e:
            logger.error(f"❌ S3 snapshot failed: {e}")
            return self._create_local_snapshots()
    
    def _create_local_snapshots(self) -> bool:
        """Create local snapshots if S3 is unavailable"""
        
        try:
            snapshot_dir = Path(f"snapshots/{self.snapshot_date}")
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            for pattern in ["encoder_*.pt", "tabnet_*.pt", "ppo_*.pt"]:
                files = list(self.models_dir.glob(pattern))
                for file in files[-3:]:  # Keep last 3 of each type
                    shutil.copy2(file, snapshot_dir / file.name)
            
            # Copy database
            if os.path.exists(self.db_path):
                shutil.copy2(self.db_path, snapshot_dir / f"policy_bandit_{self.snapshot_date}.db")
            
            logger.info(f"✅ Local snapshot created: {snapshot_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Local snapshot failed: {e}")
            return False
    
    def retrain_rl_system(self, steps: int = 250000) -> bool:
        """Retrain RL for 250k steps on fresh replay buffer (SOL & BTC only)"""
        
        try:
            logger.info(f"🔄 Starting RL retraining ({steps:,} steps)")
            
            # Run RL training with focused assets
            cmd = [
                "python", "quick_ppo_fix.py",
                "--episodes", str(steps // 1000),  # Convert steps to episodes
                "--assets", "SOL,BTC",
                "--fresh-buffer", "true",
                "--save-path", f"models/ppo_weekly_{self.snapshot_date}.pt"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info("✅ RL retraining completed successfully")
                logger.info(f"📊 Output: {result.stdout[-200:]}")  # Last 200 chars
                return True
            else:
                logger.error(f"❌ RL retraining failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ RL retraining timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"❌ RL retraining error: {e}")
            return False
    
    def optuna_sweep_timesnet(self, trials: int = 10) -> List[str]:
        """Optuna sweep (10 trials) on TimesNet—keep best 2 checkpoints"""
        
        try:
            logger.info(f"🎯 Starting Optuna sweep ({trials} trials)")
            
            # Run Optuna optimization
            cmd = [
                "python", "train_tabnet_with_encoder.py",
                "--optuna-trials", str(trials),
                "--objective", "accuracy",
                "--save-best", "2"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hours
            
            if result.returncode == 0:
                # Parse saved model paths from output
                lines = result.stdout.split('\n')
                saved_models = []
                
                for line in lines:
                    if 'Best model saved:' in line:
                        model_path = line.split('Best model saved:')[1].strip()
                        saved_models.append(model_path)
                
                logger.info(f"✅ Optuna sweep completed: {len(saved_models)} best models saved")
                return saved_models
            else:
                logger.error(f"❌ Optuna sweep failed: {result.stderr}")
                return []
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Optuna sweep timed out after 2 hours")
            return []
        except Exception as e:
            logger.error(f"❌ Optuna sweep error: {e}")
            return []
    
    def purge_stale_policies(self, days_threshold: int = 60, pf_threshold: float = 1.2) -> int:
        """Purge stale policies older than 60d or PF < 1.2"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find stale policies
                cutoff_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
                
                cursor.execute('''
                    SELECT p.id, p.name, p.created_at, ba.total_pnl, ba.total_trades
                    FROM policies p
                    JOIN bandit_arms ba ON p.id = ba.policy_id
                    WHERE p.created_at < ? OR 
                          (ba.total_trades > 50 AND ba.total_pnl / NULLIF(ba.total_trades, 0) < ?)
                ''', (cutoff_date, pf_threshold))
                
                stale_policies = cursor.fetchall()
                
                if not stale_policies:
                    logger.info("✅ No stale policies found")
                    return 0
                
                # Deactivate stale policies
                purged_count = 0
                for policy_id, name, created_at, total_pnl, total_trades in stale_policies:
                    pf = (total_pnl / total_trades) if total_trades > 0 else 0
                    
                    # Don't purge if it's the only active policy
                    cursor.execute('SELECT COUNT(*) FROM policies WHERE is_active = 1')
                    active_count = cursor.fetchone()[0]
                    
                    if active_count <= 1:
                        logger.info(f"⚠️  Skipping purge of {name} (only active policy)")
                        continue
                    
                    # Deactivate policy
                    cursor.execute('''
                        UPDATE policies 
                        SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (policy_id,))
                    
                    # Set traffic to 0
                    cursor.execute('''
                        UPDATE bandit_arms 
                        SET traffic_allocation = 0.0
                        WHERE policy_id = ?
                    ''', (policy_id,))
                    
                    # Log purge reason
                    reason = f"Stale policy purge: Age or PF={pf:.2f} < {pf_threshold}"
                    cursor.execute('''
                        INSERT INTO traffic_log (policy_id, allocation_percent, reason)
                        VALUES (?, ?, ?)
                    ''', (policy_id, 0.0, reason))
                    
                    purged_count += 1
                    logger.info(f"🗑️  Purged policy: {name} (PF={pf:.2f}, Age={created_at[:10]})")
                
                conn.commit()
                
                logger.info(f"✅ Purged {purged_count} stale policies")
                return purged_count
                
        except Exception as e:
            logger.error(f"❌ Policy purge error: {e}")
            return 0
    
    def cleanup_old_models(self, keep_days: int = 30) -> int:
        """Clean up old model files older than keep_days"""
        
        try:
            cutoff_time = datetime.now() - timedelta(days=keep_days)
            cutoff_timestamp = cutoff_time.timestamp()
            
            cleaned_count = 0
            
            for pattern in ["encoder_*.pt", "tabnet_*.pt", "ppo_*.pt"]:
                files = list(self.models_dir.glob(pattern))
                
                for file in files:
                    if file.stat().st_mtime < cutoff_timestamp:
                        file.unlink()
                        cleaned_count += 1
                        logger.info(f"🗑️  Cleaned: {file.name}")
            
            logger.info(f"✅ Cleaned {cleaned_count} old model files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Model cleanup error: {e}")
            return 0
    
    def generate_maintenance_report(self) -> Dict[str, any]:
        """Generate comprehensive maintenance report"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get policy statistics
            cursor.execute('''
                SELECT COUNT(*) as total_policies,
                       SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_policies,
                       AVG(ba.traffic_allocation) as avg_traffic,
                       SUM(ba.total_trades) as total_trades,
                       AVG(ba.total_pnl / NULLIF(ba.total_trades, 0)) as avg_pnl_per_trade
                FROM policies p
                JOIN bandit_arms ba ON p.id = ba.policy_id
            ''')
            
            stats = cursor.fetchone()
            
            # Get model file counts
            model_counts = {
                'encoders': len(list(self.models_dir.glob("encoder_*.pt"))),
                'tabnet': len(list(self.models_dir.glob("tabnet_*.pt"))),
                'ppo': len(list(self.models_dir.glob("ppo_*.pt")))
            }
            
            report = {
                'maintenance_date': self.snapshot_date,
                'policy_stats': {
                    'total_policies': stats[0] if stats else 0,
                    'active_policies': stats[1] if stats else 0,
                    'avg_traffic_allocation': stats[2] if stats else 0,
                    'total_trades': stats[3] if stats else 0,
                    'avg_pnl_per_trade': stats[4] if stats else 0
                },
                'model_counts': model_counts,
                'disk_usage_mb': sum(f.stat().st_size for f in self.models_dir.glob("*.pt")) / 1024 / 1024
            }
            
            return report
    
    def run_full_maintenance(self) -> Dict[str, any]:
        """Run complete weekly maintenance routine"""
        
        logger.info("🔧 Starting weekly maintenance routine")
        logger.info(f"📅 Date: {self.snapshot_date}")
        
        results = {
            'date': self.snapshot_date,
            'started_at': datetime.now().isoformat(),
            'tasks': {}
        }
        
        # 1. Snapshot models to S3
        logger.info("📤 Step 1/5: Creating model snapshots...")
        results['tasks']['snapshot'] = self.snapshot_models_to_s3()
        
        # 2. Retrain RL system
        logger.info("🔄 Step 2/5: Retraining RL system...")
        results['tasks']['rl_retrain'] = self.retrain_rl_system()
        
        # 3. Optuna sweep
        logger.info("🎯 Step 3/5: Running Optuna sweep...")
        best_models = self.optuna_sweep_timesnet()
        results['tasks']['optuna_sweep'] = len(best_models) > 0
        results['tasks']['optuna_models'] = best_models
        
        # 4. Purge stale policies
        logger.info("🗑️  Step 4/5: Purging stale policies...")
        purged_count = self.purge_stale_policies()
        results['tasks']['policy_purge'] = purged_count
        
        # 5. Clean up old models
        logger.info("🧹 Step 5/5: Cleaning old models...")
        cleaned_count = self.cleanup_old_models()
        results['tasks']['model_cleanup'] = cleaned_count
        
        # Generate final report
        results['report'] = self.generate_maintenance_report()
        results['completed_at'] = datetime.now().isoformat()
        results['duration_minutes'] = (
            datetime.fromisoformat(results['completed_at']) - 
            datetime.fromisoformat(results['started_at'])
        ).total_seconds() / 60
        
        # Log summary
        logger.info("="*60)
        logger.info("🔧 WEEKLY MAINTENANCE COMPLETE")
        logger.info("="*60)
        logger.info(f"⏱️  Duration: {results['duration_minutes']:.1f} minutes")
        logger.info(f"📤 Snapshot: {'✅' if results['tasks']['snapshot'] else '❌'}")
        logger.info(f"🔄 RL Retrain: {'✅' if results['tasks']['rl_retrain'] else '❌'}")
        logger.info(f"🎯 Optuna: {'✅' if results['tasks']['optuna_sweep'] else '❌'} ({len(best_models)} models)")
        logger.info(f"🗑️  Policies Purged: {purged_count}")
        logger.info(f"🧹 Models Cleaned: {cleaned_count}")
        logger.info("="*60)
        
        return results

def main():
    """Main maintenance function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Weekly Maintenance Routine')
    parser.add_argument('--task', choices=['snapshot', 'retrain', 'optuna', 'purge', 'cleanup', 'full'], 
                       default='full', help='Maintenance task to run')
    parser.add_argument('--s3-bucket', default='elite-policies', help='S3 bucket for snapshots')
    
    args = parser.parse_args()
    
    maintenance = WeeklyMaintenance(s3_bucket=args.s3_bucket)
    
    if args.task == 'snapshot':
        success = maintenance.snapshot_models_to_s3()
        logger.info(f"Snapshot: {'✅' if success else '❌'}")
    
    elif args.task == 'retrain':
        success = maintenance.retrain_rl_system()
        logger.info(f"RL Retrain: {'✅' if success else '❌'}")
    
    elif args.task == 'optuna':
        models = maintenance.optuna_sweep_timesnet()
        logger.info(f"Optuna: {len(models)} best models saved")
    
    elif args.task == 'purge':
        count = maintenance.purge_stale_policies()
        logger.info(f"Purged: {count} stale policies")
    
    elif args.task == 'cleanup':
        count = maintenance.cleanup_old_models()
        logger.info(f"Cleaned: {count} old models")
    
    elif args.task == 'full':
        results = maintenance.run_full_maintenance()
        
        # Save results
        results_file = f"maintenance_results_{maintenance.snapshot_date}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📋 Results saved: {results_file}")

if __name__ == "__main__":
    main() 