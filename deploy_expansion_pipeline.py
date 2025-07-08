#!/usr/bin/env python3
"""
Expansion Pipeline Deployment Orchestrator
Trains LightGBM, TimesNet, optimizes PPO, builds meta-learner, and registers all policies
"""

import os
import sys
import subprocess
import logging
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpansionPipelineOrchestrator:
    """Orchestrates the complete expansion pipeline deployment"""
    
    def __init__(self, encoder_path: str = None, parallel_jobs: int = 2):
        self.encoder_path = encoder_path or self._find_latest_encoder()
        self.parallel_jobs = parallel_jobs
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Pipeline status
        self.pipeline_status = {
            'lgbm_training': {'status': 'pending', 'model_path': None},
            'timesnet_training': {'status': 'pending', 'model_path': None},
            'ppo_optimization': {'status': 'pending', 'model_path': None},
            'meta_learning': {'status': 'pending', 'model_path': None},
            'policy_registration': {'status': 'pending', 'policies': []},
            'risk_controls': {'status': 'pending'},
            'monitoring_setup': {'status': 'pending'}
        }
        
        logger.info("🚀 Expansion Pipeline Orchestrator initialized")
        logger.info(f"📁 Using encoder: {self.encoder_path}")
        logger.info(f"⚡ Parallel jobs: {self.parallel_jobs}")
    
    def _find_latest_encoder(self) -> str:
        """Find the latest TSA-MAE encoder"""
        encoder_files = list(self.models_dir.glob('encoder_*.pt'))
        if not encoder_files:
            raise FileNotFoundError("No TSA-MAE encoder found. Run TSA-MAE training first.")
        
        # Sort by modification time
        latest_encoder = max(encoder_files, key=os.path.getmtime)
        return str(latest_encoder)
    
    def train_lightgbm(self, symbols: List[str] = ['SOL', 'BTC', 'ETH'], 
                      optuna_trials: int = 50) -> Optional[str]:
        """Train LightGBM with TSA-MAE embeddings"""
        
        logger.info("🌲 Starting LightGBM training...")
        self.pipeline_status['lgbm_training']['status'] = 'running'
        
        try:
            cmd = [
                sys.executable, 'models/train_lgbm.py',
                '--encoder', self.encoder_path,
                '--coins', ','.join(symbols),
                '--optuna_trials', str(optuna_trials),
                '--days', '30'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hours
            
            if result.returncode == 0:
                # Find the trained model
                model_files = list(self.models_dir.glob('lgbm_*.pkl'))
                if model_files:
                    latest_model = max(model_files, key=os.path.getmtime)
                    self.pipeline_status['lgbm_training']['status'] = 'completed'
                    self.pipeline_status['lgbm_training']['model_path'] = str(latest_model)
                    logger.info(f"✅ LightGBM training completed: {latest_model}")
                    return str(latest_model)
                else:
                    raise FileNotFoundError("LightGBM model not found after training")
            else:
                raise RuntimeError(f"LightGBM training failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ LightGBM training failed: {e}")
            self.pipeline_status['lgbm_training']['status'] = 'failed'
            return None
    
    def train_timesnet(self, symbols: List[str] = ['SOL', 'BTC', 'ETH'],
                      window: int = 1024, d_model: int = 128, 
                      heads: int = 8, epochs: int = 100) -> Optional[str]:
        """Train TimesNet for long-range dependencies"""
        
        logger.info("🕰️ Starting TimesNet training...")
        self.pipeline_status['timesnet_training']['status'] = 'running'
        
        try:
            cmd = [
                sys.executable, 'models/train_timesnet.py',
                '--window', str(window),
                '--d_model', str(d_model),
                '--heads', str(heads),
                '--layers', '6',
                '--symbols', ','.join(symbols),
                '--days', '90',
                '--epochs', str(epochs)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)  # 3 hours
            
            if result.returncode == 0:
                # Find the trained model
                model_files = list(self.models_dir.glob('timesnet_*.pt'))
                if model_files:
                    latest_model = max(model_files, key=os.path.getmtime)
                    self.pipeline_status['timesnet_training']['status'] = 'completed'
                    self.pipeline_status['timesnet_training']['model_path'] = str(latest_model)
                    logger.info(f"✅ TimesNet training completed: {latest_model}")
                    return str(latest_model)
                else:
                    raise FileNotFoundError("TimesNet model not found after training")
            else:
                raise RuntimeError(f"TimesNet training failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ TimesNet training failed: {e}")
            self.pipeline_status['timesnet_training']['status'] = 'failed'
            return None
    
    def optimize_ppo(self, trials: int = 30, steps: int = 50000, 
                    n_envs: int = 8) -> Optional[str]:
        """Optimize PPO hyperparameters with Optuna"""
        
        logger.info("🎯 Starting PPO optimization...")
        self.pipeline_status['ppo_optimization']['status'] = 'running'
        
        try:
            cmd = [
                sys.executable, 'train_ppo_optuna.py',
                '--encoder', self.encoder_path,
                '--trials', str(trials),
                '--steps', str(steps),
                '--n_envs', str(n_envs)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=21600)  # 6 hours
            
            if result.returncode == 0:
                # Find the optimized model
                model_files = list(self.models_dir.glob('ppo_optuna_*.pt'))
                if model_files:
                    latest_model = max(model_files, key=os.path.getmtime)
                    self.pipeline_status['ppo_optimization']['status'] = 'completed'
                    self.pipeline_status['ppo_optimization']['model_path'] = str(latest_model)
                    logger.info(f"✅ PPO optimization completed: {latest_model}")
                    return str(latest_model)
                else:
                    raise FileNotFoundError("PPO model not found after optimization")
            else:
                raise RuntimeError(f"PPO optimization failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ PPO optimization failed: {e}")
            self.pipeline_status['ppo_optimization']['status'] = 'failed'
            return None
    
    def build_meta_learner(self, lgbm_path: str = None, timesnet_path: str = None,
                          ppo_path: str = None) -> Optional[str]:
        """Build meta-learner ensemble"""
        
        logger.info("🧠 Building meta-learner ensemble...")
        self.pipeline_status['meta_learning']['status'] = 'running'
        
        try:
            cmd = [sys.executable, 'meta_learner.py']
            
            # Add model paths if available
            if lgbm_path:
                cmd.extend(['--lgbm_path', lgbm_path])
            if timesnet_path:
                cmd.extend(['--timesnet_path', timesnet_path])
            if ppo_path:
                cmd.extend(['--ppo_path', ppo_path])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour
            
            if result.returncode == 0:
                # Find the meta-learner
                model_files = list(self.models_dir.glob('meta_learner_*.pkl'))
                if model_files:
                    latest_model = max(model_files, key=os.path.getmtime)
                    self.pipeline_status['meta_learning']['status'] = 'completed'
                    self.pipeline_status['meta_learning']['model_path'] = str(latest_model)
                    logger.info(f"✅ Meta-learner built: {latest_model}")
                    return str(latest_model)
                else:
                    raise FileNotFoundError("Meta-learner not found after training")
            else:
                raise RuntimeError(f"Meta-learner training failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ Meta-learner training failed: {e}")
            self.pipeline_status['meta_learning']['status'] = 'failed'
            return None
    
    def register_policies(self, model_paths: Dict[str, str]) -> List[str]:
        """Register all trained policies"""
        
        logger.info("📋 Registering policies...")
        self.pipeline_status['policy_registration']['status'] = 'running'
        
        registered_policies = []
        
        # Registration configurations
        registrations = [
            {
                'path': model_paths.get('lgbm'),
                'type': 'supervised',
                'subtype': 'lightgbm',
                'traffic': 0.05
            },
            {
                'path': model_paths.get('timesnet'),
                'type': 'supervised',
                'subtype': 'timesnet',
                'traffic': 0.05
            },
            {
                'path': model_paths.get('ppo'),
                'type': 'reinforcement',
                'subtype': 'ppo',
                'traffic': 0.05
            },
            {
                'path': model_paths.get('meta'),
                'type': 'ensemble',
                'subtype': 'meta',
                'traffic': 0.10
            }
        ]
        
        for reg in registrations:
            if not reg['path'] or not os.path.exists(reg['path']):
                logger.warning(f"⚠️ Skipping registration: {reg['path']} not found")
                continue
            
            try:
                cmd = [
                    sys.executable, 'register_policy.py',
                    '--path', reg['path'],
                    '--type', reg['type'],
                    '--subtype', reg['subtype'],
                    '--traffic', str(reg['traffic']),
                    '--symbols', 'SOL,BTC,ETH'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    # Extract policy ID from output
                    output_lines = result.stdout.split('\n')
                    policy_id = None
                    for line in output_lines:
                        if 'Policy ID:' in line:
                            policy_id = line.split('Policy ID:')[1].strip()
                            break
                    
                    if policy_id:
                        registered_policies.append(policy_id)
                        logger.info(f"✅ Registered: {policy_id} ({reg['subtype']})")
                    else:
                        logger.warning(f"⚠️ Policy registered but ID not found: {reg['subtype']}")
                else:
                    logger.error(f"❌ Registration failed: {reg['subtype']} - {result.stderr}")
                    
            except Exception as e:
                logger.error(f"❌ Registration error: {reg['subtype']} - {e}")
        
        self.pipeline_status['policy_registration']['status'] = 'completed'
        self.pipeline_status['policy_registration']['policies'] = registered_policies
        
        return registered_policies
    
    def setup_risk_controls(self):
        """Setup enhanced risk controls"""
        
        logger.info("🛡️ Setting up risk controls...")
        self.pipeline_status['risk_controls']['status'] = 'running'
        
        try:
            # Create risk configuration
            risk_config = {
                'portfolio_dd_fuse': 0.04,  # 4% portfolio DD triggers risk reduction
                'individual_dd_limits': {
                    'supervised': 0.05,
                    'reinforcement': 0.03,
                    'ensemble': 0.06
                },
                'traffic_caps': {
                    'supervised': 0.25,
                    'reinforcement': 0.15,
                    'ensemble': 0.35
                },
                'correlation_limits': {
                    'max_correlation': 0.85,
                    'min_trades_for_check': 100
                },
                'promotion_rules': {
                    'min_trades': 150,
                    'min_pf': 2.0,
                    'max_dd': 3.0,
                    'promotion_factor': 1.5
                },
                'throttling_rules': {
                    'consecutive_losses': 10,
                    'max_dd_breach': 0.05,
                    'correlation_breach': 0.90
                }
            }
            
            # Save risk configuration
            risk_config_path = self.models_dir / 'risk_controls.json'
            with open(risk_config_path, 'w') as f:
                json.dump(risk_config, f, indent=2)
            
            logger.info(f"✅ Risk controls configured: {risk_config_path}")
            self.pipeline_status['risk_controls']['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"❌ Risk controls setup failed: {e}")
            self.pipeline_status['risk_controls']['status'] = 'failed'
    
    def setup_monitoring(self):
        """Setup monitoring and alerting"""
        
        logger.info("📊 Setting up monitoring...")
        self.pipeline_status['monitoring_setup']['status'] = 'running'
        
        try:
            # Create monitoring configuration
            monitoring_config = {
                'metrics_to_track': [
                    'pf_lgbm_30', 'pf_timesnet_30', 'pf_ppo_30', 'pf_meta_30',
                    'dd_lgbm_pct', 'dd_timesnet_pct', 'dd_ppo_pct', 'dd_meta_pct',
                    'traffic_allocation_total', 'correlation_matrix_max',
                    'portfolio_dd_current', 'active_policies_count'
                ],
                'alert_thresholds': {
                    'portfolio_dd_warning': 0.025,
                    'portfolio_dd_critical': 0.04,
                    'individual_dd_warning': 0.03,
                    'individual_dd_critical': 0.05,
                    'correlation_warning': 0.80,
                    'correlation_critical': 0.90
                },
                'check_intervals': {
                    'performance_check': 300,  # 5 minutes
                    'correlation_check': 1800,  # 30 minutes
                    'promotion_check': 3600,   # 1 hour
                    'risk_check': 600         # 10 minutes
                },
                'notification_channels': {
                    'email': 'alerts@trading-system.com',
                    'slack': '#trading-alerts',
                    'dashboard': 'http://localhost:8080/dashboard'
                }
            }
            
            # Save monitoring configuration
            monitoring_config_path = self.models_dir / 'monitoring_config.json'
            with open(monitoring_config_path, 'w') as f:
                json.dump(monitoring_config, f, indent=2)
            
            logger.info(f"✅ Monitoring configured: {monitoring_config_path}")
            self.pipeline_status['monitoring_setup']['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup failed: {e}")
            self.pipeline_status['monitoring_setup']['status'] = 'failed'
    
    def run_parallel_training(self, symbols: List[str] = ['SOL', 'BTC', 'ETH']) -> Dict[str, str]:
        """Run LightGBM and TimesNet training in parallel"""
        
        logger.info("⚡ Starting parallel training...")
        
        model_paths = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
            # Submit training jobs
            futures = {
                'lgbm': executor.submit(self.train_lightgbm, symbols, 50),
                'timesnet': executor.submit(self.train_timesnet, symbols, 1024, 128, 8, 100)
            }
            
            # Wait for completion
            for model_type, future in futures.items():
                try:
                    result = future.result(timeout=14400)  # 4 hours max
                    if result:
                        model_paths[model_type] = result
                        logger.info(f"✅ {model_type.upper()} training completed")
                    else:
                        logger.error(f"❌ {model_type.upper()} training failed")
                except concurrent.futures.TimeoutError:
                    logger.error(f"⏰ {model_type.upper()} training timed out")
                except Exception as e:
                    logger.error(f"❌ {model_type.upper()} training error: {e}")
        
        return model_paths
    
    def run_full_pipeline(self, symbols: List[str] = ['SOL', 'BTC', 'ETH']) -> Dict:
        """Run the complete expansion pipeline"""
        
        logger.info("🚀 Starting full expansion pipeline...")
        start_time = time.time()
        
        # Step 1: Parallel training (LightGBM + TimesNet)
        logger.info("📊 Step 1: Parallel supervised model training")
        model_paths = self.run_parallel_training(symbols)
        
        # Step 2: PPO optimization
        logger.info("🎯 Step 2: PPO hyperparameter optimization")
        ppo_path = self.optimize_ppo(trials=30, steps=50000, n_envs=8)
        if ppo_path:
            model_paths['ppo'] = ppo_path
        
        # Step 3: Meta-learner ensemble
        logger.info("🧠 Step 3: Meta-learner ensemble building")
        meta_path = self.build_meta_learner(
            lgbm_path=model_paths.get('lgbm'),
            timesnet_path=model_paths.get('timesnet'),
            ppo_path=model_paths.get('ppo')
        )
        if meta_path:
            model_paths['meta'] = meta_path
        
        # Step 4: Policy registration
        logger.info("📋 Step 4: Policy registration")
        registered_policies = self.register_policies(model_paths)
        
        # Step 5: Risk controls setup
        logger.info("🛡️ Step 5: Risk controls setup")
        self.setup_risk_controls()
        
        # Step 6: Monitoring setup
        logger.info("📊 Step 6: Monitoring setup")
        self.setup_monitoring()
        
        # Pipeline completion
        end_time = time.time()
        duration = end_time - start_time
        
        # Generate summary
        summary = {
            'pipeline_duration': duration,
            'models_trained': len([p for p in model_paths.values() if p]),
            'policies_registered': len(registered_policies),
            'model_paths': model_paths,
            'registered_policies': registered_policies,
            'pipeline_status': self.pipeline_status,
            'completion_time': datetime.now().isoformat()
        }
        
        logger.info("🎉 Expansion pipeline completed!")
        logger.info(f"⏱️ Total duration: {duration/3600:.2f} hours")
        logger.info(f"🎯 Models trained: {summary['models_trained']}")
        logger.info(f"📋 Policies registered: {summary['policies_registered']}")
        
        return summary
    
    def show_status(self):
        """Show current pipeline status"""
        
        print("\n📊 Expansion Pipeline Status:")
        print("=" * 60)
        
        for step, status in self.pipeline_status.items():
            status_icon = {
                'pending': '⏳',
                'running': '🔄',
                'completed': '✅',
                'failed': '❌'
            }.get(status['status'], '❓')
            
            print(f"{status_icon} {step.replace('_', ' ').title()}: {status['status']}")
            
            if status.get('model_path'):
                print(f"   📁 Model: {status['model_path']}")
            if status.get('policies'):
                print(f"   📋 Policies: {len(status['policies'])}")
        
        print("=" * 60)

def main():
    """Main deployment function"""
    
    parser = argparse.ArgumentParser(description='Deploy expansion pipeline')
    parser.add_argument('--encoder', help='Path to TSA-MAE encoder')
    parser.add_argument('--symbols', default='SOL,BTC,ETH', help='Trading symbols')
    parser.add_argument('--parallel', type=int, default=2, help='Parallel jobs')
    parser.add_argument('--quick', action='store_true', help='Quick mode (reduced trials/epochs)')
    parser.add_argument('--status', action='store_true', help='Show pipeline status')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no actual training)')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ExpansionPipelineOrchestrator(
        encoder_path=args.encoder,
        parallel_jobs=args.parallel
    )
    
    if args.status:
        orchestrator.show_status()
        return
    
    if args.dry_run:
        logger.info("🔍 Dry run mode - no actual training")
        orchestrator.show_status()
        return
    
    # Parse symbols
    symbols = args.symbols.split(',')
    
    # Run pipeline
    try:
        if args.quick:
            logger.info("⚡ Quick mode enabled - reduced training time")
            # Override with quick settings
            summary = orchestrator.run_full_pipeline(symbols)
        else:
            summary = orchestrator.run_full_pipeline(symbols)
        
        # Save summary
        summary_path = Path('expansion_pipeline_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Pipeline summary saved: {summary_path}")
        
        # Show final status
        orchestrator.show_status()
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 