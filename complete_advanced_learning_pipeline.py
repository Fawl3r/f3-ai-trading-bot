#!/usr/bin/env python3
"""
Complete Advanced Learning Layer Pipeline
1. TSA-MAE pre-training on 12 months SOL/BTC/ETH
2. PPO fine-tuning for dynamic pyramiding
3. Thompson Sampling bandit registration with 10% allocation
"""

import os
import sys
import subprocess
import logging
import argparse
import torch
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedLearningPipeline:
    """Complete Advanced Learning Layer Pipeline"""
    
    def __init__(self, device='auto'):
        self.device = self._select_device(device)
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Advanced Learning Layer Pipeline")
        logger.info(f"🖥️  Device: {self.device}")
        logger.info(f"📁 Models directory: {self.models_dir}")
    
    def _select_device(self, device):
        """Auto-select best available device"""
        if device == 'auto':
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                return 'cuda'
            else:
                logger.warning("⚠️  CUDA not available, using CPU")
                return 'cpu'
        return device
    
    def step1_train_tsa_mae(self, months=12, epochs=50, batch_size=None, quick=False):
        """Step 1: Train TSA-MAE on crypto data"""
        
        logger.info("\n" + "="*60)
        logger.info("📊 STEP 1: TSA-MAE Pre-training")
        logger.info("="*60)
        
        if quick:
            logger.info("🧪 Quick mode: 1 month, 5 epochs")
            months = 1
            epochs = 5
            batch_size = 32
        elif batch_size is None:
            # Auto-select batch size based on GPU
            if self.device == 'cuda':
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory >= 11:  # RTX 2080 Ti or better
                    batch_size = 192
                elif gpu_memory >= 8:
                    batch_size = 128
                else:
                    batch_size = 64
            else:
                batch_size = 32
        
        logger.info(f"📈 Training: {months} months, {epochs} epochs, batch size {batch_size}")
        
        # Run TSA-MAE training
        cmd = [
            'python', 'rtx2080ti_trainer.py',
            '--months', str(months),
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--symbols', 'SOL', 'BTC', 'ETH'
        ]
        
        logger.info(f"🔥 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600*4)  # 4 hour timeout
            
            if result.returncode == 0:
                logger.info("✅ TSA-MAE training completed successfully")
                
                # Extract model hash from output
                output_lines = result.stdout.split('\n')
                encoder_path = None
                model_hash = None
                
                for line in output_lines:
                    if 'Model:' in line:
                        model_hash = line.split('Model:')[1].strip()
                    if 'encoder_' in line and '.pt' in line:
                        # Find the encoder file
                        for model_file in self.models_dir.glob('encoder_*.pt'):
                            if model_hash in str(model_file):
                                encoder_path = str(model_file)
                                break
                
                if not encoder_path:
                    # Find latest encoder
                    encoder_files = list(self.models_dir.glob('encoder_*.pt'))
                    if encoder_files:
                        encoder_path = str(max(encoder_files, key=os.path.getctime))
                
                logger.info(f"📁 Encoder saved: {encoder_path}")
                return encoder_path
                
            else:
                logger.error(f"❌ TSA-MAE training failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ TSA-MAE training timed out")
            return None
        except Exception as e:
            logger.error(f"💥 TSA-MAE training error: {e}")
            return None
    
    def step2_train_ppo(self, encoder_path, episodes=250, steps=1000):
        """Step 2: Train PPO agent with TSA-MAE embeddings"""
        
        logger.info("\n" + "="*60)
        logger.info("🎮 STEP 2: PPO Dynamic Pyramiding")
        logger.info("="*60)
        
        if not encoder_path or not os.path.exists(encoder_path):
            logger.warning("⚠️  No encoder found, training PPO without embeddings")
            encoder_path = None
        
        logger.info(f"📁 Using encoder: {encoder_path}")
        logger.info(f"🎯 Training: {episodes} episodes, {steps} steps each")
        
        # Run PPO training
        cmd = [
            'python', 'train_ppo_pyramiding.py',
            '--episodes', str(episodes),
            '--steps', str(steps)
        ]
        
        if encoder_path:
            cmd.extend(['--encoder', encoder_path])
        
        logger.info(f"🔥 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                logger.info("✅ PPO training completed successfully")
                
                # Find PPO model
                ppo_files = list(self.models_dir.glob('ppo_*.pt'))
                if ppo_files:
                    ppo_path = str(max(ppo_files, key=os.path.getctime))
                    logger.info(f"📁 PPO model saved: {ppo_path}")
                    return ppo_path
                else:
                    logger.error("❌ PPO model not found")
                    return None
                    
            else:
                logger.error(f"❌ PPO training failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ PPO training timed out")
            return None
        except Exception as e:
            logger.error(f"💥 PPO training error: {e}")
            return None
    
    def step3_register_policy(self, model_path, policy_type='reinforcement'):
        """Step 3: Register policy with Thompson Sampling bandit"""
        
        logger.info("\n" + "="*60)
        logger.info("🎰 STEP 3: Policy Registration")
        logger.info("="*60)
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"❌ Model not found: {model_path}")
            return False
        
        logger.info(f"📁 Registering model: {model_path}")
        logger.info(f"🏷️  Policy type: {policy_type}")
        logger.info(f"📊 Initial allocation: 10%")
        
        # Generate policy name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = Path(model_path).stem
        policy_name = f"{policy_type}_{timestamp}_{model_name}"
        
        # Run policy registration
        cmd = [
            'python', 'enhanced_register_policy.py',
            '--path', model_path,
            '--name', policy_name,
            '--type', policy_type,
            '--simulate'  # Add some simulation data
        ]
        
        logger.info(f"🔥 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Policy registration completed successfully")
                logger.info(f"🎯 Policy name: {policy_name}")
                logger.info("📊 Thompson Sampling bandit updated")
                return True
                
            else:
                logger.error(f"❌ Policy registration failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"💥 Policy registration error: {e}")
            return False
    
    def run_complete_pipeline(self, quick=False, tsa_mae_epochs=50, ppo_episodes=250):
        """Run the complete Advanced Learning Layer pipeline"""
        
        logger.info("🌟 Starting Complete Advanced Learning Layer Pipeline")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # Step 1: TSA-MAE pre-training
        logger.info("Phase 1: Time Series Masked AutoEncoder")
        encoder_path = self.step1_train_tsa_mae(
            months=12 if not quick else 1,
            epochs=tsa_mae_epochs if not quick else 5,
            quick=quick
        )
        
        if not encoder_path:
            logger.error("❌ Pipeline failed at TSA-MAE training")
            return False
        
        # Step 2: PPO fine-tuning
        logger.info("Phase 2: PPO Dynamic Pyramiding")
        ppo_path = self.step2_train_ppo(
            encoder_path=encoder_path,
            episodes=ppo_episodes if not quick else 50,
            steps=1000 if not quick else 200
        )
        
        if not ppo_path:
            logger.error("❌ Pipeline failed at PPO training")
            return False
        
        # Step 3: Policy registration
        logger.info("Phase 3: Thompson Sampling Registration")
        success = self.step3_register_policy(
            model_path=ppo_path,
            policy_type='reinforcement'
        )
        
        if not success:
            logger.error("❌ Pipeline failed at policy registration")
            return False
        
        # Pipeline complete
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("🎉 ADVANCED LEARNING LAYER PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"⏱️  Total time: {duration}")
        logger.info(f"📁 TSA-MAE encoder: {encoder_path}")
        logger.info(f"📁 PPO agent: {ppo_path}")
        logger.info("🎰 Thompson Sampling bandit: 10% traffic allocated")
        logger.info("\n🚀 Ready for live deployment!")
        
        # Show next steps
        logger.info("\n📋 Next Steps:")
        logger.info("1. python enhanced_register_policy.py --stats  # View bandit status")
        logger.info("2. python enhanced_register_policy.py --select  # Test policy selection")
        logger.info("3. Integrate with live trading system")
        logger.info("4. Monitor performance and auto-scaling")
        
        return True
    
    def show_status(self):
        """Show current pipeline status"""
        
        logger.info("📊 Advanced Learning Layer Status")
        logger.info("="*50)
        
        # Check models
        encoder_files = list(self.models_dir.glob('encoder_*.pt'))
        ppo_files = list(self.models_dir.glob('ppo_*.pt'))
        tabnet_files = list(self.models_dir.glob('tabnet_*.pt'))
        
        logger.info(f"🧠 TSA-MAE encoders: {len(encoder_files)}")
        if encoder_files:
            latest_encoder = max(encoder_files, key=os.path.getctime)
            logger.info(f"   Latest: {latest_encoder.name}")
        
        logger.info(f"🎮 PPO agents: {len(ppo_files)}")
        if ppo_files:
            latest_ppo = max(ppo_files, key=os.path.getctime)
            logger.info(f"   Latest: {latest_ppo.name}")
        
        logger.info(f"📊 TabNet models: {len(tabnet_files)}")
        
        # Check bandit database
        bandit_db = Path('models/policy_bandit.db')
        if bandit_db.exists():
            logger.info("🎰 Thompson Sampling bandit: Active")
            
            # Show stats
            try:
                cmd = ['python', 'enhanced_register_policy.py', '--stats']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(result.stdout)
            except:
                pass
        else:
            logger.info("🎰 Thompson Sampling bandit: Not initialized")

def main():
    """Main pipeline runner"""
    
    parser = argparse.ArgumentParser(description='Advanced Learning Layer Pipeline')
    
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test mode (1 month, 5 epochs)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--tsa-mae-epochs', type=int, default=50,
                       help='TSA-MAE training epochs')
    parser.add_argument('--ppo-episodes', type=int, default=250,
                       help='PPO training episodes')
    parser.add_argument('--status', action='store_true',
                       help='Show pipeline status')
    parser.add_argument('--step', type=int, choices=[1, 2, 3],
                       help='Run specific step only')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AdvancedLearningPipeline(device=args.device)
    
    if args.status:
        pipeline.show_status()
        return
    
    if args.step:
        if args.step == 1:
            encoder_path = pipeline.step1_train_tsa_mae(quick=args.quick, epochs=args.tsa_mae_epochs)
            print(f"Encoder: {encoder_path}")
        elif args.step == 2:
            # Find latest encoder
            encoder_files = list(Path('models').glob('encoder_*.pt'))
            encoder_path = str(max(encoder_files, key=os.path.getctime)) if encoder_files else None
            ppo_path = pipeline.step2_train_ppo(encoder_path, episodes=args.ppo_episodes)
            print(f"PPO: {ppo_path}")
        elif args.step == 3:
            # Find latest PPO model
            ppo_files = list(Path('models').glob('ppo_*.pt'))
            ppo_path = str(max(ppo_files, key=os.path.getctime)) if ppo_files else None
            success = pipeline.step3_register_policy(ppo_path)
            print(f"Registration: {'Success' if success else 'Failed'}")
    else:
        # Run complete pipeline
        success = pipeline.run_complete_pipeline(
            quick=args.quick,
            tsa_mae_epochs=args.tsa_mae_epochs,
            ppo_episodes=args.ppo_episodes
        )
        
        if success:
            logger.info("🌟 Pipeline completed successfully!")
        else:
            logger.error("💥 Pipeline failed!")
            sys.exit(1)

if __name__ == "__main__":
    main() 