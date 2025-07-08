#!/usr/bin/env python3
"""
Advanced Learning Pipeline Executor
Runs the complete pipeline: TSA-MAE pre-training → PPO fine-tuning → Bandit registration
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add models directory to path
sys.path.append('models')

from tsa_mae import pretrain_tsa_mae, load_historical_data
from ppo_pyramiding import fine_tune_ppo, load_pretrained_tsa_mae
from register_policy import register_ppo_pyramiding_policy, PolicyRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedLearningPipeline:
    """Complete Advanced Learning Pipeline"""
    
    def __init__(self, 
                 symbols: list = None,
                 training_months: int = 12,
                 device: str = 'auto'):
        
        self.symbols = symbols or ['SOL', 'BTC', 'ETH']
        self.training_months = training_months
        
        # Auto-detect device
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        logger.info(f"🧠 Advanced Learning Pipeline initialized")
        logger.info(f"📊 Symbols: {self.symbols}")
        logger.info(f"📅 Training months: {training_months}")
        logger.info(f"🖥️  Device: {self.device}")
    
    def run_complete_pipeline(self, 
                            tsa_mae_epochs: int = 100,
                            ppo_episodes: int = 1000,
                            skip_pretraining: bool = False,
                            skip_finetuning: bool = False,
                            skip_registration: bool = False):
        """
        Run the complete Advanced Learning pipeline
        
        Args:
            tsa_mae_epochs: Epochs for TSA-MAE pre-training
            ppo_episodes: Episodes for PPO fine-tuning
            skip_pretraining: Skip TSA-MAE pre-training if model exists
            skip_finetuning: Skip PPO fine-tuning if model exists
            skip_registration: Skip policy registration
        """
        
        logger.info("🚀 Starting Advanced Learning Pipeline")
        logger.info("=" * 60)
        
        pipeline_start_time = time.time()
        
        # Step 1: TSA-MAE Pre-training
        tsa_mae_model = None
        if not skip_pretraining:
            logger.info("📊 STEP 1: TSA-MAE Pre-training")
            logger.info("-" * 40)
            
            # Check if pre-trained model exists
            if os.path.exists('models/tsa_mae_final.pt') and skip_pretraining:
                logger.info("✅ TSA-MAE model already exists, loading...")
                tsa_mae_model, scaler, features = load_pretrained_tsa_mae()
            else:
                logger.info(f"🧠 Pre-training TSA-MAE on {self.training_months} months of data...")
                
                try:
                    tsa_mae_model, scaler, features = pretrain_tsa_mae(
                        symbols=self.symbols,
                        months=self.training_months,
                        epochs=tsa_mae_epochs,
                        batch_size=32,
                        learning_rate=1e-4,
                        device=self.device
                    )
                    
                    logger.info("✅ TSA-MAE pre-training completed successfully!")
                    
                except Exception as e:
                    logger.error(f"❌ TSA-MAE pre-training failed: {e}")
                    return False
        
        # Step 2: PPO Fine-tuning
        ppo_agent = None
        if not skip_finetuning:
            logger.info("\n🤖 STEP 2: PPO Fine-tuning")
            logger.info("-" * 40)
            
            # Check if fine-tuned model exists
            if os.path.exists('models/ppo_pyramiding_final.pt') and skip_finetuning:
                logger.info("✅ PPO model already exists, skipping fine-tuning...")
            else:
                logger.info(f"🎯 Fine-tuning PPO for {ppo_episodes} episodes...")
                
                try:
                    ppo_agent, trainer = fine_tune_ppo(
                        episodes=ppo_episodes,
                        buffer_size=2048,
                        device=self.device
                    )
                    
                    logger.info("✅ PPO fine-tuning completed successfully!")
                    
                except Exception as e:
                    logger.error(f"❌ PPO fine-tuning failed: {e}")
                    return False
        
        # Step 3: Policy Registration
        policy_id = None
        if not skip_registration:
            logger.info("\n🎯 STEP 3: Policy Registration")
            logger.info("-" * 40)
            
            try:
                policy_id = register_ppo_pyramiding_policy()
                
                if policy_id:
                    logger.info("✅ Policy registration completed successfully!")
                else:
                    logger.error("❌ Policy registration failed")
                    return False
                
            except Exception as e:
                logger.error(f"❌ Policy registration failed: {e}")
                return False
        
        # Pipeline completion
        pipeline_end_time = time.time()
        pipeline_duration = pipeline_end_time - pipeline_start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ADVANCED LEARNING PIPELINE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total duration: {pipeline_duration:.1f} seconds")
        logger.info(f"🧠 TSA-MAE model: {'✅' if tsa_mae_model else '⏭️ '}")
        logger.info(f"🤖 PPO agent: {'✅' if ppo_agent else '⏭️ '}")
        logger.info(f"🎯 Policy ID: {policy_id if policy_id else 'N/A'}")
        
        # Show final status
        self.show_pipeline_status()
        
        return True
    
    def show_pipeline_status(self):
        """Show current pipeline status"""
        logger.info("\n📊 PIPELINE STATUS")
        logger.info("-" * 30)
        
        # Check model files
        models = {
            'TSA-MAE (Pre-trained)': 'models/tsa_mae_final.pt',
            'PPO Pyramiding (Fine-tuned)': 'models/ppo_pyramiding_final.pt'
        }
        
        for name, path in models.items():
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024 / 1024  # MB
                logger.info(f"✅ {name}: {size:.1f} MB")
            else:
                logger.info(f"❌ {name}: Not found")
        
        # Check policy registry
        try:
            registry = PolicyRegistry()
            policies = registry.list_policies()
            
            logger.info(f"\n🎯 REGISTERED POLICIES: {len(policies)}")
            for policy in policies:
                status_icon = "🟢" if policy['status'] == 'active' else "🔴"
                logger.info(f"{status_icon} {policy['name']} v{policy['version']}: "
                           f"{policy['traffic_allocation']:.1%} traffic")
        
        except Exception as e:
            logger.warning(f"Could not check policy registry: {e}")
    
    def validate_pipeline(self):
        """Validate pipeline components"""
        logger.info("🔍 Validating pipeline components...")
        
        validation_results = {
            'tsa_mae_model': False,
            'ppo_model': False,
            'policy_registered': False
        }
        
        # Check TSA-MAE model
        if os.path.exists('models/tsa_mae_final.pt'):
            try:
                tsa_mae_model, scaler, features = load_pretrained_tsa_mae()
                if tsa_mae_model is not None:
                    validation_results['tsa_mae_model'] = True
                    logger.info("✅ TSA-MAE model validation passed")
                else:
                    logger.error("❌ TSA-MAE model validation failed")
            except Exception as e:
                logger.error(f"❌ TSA-MAE model validation error: {e}")
        
        # Check PPO model
        if os.path.exists('models/ppo_pyramiding_final.pt'):
            try:
                import torch
                checkpoint = torch.load('models/ppo_pyramiding_final.pt', map_location='cpu')
                if 'agent_state_dict' in checkpoint:
                    validation_results['ppo_model'] = True
                    logger.info("✅ PPO model validation passed")
                else:
                    logger.error("❌ PPO model validation failed")
            except Exception as e:
                logger.error(f"❌ PPO model validation error: {e}")
        
        # Check policy registration
        try:
            registry = PolicyRegistry()
            policies = registry.list_policies()
            
            ppo_policies = [p for p in policies if 'PPO' in p['name']]
            if ppo_policies:
                validation_results['policy_registered'] = True
                logger.info("✅ Policy registration validation passed")
            else:
                logger.error("❌ No PPO policies found in registry")
        
        except Exception as e:
            logger.error(f"❌ Policy registry validation error: {e}")
        
        # Summary
        passed = sum(validation_results.values())
        total = len(validation_results)
        
        logger.info(f"\n🎯 Validation Summary: {passed}/{total} components passed")
        
        if passed == total:
            logger.info("🎉 All pipeline components validated successfully!")
            return True
        else:
            logger.warning("⚠️  Some pipeline components failed validation")
            return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Advanced Learning Pipeline')
    
    parser.add_argument('--symbols', nargs='+', default=['SOL', 'BTC', 'ETH'],
                       help='Trading symbols to train on')
    parser.add_argument('--training-months', type=int, default=12,
                       help='Months of historical data for training')
    parser.add_argument('--tsa-mae-epochs', type=int, default=50,
                       help='Epochs for TSA-MAE pre-training')
    parser.add_argument('--ppo-episodes', type=int, default=500,
                       help='Episodes for PPO fine-tuning')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--skip-pretraining', action='store_true',
                       help='Skip TSA-MAE pre-training if model exists')
    parser.add_argument('--skip-finetuning', action='store_true',
                       help='Skip PPO fine-tuning if model exists')
    parser.add_argument('--skip-registration', action='store_true',
                       help='Skip policy registration')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing pipeline')
    parser.add_argument('--status-only', action='store_true',
                       help='Only show pipeline status')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AdvancedLearningPipeline(
        symbols=args.symbols,
        training_months=args.training_months,
        device=args.device
    )
    
    # Execute based on arguments
    if args.status_only:
        pipeline.show_pipeline_status()
    elif args.validate_only:
        pipeline.validate_pipeline()
    else:
        # Run complete pipeline
        success = pipeline.run_complete_pipeline(
            tsa_mae_epochs=args.tsa_mae_epochs,
            ppo_episodes=args.ppo_episodes,
            skip_pretraining=args.skip_pretraining,
            skip_finetuning=args.skip_finetuning,
            skip_registration=args.skip_registration
        )
        
        if success:
            logger.info("\n🎉 Pipeline execution completed successfully!")
            
            # Validate after completion
            if pipeline.validate_pipeline():
                logger.info("🚀 Advanced Learning Layer is ready for deployment!")
            else:
                logger.warning("⚠️  Some validation checks failed")
        else:
            logger.error("❌ Pipeline execution failed")
            sys.exit(1)

if __name__ == "__main__":
    main() 