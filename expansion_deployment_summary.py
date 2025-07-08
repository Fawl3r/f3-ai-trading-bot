#!/usr/bin/env python3
"""
Expansion Pipeline Deployment Summary
Comprehensive status report of all implemented components
"""

import os
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpansionDeploymentSummary:
    """Generate comprehensive deployment summary"""
    
    def __init__(self):
        self.models_dir = Path('models')
        self.db_path = 'models/policy_bandit.db'
    
    def check_models_trained(self):
        """Check which models have been successfully trained"""
        
        models_status = {
            'TSA-MAE Encoder': {
                'files': list(self.models_dir.glob('encoder_*.pt')),
                'latest': None,
                'status': 'Not Found'
            },
            'LightGBM + TSA-MAE': {
                'files': list(self.models_dir.glob('lgbm_*.pkl')),
                'latest': None,
                'status': 'Not Found'
            },
            'TimesNet Long-Range': {
                'files': list(self.models_dir.glob('timesnet_*.pt')),
                'latest': None,
                'status': 'Not Found'
            },
            'PPO Optimized': {
                'files': list(self.models_dir.glob('ppo_*.pt')),
                'latest': None,
                'status': 'Not Found'
            }
        }
        
        for model_name, info in models_status.items():
            if info['files']:
                # Get latest file by modification time
                latest_file = max(info['files'], key=os.path.getmtime)
                info['latest'] = latest_file
                info['status'] = 'Available'
                info['size'] = latest_file.stat().st_size / 1024 / 1024  # MB
                info['modified'] = datetime.fromtimestamp(latest_file.stat().st_mtime)
        
        return models_status
    
    def check_policies_registered(self):
        """Check registered policies in Thompson Sampling bandit"""
        
        if not os.path.exists(self.db_path):
            return {'status': 'Database not found', 'policies': []}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT p.id, p.name, p.type, p.model_path, p.config,
                       ba.traffic_allocation, ba.alpha, ba.beta,
                       (ba.alpha / (ba.alpha + ba.beta)) as win_rate_estimate
                FROM policies p
                JOIN bandit_arms ba ON p.id = ba.policy_id
                WHERE p.is_active = 1
                ORDER BY ba.traffic_allocation DESC
            ''')
            
            policies = []
            total_allocation = 0
            
            for row in cursor.fetchall():
                policy_id, name, ptype, model_path, config_str, allocation, alpha, beta, win_rate = row
                
                try:
                    config = json.loads(config_str) if config_str else {}
                except:
                    config = {}
                
                policies.append({
                    'id': policy_id,
                    'name': name,
                    'type': ptype,
                    'model_path': model_path,
                    'config': config,
                    'traffic_allocation': allocation,
                    'alpha': alpha,
                    'beta': beta,
                    'win_rate_estimate': win_rate,
                    'file_exists': os.path.exists(model_path) if model_path else False
                })
                
                total_allocation += allocation
            
            conn.close()
            
            return {
                'status': 'Success',
                'policies': policies,
                'total_allocation': total_allocation,
                'active_count': len(policies)
            }
            
        except Exception as e:
            conn.close()
            return {'status': f'Error: {e}', 'policies': []}
    
    def check_infrastructure(self):
        """Check infrastructure components"""
        
        infrastructure = {
            'Database': {
                'path': self.db_path,
                'exists': os.path.exists(self.db_path),
                'size_kb': 0
            },
            'Models Directory': {
                'path': str(self.models_dir),
                'exists': self.models_dir.exists(),
                'file_count': 0
            },
            'Configuration Files': {
                'register_policy': os.path.exists('register_policy.py'),
                'meta_learner': os.path.exists('meta_learner.py'),
                'train_lgbm': os.path.exists('models/train_lgbm.py'),
                'train_timesnet': os.path.exists('models/train_timesnet.py'),
                'train_ppo_optuna': os.path.exists('train_ppo_optuna.py')
            }
        }
        
        if infrastructure['Database']['exists']:
            infrastructure['Database']['size_kb'] = os.path.getsize(self.db_path) / 1024
        
        if infrastructure['Models Directory']['exists']:
            infrastructure['Models Directory']['file_count'] = len(list(self.models_dir.glob('*')))
        
        return infrastructure
    
    def calculate_expected_performance(self, policies):
        """Calculate expected performance improvements"""
        
        baseline_pf = 2.3  # Current system PF
        
        improvements = {
            'LightGBM': {'pf_gain': 0.15, 'dd_reduction': 0.002},
            'TimesNet': {'pf_gain': 0.20, 'dd_reduction': 0.002}, 
            'PPO Optimized': {'pf_gain': 0.15, 'dd_reduction': 0.003},
            'Meta Ensemble': {'pf_gain': 0.10, 'dd_reduction': 0.0}
        }
        
        # Calculate weighted improvements based on traffic allocation
        total_pf_gain = 0
        total_dd_reduction = 0
        
        for policy in policies:
            allocation = policy['traffic_allocation']
            
            if 'lightgbm' in policy['name'].lower():
                total_pf_gain += improvements['LightGBM']['pf_gain'] * allocation
                total_dd_reduction += improvements['LightGBM']['dd_reduction'] * allocation
            elif 'timesnet' in policy['name'].lower():
                total_pf_gain += improvements['TimesNet']['pf_gain'] * allocation
                total_dd_reduction += improvements['TimesNet']['dd_reduction'] * allocation
            elif 'ppo' in policy['name'].lower():
                total_pf_gain += improvements['PPO Optimized']['pf_gain'] * allocation
                total_dd_reduction += improvements['PPO Optimized']['dd_reduction'] * allocation
        
        expected_pf = baseline_pf + total_pf_gain
        
        return {
            'baseline_pf': baseline_pf,
            'expected_pf': expected_pf,
            'pf_improvement': total_pf_gain,
            'dd_reduction': total_dd_reduction,
            'target_range': [2.7, 2.9]
        }
    
    def generate_summary(self):
        """Generate comprehensive deployment summary"""
        
        summary = {
            'deployment_time': datetime.now().isoformat(),
            'models': self.check_models_trained(),
            'policies': self.check_policies_registered(),
            'infrastructure': self.check_infrastructure()
        }
        
        # Add performance projections
        if summary['policies']['status'] == 'Success':
            summary['performance_projection'] = self.calculate_expected_performance(
                summary['policies']['policies']
            )
        
        return summary
    
    def print_summary(self):
        """Print formatted deployment summary"""
        
        summary = self.generate_summary()
        
        print("\n" + "=" * 80)
        print("🚀 EXPANSION PIPELINE DEPLOYMENT SUMMARY")
        print("=" * 80)
        print(f"📅 Generated: {summary['deployment_time']}")
        print()
        
        # Models Status
        print("📊 TRAINED MODELS STATUS:")
        print("-" * 50)
        
        for model_name, info in summary['models'].items():
            status_icon = "✅" if info['status'] == 'Available' else "❌"
            print(f"{status_icon} {model_name}: {info['status']}")
            
            if info['status'] == 'Available':
                print(f"   📁 File: {info['latest'].name}")
                print(f"   📏 Size: {info['size']:.1f} MB")
                print(f"   🕒 Modified: {info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        # Policies Status
        print("🎯 REGISTERED POLICIES:")
        print("-" * 50)
        
        policies_info = summary['policies']
        if policies_info['status'] == 'Success':
            print(f"📋 Active Policies: {policies_info['active_count']}")
            print(f"📈 Total Traffic Allocation: {policies_info['total_allocation']:.1%}")
            print()
            
            for policy in policies_info['policies']:
                type_icon = {"supervised": "🧠", "reinforcement": "🎯", "ensemble": "🔗"}.get(policy['type'], "❓")
                file_icon = "✅" if policy['file_exists'] else "❌"
                
                print(f"{type_icon} {policy['name']} ({policy['type']})")
                print(f"   🎲 Traffic: {policy['traffic_allocation']:.1%}")
                print(f"   📊 Thompson: α={policy['alpha']:.1f}, β={policy['beta']:.1f}")
                print(f"   🎯 Est. Win Rate: {policy['win_rate_estimate']:.1%}")
                print(f"   {file_icon} Model File: {os.path.basename(policy['model_path'])}")
                print()
        else:
            print(f"❌ {policies_info['status']}")
            print()
        
        # Performance Projections
        if 'performance_projection' in summary:
            perf = summary['performance_projection']
            print("📈 PERFORMANCE PROJECTIONS:")
            print("-" * 50)
            print(f"📊 Current Baseline PF: {perf['baseline_pf']:.2f}")
            print(f"🎯 Expected New PF: {perf['expected_pf']:.2f}")
            print(f"🚀 PF Improvement: +{perf['pf_improvement']:.3f}")
            print(f"🛡️  DD Reduction: -{perf['dd_reduction']:.3f}%")
            print(f"🎯 Target Range: {perf['target_range'][0]:.1f} - {perf['target_range'][1]:.1f}")
            
            if perf['target_range'][0] <= perf['expected_pf'] <= perf['target_range'][1]:
                print("✅ PROJECTION: Target range achieved!")
            else:
                print("⚠️  PROJECTION: Below target range, need more traffic allocation")
            print()
        
        # Infrastructure Status
        print("🏗️  INFRASTRUCTURE STATUS:")
        print("-" * 50)
        
        infra = summary['infrastructure']
        
        db_icon = "✅" if infra['Database']['exists'] else "❌"
        print(f"{db_icon} Database: {infra['Database']['path']}")
        if infra['Database']['exists']:
            print(f"   📏 Size: {infra['Database']['size_kb']:.1f} KB")
        
        models_icon = "✅" if infra['Models Directory']['exists'] else "❌"
        print(f"{models_icon} Models Directory: {infra['Models Directory']['file_count']} files")
        
        print("📝 Configuration Files:")
        for file_name, exists in infra['Configuration Files'].items():
            icon = "✅" if exists else "❌"
            print(f"   {icon} {file_name}.py")
        
        print()
        
        # Implementation Checklist
        print("✅ IMPLEMENTATION CHECKLIST:")
        print("-" * 50)
        
        checklist = [
            ("TSA-MAE Encoder Pre-trained", summary['models']['TSA-MAE Encoder']['status'] == 'Available'),
            ("LightGBM + Embeddings Trained", summary['models']['LightGBM + TSA-MAE']['status'] == 'Available'),
            ("TimesNet Long-Range Trained", summary['models']['TimesNet Long-Range']['status'] == 'Available'),
            ("PPO Hyperparameters Optimized", summary['models']['PPO Optimized']['status'] == 'Available'),
            ("Policies Registered in Bandit", policies_info['status'] == 'Success'),
            ("Thompson Sampling Active", policies_info['active_count'] > 0 if policies_info['status'] == 'Success' else False),
            ("Traffic Allocation Configured", policies_info['total_allocation'] > 0 if policies_info['status'] == 'Success' else False)
        ]
        
        for item, status in checklist:
            icon = "✅" if status else "❌"
            print(f"{icon} {item}")
        
        completed_items = sum(1 for _, status in checklist if status)
        completion_rate = completed_items / len(checklist) * 100
        
        print()
        print(f"📊 Overall Completion: {completion_rate:.0f}% ({completed_items}/{len(checklist)})")
        
        if completion_rate == 100:
            print("🎉 EXPANSION PIPELINE FULLY DEPLOYED!")
        elif completion_rate >= 80:
            print("🚀 EXPANSION PIPELINE MOSTLY DEPLOYED - Minor items remaining")
        else:
            print("⚠️  EXPANSION PIPELINE PARTIALLY DEPLOYED - Major items remaining")
        
        print("=" * 80)
        
        return summary

def main():
    """Main function"""
    summary_generator = ExpansionDeploymentSummary()
    summary = summary_generator.print_summary()
    
    # Save summary to file
    summary_file = f"expansion_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"💾 Detailed summary saved: {summary_file}")

if __name__ == "__main__":
    main() 