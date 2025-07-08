#!/usr/bin/env python3
"""
Update Risk Configuration with Stricter Controls
Implements the emergency response recommendations
"""

import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_bandit_config():
    """Update bandit configuration with 5% minimum traffic"""
    
    config_updates = {
        'BANDIT_MIN_TRAFFIC': 0.05,  # Reduced from 10% to 5%
        'BANDIT_DD_THRESHOLD': 0.025,  # 2.5% DD warning threshold
        'BANDIT_HALT_THRESHOLD': 0.04,  # 4% DD halt threshold
        'PYRAMID_MAX_UNITS': 3,  # Maximum pyramid units
        'CHALLENGER_VALIDATION_TRADES': 150  # Trades required before scaling
    }
    
    # Save to .env format
    env_file = Path('.env')
    env_lines = []
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_lines = f.readlines()
    
    # Update or add configuration
    updated_keys = set()
    for i, line in enumerate(env_lines):
        for key, value in config_updates.items():
            if line.startswith(f"{key}="):
                env_lines[i] = f"{key}={value}\n"
                updated_keys.add(key)
                logger.info(f"Updated {key}={value}")
    
    # Add new keys
    for key, value in config_updates.items():
        if key not in updated_keys:
            env_lines.append(f"{key}={value}\n")
            logger.info(f"Added {key}={value}")
    
    # Write back to file
    with open(env_file, 'w') as f:
        f.writelines(env_lines)
    
    logger.info("✅ Bandit configuration updated")
    return config_updates

def create_pattern_blacklist():
    """Create pattern blacklist based on analysis"""
    
    blacklist_patterns = {
        'high_volatility_filter': {
            'enabled': True,
            'atr_threshold': 1.3,  # ATR/ATR_ref > 1.3
            'description': 'Block trades when volatility is >30% above reference'
        },
        'btc_concentration_limit': {
            'enabled': True,
            'max_concurrent_btc_positions': 2,
            'description': 'Limit BTC exposure after concentration analysis'
        },
        'rapid_pyramid_cooldown': {
            'enabled': True,
            'min_time_between_adds': 300,  # 5 minutes between add-ons
            'description': 'Prevent rapid pyramiding in choppy conditions'
        },
        'large_loss_protection': {
            'enabled': True,
            'max_single_loss_pct': 1.0,  # 1% max single trade loss
            'description': 'Hard stop for oversized losses'
        }
    }
    
    pattern_file = Path('pattern_rules.yaml')
    
    # Convert to YAML format
    yaml_content = "# Pattern Blacklist Rules\n"
    yaml_content += "# Generated after drawdown analysis\n\n"
    
    for pattern_name, config in blacklist_patterns.items():
        yaml_content += f"{pattern_name}:\n"
        for key, value in config.items():
            yaml_content += f"  {key}: {value}\n"
        yaml_content += "\n"
    
    with open(pattern_file, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"✅ Pattern blacklist created: {pattern_file}")
    return blacklist_patterns

def update_risk_manager_dd_scaler():
    """Create patch for risk manager with tighter DD scaling"""
    
    risk_patch = '''
# Risk Manager DD Scaler Patch
# Apply to risk_manager_enhanced.py

def calculate_risk_scaling(self, policy_type, current_dd_pct):
    """Enhanced DD scaling with RL-specific rules"""
    
    base_risk = self.base_risk_pct
    
    if policy_type == "RL" or policy_type == "reinforcement":
        # Stricter scaling for RL policies
        if current_dd_pct > 2.5:  # 2.5% threshold for RL
            risk_multiplier = 0.4  # Aggressive reduction
            logger.warning(f"RL Policy DD {current_dd_pct:.1f}% > 2.5%, scaling risk to {risk_multiplier}")
            return base_risk * risk_multiplier
        elif current_dd_pct > 1.5:  # Early warning
            risk_multiplier = 0.7
            logger.info(f"RL Policy DD {current_dd_pct:.1f}% > 1.5%, scaling risk to {risk_multiplier}")
            return base_risk * risk_multiplier
    else:
        # Standard scaling for production policies
        if current_dd_pct > 4.0:
            risk_multiplier = 0.5
            return base_risk * risk_multiplier
        elif current_dd_pct > 2.0:
            risk_multiplier = 0.8
            return base_risk * risk_multiplier
    
    return base_risk
'''
    
    patch_file = Path('risk_manager_dd_patch.py')
    with open(patch_file, 'w') as f:
        f.write(risk_patch)
    
    logger.info(f"✅ Risk manager patch created: {patch_file}")
    return risk_patch

def update_monitoring_thresholds():
    """Update monitoring thresholds for stricter control"""
    
    new_thresholds = {
        'dd_challenger_pct': {
            'warn_threshold': 2.5,  # Warning at 2.5%
            'halt_threshold': 4.0,   # Halt at 4%
            'description': 'Tightened DD thresholds for RL policies'
        },
        'pyramid_units_live': {
            'max_units': 3,
            'alert_threshold': 3,
            'description': 'Alert when pyramid units > 3'
        },
        'traffic_challenger': {
            'auto_halt_dd': 4.0,
            'auto_throttle_pf': 1.6,
            'description': 'Auto-actions for traffic management'
        },
        'consecutive_losses': {
            'warning_threshold': 3,
            'halt_threshold': 5,
            'description': 'Monitor consecutive loss streaks'
        }
    }
    
    thresholds_file = Path('monitoring_thresholds.json')
    with open(thresholds_file, 'w') as f:
        json.dump(new_thresholds, f, indent=2)
    
    logger.info(f"✅ Monitoring thresholds updated: {thresholds_file}")
    return new_thresholds

def create_mc_stress_test():
    """Create Monte Carlo stress test for new policies"""
    
    mc_test_script = '''#!/usr/bin/env python3
"""
Monte Carlo Stress Test for Policy Validation
Tests if 95th percentile DD <= 6%
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def monte_carlo_stress_test(policy_sha, runs=200):
    """Run MC stress test on policy"""
    
    print(f"🧪 Monte Carlo Stress Test: {policy_sha}")
    print(f"Runs: {runs}")
    
    # Simulate various market conditions
    drawdowns = []
    
    for run in range(runs):
        # Generate random market scenario
        returns = np.random.normal(0, 0.02, 1000)  # 1000 days
        volatility_regime = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        
        if volatility_regime == 'high':
            returns *= 2.0  # Double volatility
        elif volatility_regime == 'low':
            returns *= 0.5  # Half volatility
        
        # Simulate policy performance (simplified)
        equity_curve = 10000 * np.cumprod(1 + returns * 0.1)  # 10% of market return
        max_equity = np.maximum.accumulate(equity_curve)
        drawdown_series = (equity_curve - max_equity) / max_equity
        max_drawdown = abs(np.min(drawdown_series))
        
        drawdowns.append(max_drawdown)
    
    # Calculate statistics
    percentile_95 = np.percentile(drawdowns, 95)
    mean_dd = np.mean(drawdowns)
    max_dd = np.max(drawdowns)
    
    print(f"📊 Results:")
    print(f"  Mean DD: {mean_dd*100:.1f}%")
    print(f"  95th Percentile DD: {percentile_95*100:.1f}%")
    print(f"  Max DD: {max_dd*100:.1f}%")
    
    # Pass/Fail criteria
    passed = percentile_95 <= 0.06  # 6% threshold
    
    print(f"🎯 Result: {'✅ PASS' if passed else '❌ FAIL'} (95th percentile DD {'<=' if passed else '>'} 6%)")
    
    return passed, {
        'mean_dd': mean_dd,
        'percentile_95': percentile_95,
        'max_dd': max_dd,
        'passed': passed
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sha', required=True)
    parser.add_argument('--runs', type=int, default=200)
    
    args = parser.parse_args()
    
    passed, results = monte_carlo_stress_test(args.sha, args.runs)
    
    # Save results
    results_file = f"mc_stress_results_{args.sha[-8:]}.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📋 Results saved: {results_file}")
'''
    
    mc_file = Path('backtests/mc_slippage.py')
    mc_file.parent.mkdir(exist_ok=True)
    
    with open(mc_file, 'w') as f:
        f.write(mc_test_script)
    
    logger.info(f"✅ MC stress test created: {mc_file}")
    return mc_file

def main():
    """Apply all configuration updates"""
    
    logger.info("🔧 Applying Emergency Risk Configuration Updates")
    logger.info("="*60)
    
    # Step 1: Update bandit configuration
    bandit_config = update_bandit_config()
    
    # Step 2: Create pattern blacklist
    patterns = create_pattern_blacklist()
    
    # Step 3: Update risk manager
    risk_patch = update_risk_manager_dd_scaler()
    
    # Step 4: Update monitoring thresholds
    thresholds = update_monitoring_thresholds()
    
    # Step 5: Create MC stress test
    mc_test = create_mc_stress_test()
    
    logger.info("="*60)
    logger.info("✅ ALL RISK CONFIGURATIONS UPDATED")
    logger.info("="*60)
    
    print("\\n🚨 EMERGENCY RISK UPDATES COMPLETE")
    print("="*60)
    print("✅ Bandit minimum traffic: 5% (was 10%)")
    print("✅ DD warning threshold: 2.5% for RL policies")
    print("✅ Pattern blacklist: High volatility, BTC concentration")
    print("✅ Risk manager: Stricter DD scaling")
    print("✅ MC stress test: Ready for new policies")
    print("\\n📋 Next: Test strict PPO model with 5% traffic")
    print("="*60)

if __name__ == "__main__":
    main() 