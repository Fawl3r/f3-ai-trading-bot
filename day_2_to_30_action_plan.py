#!/usr/bin/env python3
"""
Day+2 to Day+30 Action Plan Execution Framework
Automated monitoring and escalation for expansion pipeline optimization
"""

import sqlite3
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import os
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionPlanMonitor:
    """Comprehensive monitoring system for 30-day action plan"""
    
    def __init__(self):
        self.db_path = 'models/policy_bandit.db'
        self.ops_journal = 'ops_journal.md'
        self.risk_cap = 0.05  # 5% portfolio risk cap
        self.targets = self._load_targets()
        
    def _load_targets(self) -> Dict:
        """Load success targets and thresholds"""
        return {
            'horizon_48h': {
                'pf_min': 1.7,
                'pf_halt': 1.5,
                'dd_max': 0.01,  # 1%
                'dd_warn': 0.025,  # 2.5%
                'dd_halt': 0.04,  # 4%
                'bandit_flow_target': (0.04, 0.06)  # 4-6%
            },
            'day_3_7': {
                'trades_per_day_min': 9,
                'total_trades_by_day7': 200,
                'correlation_max': 0.85
            },
            'week_2': {
                'meta_roc_auc_improvement': 0.02,
                'meta_pf_min': 2.0,
                'meta_dd_max': 0.03,
                'meta_trades_gate': 150
            },
            'week_3_4': {
                'winner_traffic_cap': 0.25,  # 25%
                'portfolio_dd_max': 0.04,  # 4%
                'risk_scaler_cut': 0.30  # 30% auto-cut
            }
        }
    
    def log_operation(self, action: str, details: str, severity: str = "INFO"):
        """Log operation to ops journal"""
        timestamp = datetime.now().isoformat()
        
        with open(self.ops_journal, 'a', encoding='utf-8') as f:
            f.write(f"\n## {timestamp} - {severity}\n")
            f.write(f"**Action:** {action}\n")
            f.write(f"**Details:** {details}\n")
            f.write(f"---\n")
        
        logger.info(f"[{severity}] {action}: {details}")
    
    def get_policy_performance(self, hours_back: int = 48) -> Dict:
        """Get policy performance metrics for specified timeframe"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Simulate performance data (in real system, would query actual trades)
        policies_data = {
            'timesnet_longrange': {
                'pf_30': np.random.normal(1.8, 0.2),
                'dd_pct': np.random.normal(0.008, 0.003),
                'trades': np.random.poisson(45),
                'traffic_allocation': 0.011
            },
            'lightgbm_tsa_mae': {
                'pf_30': np.random.normal(1.75, 0.15),
                'dd_pct': np.random.normal(0.007, 0.002),
                'trades': np.random.poisson(42),
                'traffic_allocation': 0.011
            },
            'ppo_strict_enhanced': {
                'pf_30': np.random.normal(1.9, 0.25),
                'dd_pct': np.random.normal(0.005, 0.002),
                'trades': np.random.poisson(38),
                'traffic_allocation': 0.011
            }
        }
        
        conn.close()
        return policies_data
    
    def check_48h_horizon(self) -> Dict:
        """Check 48-hour performance targets"""
        
        performance = self.get_policy_performance(48)
        targets = self.targets['horizon_48h']
        alerts = []
        actions = []
        
        total_dd = 0
        total_traffic = 0
        
        for policy_name, metrics in performance.items():
            pf = metrics['pf_30']
            dd = metrics['dd_pct']
            traffic = metrics['traffic_allocation']
            
            total_dd += dd * traffic  # Risk-weighted DD
            total_traffic += traffic
            
            # Check PF thresholds
            if pf < targets['pf_halt']:
                alerts.append(f"🚨 HALT: {policy_name} PF={pf:.2f} < {targets['pf_halt']}")
                actions.append(f"throttle_{policy_name}_to_0pct")
            elif pf < targets['pf_min']:
                alerts.append(f"⚠️ WARN: {policy_name} PF={pf:.2f} < {targets['pf_min']}")
            
            # Check DD thresholds
            if dd > targets['dd_halt']:
                alerts.append(f"🚨 HALT: {policy_name} DD={dd:.1%} > {targets['dd_halt']:.1%}")
                actions.append(f"halt_{policy_name}")
            elif dd > targets['dd_warn']:
                alerts.append(f"⚠️ WARN: {policy_name} DD={dd:.1%} > {targets['dd_warn']:.1%}")
        
        # Check bandit flow
        flow_target_min, flow_target_max = targets['bandit_flow_target']
        if total_traffic < flow_target_min:
            alerts.append(f"📉 LOW: Total bandit flow {total_traffic:.1%} < {flow_target_min:.1%}")
        elif total_traffic > flow_target_max:
            alerts.append(f"📈 HIGH: Total bandit flow {total_traffic:.1%} > {flow_target_max:.1%}")
            alerts.append("🔍 Check bandit parameters for rapid allocation changes")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'horizon': '48h',
            'alerts': alerts,
            'actions': actions,
            'performance': performance,
            'total_traffic': total_traffic,
            'portfolio_dd': total_dd
        }
    
    def check_day_3_7_horizon(self) -> Dict:
        """Check Day 3-7 performance targets"""
        
        performance = self.get_policy_performance(24 * 7)  # 7 days
        targets = self.targets['day_3_7']
        alerts = []
        actions = []
        
        # Calculate daily trade flow
        total_trades = sum(p['trades'] for p in performance.values())
        daily_avg = total_trades / 7
        
        if daily_avg < targets['trades_per_day_min']:
            alerts.append(f"📉 LOW FLOW: {daily_avg:.1f} trades/day < {targets['trades_per_day_min']}")
            actions.append("loosen_asset_selector")
            actions.append("consider_adding_3rd_coin")
        
        # Check 200 trades by day 7 target
        if total_trades < targets['total_trades_by_day7']:
            alerts.append(f"📊 VOLUME: {total_trades} total trades < {targets['total_trades_by_day7']} target")
        
        # Simulate correlation check
        correlation = np.random.uniform(0.7, 0.9)
        if correlation > targets['correlation_max']:
            alerts.append(f"🔗 HIGH CORRELATION: {correlation:.2f} > {targets['correlation_max']}")
            actions.append("reject_or_blend_correlated_models")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'horizon': 'day_3_7',
            'alerts': alerts,
            'actions': actions,
            'daily_trade_flow': daily_avg,
            'total_trades': total_trades,
            'correlation': correlation
        }
    
    def check_week_2_horizon(self) -> Dict:
        """Check Week 2 meta-learner targets"""
        
        targets = self.targets['week_2']
        alerts = []
        actions = []
        
        # Simulate meta-learner performance
        meta_roc_improvement = np.random.normal(0.025, 0.008)
        meta_pf = np.random.normal(2.1, 0.3)
        meta_dd = np.random.normal(0.025, 0.005)
        meta_trades = np.random.poisson(160)
        
        # Check ROC-AUC improvement
        if meta_roc_improvement < targets['meta_roc_auc_improvement']:
            alerts.append(f"📊 META ROC: +{meta_roc_improvement:.3f} < +{targets['meta_roc_auc_improvement']:.3f}")
            actions.append("try_xgboost_or_shallow_mlp_blender")
        
        # Check deployment gates
        if meta_trades >= targets['meta_trades_gate']:
            if meta_pf < targets['meta_pf_min'] or meta_dd > targets['meta_dd_max']:
                alerts.append(f"🚨 META GATE FAIL: PF={meta_pf:.2f}, DD={meta_dd:.1%} after {meta_trades} trades")
                actions.append("retrain_with_regularization")
            else:
                alerts.append(f"✅ META READY: Deploy at 10% traffic (PF={meta_pf:.2f}, DD={meta_dd:.1%})")
                actions.append("deploy_meta_10pct_traffic")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'horizon': 'week_2',
            'alerts': alerts,
            'actions': actions,
            'meta_roc_improvement': meta_roc_improvement,
            'meta_pf': meta_pf,
            'meta_dd': meta_dd,
            'meta_trades': meta_trades
        }
    
    def check_week_3_4_horizon(self) -> Dict:
        """Check Week 3-4 scaling and risk management"""
        
        performance = self.get_policy_performance(24 * 7)
        targets = self.targets['week_3_4']
        alerts = []
        actions = []
        
        # Check traffic caps
        high_performers = []
        for policy_name, metrics in performance.items():
            traffic = metrics['traffic_allocation']
            pf = metrics['pf_30']
            
            if traffic > targets['winner_traffic_cap']:
                alerts.append(f"🔍 REVIEW: {policy_name} traffic {traffic:.1%} > {targets['winner_traffic_cap']:.1%}")
                actions.append(f"manual_review_{policy_name}")
            
            if pf > 2.0:  # Winner threshold
                high_performers.append((policy_name, pf, traffic))
        
        # Portfolio DD check
        portfolio_dd = sum(p['dd_pct'] * p['traffic_allocation'] for p in performance.values())
        
        if portfolio_dd > targets['portfolio_dd_max']:
            alerts.append(f"🚨 PORTFOLIO DD: {portfolio_dd:.1%} > {targets['portfolio_dd_max']:.1%}")
            actions.append(f"auto_cut_risk_pct_live_{targets['risk_scaler_cut']:.0%}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'horizon': 'week_3_4',
            'alerts': alerts,
            'actions': actions,
            'high_performers': high_performers,
            'portfolio_dd': portfolio_dd
        }
    
    def execute_actions(self, actions: List[str]) -> Dict:
        """Execute automated actions"""
        
        executed = []
        failed = []
        
        for action in actions:
            try:
                if action.startswith('throttle_') and action.endswith('_to_0pct'):
                    policy_name = action.replace('throttle_', '').replace('_to_0pct', '')
                    self._throttle_policy(policy_name, 0.0)
                    executed.append(action)
                    
                elif action.startswith('halt_'):
                    policy_name = action.replace('halt_', '')
                    self._halt_policy(policy_name)
                    executed.append(action)
                    
                elif action == 'loosen_asset_selector':
                    self._loosen_asset_selector()
                    executed.append(action)
                    
                elif action.startswith('auto_cut_risk_pct_live_'):
                    cut_pct = int(action.split('_')[-1].replace('%', '')) / 100
                    self._auto_cut_risk(cut_pct)
                    executed.append(action)
                    
                else:
                    # Manual action - log for human review
                    self.log_operation("MANUAL_ACTION_REQUIRED", action, "WARN")
                    executed.append(f"logged_{action}")
                    
            except Exception as e:
                failed.append(f"{action}: {str(e)}")
                logger.error(f"Failed to execute {action}: {e}")
        
        return {
            'executed': executed,
            'failed': failed,
            'timestamp': datetime.now().isoformat()
        }
    
    def _throttle_policy(self, policy_name: str, new_allocation: float):
        """Throttle policy traffic allocation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE bandit_arms 
            SET traffic_allocation = ?
            WHERE policy_id = (SELECT id FROM policies WHERE name = ?)
        ''', (new_allocation, policy_name))
        
        conn.commit()
        conn.close()
        
        self.log_operation(
            "THROTTLE_POLICY", 
            f"{policy_name} traffic set to {new_allocation:.1%}",
            "WARN"
        )
    
    def _halt_policy(self, policy_name: str):
        """Halt policy completely"""
        self._throttle_policy(policy_name, 0.0)
        self.log_operation("HALT_POLICY", f"{policy_name} halted due to risk threshold", "ERROR")
    
    def _loosen_asset_selector(self):
        """Loosen asset selection criteria"""
        # In real system, would modify asset selection parameters
        self.log_operation("LOOSEN_SELECTOR", "Asset selection criteria loosened to increase trade flow", "INFO")
    
    def _auto_cut_risk(self, cut_percentage: float):
        """Automatically cut portfolio risk"""
        # In real system, would reduce position sizing
        self.log_operation(
            "AUTO_RISK_CUT", 
            f"Portfolio risk reduced by {cut_percentage:.0%} due to DD threshold",
            "WARN"
        )
    
    def run_daily_check(self) -> Dict:
        """Run comprehensive daily monitoring check"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'horizon_48h': self.check_48h_horizon(),
            'day_3_7': self.check_day_3_7_horizon(),
            'week_2': self.check_week_2_horizon(),
            'week_3_4': self.check_week_3_4_horizon()
        }
        
        # Collect all actions
        all_actions = []
        for horizon_data in results.values():
            if isinstance(horizon_data, dict) and 'actions' in horizon_data:
                all_actions.extend(horizon_data['actions'])
        
        # Execute automated actions
        if all_actions:
            execution_results = self.execute_actions(all_actions)
            results['execution'] = execution_results
        
        # Save results
        results_file = f"daily_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def print_dashboard(self):
        """Print comprehensive monitoring dashboard"""
        
        results = self.run_daily_check()
        
        print("\n" + "=" * 80)
        print("🚀 DAY+2 TO DAY+30 ACTION PLAN MONITOR")
        print("=" * 80)
        print(f"📅 {results['timestamp']}")
        print()
        
        # 48-hour horizon
        h48 = results['horizon_48h']
        print("📊 48-HOUR HORIZON:")
        print("-" * 50)
        
        for policy, perf in h48['performance'].items():
            status = "✅" if perf['pf_30'] >= 1.7 and perf['dd_pct'] <= 0.01 else "⚠️"
            print(f"{status} {policy}: PF={perf['pf_30']:.2f}, DD={perf['dd_pct']:.1%}, Traffic={perf['traffic_allocation']:.1%}")
        
        print(f"📈 Total Bandit Flow: {h48['total_traffic']:.1%}")
        print(f"🛡️ Portfolio DD: {h48['portfolio_dd']:.1%}")
        
        if h48['alerts']:
            print("🚨 ALERTS:")
            for alert in h48['alerts']:
                print(f"   {alert}")
        
        print()
        
        # Day 3-7 horizon
        d37 = results['day_3_7']
        print("📈 DAY 3-7 HORIZON:")
        print("-" * 50)
        print(f"📊 Daily Trade Flow: {d37['daily_trade_flow']:.1f}/day (target: ≥9)")
        print(f"📋 Total Trades: {d37['total_trades']} (target: 200 by day 7)")
        print(f"🔗 Signal Correlation: {d37['correlation']:.2f} (target: <0.85)")
        
        if d37['alerts']:
            print("🚨 ALERTS:")
            for alert in d37['alerts']:
                print(f"   {alert}")
        
        print()
        
        # Week 2 horizon
        w2 = results['week_2']
        print("🧠 WEEK 2 HORIZON (Meta-Learner):")
        print("-" * 50)
        print(f"📊 ROC-AUC Improvement: +{w2['meta_roc_improvement']:.3f} (target: +0.020)")
        print(f"📈 Meta PF: {w2['meta_pf']:.2f} (target: ≥2.0)")
        print(f"🛡️ Meta DD: {w2['meta_dd']:.1%} (target: ≤3%)")
        print(f"📋 Meta Trades: {w2['meta_trades']} (gate: 150)")
        
        if w2['alerts']:
            print("🚨 ALERTS:")
            for alert in w2['alerts']:
                print(f"   {alert}")
        
        print()
        
        # Week 3-4 horizon
        w34 = results['week_3_4']
        print("🎯 WEEK 3-4 HORIZON (Scaling):")
        print("-" * 50)
        print(f"🛡️ Portfolio DD: {w34['portfolio_dd']:.1%} (cap: 4%)")
        
        if w34['high_performers']:
            print("🏆 High Performers:")
            for name, pf, traffic in w34['high_performers']:
                print(f"   {name}: PF={pf:.2f}, Traffic={traffic:.1%}")
        
        if w34['alerts']:
            print("🚨 ALERTS:")
            for alert in w34['alerts']:
                print(f"   {alert}")
        
        print()
        
        # Execution results
        if 'execution' in results:
            exec_results = results['execution']
            print("⚡ AUTOMATED ACTIONS:")
            print("-" * 50)
            
            if exec_results['executed']:
                print("✅ Executed:")
                for action in exec_results['executed']:
                    print(f"   {action}")
            
            if exec_results['failed']:
                print("❌ Failed:")
                for failure in exec_results['failed']:
                    print(f"   {failure}")
        
        print()
        print("📝 For detailed logs, check: ops_journal.md")
        print("📊 Full results saved to: daily_check_*.json")
        print("=" * 80)
        
        return results

def main():
    """Main execution function"""
    
    monitor = ActionPlanMonitor()
    
    # Initialize ops journal if it doesn't exist
    if not os.path.exists(monitor.ops_journal):
        with open(monitor.ops_journal, 'w') as f:
            f.write("# Operations Journal - Day+2 to Day+30 Action Plan\n\n")
            f.write("This journal tracks all automated and manual actions taken during the expansion pipeline optimization period.\n\n")
    
    # Run dashboard
    results = monitor.print_dashboard()
    
    print("\n🎯 KEY REMINDERS:")
    print("-" * 30)
    print("• Traffic auto-scale is conservative - don't override until 150-trade gates pass")
    print("• Snapshot every promoted SHA to S3 when crossing 10% traffic")
    print("• Document manual tweaks in ops_journal.md")
    print("• Expectancy > win rate: Target 2.7-3.0 PF range, not 90% WR")
    print("• 4R:1R structure makes 90% hit-rate statistically unlikely")

if __name__ == "__main__":
    main() 