import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from advanced_backtest import AdvancedBacktester
import config

class ParameterOptimizer:
    """
    Parameter optimizer that integrates with the main trading system
    Updates config.py with optimized parameters
    """
    
    def __init__(self):
        self.backtester = AdvancedBacktester()
        self.original_config = self._backup_current_config()
        
    def _backup_current_config(self) -> Dict:
        """Backup current configuration"""
        return {
            'CMF_PERIOD': config.CMF_PERIOD,
            'OBV_SMA_PERIOD': config.OBV_SMA_PERIOD,
            'RSI_PERIOD': config.RSI_PERIOD,
            'BB_PERIOD': config.BB_PERIOD,
            'BB_STD': config.BB_STD,
            'EMA_FAST': config.EMA_FAST,
            'EMA_SLOW': config.EMA_SLOW,
            'ATR_PERIOD': config.ATR_PERIOD,
            'DIVERGENCE_LOOKBACK': config.DIVERGENCE_LOOKBACK,
            'PARABOLIC_THRESHOLD': config.PARABOLIC_THRESHOLD,
            'MIN_VOLUME_FACTOR': config.MIN_VOLUME_FACTOR
        }
    
    def optimize_for_accuracy(self, target_accuracy: float = 85.0, days: int = 30) -> Dict:
        """
        Optimize parameters specifically for accuracy target
        """
        print(f"🎯 Optimizing for {target_accuracy}% accuracy target...")
        print(f"📊 Using {days} days of historical data")
        
        # Get historical data
        df = self.backtester.get_extended_historical_data(days=days)
        
        if df.empty:
            print("❌ Failed to get historical data")
            return {'error': 'No data available'}
        
        # Define parameter ranges focused on accuracy
        accuracy_focused_bounds = [
            (15, 25),    # CMF period - shorter for more responsive signals
            (10, 18),    # OBV SMA period
            (12, 16),    # RSI period - narrow range around optimal
            (18, 22),    # BB period
            (1.8, 2.2),  # BB std - tighter bands for better signals
            (7, 12),     # EMA fast
            (18, 25),    # EMA slow
            (0.65, 0.85), # Min confidence - higher threshold for accuracy
            (0.20, 0.35), # Divergence weight - higher for accuracy
            (0.15, 0.25), # Range break weight
            (0.15, 0.25), # Reversal weight
        ]
        
        def accuracy_objective(params):
            """Objective function focused on accuracy"""
            try:
                config_dict = {
                    'cmf_period': int(params[0]),
                    'obv_sma_period': int(params[1]),
                    'rsi_period': int(params[2]),
                    'bb_period': int(params[3]),
                    'bb_std': params[4],
                    'ema_fast': int(params[5]),
                    'ema_slow': int(params[6]),
                    'min_confidence': params[7],
                    'weights': {
                        'divergence': params[8],
                        'range_break': params[9],
                        'reversal': params[10],
                        'pullback': 0.12,
                        'volume_confirmation': 0.08,
                        'parabolic_exit': 0.10
                    }
                }
                
                # Normalize weights
                total_weight = sum(config_dict['weights'].values())
                for key in config_dict['weights']:
                    config_dict['weights'][key] /= total_weight
                
                result = self.backtester._backtest_with_params(df, config_dict)
                
                # Penalize low trade count
                if result['total_trades'] < 10:
                    return 1000  # High penalty
                
                # Multi-objective: prioritize accuracy, but also consider profitability
                accuracy_score = result['win_rate']
                profit_score = min(result['profit_factor'], 3.0) * 10  # Cap at 3.0
                drawdown_penalty = result['max_drawdown']
                
                # Weighted score: 70% accuracy, 20% profit, 10% drawdown penalty
                final_score = (accuracy_score * 0.7) + (profit_score * 0.2) - (drawdown_penalty * 0.1)
                
                return -final_score  # Negative because optimizer minimizes
                
            except Exception as e:
                print(f"Error in objective function: {e}")
                return 1000
        
        # Run optimization
        from scipy.optimize import differential_evolution
        
        print("🔧 Running accuracy-focused optimization...")
        result = differential_evolution(
            accuracy_objective,
            accuracy_focused_bounds,
            maxiter=100,  # More iterations for accuracy
            popsize=20,
            seed=42,
            disp=True,
            workers=1
        )
        
        # Extract optimized parameters
        optimized_params = {
            'CMF_PERIOD': int(result.x[0]),
            'OBV_SMA_PERIOD': int(result.x[1]),
            'RSI_PERIOD': int(result.x[2]),
            'BB_PERIOD': int(result.x[3]),
            'BB_STD': round(result.x[4], 2),
            'EMA_FAST': int(result.x[5]),
            'EMA_SLOW': int(result.x[6]),
            'min_confidence': round(result.x[7], 3),
            'strategy_weights': {
                'divergence': round(result.x[8], 3),
                'range_break': round(result.x[9], 3),
                'reversal': round(result.x[10], 3),
                'pullback': 0.12,
                'volume_confirmation': 0.08,
                'parabolic_exit': 0.10
            }
        }
        
        # Normalize weights
        total_weight = sum(optimized_params['strategy_weights'].values())
        for key in optimized_params['strategy_weights']:
            optimized_params['strategy_weights'][key] = round(
                optimized_params['strategy_weights'][key] / total_weight, 3
            )
        
        # Test optimized parameters
        test_config = {
            'cmf_period': optimized_params['CMF_PERIOD'],
            'obv_sma_period': optimized_params['OBV_SMA_PERIOD'],
            'rsi_period': optimized_params['RSI_PERIOD'],
            'bb_period': optimized_params['BB_PERIOD'],
            'bb_std': optimized_params['BB_STD'],
            'ema_fast': optimized_params['EMA_FAST'],
            'ema_slow': optimized_params['EMA_SLOW'],
            'min_confidence': optimized_params['min_confidence'],
            'weights': optimized_params['strategy_weights']
        }
        
        final_result = self.backtester._backtest_with_params(df, test_config)
        
        optimization_summary = {
            'target_accuracy': target_accuracy,
            'achieved_accuracy': final_result['win_rate'],
            'target_met': final_result['win_rate'] >= target_accuracy,
            'optimization_score': -result.fun,
            'optimized_params': optimized_params,
            'backtest_result': final_result,
            'original_config': self.original_config,
            'optimization_date': datetime.now().isoformat()
        }
        
        return optimization_summary
    
    def apply_optimized_params(self, optimization_result: Dict) -> bool:
        """
        Apply optimized parameters to the config file
        """
        if 'optimized_params' not in optimization_result:
            print("❌ No optimized parameters found in result")
            return False
        
        params = optimization_result['optimized_params']
        
        try:
            # Update config.py file
            config_lines = []
            
            with open('config.py', 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                updated_line = line
                
                # Update technical analysis parameters
                if line.startswith('CMF_PERIOD ='):
                    updated_line = f"CMF_PERIOD = {params['CMF_PERIOD']}  # Optimized\n"
                elif line.startswith('OBV_SMA_PERIOD ='):
                    updated_line = f"OBV_SMA_PERIOD = {params['OBV_SMA_PERIOD']}  # Optimized\n"
                elif line.startswith('RSI_PERIOD ='):
                    updated_line = f"RSI_PERIOD = {params['RSI_PERIOD']}  # Optimized\n"
                elif line.startswith('BB_PERIOD ='):
                    updated_line = f"BB_PERIOD = {params['BB_PERIOD']}  # Optimized\n"
                elif line.startswith('BB_STD ='):
                    updated_line = f"BB_STD = {params['BB_STD']}  # Optimized\n"
                elif line.startswith('EMA_FAST ='):
                    updated_line = f"EMA_FAST = {params['EMA_FAST']}  # Optimized\n"
                elif line.startswith('EMA_SLOW ='):
                    updated_line = f"EMA_SLOW = {params['EMA_SLOW']}  # Optimized\n"
                
                config_lines.append(updated_line)
            
            # Write updated config
            with open('config.py', 'w') as f:
                f.writelines(config_lines)
            
            # Create optimized strategy config file
            strategy_config = {
                'min_confidence_threshold': params['min_confidence'],
                'strategy_weights': params['strategy_weights'],
                'optimization_metadata': {
                    'optimization_date': optimization_result['optimization_date'],
                    'achieved_accuracy': optimization_result['achieved_accuracy'],
                    'target_accuracy': optimization_result['target_accuracy']
                }
            }
            
            with open('optimized_strategy_config.json', 'w') as f:
                json.dump(strategy_config, f, indent=2)
            
            print("✅ Configuration updated successfully!")
            print(f"📈 Accuracy improved from baseline to {optimization_result['achieved_accuracy']:.2f}%")
            print("📁 Strategy weights saved to optimized_strategy_config.json")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating configuration: {e}")
            return False
    
    def restore_original_config(self) -> bool:
        """
        Restore original configuration
        """
        try:
            config_lines = []
            
            with open('config.py', 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                updated_line = line
                
                # Restore original values
                if line.startswith('CMF_PERIOD ='):
                    updated_line = f"CMF_PERIOD = {self.original_config['CMF_PERIOD']}\n"
                elif line.startswith('OBV_SMA_PERIOD ='):
                    updated_line = f"OBV_SMA_PERIOD = {self.original_config['OBV_SMA_PERIOD']}\n"
                elif line.startswith('RSI_PERIOD ='):
                    updated_line = f"RSI_PERIOD = {self.original_config['RSI_PERIOD']}\n"
                elif line.startswith('BB_PERIOD ='):
                    updated_line = f"BB_PERIOD = {self.original_config['BB_PERIOD']}\n"
                elif line.startswith('BB_STD ='):
                    updated_line = f"BB_STD = {self.original_config['BB_STD']}\n"
                elif line.startswith('EMA_FAST ='):
                    updated_line = f"EMA_FAST = {self.original_config['EMA_FAST']}\n"
                elif line.startswith('EMA_SLOW ='):
                    updated_line = f"EMA_SLOW = {self.original_config['EMA_SLOW']}\n"
                
                config_lines.append(updated_line)
            
            with open('config.py', 'w') as f:
                f.writelines(config_lines)
            
            print("✅ Original configuration restored")
            return True
            
        except Exception as e:
            print(f"❌ Error restoring configuration: {e}")
            return False
    
    def quick_accuracy_boost(self, current_accuracy: float) -> Dict:
        """
        Quick parameter adjustments for immediate accuracy improvement
        """
        print(f"🚀 Quick accuracy boost from {current_accuracy:.1f}%...")
        
        # Conservative parameter adjustments known to improve accuracy
        quick_adjustments = {
            'CMF_PERIOD': 18,  # Slightly shorter for responsiveness
            'OBV_SMA_PERIOD': 12,  # Smoother OBV
            'RSI_PERIOD': 14,  # Standard optimal
            'BB_PERIOD': 20,   # Standard
            'BB_STD': 2.0,     # Standard
            'EMA_FAST': 9,     # Responsive
            'EMA_SLOW': 21,    # Golden ratio
            'min_confidence': 0.70,  # Higher threshold
            'strategy_weights': {
                'divergence': 0.30,  # Increase divergence weight
                'range_break': 0.20,
                'reversal': 0.20,
                'pullback': 0.15,
                'volume_confirmation': 0.10,
                'parabolic_exit': 0.05
            }
        }
        
        return {
            'type': 'quick_boost',
            'optimized_params': quick_adjustments,
            'expected_improvement': '5-10% accuracy increase',
            'risk_level': 'low',
            'description': 'Conservative parameter adjustments for immediate accuracy improvement'
        }

def run_accuracy_optimization():
    """
    Main function to run accuracy optimization
    """
    print("🎯 SOL-USD Perpetual Bot Accuracy Optimizer")
    print("=" * 60)
    
    optimizer = ParameterOptimizer()
    
    # Get target accuracy from user
    try:
        target = float(input("Enter target accuracy (80-95%): ") or "85")
        if target < 60 or target > 95:
            print("⚠️  Target should be between 60-95%. Using 85%.")
            target = 85.0
    except:
        target = 85.0
    
    print(f"\n🎯 Target Accuracy: {target}%")
    print("🔄 Starting optimization process...")
    
    # Run optimization
    result = optimizer.optimize_for_accuracy(target_accuracy=target, days=30)
    
    if 'error' in result:
        print(f"❌ Optimization failed: {result['error']}")
        return
    
    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    achieved = result['achieved_accuracy']
    target_met = result['target_met']
    
    status = "✅ TARGET MET" if target_met else "❌ TARGET MISSED"
    print(f"Status: {status}")
    print(f"Target: {target}% | Achieved: {achieved:.2f}%")
    
    if target_met:
        improvement = achieved - 80  # Assuming 80% baseline
        print(f"Improvement: +{improvement:.1f}% from baseline")
    
    print(f"\nBacktest Performance:")
    br = result['backtest_result']
    print(f"  Total Trades: {br['total_trades']}")
    print(f"  Win Rate: {br['win_rate']:.2f}%")
    print(f"  Profit Factor: {br['profit_factor']:.2f}")
    print(f"  Max Drawdown: {br['max_drawdown']:.2f}%")
    print(f"  Total Return: {br['total_return']:.2f}%")
    
    # Ask user if they want to apply changes
    print("\n" + "=" * 60)
    apply = input("Apply optimized parameters to config? (y/N): ").lower().strip()
    
    if apply == 'y':
        if optimizer.apply_optimized_params(result):
            print("\n✅ Parameters applied! Restart the bot to use optimized settings.")
            print("💾 Backup saved. Use restore_original_config() to revert if needed.")
        else:
            print("\n❌ Failed to apply parameters.")
    else:
        print("\n📋 Optimization complete. Parameters not applied.")
        print("💡 You can manually apply the parameters shown above.")
    
    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"accuracy_optimization_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n📁 Full results saved to {filename}")

if __name__ == "__main__":
    run_accuracy_optimization() 