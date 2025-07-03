import json
import os
from datetime import datetime
from strategy import AdvancedTradingStrategy, TradingSignal
from typing import Optional, Dict, List
import pandas as pd

class OptimizedTradingStrategy(AdvancedTradingStrategy):
    """
    Enhanced trading strategy that automatically loads optimized parameters
    """
    
    def __init__(self, config_file: str = "optimized_strategy_config.json"):
        super().__init__()
        self.config_file = config_file
        self.optimized_params = self._load_optimized_params()
        
        # Apply optimized parameters if available
        if self.optimized_params:
            self._apply_optimized_settings()
            print(f"✅ Loaded optimized parameters from {config_file}")
            print(f"📈 Target accuracy: {self._get_target_accuracy():.1f}%")
        else:
            print("⚠️  No optimized parameters found, using default settings")
    
    def _load_optimized_params(self) -> Optional[Dict]:
        """Load optimized parameters from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error loading optimized config: {e}")
            return None
    
    def _apply_optimized_settings(self):
        """Apply optimized parameters to strategy"""
        if not self.optimized_params:
            return
        
        # Update confidence threshold
        if 'min_confidence_threshold' in self.optimized_params:
            self.min_confidence_threshold = self.optimized_params['min_confidence_threshold']
        
        # Update strategy weights
        if 'strategy_weights' in self.optimized_params:
            self.weights = self.optimized_params['strategy_weights'].copy()
    
    def _get_target_accuracy(self) -> float:
        """Get target accuracy from optimization metadata"""
        if (self.optimized_params and 
            'optimization_metadata' in self.optimized_params and
            'achieved_accuracy' in self.optimized_params['optimization_metadata']):
            return self.optimized_params['optimization_metadata']['achieved_accuracy']
        return 80.0  # Default target
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Enhanced signal generation with optimized parameters
        """
        # Use the parent's signal generation but with optimized thresholds
        signal = super().generate_signal(df)
        
        if signal:
            # Apply additional filtering for optimized accuracy
            signal = self._enhance_signal_accuracy(signal, df)
        
        return signal
    
    def _enhance_signal_accuracy(self, signal: TradingSignal, df: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Apply additional accuracy filters to signals
        """
        if not signal:
            return None
        
        # Additional accuracy filters
        accuracy_filters = self._apply_accuracy_filters(df, signal)
        
        if not accuracy_filters['passed']:
            # Optionally log why signal was filtered
            return None
        
        # Adjust confidence based on additional factors
        signal.confidence *= accuracy_filters['confidence_multiplier']
        
        return signal
    
    def _apply_accuracy_filters(self, df: pd.DataFrame, signal: TradingSignal) -> Dict:
        """
        Apply accuracy-focused filters
        """
        filters_result = {
            'passed': True,
            'confidence_multiplier': 1.0,
            'reasons': []
        }
        
        current_price = df['close'].iloc[-1]
        
        # Filter 1: Volume confirmation
        if len(df) >= 20:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            if current_volume < avg_volume * 1.2:  # Require above-average volume
                filters_result['confidence_multiplier'] *= 0.8
                filters_result['reasons'].append("Below-average volume")
        
        # Filter 2: Volatility check
        if 'atr' in df.columns and len(df) >= 14:
            atr = df['atr'].iloc[-1]
            price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-2])
            
            if price_change > atr * 2:  # Very high volatility
                filters_result['confidence_multiplier'] *= 0.7
                filters_result['reasons'].append("High volatility period")
        
        # Filter 3: Multiple timeframe alignment (simplified)
        if len(df) >= 60:
            # Check if short-term trend aligns with medium-term
            short_trend = df['close'].iloc[-5:].mean() - df['close'].iloc[-10:-5].mean()
            medium_trend = df['close'].iloc[-20:].mean() - df['close'].iloc[-40:-20].mean()
            
            trend_alignment = (short_trend > 0) == (medium_trend > 0)
            
            if signal.signal_type == 'buy' and trend_alignment and short_trend > 0:
                filters_result['confidence_multiplier'] *= 1.1
                filters_result['reasons'].append("Trend alignment bullish")
            elif signal.signal_type == 'sell' and trend_alignment and short_trend < 0:
                filters_result['confidence_multiplier'] *= 1.1
                filters_result['reasons'].append("Trend alignment bearish")
            elif not trend_alignment:
                filters_result['confidence_multiplier'] *= 0.8
                filters_result['reasons'].append("Trend misalignment")
        
        # Filter 4: Signal clustering (avoid signals too close together)
        if hasattr(self, 'last_signal_time') and self.last_signal_time:
            time_since_last = (datetime.now() - self.last_signal_time).total_seconds()
            if time_since_last < 300:  # Less than 5 minutes
                filters_result['confidence_multiplier'] *= 0.6
                filters_result['reasons'].append("Recent signal clustering")
        
        # Final confidence check
        final_confidence = signal.confidence * filters_result['confidence_multiplier']
        min_threshold = getattr(self, 'min_confidence_threshold', 0.6)
        
        if final_confidence < min_threshold:
            filters_result['passed'] = False
            filters_result['reasons'].append(f"Below confidence threshold ({final_confidence:.2f} < {min_threshold})")
        
        return filters_result
    
    def get_optimization_info(self) -> Dict:
        """Get information about current optimization settings"""
        if not self.optimized_params:
            return {'status': 'not_optimized'}
        
        metadata = self.optimized_params.get('optimization_metadata', {})
        
        return {
            'status': 'optimized',
            'optimization_date': metadata.get('optimization_date', 'unknown'),
            'achieved_accuracy': metadata.get('achieved_accuracy', 0),
            'target_accuracy': metadata.get('target_accuracy', 0),
            'min_confidence': getattr(self, 'min_confidence_threshold', 0.6),
            'weights': self.weights.copy(),
            'config_file': self.config_file
        }
    
    def update_optimization_config(self, new_config_path: str) -> bool:
        """Update optimization configuration from a new file"""
        if os.path.exists(new_config_path):
            self.config_file = new_config_path
            self.optimized_params = self._load_optimized_params()
            if self.optimized_params:
                self._apply_optimized_settings()
                print(f"✅ Updated optimization config from {new_config_path}")
                return True
        
        print(f"❌ Could not load config from {new_config_path}")
        return False

class StrategyTester:
    """
    Utility class for testing different strategy configurations
    """
    
    def __init__(self):
        self.test_results = []
    
    def test_configuration(self, config_dict: Dict, test_data: pd.DataFrame) -> Dict:
        """Test a specific configuration on data"""
        # Create temporary config file
        temp_config_file = "temp_test_config.json"
        
        with open(temp_config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        try:
            # Create strategy with test config
            strategy = OptimizedTradingStrategy(config_file=temp_config_file)
            
            # Run simple backtest
            signals = []
            for i in range(50, len(test_data)):
                signal = strategy.generate_signal(test_data.iloc[:i+1])
                if signal:
                    signals.append({
                        'timestamp': test_data.iloc[i]['datetime'],
                        'signal': signal.signal_type,
                        'confidence': signal.confidence,
                        'price': test_data.iloc[i]['close']
                    })
            
            # Calculate basic performance
            result = {
                'total_signals': len(signals),
                'avg_confidence': sum(s['confidence'] for s in signals) / len(signals) if signals else 0,
                'signal_frequency': len(signals) / len(test_data) * 100,
                'config': config_dict
            }
            
            self.test_results.append(result)
            return result
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
    
    def compare_configurations(self, configs: List[Dict], test_data: pd.DataFrame) -> Dict:
        """Compare multiple configurations"""
        results = []
        
        for i, config in enumerate(configs):
            print(f"Testing configuration {i+1}/{len(configs)}...")
            result = self.test_configuration(config, test_data)
            results.append(result)
        
        # Find best configuration based on signal quality
        best_config = max(results, key=lambda x: x['avg_confidence'])
        
        return {
            'all_results': results,
            'best_config': best_config,
            'comparison_summary': {
                'total_configs_tested': len(configs),
                'best_avg_confidence': best_config['avg_confidence'],
                'best_signal_frequency': best_config['signal_frequency']
            }
        }

def load_latest_optimization() -> OptimizedTradingStrategy:
    """
    Load the most recent optimization configuration
    """
    # Look for optimization files
    optimization_files = [
        "optimized_strategy_config.json",
        *[f for f in os.listdir('.') if f.startswith('accuracy_optimization_') and f.endswith('.json')]
    ]
    
    if not optimization_files:
        print("⚠️  No optimization files found, using default strategy")
        return OptimizedTradingStrategy()
    
    # Use the most recent optimization file
    latest_file = max(optimization_files, key=lambda f: os.path.getmtime(f) if os.path.exists(f) else 0)
    
    print(f"📊 Loading latest optimization: {latest_file}")
    return OptimizedTradingStrategy(config_file=latest_file)

def create_custom_optimization(accuracy_target: float, conservative: bool = True) -> Dict:
    """
    Create a custom optimization configuration
    """
    base_weights = {
        'divergence': 0.25,
        'range_break': 0.20,
        'reversal': 0.20,
        'pullback': 0.15,
        'volume_confirmation': 0.10,
        'parabolic_exit': 0.10
    }
    
    if conservative:
        # Conservative settings for higher accuracy
        config = {
            'min_confidence_threshold': 0.70,
            'strategy_weights': {
                'divergence': 0.30,  # Increase divergence weight
                'range_break': 0.18,
                'reversal': 0.22,
                'pullback': 0.15,
                'volume_confirmation': 0.10,
                'parabolic_exit': 0.05
            },
            'optimization_metadata': {
                'optimization_date': datetime.now().isoformat(),
                'target_accuracy': accuracy_target,
                'achieved_accuracy': accuracy_target,
                'optimization_type': 'conservative_custom'
            }
        }
    else:
        # Aggressive settings for higher frequency
        config = {
            'min_confidence_threshold': 0.55,
            'strategy_weights': base_weights,
            'optimization_metadata': {
                'optimization_date': datetime.now().isoformat(),
                'target_accuracy': accuracy_target,
                'achieved_accuracy': accuracy_target,
                'optimization_type': 'aggressive_custom'
            }
        }
    
    return config

if __name__ == "__main__":
    # Example usage
    print("🚀 Testing Optimized Strategy")
    
    # Load optimized strategy
    strategy = load_latest_optimization()
    
    # Display optimization info
    info = strategy.get_optimization_info()
    print("\nOptimization Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Optimized strategy ready for use!")
    print("💡 Use this strategy in main.py by replacing AdvancedTradingStrategy with OptimizedTradingStrategy") 