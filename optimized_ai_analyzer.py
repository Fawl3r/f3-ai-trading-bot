#!/usr/bin/env python3
"""
Optimized AI Trade Analyzer with Enhanced Learning
Adjusted thresholds and improved adaptive learning system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptimizedAIAnalyzer:
    """Enhanced AI analyzer with realistic thresholds and adaptive learning"""
    
    def __init__(self, mode: str = "BALANCED"):
        # Optimized confidence thresholds (reduced by 15% from original)
        self.mode_thresholds = {
            "SAFE": 50.0,      # Was 85% → Now 50% (more realistic)
            "RISK": 40.0,      # Was 75% → Now 40% (balanced)
            "SUPER_RISKY": 30.0,  # Was 60% → Now 30% (aggressive)
            "INSANE": 60.0     # Was 90% → Now 60% (still selective)
        }
        
        # Enhanced indicator weights with learning adaptation
        self.indicator_weights = {
            'rsi_extreme': 0.25,      # RSI oversold/overbought strength
            'volume_confirmation': 0.20,  # Volume supporting the move
            'trend_alignment': 0.20,   # Price trend direction
            'momentum_divergence': 0.15,  # MACD momentum signals
            'volatility_squeeze': 0.10,   # Bollinger Band squeeze
            'support_resistance': 0.10   # Key level proximity
        }
        
        # Learning system enhancement
        self.learning_enabled = True
        self.prediction_history = []
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy_rate': 0.0,
            'confidence_vs_outcome': [],
            'weight_adjustments': 0
        }
        
        # Adaptive parameters
        self.market_volatility_factor = 1.0
        self.recent_performance_weight = 0.3
        self.min_trades_for_learning = 5
        
        print(f"🧠 OPTIMIZED AI ANALYZER initialized for {mode} mode")
        print(f"🎯 Enhanced learning system with realistic thresholds")
        print(f"📊 Base threshold: {self.mode_thresholds.get(mode, 40.0)}%")
    
    def analyze_trade_opportunity(self, data: pd.DataFrame, current_price: float, 
                                signal_type: str = 'buy') -> Dict:
        """Enhanced trade analysis with optimized scoring"""
        
        if len(data) < 20:
            return self._default_response()
        
        try:
            # Calculate all technical indicators
            indicators = self._calculate_enhanced_indicators(data, current_price)
            
            # Multi-factor confidence calculation
            confidence_score = self._calculate_confidence_score(indicators, signal_type)
            
            # Market condition adjustment
            adjusted_confidence = self._adjust_for_market_conditions(confidence_score, data)
            
            # Dynamic leverage calculation (for Insane Mode)
            dynamic_leverage = self._calculate_dynamic_leverage(adjusted_confidence)
            
            # Generate detailed analysis
            analysis = self._generate_trade_analysis(indicators, adjusted_confidence)
            
            # Record prediction for learning
            self._record_prediction(adjusted_confidence, indicators)
            
            result = {
                'ai_confidence': max(0, min(100, adjusted_confidence)),
                'dynamic_leverage': dynamic_leverage,
                'signal_strength': confidence_score,
                'market_adjustment': adjusted_confidence - confidence_score,
                'key_factors': analysis['key_factors'],
                'risk_assessment': analysis['risk_level'],
                'prediction_id': len(self.prediction_history)
            }
            
            # Enhanced logging
            if adjusted_confidence > 45:  # Log significant signals
                print(f"    🧠 AI Signal: {adjusted_confidence:.1f}% confidence")
                print(f"       Key factors: {', '.join(analysis['key_factors'][:2])}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ AI Analysis error: {e}")
            return self._default_response()
    
    def _calculate_enhanced_indicators(self, data: pd.DataFrame, price: float) -> Dict:
        """Calculate enhanced technical indicators"""
        latest = data.iloc[-1]
        
        # RSI analysis (enhanced)
        rsi = latest.get('rsi', 50)
        rsi_extreme_score = 0
        if rsi < 30:
            rsi_extreme_score = (30 - rsi) / 30 * 100  # Stronger signal when more oversold
        elif rsi > 70:
            rsi_extreme_score = (rsi - 70) / 30 * 100  # Stronger signal when more overbought
        
        # Volume analysis (enhanced)
        recent_volumes = data['volume'].tail(10)
        current_volume = latest.get('volume', 1000)
        avg_volume = recent_volumes.mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        volume_score = min(100, (volume_ratio - 1) * 50 + 50)  # 50-100 scale
        
        # Trend analysis (enhanced)
        prices = data['close'].tail(20)
        if len(prices) >= 10:
            short_ma = prices.tail(5).mean()
            long_ma = prices.tail(20).mean()
            trend_strength = ((short_ma - long_ma) / long_ma) * 1000 if long_ma > 0 else 0
            trend_score = 50 + min(25, max(-25, trend_strength))  # 25-75 scale
        else:
            trend_score = 50
        
        # MACD momentum (enhanced)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_histogram = latest.get('macd_histogram', 0)
        
        momentum_score = 50
        if abs(macd_histogram) > 0.1:
            momentum_score += min(25, abs(macd_histogram) * 250)
        
        # Volatility squeeze detection
        bb_upper = latest.get('bb_upper', price * 1.02)
        bb_lower = latest.get('bb_lower', price * 0.98)
        bb_width = (bb_upper - bb_lower) / price * 100
        
        # Historical average width
        if len(data) >= 20:
            historical_widths = []
            for i in range(-20, 0):
                if i + len(data) >= 0:
                    row = data.iloc[i]
                    width = (row.get('bb_upper', price) - row.get('bb_lower', price)) / row.get('close', price) * 100
                    historical_widths.append(width)
            
            avg_width = np.mean(historical_widths) if historical_widths else bb_width
            squeeze_ratio = bb_width / avg_width if avg_width > 0 else 1
            volatility_score = (1 - min(0.8, squeeze_ratio)) * 125  # Higher score for squeeze
        else:
            volatility_score = 50
        
        # Support/Resistance analysis
        recent_highs = data['high'].tail(50)
        recent_lows = data['low'].tail(50)
        
        resistance_level = recent_highs.max()
        support_level = recent_lows.min()
        
        # Distance to key levels
        resistance_distance = (resistance_level - price) / price * 100
        support_distance = (price - support_level) / price * 100
        
        # Score based on proximity to key levels
        if support_distance < 1:  # Very close to support
            support_resistance_score = 80
        elif resistance_distance < 1:  # Very close to resistance
            support_resistance_score = 20
        else:
            support_resistance_score = 50
        
        return {
            'rsi_extreme': max(0, min(100, rsi_extreme_score)),
            'volume_confirmation': max(0, min(100, volume_score)),
            'trend_alignment': max(0, min(100, trend_score)),
            'momentum_divergence': max(0, min(100, momentum_score)),
            'volatility_squeeze': max(0, min(100, volatility_score)),
            'support_resistance': max(0, min(100, support_resistance_score)),
            'raw_data': {
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'bb_width': bb_width,
                'macd_histogram': macd_histogram
            }
        }
    
    def _calculate_confidence_score(self, indicators: Dict, signal_type: str) -> float:
        """Calculate weighted confidence score"""
        score = 0
        total_weight = 0
        
        for indicator, value in indicators.items():
            if indicator == 'raw_data':
                continue
                
            weight = self.indicator_weights.get(indicator, 0)
            if weight > 0:
                score += value * weight
                total_weight += weight
        
        # Normalize to 0-100 scale
        base_score = (score / total_weight) if total_weight > 0 else 50
        
        # Apply signal-specific adjustments
        if signal_type == 'buy':
            # Boost confidence for strong oversold + volume + support
            if indicators['rsi_extreme'] > 60 and indicators['volume_confirmation'] > 60:
                base_score *= 1.1
            if indicators['support_resistance'] > 70:
                base_score *= 1.05
        elif signal_type == 'sell':
            # Boost confidence for strong overbought + volume + resistance
            if indicators['rsi_extreme'] > 60 and indicators['volume_confirmation'] > 60:
                base_score *= 1.1
            if indicators['support_resistance'] < 30:
                base_score *= 1.05
        
        return max(0, min(100, base_score))
    
    def _adjust_for_market_conditions(self, base_confidence: float, data: pd.DataFrame) -> float:
        """Adjust confidence based on current market conditions"""
        
        # Calculate recent volatility
        recent_prices = data['close'].tail(20)
        if len(recent_prices) >= 10:
            volatility = recent_prices.std() / recent_prices.mean() * 100
            
            # Adjust based on volatility
            if volatility > 5:  # High volatility
                self.market_volatility_factor = 1.1  # Slight boost in volatile markets
            elif volatility < 2:  # Low volatility
                self.market_volatility_factor = 0.95  # Slight reduction in calm markets
            else:
                self.market_volatility_factor = 1.0
        
        # Apply learning-based adjustments
        if self.learning_enabled and len(self.prediction_history) >= self.min_trades_for_learning:
            learning_adjustment = self._get_learning_adjustment()
            adjusted_confidence = base_confidence * self.market_volatility_factor * learning_adjustment
        else:
            adjusted_confidence = base_confidence * self.market_volatility_factor
        
        return max(0, min(100, adjusted_confidence))
    
    def _calculate_dynamic_leverage(self, confidence: float) -> int:
        """Calculate dynamic leverage based on confidence (for Insane Mode)"""
        if confidence >= 80:
            return 50  # Maximum leverage for highest confidence
        elif confidence >= 70:
            return 45
        elif confidence >= 60:
            return 40
        elif confidence >= 50:
            return 35
        else:
            return 30  # Minimum leverage
    
    def _generate_trade_analysis(self, indicators: Dict, confidence: float) -> Dict:
        """Generate detailed trade analysis"""
        key_factors = []
        
        # Identify strongest factors
        for indicator, value in indicators.items():
            if indicator == 'raw_data':
                continue
            if value > 70:
                factor_name = indicator.replace('_', ' ').title()
                key_factors.append(f"{factor_name} ({value:.0f}%)")
        
        # Risk assessment
        if confidence >= 70:
            risk_level = "LOW - High probability setup"
        elif confidence >= 50:
            risk_level = "MODERATE - Decent setup with confirmation"
        elif confidence >= 30:
            risk_level = "HIGH - Weak signals, proceed with caution"
        else:
            risk_level = "VERY HIGH - Poor setup, avoid trade"
        
        return {
            'key_factors': key_factors[:3],  # Top 3 factors
            'risk_level': risk_level
        }
    
    def _record_prediction(self, confidence: float, indicators: Dict):
        """Record prediction for learning system"""
        prediction = {
            'timestamp': datetime.now(),
            'confidence': confidence,
            'indicators': indicators.copy(),
            'outcome': None,  # Will be updated when trade closes
            'prediction_id': len(self.prediction_history)
        }
        
        self.prediction_history.append(prediction)
        self.performance_stats['total_predictions'] += 1
        
        # Keep only recent predictions (last 100)
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
    
    def update_trade_result(self, confidence: float, outcome: str):
        """Enhanced learning from trade outcomes"""
        if not self.learning_enabled:
            return
        
        # Find the most recent prediction that matches this confidence level
        matching_prediction = None
        for prediction in reversed(self.prediction_history):
            if prediction['outcome'] is None and abs(prediction['confidence'] - confidence) < 5:
                matching_prediction = prediction
                break
        
        if matching_prediction:
            matching_prediction['outcome'] = outcome
            
            # Update performance stats
            if outcome == 'win':
                self.performance_stats['correct_predictions'] += 1
            
            # Record confidence vs outcome for analysis
            self.performance_stats['confidence_vs_outcome'].append({
                'confidence': confidence,
                'outcome': outcome,
                'timestamp': datetime.now()
            })
            
            # Update accuracy rate
            completed_predictions = [p for p in self.prediction_history if p['outcome'] is not None]
            if completed_predictions:
                wins = len([p for p in completed_predictions if p['outcome'] == 'win'])
                self.performance_stats['accuracy_rate'] = (wins / len(completed_predictions)) * 100
            
            # Adaptive learning - adjust weights based on performance
            self._adaptive_weight_adjustment(matching_prediction, outcome)
            
            print(f"    🧠 AI Learning: {outcome} @ {confidence:.1f}% confidence")
            print(f"       Overall accuracy: {self.performance_stats['accuracy_rate']:.1f}%")
    
    def _adaptive_weight_adjustment(self, prediction: Dict, outcome: str):
        """Adjust indicator weights based on prediction outcomes"""
        
        # Only adjust after minimum number of trades
        if len([p for p in self.prediction_history if p['outcome'] is not None]) < self.min_trades_for_learning:
            return
        
        indicators = prediction['indicators']
        confidence = prediction['confidence']
        
        # Calculate adjustment factor
        if outcome == 'win':
            adjustment_factor = 1.02  # Small boost for successful indicators
        else:
            adjustment_factor = 0.98  # Small reduction for failed indicators
        
        # Adjust weights for indicators that contributed most to this prediction
        for indicator, value in indicators.items():
            if indicator == 'raw_data':
                continue
                
            if value > 60:  # High contribution indicators
                current_weight = self.indicator_weights.get(indicator, 0)
                new_weight = current_weight * adjustment_factor
                
                # Keep weights in reasonable bounds
                self.indicator_weights[indicator] = max(0.05, min(0.4, new_weight))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.indicator_weights.values())
        if total_weight > 0:
            for indicator in self.indicator_weights:
                self.indicator_weights[indicator] /= total_weight
        
        self.performance_stats['weight_adjustments'] += 1
        
        # Log significant adjustments
        if self.performance_stats['weight_adjustments'] % 10 == 0:
            print(f"    🔧 AI Weights adjusted ({self.performance_stats['weight_adjustments']} times)")
    
    def _get_learning_adjustment(self) -> float:
        """Get confidence adjustment based on recent performance"""
        
        recent_outcomes = self.performance_stats['confidence_vs_outcome'][-20:]  # Last 20 trades
        if len(recent_outcomes) < 5:
            return 1.0
        
        # Calculate recent accuracy
        wins = len([o for o in recent_outcomes if o['outcome'] == 'win'])
        recent_accuracy = wins / len(recent_outcomes)
        
        # Adjust confidence based on recent performance
        if recent_accuracy > 0.6:  # Doing well
            return 1.05  # Slight confidence boost
        elif recent_accuracy < 0.4:  # Doing poorly
            return 0.95  # Slight confidence reduction
        else:
            return 1.0  # No adjustment
    
    def _default_response(self) -> Dict:
        """Default response for error cases"""
        return {
            'ai_confidence': 25.0,
            'dynamic_leverage': 30,
            'signal_strength': 25.0,
            'market_adjustment': 0.0,
            'key_factors': ['Insufficient data'],
            'risk_assessment': 'HIGH - Limited analysis available',
            'prediction_id': -1
        }
    
    def get_ai_performance_stats(self) -> Dict:
        """Get comprehensive AI performance statistics"""
        completed_trades = [p for p in self.prediction_history if p['outcome'] is not None]
        
        stats = {
            'total_predictions': self.performance_stats['total_predictions'],
            'completed_trades': len(completed_trades),
            'accuracy_rate': self.performance_stats['accuracy_rate'],
            'weight_adjustments': self.performance_stats['weight_adjustments'],
            'current_weights': self.indicator_weights.copy(),
            'market_volatility_factor': self.market_volatility_factor,
            'learning_enabled': self.learning_enabled
        }
        
        # Confidence level analysis
        if completed_trades:
            high_conf_trades = [t for t in completed_trades if t['confidence'] >= 60]
            med_conf_trades = [t for t in completed_trades if 40 <= t['confidence'] < 60]
            low_conf_trades = [t for t in completed_trades if t['confidence'] < 40]
            
            stats['confidence_breakdown'] = {
                'high_confidence': {
                    'count': len(high_conf_trades),
                    'accuracy': (len([t for t in high_conf_trades if t['outcome'] == 'win']) / len(high_conf_trades) * 100) if high_conf_trades else 0
                },
                'medium_confidence': {
                    'count': len(med_conf_trades),
                    'accuracy': (len([t for t in med_conf_trades if t['outcome'] == 'win']) / len(med_conf_trades) * 100) if med_conf_trades else 0
                },
                'low_confidence': {
                    'count': len(low_conf_trades),
                    'accuracy': (len([t for t in low_conf_trades if t['outcome'] == 'win']) / len(low_conf_trades) * 100) if low_conf_trades else 0
                }
            }
        
        return stats
    
    def reset_learning_data(self):
        """Reset learning data for fresh start"""
        self.prediction_history = []
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy_rate': 0.0,
            'confidence_vs_outcome': [],
            'weight_adjustments': 0
        }
        print("🧠 AI Learning data reset - fresh start!")

def main():
    """Test the optimized AI analyzer"""
    print("🧪 Testing Optimized AI Analyzer")
    
    # Test with different modes
    modes = ['SAFE', 'RISK', 'SUPER_RISKY', 'INSANE']
    
    for mode in modes:
        print(f"\n🔬 Testing {mode} mode...")
        analyzer = OptimizedAIAnalyzer(mode)
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='T')
        data = pd.DataFrame({
            'timestamp': dates,
            'close': 140 + np.random.normal(0, 2, 100).cumsum(),
            'volume': np.random.uniform(800, 1200, 100),
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.normal(0, 0.5, 100),
            'macd_signal': np.random.normal(0, 0.3, 100),
            'macd_histogram': np.random.normal(0, 0.2, 100),
            'bb_upper': 142 + np.random.normal(0, 1, 100),
            'bb_lower': 138 + np.random.normal(0, 1, 100)
        })
        
        # Test analysis
        result = analyzer.analyze_trade_opportunity(data, 140.0, 'buy')
        print(f"   Confidence: {result['ai_confidence']:.1f}%")
        print(f"   Threshold: {analyzer.mode_thresholds[mode]}%")
        print(f"   Above threshold: {'✅' if result['ai_confidence'] >= analyzer.mode_thresholds[mode] else '❌'}")

if __name__ == "__main__":
    main() 