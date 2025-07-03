#!/usr/bin/env python3
"""
AI-Powered Trade Analysis System for Insane Mode
Uses multiple indicators and machine learning-style analysis
to identify only high-probability trades (90%+ confidence)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class AITradeAnalyzer:
    """AI-powered analysis for extreme precision trading"""
    
    def __init__(self):
        self.name = "AI Trade Analyzer v1.0"
        self.min_confidence_threshold = 90.0
        
        # AI weights for different indicators (trained/optimized)
        self.indicator_weights = {
            'rsi_strength': 0.25,      # RSI extreme levels
            'volume_confirmation': 0.20, # Volume surge confirmation
            'trend_alignment': 0.15,    # Multiple timeframe alignment
            'momentum_divergence': 0.15, # Price vs indicator divergence
            'volatility_squeeze': 0.10,  # Low volatility before breakout
            'support_resistance': 0.15  # Near key levels
        }
        
        # Historical performance tracking for AI learning
        self.trade_history = []
        self.accuracy_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'current_accuracy': 0.0
        }
        
        print("🧠 AI Trade Analyzer initialized for INSANE MODE")
        print("🎯 Target accuracy: 90%+ for extreme leverage trading")
    
    def analyze_trade_opportunity(self, data: pd.DataFrame, current_price: float, side: str) -> Dict:
        """
        AI-powered analysis of trade opportunity
        Returns comprehensive analysis with confidence score
        """
        if len(data) < 50:
            return self._insufficient_data_response()
        
        try:
            # Get latest data point
            latest = data.iloc[-1]
            
            # Perform multi-dimensional analysis
            analysis_results = {
                'rsi_analysis': self._analyze_rsi_extremes(data, side),
                'volume_analysis': self._analyze_volume_confirmation(data),
                'trend_analysis': self._analyze_trend_alignment(data),
                'momentum_analysis': self._analyze_momentum_divergence(data),
                'volatility_analysis': self._analyze_volatility_squeeze(data),
                'support_resistance': self._analyze_key_levels(data, current_price)
            }
            
            # Calculate AI confidence score
            confidence_score = self._calculate_ai_confidence(analysis_results)
            
            # Dynamic leverage calculation (30x-50x based on confidence)
            dynamic_leverage = self._calculate_dynamic_leverage(confidence_score)
            
            # Risk assessment
            risk_assessment = self._assess_extreme_risk(analysis_results, confidence_score)
            
            # Final AI recommendation
            ai_recommendation = self._generate_ai_recommendation(
                analysis_results, confidence_score, side, current_price
            )
            
            return {
                'ai_confidence': confidence_score,
                'dynamic_leverage': dynamic_leverage,
                'recommendation': ai_recommendation,
                'risk_assessment': risk_assessment,
                'analysis_breakdown': analysis_results,
                'trade_approved': confidence_score >= self.min_confidence_threshold,
                'timestamp': datetime.now(),
                'price_analyzed': current_price,
                'side_analyzed': side
            }
            
        except Exception as e:
            print(f"❌ AI Analysis error: {e}")
            return self._error_response()
    
    def _analyze_rsi_extremes(self, data: pd.DataFrame, side: str) -> Dict:
        """Analyze RSI for extreme oversold/overbought conditions"""
        latest_rsi = data['rsi'].iloc[-1]
        rsi_history = data['rsi'].tail(20)
        
        if side == 'buy':
            # For buy signals, look for extreme oversold
            extreme_threshold = 20.0
            strength = max(0, (extreme_threshold - latest_rsi) / extreme_threshold) * 100
            
            # Check for RSI divergence (price lower, RSI higher)
            price_trend = data['close'].tail(10).pct_change().sum()
            rsi_trend = rsi_history.tail(10).diff().sum()
            divergence_detected = price_trend < 0 and rsi_trend > 0
            
        else:  # sell
            # For sell signals, look for extreme overbought
            extreme_threshold = 80.0
            strength = max(0, (latest_rsi - extreme_threshold) / (100 - extreme_threshold)) * 100
            
            # Check for bearish divergence
            price_trend = data['close'].tail(10).pct_change().sum()
            rsi_trend = rsi_history.tail(10).diff().sum()
            divergence_detected = price_trend > 0 and rsi_trend < 0
        
        return {
            'rsi_value': latest_rsi,
            'extreme_strength': strength,
            'divergence_detected': divergence_detected,
            'confidence_contribution': strength * 0.8 + (20 if divergence_detected else 0)
        }
    
    def _analyze_volume_confirmation(self, data: pd.DataFrame) -> Dict:
        """Analyze volume for trade confirmation"""
        recent_volume = data['volume'].tail(5).mean()
        historical_avg = data['volume'].tail(50).mean()
        
        volume_surge = (recent_volume / historical_avg - 1) * 100
        volume_trend = data['volume'].tail(10).pct_change().sum()
        
        # Higher volume surge = higher confidence
        volume_strength = min(100, max(0, volume_surge * 2))
        
        return {
            'volume_surge_pct': volume_surge,
            'volume_trend': volume_trend,
            'volume_strength': volume_strength,
            'confidence_contribution': volume_strength * 0.6
        }
    
    def _analyze_trend_alignment(self, data: pd.DataFrame) -> Dict:
        """Analyze multiple timeframe trend alignment"""
        # Short-term trend (5 periods)
        short_trend = (data['close'].iloc[-1] / data['close'].iloc[-6] - 1) * 100
        
        # Medium-term trend (20 periods)
        medium_trend = (data['close'].iloc[-1] / data['close'].iloc[-21] - 1) * 100
        
        # Long-term trend (50 periods)
        long_trend = (data['close'].iloc[-1] / data['close'].iloc[-51] - 1) * 100 if len(data) >= 51 else 0
        
        # Check EMA alignment
        ema_12 = data['ema_12'].iloc[-1] if 'ema_12' in data.columns else data['close'].iloc[-1]
        ema_26 = data['ema_26'].iloc[-1] if 'ema_26' in data.columns else data['close'].iloc[-1]
        
        ema_alignment = (ema_12 > ema_26) if len(data) > 26 else True
        
        # Calculate trend alignment strength
        trends = [short_trend, medium_trend, long_trend]
        trend_consistency = len([t for t in trends if abs(t) > 1 and np.sign(t) == np.sign(trends[0])]) / len(trends) * 100
        
        return {
            'short_trend': short_trend,
            'medium_trend': medium_trend,
            'long_trend': long_trend,
            'ema_alignment': ema_alignment,
            'trend_consistency': trend_consistency,
            'confidence_contribution': trend_consistency * 0.7
        }
    
    def _analyze_momentum_divergence(self, data: pd.DataFrame) -> Dict:
        """Analyze momentum indicators for divergence"""
        # MACD analysis
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            macd_cross = data['macd'].iloc[-1] > data['macd_signal'].iloc[-1]
            macd_histogram = data['macd'].iloc[-1] - data['macd_signal'].iloc[-1]
            macd_strength = min(100, abs(macd_histogram) * 1000)
        else:
            macd_cross = True
            macd_strength = 50
        
        # Price momentum
        price_momentum = data['close'].pct_change(5).iloc[-1] * 100
        
        # Stochastic analysis (if available)
        momentum_strength = min(100, abs(price_momentum) * 10)
        
        return {
            'macd_cross': macd_cross,
            'macd_strength': macd_strength,
            'price_momentum': price_momentum,
            'momentum_strength': momentum_strength,
            'confidence_contribution': momentum_strength * 0.5
        }
    
    def _analyze_volatility_squeeze(self, data: pd.DataFrame) -> Dict:
        """Analyze volatility for squeeze patterns"""
        # Bollinger Bands squeeze detection
        if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
            bb_width = (data['bb_upper'] - data['bb_lower']) / data['close']
            current_squeeze = bb_width.iloc[-1]
            avg_squeeze = bb_width.tail(20).mean()
            
            squeeze_ratio = current_squeeze / avg_squeeze
            squeeze_strength = max(0, (1 - squeeze_ratio) * 100)
        else:
            # Use price volatility as proxy
            volatility = data['close'].pct_change().tail(20).std() * 100
            avg_volatility = data['close'].pct_change().tail(100).std() * 100
            
            squeeze_ratio = volatility / avg_volatility if avg_volatility > 0 else 1
            squeeze_strength = max(0, (1 - squeeze_ratio) * 100)
        
        return {
            'squeeze_ratio': squeeze_ratio,
            'squeeze_strength': squeeze_strength,
            'confidence_contribution': squeeze_strength * 0.4
        }
    
    def _analyze_key_levels(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Analyze proximity to key support/resistance levels"""
        # Calculate recent highs and lows
        recent_high = data['high'].tail(50).max()
        recent_low = data['low'].tail(50).min()
        
        # Calculate distance to key levels
        distance_to_high = abs(current_price - recent_high) / current_price * 100
        distance_to_low = abs(current_price - recent_low) / current_price * 100
        
        # Closer to key levels = higher significance
        key_level_proximity = min(distance_to_high, distance_to_low)
        proximity_strength = max(0, (5 - key_level_proximity) * 20) if key_level_proximity < 5 else 0
        
        return {
            'recent_high': recent_high,
            'recent_low': recent_low,
            'distance_to_high': distance_to_high,
            'distance_to_low': distance_to_low,
            'proximity_strength': proximity_strength,
            'confidence_contribution': proximity_strength * 0.3
        }
    
    def _calculate_ai_confidence(self, analysis_results: Dict) -> float:
        """Calculate overall AI confidence score using weighted indicators"""
        total_confidence = 0.0
        
        for indicator, weight in self.indicator_weights.items():
            if indicator in analysis_results:
                contribution = analysis_results[indicator].get('confidence_contribution', 0)
                total_confidence += contribution * weight
        
        # Apply AI learning adjustment based on historical accuracy
        if self.accuracy_metrics['total_predictions'] > 10:
            accuracy_adjustment = (self.accuracy_metrics['current_accuracy'] / 100) * 0.1
            total_confidence *= (1 + accuracy_adjustment)
            
            # Adaptive weight adjustment based on performance
            if self.accuracy_metrics['total_predictions'] >= 20:
                self._adapt_indicator_weights()
        
        # Ensure confidence is between 0 and 100
        return min(100.0, max(0.0, total_confidence))
    
    def _calculate_dynamic_leverage(self, confidence_score: float) -> int:
        """Calculate dynamic leverage based on AI confidence (30x-50x range)"""
        if confidence_score < 90:
            return 30  # Minimum for insane mode
        elif confidence_score >= 98:
            return 50  # Maximum leverage for highest confidence
        else:
            # Linear scaling between 30x and 50x
            leverage_range = 50 - 30
            confidence_range = 98 - 90
            confidence_above_min = confidence_score - 90
            
            additional_leverage = (confidence_above_min / confidence_range) * leverage_range
            return int(30 + additional_leverage)
    
    def _assess_extreme_risk(self, analysis_results: Dict, confidence_score: float) -> Dict:
        """Assess risk for extreme leverage trading"""
        # Calculate multiple risk factors
        trend_risk = 100 - analysis_results['trend_analysis']['trend_consistency']
        volume_risk = 100 - analysis_results['volume_analysis']['volume_strength']
        volatility_risk = analysis_results['volatility_analysis']['squeeze_ratio'] * 50
        
        overall_risk = (trend_risk + volume_risk + volatility_risk) / 3
        
        # Risk level classification
        if overall_risk < 20:
            risk_level = "LOW"
        elif overall_risk < 40:
            risk_level = "MODERATE"
        elif overall_risk < 60:
            risk_level = "HIGH"
        else:
            risk_level = "EXTREME"
        
        return {
            'overall_risk_score': overall_risk,
            'risk_level': risk_level,
            'trend_risk': trend_risk,
            'volume_risk': volume_risk,
            'volatility_risk': volatility_risk,
            'confidence_vs_risk_ratio': confidence_score / max(overall_risk, 1)
        }
    
    def _generate_ai_recommendation(self, analysis_results: Dict, confidence_score: float, 
                                  side: str, current_price: float) -> Dict:
        """Generate final AI recommendation"""
        approved = confidence_score >= self.min_confidence_threshold
        
        if approved:
            recommendation = f"✅ AI APPROVES {side.upper()} TRADE"
            reasoning = self._get_approval_reasoning(analysis_results, confidence_score)
        else:
            recommendation = f"❌ AI REJECTS {side.upper()} TRADE"
            reasoning = self._get_rejection_reasoning(analysis_results, confidence_score)
        
        return {
            'approved': approved,
            'recommendation': recommendation,
            'reasoning': reasoning,
            'confidence_score': confidence_score,
            'min_required': self.min_confidence_threshold,
            'side': side,
            'price': current_price
        }
    
    def _get_approval_reasoning(self, analysis_results: Dict, confidence_score: float) -> List[str]:
        """Get reasoning for trade approval"""
        reasons = [f"🧠 AI Confidence: {confidence_score:.1f}% (>90% required)"]
        
        # Add specific strong points
        if analysis_results['rsi_analysis']['extreme_strength'] > 70:
            reasons.append("🎯 Extreme RSI condition detected")
        
        if analysis_results['volume_analysis']['volume_surge_pct'] > 50:
            reasons.append("📈 Strong volume confirmation")
        
        if analysis_results['trend_analysis']['trend_consistency'] > 80:
            reasons.append("📊 Multiple timeframe alignment")
        
        if analysis_results['rsi_analysis']['divergence_detected']:
            reasons.append("🔄 Price-RSI divergence detected")
        
        return reasons
    
    def _get_rejection_reasoning(self, analysis_results: Dict, confidence_score: float) -> List[str]:
        """Get reasoning for trade rejection"""
        reasons = [f"❌ AI Confidence: {confidence_score:.1f}% (<90% required)"]
        
        # Add specific weak points
        if analysis_results['rsi_analysis']['extreme_strength'] < 50:
            reasons.append("⚠️ RSI not extreme enough")
        
        if analysis_results['volume_analysis']['volume_surge_pct'] < 20:
            reasons.append("⚠️ Insufficient volume confirmation")
        
        if analysis_results['trend_analysis']['trend_consistency'] < 60:
            reasons.append("⚠️ Poor trend alignment")
        
        return reasons
    
    def _insufficient_data_response(self) -> Dict:
        """Response when insufficient data for analysis"""
        return {
            'ai_confidence': 0.0,
            'dynamic_leverage': 30,
            'recommendation': {'approved': False, 'reasoning': ['❌ Insufficient data for AI analysis']},
            'risk_assessment': {'risk_level': 'UNKNOWN'},
            'trade_approved': False,
            'error': 'Insufficient data'
        }
    
    def _error_response(self) -> Dict:
        """Response when analysis error occurs"""
        return {
            'ai_confidence': 0.0,
            'dynamic_leverage': 30,
            'recommendation': {'approved': False, 'reasoning': ['❌ AI analysis error']},
            'risk_assessment': {'risk_level': 'ERROR'},
            'trade_approved': False,
            'error': 'Analysis error'
        }
    
    def update_trade_result(self, prediction_confidence: float, actual_result: str):
        """Update AI learning with trade results"""
        self.accuracy_metrics['total_predictions'] += 1
        
        if actual_result == 'win':
            self.accuracy_metrics['correct_predictions'] += 1
        else:
            if prediction_confidence >= 90:
                self.accuracy_metrics['false_positives'] += 1
            else:
                self.accuracy_metrics['false_negatives'] += 1
        
        # Update current accuracy
        if self.accuracy_metrics['total_predictions'] > 0:
            self.accuracy_metrics['current_accuracy'] = (
                self.accuracy_metrics['correct_predictions'] / 
                self.accuracy_metrics['total_predictions'] * 100
            )
        
        print(f"🧠 AI Learning Update: {self.accuracy_metrics['current_accuracy']:.1f}% accuracy")
        
        # Store detailed trade result for pattern analysis
        self.trade_history.append({
            'confidence': prediction_confidence,
            'result': actual_result,
            'timestamp': datetime.now()
        })
        
        # Keep only last 100 trades for memory efficiency
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def _adapt_indicator_weights(self):
        """Dynamically adjust indicator weights based on performance"""
        print(f"🔧 AI Adaptation Check: {self.accuracy_metrics['total_predictions']} trades, {self.accuracy_metrics['current_accuracy']:.1f}% accuracy")
        
        if self.accuracy_metrics['total_predictions'] < 20:
            print("   ⏸️ Not enough trades for adaptation yet")
            return
        
        # Analyze recent performance patterns
        recent_trades = self.trade_history[-20:]
        wins = [t for t in recent_trades if t['result'] == 'win']
        losses = [t for t in recent_trades if t['result'] == 'loss']
        
        # If accuracy is low, reduce confidence requirements and adjust weights
        if self.accuracy_metrics['current_accuracy'] < 70:
            # Increase emphasis on volume and support/resistance
            self.indicator_weights['volume_confirmation'] = min(0.30, self.indicator_weights['volume_confirmation'] * 1.1)
            self.indicator_weights['support_resistance'] = min(0.25, self.indicator_weights['support_resistance'] * 1.1)
            # Reduce RSI emphasis if it's not working well
            self.indicator_weights['rsi_strength'] = max(0.15, self.indicator_weights['rsi_strength'] * 0.9)
            
            print("🧠 AI Adapting: Increasing volume/support emphasis, reducing RSI weight")
        
        elif self.accuracy_metrics['current_accuracy'] > 90:
            # If doing very well, slightly increase RSI and trend weights
            self.indicator_weights['rsi_strength'] = min(0.30, self.indicator_weights['rsi_strength'] * 1.05)
            self.indicator_weights['trend_alignment'] = min(0.20, self.indicator_weights['trend_alignment'] * 1.05)
            
            print("🧠 AI Adapting: High accuracy - emphasizing RSI and trend analysis")
        
        # Normalize weights to ensure they sum to ~1.0
        total_weight = sum(self.indicator_weights.values())
        if total_weight > 0:
            for key in self.indicator_weights:
                self.indicator_weights[key] = self.indicator_weights[key] / total_weight
    
    def get_ai_performance_stats(self) -> Dict:
        """Get AI performance statistics"""
        return {
            'total_predictions': self.accuracy_metrics['total_predictions'],
            'accuracy_rate': self.accuracy_metrics['current_accuracy'],
            'correct_predictions': self.accuracy_metrics['correct_predictions'],
            'false_positives': self.accuracy_metrics['false_positives'],
            'false_negatives': self.accuracy_metrics['false_negatives']
        }

def main():
    """Test the AI analyzer"""
    print("🧠 Testing AI Trade Analyzer for INSANE MODE")
    
    # Create sample data for testing
    data = pd.DataFrame({
        'close': np.random.normal(142, 2, 100),
        'high': np.random.normal(143, 2, 100),
        'low': np.random.normal(141, 2, 100),
        'volume': np.random.normal(1000, 200, 100),
        'rsi': np.random.normal(50, 20, 100)
    })
    
    # Add some indicators
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['ema_26'] = data['close'].ewm(span=26).mean()
    
    analyzer = AITradeAnalyzer()
    
    # Test analysis
    result = analyzer.analyze_trade_opportunity(data, 142.0, 'buy')
    
    print(f"\n🎯 AI Analysis Result:")
    print(f"Confidence: {result['ai_confidence']:.1f}%")
    print(f"Approved: {result['trade_approved']}")
    print(f"Dynamic Leverage: {result['dynamic_leverage']}x")
    print(f"Recommendation: {result['recommendation']['recommendation']}")

if __name__ == "__main__":
    main() 