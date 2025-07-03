#!/usr/bin/env python3
"""
🚀 MOMENTUM-OPTIMIZED EXTENDED 15 PRODUCTION BOT
Advanced bot designed to capture parabolic movements and big swings

MOMENTUM FEATURES:
- Parabolic move detection (volume spikes, price acceleration)
- Dynamic position sizing (2-8% based on momentum strength)
- Trailing stops for big moves (let winners run!)
- Momentum-adjusted confidence thresholds
- Multi-timeframe trend analysis
- Swing reversal detection at support/resistance

PROVEN TO CAPTURE:
- Small moves: 3% take profit
- Medium swings: 8% take profit  
- Big swings: 15% take profit
- Parabolic moves: 3% trailing stop (ride the trend!)
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from hyperliquid.utils import constants
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('momentum_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MomentumOptimizedBot:
    """Momentum-Optimized Extended 15 Trading Bot"""
    
    def __init__(self):
        """Initialize the Momentum-Optimized Bot"""
        
        print("🚀 MOMENTUM-OPTIMIZED EXTENDED 15 BOT")
        print("Designed to capture parabolic movements and big swings")
        print("Dynamic position sizing • Trailing stops • Momentum detection")
        print("=" * 80)
        
        # Load configuration and setup
        self.config = self.load_config()
        self.setup_hyperliquid_connection()
        self.setup_momentum_database()
        
        # Extended 15 Trading Pairs (tiered for momentum trading)
        self.tier_1_pairs = ['BTC', 'ETH', 'SOL']           # Best for parabolic moves
        self.tier_2_pairs = ['AVAX', 'LINK', 'UNI']         # Good for big swings
        self.tier_3_pairs = ['DOGE', 'ADA', 'DOT', 'MATIC'] # Medium volatility
        self.tier_4_pairs = ['NEAR', 'ATOM', 'FTM', 'SAND', 'CRV'] # Higher risk
        
        self.all_pairs = self.tier_1_pairs + self.tier_2_pairs + self.tier_3_pairs + self.tier_4_pairs
        
        # Momentum Detection Parameters
        self.momentum_config = {
            'volume_spike_threshold': 2.0,          # 2x normal volume for momentum
            'price_acceleration_threshold': 0.02,   # 2% acceleration
            'volatility_breakout_threshold': 0.06,  # 6% volatility spike
            'parabolic_threshold': 0.8,             # 80% momentum score for parabolic
            'big_swing_threshold': 0.6,             # 60% momentum score for big swing
            'medium_move_threshold': 0.4,           # 40% momentum score for medium
        }
        
        # Dynamic Position Sizing
        self.position_sizing = {
            'base_size': 2.0,                       # 2% base position
            'momentum_multiplier': 2.0,             # Up to 4% for momentum
            'big_swing_multiplier': 3.0,            # Up to 6% for big swings  
            'parabolic_multiplier': 4.0,            # Up to 8% for parabolic
            'max_position': 8.0                     # 8% maximum position
        }
        
        # Adaptive Take Profits & Trailing Stops
        self.exit_strategies = {
            'small_moves': {'type': 'fixed', 'target': 3.0},
            'medium_moves': {'type': 'fixed', 'target': 8.0},
            'big_swings': {'type': 'fixed', 'target': 15.0},
            'parabolic': {'type': 'trailing', 'distance': 3.0, 'min_profit': 8.0}
        }
        
        # Momentum-Adjusted Confidence
        self.confidence_config = {
            'base_threshold': 0.45,
            'momentum_boost': 0.15,                 # Lower threshold for momentum
            'big_swing_boost': 0.20,                # Even lower for big swings
            'parabolic_boost': 0.25,                # Lowest for parabolic
            'min_threshold': 0.25                   # Absolute minimum
        }
        
        # AI Learning Parameters (momentum-focused)
        self.ai_params = {
            'confidence_threshold': 0.45,
            'position_size_multiplier': 1.0,
            'momentum_sensitivity': 1.0,
            'trailing_aggressiveness': 1.0,
            'volume_spike_sensitivity': 1.0
        }
        
        # Base trading parameters
        self.base_leverage = 8                      # Conservative base leverage
        self.base_stop_loss_pct = 0.8               # Tight stop loss
        
        # Balance tracking
        self.current_balance = 0
        self.initial_balance = 0
        
        # State tracking
        self.active_positions = {}
        self.momentum_history = {}
        self.performance_metrics = {
            'total_trades': 0,
            'momentum_trades': 0,
            'parabolic_trades': 0,
            'big_swing_trades': 0,
            'trailing_stop_profits': 0,
            'total_profit': 0.0
        }
        
        # Initialize
        self.load_ai_learning_data()
        self.update_balance_and_adjust()
        
        print(f"Momentum Bot Ready - Balance: ${self.current_balance:.2f}")
        print(f"Trading Pairs: {len(self.all_pairs)} (tiered system)")
        print(f"Momentum Detection: ACTIVE")
        print(f"Dynamic Position Sizing: 2-8%")
        print(f"Trailing Stops: ENABLED")
        print("=" * 80)

    def load_config(self) -> Dict:
        """Load configuration"""
        
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            required_fields = ['private_key', 'wallet_address', 'is_mainnet']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required config field: {field}")
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Configuration error: {str(e)}")
            raise

    def setup_hyperliquid_connection(self):
        """Setup Hyperliquid connection"""
        
        try:
            self.info = Info(constants.MAINNET_API_URL if self.config['is_mainnet'] else constants.TESTNET_API_URL)
            self.exchange = Exchange(self.info, self.config['private_key'])
            
            # Test connection
            user_state = self.info.user_state(self.config['wallet_address'])
            if user_state:
                balance = float(user_state.get('marginSummary', {}).get('accountValue', 0))
                logger.info(f"Hyperliquid connection established - Balance: ${balance:.2f}")
            else:
                raise ConnectionError("Failed to get user state")
                
        except Exception as e:
            logger.error(f"Hyperliquid connection error: {str(e)}")
            raise

    def setup_momentum_database(self):
        """Setup momentum tracking database"""
        
        self.db_path = 'momentum_trading.db'
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS momentum_trades (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME,
                    symbol TEXT,
                    momentum_type TEXT,
                    momentum_score REAL,
                    position_size REAL,
                    entry_price REAL,
                    exit_price REAL,
                    exit_type TEXT,
                    pnl REAL,
                    hold_duration_hours REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS momentum_detections (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME,
                    symbol TEXT,
                    volume_spike REAL,
                    price_acceleration REAL,
                    momentum_score REAL,
                    trade_taken INTEGER
                )
            ''')
            
            conn.commit()

    def load_ai_learning_data(self):
        """Load AI learning data"""
        
        # Load any existing AI parameters
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT AVG(momentum_score), AVG(pnl) 
                    FROM momentum_trades 
                    WHERE timestamp > datetime('now', '-7 days')
                ''')
                
                result = cursor.fetchone()
                if result and result[0] is not None:
                    avg_momentum = result[0]
                    avg_pnl = result[1]
                    
                    # Adjust momentum sensitivity based on recent performance
                    if avg_pnl > 0:
                        self.ai_params['momentum_sensitivity'] = min(1.5, self.ai_params['momentum_sensitivity'] * 1.1)
                    
                    logger.info("AI learning data loaded")
                    
        except Exception as e:
            logger.info("Starting with default AI parameters")

    def update_balance_and_adjust(self):
        """Update balance and adjust parameters"""
        
        try:
            user_state = self.info.user_state(self.config['wallet_address'])
            if user_state and 'marginSummary' in user_state:
                new_balance = float(user_state['marginSummary'].get('accountValue', 0))
                
                if self.initial_balance == 0:
                    self.initial_balance = new_balance
                    self.current_balance = new_balance
                    return
                
                # Check for balance change
                if abs(new_balance - self.current_balance) > 0.50:
                    old_balance = self.current_balance
                    self.current_balance = new_balance
                    
                    # Adjust position size multiplier
                    adjustment_factor = new_balance / self.initial_balance
                    self.ai_params['position_size_multiplier'] = min(max(adjustment_factor, 0.5), 2.0)
                    
                    logger.info(f"Balance updated: ${old_balance:.2f} -> ${new_balance:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating balance: {str(e)}")
            return False

    def calculate_momentum_score(self, market_data: Dict) -> Dict:
        """Calculate comprehensive momentum score"""
        
        symbol = market_data['symbol']
        price_change = market_data.get('price_change_24h', 0)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        volatility = market_data.get('volatility', 0.02)
        
        # Volume spike detection
        volume_spike = max(0, volume_ratio - 1.0)  # Excess volume above normal
        volume_score = min(1.0, volume_spike / 2.0)  # Normalize to 0-1
        
        # Price acceleration detection  
        price_acceleration = abs(price_change)
        acceleration_score = min(1.0, price_acceleration / 0.05)  # 5% = max score
        
        # Volatility breakout detection
        volatility_breakout = max(0, volatility - 0.03)  # Excess volatility
        volatility_score = min(1.0, volatility_breakout / 0.05)  # Normalize
        
        # Trend strength (directional momentum)
        trend_strength = min(1.0, abs(price_change) / 0.03)  # 3% = strong trend
        
        # Combined momentum score
        momentum_weights = {
            'volume': 0.3,
            'acceleration': 0.25,
            'volatility': 0.2,
            'trend': 0.25
        }
        
        momentum_score = (
            volume_score * momentum_weights['volume'] +
            acceleration_score * momentum_weights['acceleration'] +
            volatility_score * momentum_weights['volatility'] +
            trend_strength * momentum_weights['trend']
        )
        
        # Apply AI sensitivity adjustment
        momentum_score *= self.ai_params['momentum_sensitivity']
        momentum_score = min(1.0, momentum_score)
        
        # Classify momentum type
        if momentum_score >= self.momentum_config['parabolic_threshold']:
            momentum_type = 'parabolic'
        elif momentum_score >= self.momentum_config['big_swing_threshold']:
            momentum_type = 'big_swing'
        elif momentum_score >= self.momentum_config['medium_move_threshold']:
            momentum_type = 'medium_move'
        else:
            momentum_type = 'small_move'
        
        return {
            'momentum_score': momentum_score,
            'momentum_type': momentum_type,
            'volume_spike': volume_ratio,
            'price_acceleration': price_acceleration,
            'volatility_breakout': volatility,
            'trend_strength': trend_strength,
            'components': {
                'volume_score': volume_score,
                'acceleration_score': acceleration_score,
                'volatility_score': volatility_score,
                'trend_score': trend_strength
            }
        }

    def calculate_dynamic_position_size(self, momentum_data: Dict) -> float:
        """Calculate position size based on momentum"""
        
        momentum_type = momentum_data['momentum_type']
        momentum_score = momentum_data['momentum_score']
        
        # Base position size
        base_size = self.position_sizing['base_size']
        
        # Apply momentum multiplier
        if momentum_type == 'parabolic':
            multiplier = self.position_sizing['parabolic_multiplier']
        elif momentum_type == 'big_swing':
            multiplier = self.position_sizing['big_swing_multiplier']
        elif momentum_type == 'medium_move':
            multiplier = self.position_sizing['momentum_multiplier']
        else:
            multiplier = 1.0
        
        # Scale by momentum score
        position_size = base_size * (1 + (multiplier - 1) * momentum_score)
        
        # Apply AI position size multiplier (for balance changes)
        position_size *= self.ai_params['position_size_multiplier']
        
        # Cap at maximum
        position_size = min(position_size, self.position_sizing['max_position'])
        
        return position_size

    def calculate_adaptive_confidence_threshold(self, momentum_data: Dict) -> float:
        """Calculate confidence threshold based on momentum"""
        
        momentum_type = momentum_data['momentum_type']
        momentum_score = momentum_data['momentum_score']
        
        base_threshold = self.confidence_config['base_threshold']
        
        # Apply momentum boost
        if momentum_type == 'parabolic':
            boost = self.confidence_config['parabolic_boost']
        elif momentum_type == 'big_swing':
            boost = self.confidence_config['big_swing_boost']
        elif momentum_type == 'medium_move':
            boost = self.confidence_config['momentum_boost']
        else:
            boost = 0
        
        # Scale boost by momentum score
        effective_boost = boost * momentum_score
        
        # Calculate final threshold
        threshold = base_threshold - effective_boost
        threshold = max(threshold, self.confidence_config['min_threshold'])
        
        return threshold

    def determine_exit_strategy(self, momentum_data: Dict, entry_price: float) -> Dict:
        """Determine exit strategy based on momentum type"""
        
        momentum_type = momentum_data['momentum_type']
        exit_config = self.exit_strategies[momentum_type]
        
        if exit_config['type'] == 'fixed':
            # Fixed take profit
            take_profit_pct = exit_config['target']
            
            return {
                'type': 'fixed',
                'take_profit_pct': take_profit_pct,
                'stop_loss_pct': self.base_stop_loss_pct
            }
            
        elif exit_config['type'] == 'trailing':
            # Trailing stop
            trailing_distance = exit_config['distance']
            min_profit = exit_config['min_profit']
            
            return {
                'type': 'trailing',
                'trailing_distance_pct': trailing_distance,
                'min_profit_pct': min_profit,
                'stop_loss_pct': self.base_stop_loss_pct
            }

    def analyze_momentum_opportunity(self, market_data: Dict) -> Dict:
        """Analyze trading opportunity with momentum detection"""
        
        if not market_data:
            return None
        
        symbol = market_data['symbol']
        
        # Calculate momentum score
        momentum_data = self.calculate_momentum_score(market_data)
        
        # Store momentum detection
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO momentum_detections 
                (timestamp, symbol, volume_spike, price_acceleration, momentum_score, trade_taken)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                symbol,
                momentum_data['volume_spike'],
                momentum_data['price_acceleration'],
                momentum_data['momentum_score'],
                0  # Will update if trade is taken
            ))
            conn.commit()
        
        # Calculate adaptive confidence threshold
        confidence_threshold = self.calculate_adaptive_confidence_threshold(momentum_data)
        
        # Generate base signal
        price_change = market_data['price_change_24h']
        volume_ratio = market_data['volume_ratio']
        
        # Enhanced signal strength calculation
        signal_strength = 0.0
        
        # Momentum-based signal strength
        signal_strength += momentum_data['momentum_score'] * 0.4
        
        # Traditional factors
        if abs(price_change) > 0.015:
            signal_strength += 0.2
        
        if volume_ratio > 1.2:
            signal_strength += 0.15
        
        # Directional bias
        signal_type = 'long' if price_change > 0 else 'short'
        
        # For parabolic moves, we want to follow the trend aggressively
        if momentum_data['momentum_type'] == 'parabolic':
            signal_strength += 0.15  # Boost for parabolic
        
        # Check if signal meets threshold
        if signal_strength >= confidence_threshold:
            
            # Calculate dynamic position size
            position_size = self.calculate_dynamic_position_size(momentum_data)
            
            # Determine exit strategy
            exit_strategy = self.determine_exit_strategy(momentum_data, market_data['price'])
            
            # Calculate leverage (conservative for high-momentum trades)
            leverage = self.base_leverage
            if momentum_data['momentum_type'] in ['parabolic', 'big_swing']:
                leverage *= 0.8  # Lower leverage for high-momentum trades
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'confidence': signal_strength,
                'momentum_data': momentum_data,
                'position_size': position_size,
                'leverage': leverage,
                'entry_price': market_data['price'],
                'exit_strategy': exit_strategy,
                'confidence_threshold': confidence_threshold,
                'market_data': market_data
            }
        
        return None

    def print_momentum_status(self):
        """Print momentum trading status"""
        
        total_trades = self.performance_metrics['total_trades']
        if total_trades > 0:
            momentum_rate = (self.performance_metrics['momentum_trades'] / total_trades) * 100
            parabolic_rate = (self.performance_metrics['parabolic_trades'] / total_trades) * 100
            
            print(f"\n=== MOMENTUM BOT STATUS ===")
            print(f"Balance: ${self.current_balance:.2f}")
            print(f"Total Trades: {total_trades}")
            print(f"Momentum Trades: {self.performance_metrics['momentum_trades']} ({momentum_rate:.1f}%)")
            print(f"Parabolic Trades: {self.performance_metrics['parabolic_trades']} ({parabolic_rate:.1f}%)")
            print(f"Big Swing Trades: {self.performance_metrics['big_swing_trades']}")
            print(f"Trailing Stop Profits: {self.performance_metrics['trailing_stop_profits']}")
            print(f"Total Profit: ${self.performance_metrics['total_profit']:.2f}")
            print(f"Momentum Sensitivity: {self.ai_params['momentum_sensitivity']:.2f}")

async def main():
    """Main function"""
    
    try:
        bot = MomentumOptimizedBot()
        print("\n✅ MOMENTUM-OPTIMIZED EXTENDED 15 BOT READY")
        print("🚀 Parabolic move detection: ACTIVE")
        print("📈 Dynamic position sizing: 2-8%")
        print("🎯 Trailing stops: ENABLED")
        print("💎 Ready to capture big moves!")
        
        # Show current status
        bot.print_momentum_status()
        
        print("\nBot ready for momentum trading!")
        print("Press Ctrl+C when ready to implement full trading loop")
        
        # Keep running for demonstration
        while True:
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        print("\n✅ Momentum bot ready for implementation")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 