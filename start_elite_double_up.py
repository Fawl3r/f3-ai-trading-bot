#!/usr/bin/env python3
"""
Elite Double-Up Trading System - Main Execution Script
$50 → $100 in 30 days with 0.75% risk per trade
"""

import asyncio
import logging
import argparse
import yaml
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple
import signal
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Core components
from data.fetch_hyperliquid import HyperliquidDataFetcher
from models.elite_ai_trainer import EliteAITrainer
from backtests.elite_walk_forward_backtest import EliteWalkForwardBacktest

# ML libraries
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler

# Hyperliquid API
import requests
import websockets
import hmac
import hashlib
from eth_account import Account

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('elite_double_up.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EliteDoubleUpTrader:
    """Elite Double-Up Trading System - Main Trading Engine"""
    
    def __init__(self, config_file: str = "deployment_config_double_up.yaml"):
        self.config = self.load_config(config_file)
        self.running = False
        self.emergency_stop = False
        self.current_phase = 1
        self.positions = {}
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl_r': 0.0,
            'total_pnl_pct': 0.0,
            'current_drawdown': 0.0,
            'max_drawdown': 0.0,
            'consecutive_losses': 0,
            'account_balance': 50.0,  # Starting balance
            'daily_pnl': 0.0,
            'weekly_pnl': 0.0,
            'monthly_pnl': 0.0
        }
        
        # Load models and scalers
        self.models = {}
        self.scalers = {}
        self.load_trained_models()
        
        # Initialize Hyperliquid client
        self.init_hyperliquid_client()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🚀 Elite Double-Up Trader initialized")
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"📋 Configuration loaded from {config_file}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise
    
    def load_trained_models(self):
        """Load trained AI models for all coins"""
        models_dir = Path("models/trained")
        
        for coin in ['SOL', 'BTC', 'ETH']:
            coin_dir = models_dir / coin
            if not coin_dir.exists():
                logger.warning(f"⚠️ No models found for {coin}")
                continue
            
            coin_models = {}
            
            # Load tree-based models
            for model_name in ['lightgbm', 'catboost', 'ensemble']:
                model_file = coin_dir / f"{model_name}_model.pkl"
                if model_file.exists():
                    coin_models[model_name] = joblib.load(model_file)
            
            # Load deep learning models
            for model_name in ['bilstm', 'transformer']:
                model_file = coin_dir / f"{model_name}_model.pth"
                if model_file.exists():
                    # Load metadata to get input size
                    try:
                        metadata = joblib.load(Path("data/processed") / coin / "metadata.pkl")
                        input_size = len(metadata['feature_names'])
                        
                        if model_name == 'bilstm':
                            from models.elite_ai_trainer import BiLSTMModel
                            model = BiLSTMModel(input_size=input_size)
                        elif model_name == 'transformer':
                            from models.elite_ai_trainer import TransformerModel
                            model = TransformerModel(input_size=input_size)
                        
                        model.load_state_dict(torch.load(model_file, map_location='cpu'))
                        model.eval()
                        coin_models[model_name] = model
                    except Exception as e:
                        logger.error(f"❌ Failed to load {model_name} for {coin}: {e}")
            
            # Load scaler
            scaler_file = Path("data/processed") / coin / "scaler.pkl"
            if scaler_file.exists():
                self.scalers[coin] = joblib.load(scaler_file)
            
            self.models[coin] = coin_models
            logger.info(f"✅ Loaded {len(coin_models)} models for {coin}")
    
    def init_hyperliquid_client(self):
        """Initialize Hyperliquid client"""
        try:
            self.private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
            if not self.private_key:
                raise ValueError("HYPERLIQUID_PRIVATE_KEY not set")
            
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
            
            # Test connection
            response = requests.get('https://api.hyperliquid.xyz/info')
            if response.status_code == 200:
                logger.info("✅ Hyperliquid API connection successful")
            else:
                raise Exception(f"API connection failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hyperliquid client: {e}")
            raise
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.emergency_stop = True
        self.running = False
    
    def get_current_price(self, coin: str) -> float:
        """Get current price for a coin"""
        try:
            payload = {
                "type": "l2Book",
                "coin": coin
            }
            response = requests.post('https://api.hyperliquid.xyz/info', json=payload)
            data = response.json()
            
            if 'levels' in data and len(data['levels']) > 0:
                # Get mid price
                bids = [level for level in data['levels'] if float(level['n']) > 0]
                asks = [level for level in data['levels'] if float(level['n']) < 0]
                
                if bids and asks:
                    best_bid = max(bids, key=lambda x: float(x['px']))['px']
                    best_ask = min(asks, key=lambda x: float(x['px']))['px']
                    return (float(best_bid) + float(best_ask)) / 2
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Failed to get price for {coin}: {e}")
            return 0.0
    
    def get_account_balance(self) -> float:
        """Get account balance from Hyperliquid"""
        try:
            # This would be implemented with actual Hyperliquid API calls
            # For now, return tracked balance
            return self.metrics['account_balance']
        except Exception as e:
            logger.error(f"❌ Failed to get account balance: {e}")
            return self.metrics['account_balance']
    
    def calculate_position_size(self, price: float, atr: float) -> float:
        """Calculate position size based on risk management"""
        account_balance = self.get_account_balance()
        risk_amount = account_balance * self.config['risk_config']['risk_pct']
        
        # Position size = Risk Amount / (ATR * 1) since we use 1 ATR stop
        position_size = risk_amount / atr
        
        # Apply limits
        min_size_usd = self.config['risk_config']['min_position_size_usd']
        max_size_pct = self.config['risk_config']['max_position_size_pct']
        
        min_size = min_size_usd / price
        max_size = (account_balance * max_size_pct) / price
        
        position_size = max(min_size, min(position_size, max_size))
        
        return position_size
    
    def get_market_data(self, coin: str) -> Dict:
        """Get comprehensive market data for a coin"""
        try:
            # Get current price and order book
            price = self.get_current_price(coin)
            
            # Get recent candles for ATR calculation
            payload = {
                "type": "candles",
                "coin": coin,
                "interval": "1m",
                "startTime": int((datetime.now() - timedelta(hours=2)).timestamp() * 1000),
                "endTime": int(datetime.now().timestamp() * 1000)
            }
            
            response = requests.post('https://api.hyperliquid.xyz/info', json=payload)
            candles = response.json()
            
            if not candles:
                return {}
            
            # Calculate ATR
            highs = [float(c['h']) for c in candles[-20:]]
            lows = [float(c['l']) for c in candles[-20:]]
            closes = [float(c['c']) for c in candles[-20:]]
            
            if len(highs) < 2:
                return {}
            
            # Simple ATR calculation
            true_ranges = []
            for i in range(1, len(highs)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                true_ranges.append(tr)
            
            atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.mean(true_ranges)
            
            return {
                'price': price,
                'atr': atr,
                'candles': candles,
                'high': highs[-1],
                'low': lows[-1],
                'close': closes[-1],
                'volume': float(candles[-1]['v']) if candles else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get market data for {coin}: {e}")
            return {}
    
    def get_ensemble_prediction(self, coin: str, features: np.ndarray) -> float:
        """Get ensemble prediction for a coin"""
        if coin not in self.models or not self.models[coin]:
            return 0.5
        
        predictions = []
        weights = self.config['signals']['model_weights']
        
        # Flatten features for tree-based models
        features_flat = features.reshape(1, -1)
        
        for model_name, model in self.models[coin].items():
            if model_name == 'ensemble':
                continue
            
            try:
                if model_name == 'lightgbm':
                    pred = model.predict(features_flat)[0]
                    predictions.append((pred, weights.get('lightgbm', 0.3)))
                elif model_name == 'catboost':
                    pred = model.predict_proba(features_flat)[0, 1]
                    predictions.append((pred, weights.get('catboost', 0.3)))
                elif model_name == 'bilstm':
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features).unsqueeze(0)
                        outputs = model(features_tensor)
                        pred = F.softmax(outputs, dim=1)[0, 1].numpy()
                        predictions.append((pred, weights.get('bilstm', 0.25)))
                elif model_name == 'transformer':
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features).unsqueeze(0)
                        outputs = model(features_tensor)
                        pred = F.softmax(outputs, dim=1)[0, 1].numpy()
                        predictions.append((pred, weights.get('transformer', 0.15)))
            except Exception as e:
                logger.error(f"❌ Error getting prediction from {model_name}: {e}")
                continue
        
        if not predictions:
            return 0.5
        
        # Weighted average
        total_weight = sum(weight for _, weight in predictions)
        if total_weight == 0:
            return 0.5
        
        weighted_pred = sum(pred * weight for pred, weight in predictions) / total_weight
        return weighted_pred
    
    def should_enter_trade(self, coin: str, prediction: float, market_data: Dict) -> bool:
        """Determine if we should enter a trade"""
        # Check prediction threshold
        if prediction < self.config['signals']['entry_threshold']:
            return False
        
        # Check if we already have max positions
        current_positions = len([p for p in self.positions.values() if p['status'] == 'open'])
        if current_positions >= self.config['risk_config']['max_positions']:
            return False
        
        # Check if we already have a position in this coin
        if coin in self.positions and self.positions[coin]['status'] == 'open':
            return False
        
        # Check market conditions
        if not market_data or market_data.get('atr', 0) <= 0:
            return False
        
        # Check daily loss limit
        if self.metrics['daily_pnl'] <= self.config['risk_config']['daily_loss_halt']:
            logger.warning(f"⚠️ Daily loss limit reached: {self.metrics['daily_pnl']}")
            return False
        
        # Check drawdown limit
        if self.metrics['current_drawdown'] >= self.config['risk_config']['weekly_drawdown_halt']:
            logger.warning(f"⚠️ Drawdown limit reached: {self.metrics['current_drawdown']}")
            return False
        
        return True
    
    def place_order(self, coin: str, side: str, size: float, price: float, order_type: str = 'limit') -> Dict:
        """Place order on Hyperliquid (placeholder implementation)"""
        try:
            # This would be implemented with actual Hyperliquid API calls
            # For now, simulate order placement
            order_id = f"{coin}_{side}_{int(time.time())}"
            
            # Simulate order execution
            executed_price = price * (1 + np.random.normal(0, 0.0001))  # Small slippage
            
            order = {
                'order_id': order_id,
                'coin': coin,
                'side': side,
                'size': size,
                'price': executed_price,
                'status': 'filled',
                'timestamp': datetime.now()
            }
            
            logger.info(f"📈 {side.upper()} order placed: {coin} {size:.4f} @ ${executed_price:.4f}")
            return order
            
        except Exception as e:
            logger.error(f"❌ Failed to place order: {e}")
            return {}
    
    def enter_position(self, coin: str, prediction: float, market_data: Dict):
        """Enter a new position"""
        try:
            price = market_data['price']
            atr = market_data['atr']
            
            # Calculate position size
            size = self.calculate_position_size(price, atr)
            
            # Calculate stop loss and take profit
            stop_loss = price - atr
            take_profit = price + (atr * self.config['risk_config']['risk_reward_ratio'])
            
            # Place entry order
            entry_order = self.place_order(coin, 'buy', size, price)
            
            if not entry_order:
                return
            
            # Place stop loss
            stop_order = self.place_order(coin, 'sell', size, stop_loss, 'stop')
            
            # Place take profit
            tp_order = self.place_order(coin, 'sell', size, take_profit, 'limit')
            
            # Record position
            position = {
                'coin': coin,
                'entry_price': entry_order['price'],
                'size': size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'status': 'open',
                'prediction': prediction,
                'atr': atr,
                'entry_order': entry_order,
                'stop_order': stop_order,
                'tp_order': tp_order
            }
            
            self.positions[coin] = position
            
            logger.info(f"🎯 Position entered: {coin}")
            logger.info(f"   Entry: ${entry_order['price']:.4f}")
            logger.info(f"   Stop Loss: ${stop_loss:.4f}")
            logger.info(f"   Take Profit: ${take_profit:.4f}")
            logger.info(f"   Size: {size:.4f}")
            logger.info(f"   Prediction: {prediction:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to enter position for {coin}: {e}")
    
    def check_positions(self):
        """Check and manage open positions"""
        for coin, position in list(self.positions.items()):
            if position['status'] != 'open':
                continue
            
            try:
                # Get current price
                current_price = self.get_current_price(coin)
                if current_price <= 0:
                    continue
                
                # Check if stop loss or take profit hit
                if current_price <= position['stop_loss']:
                    self.close_position(coin, current_price, 'stop_loss')
                elif current_price >= position['take_profit']:
                    self.close_position(coin, current_price, 'take_profit')
                
            except Exception as e:
                logger.error(f"❌ Error checking position for {coin}: {e}")
    
    def close_position(self, coin: str, exit_price: float, reason: str):
        """Close a position"""
        if coin not in self.positions:
            return
        
        position = self.positions[coin]
        if position['status'] != 'open':
            return
        
        try:
            # Calculate P&L
            entry_price = position['entry_price']
            size = position['size']
            
            pnl_usd = (exit_price - entry_price) * size
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_r = pnl_usd / (position['atr'] * size)  # P&L in R terms
            
            # Update position
            position['status'] = 'closed'
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['pnl_usd'] = pnl_usd
            position['pnl_pct'] = pnl_pct
            position['pnl_r'] = pnl_r
            position['exit_reason'] = reason
            
            # Update metrics
            self.metrics['total_trades'] += 1
            self.metrics['total_pnl_r'] += pnl_r
            self.metrics['total_pnl_pct'] += pnl_pct
            self.metrics['account_balance'] += pnl_usd
            self.metrics['daily_pnl'] += pnl_r
            self.metrics['weekly_pnl'] += pnl_r
            self.metrics['monthly_pnl'] += pnl_r
            
            if pnl_r > 0:
                self.metrics['winning_trades'] += 1
                self.metrics['consecutive_losses'] = 0
            else:
                self.metrics['consecutive_losses'] += 1
            
            # Update drawdown
            if pnl_r < 0:
                self.metrics['current_drawdown'] += abs(pnl_r)
                self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], self.metrics['current_drawdown'])
            else:
                self.metrics['current_drawdown'] = max(0, self.metrics['current_drawdown'] - pnl_r)
            
            logger.info(f"🔄 Position closed: {coin}")
            logger.info(f"   Exit: ${exit_price:.4f} ({reason})")
            logger.info(f"   P&L: ${pnl_usd:.2f} ({pnl_r:.2f}R)")
            logger.info(f"   Account: ${self.metrics['account_balance']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error closing position for {coin}: {e}")
    
    def get_trading_signals(self) -> Dict[str, float]:
        """Get trading signals for all coins"""
        signals = {}
        
        for coin in self.config['trading_pairs']['primary']:
            coin_symbol = coin['symbol']
            
            try:
                # Get market data
                market_data = self.get_market_data(coin_symbol)
                if not market_data:
                    continue
                
                # Get features (simplified - in real implementation, would use full feature pipeline)
                # For now, use basic features
                features = np.array([
                    market_data['price'],
                    market_data['atr'],
                    market_data['volume'],
                    market_data['high'],
                    market_data['low'],
                    market_data['close']
                ])
                
                # Pad to expected size (would normally be done by feature pipeline)
                if len(features) < 100:  # Assuming 100 features
                    features = np.pad(features, (0, 100 - len(features)), 'constant')
                
                # Reshape for sequence model
                features = features[:100].reshape(1, 100)  # Simplified
                
                # Get prediction
                prediction = self.get_ensemble_prediction(coin_symbol, features)
                signals[coin_symbol] = prediction
                
            except Exception as e:
                logger.error(f"❌ Error getting signal for {coin_symbol}: {e}")
                continue
        
        return signals
    
    def run_trading_cycle(self):
        """Run one trading cycle"""
        try:
            # Get trading signals
            signals = self.get_trading_signals()
            
            # Check for new entry opportunities
            for coin, prediction in signals.items():
                market_data = self.get_market_data(coin)
                
                if self.should_enter_trade(coin, prediction, market_data):
                    self.enter_position(coin, prediction, market_data)
            
            # Check existing positions
            self.check_positions()
            
            # Log status
            if self.metrics['total_trades'] % 10 == 0 and self.metrics['total_trades'] > 0:
                self.log_performance()
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
    
    def log_performance(self):
        """Log current performance metrics"""
        win_rate = self.metrics['winning_trades'] / self.metrics['total_trades'] if self.metrics['total_trades'] > 0 else 0
        expectancy = self.metrics['total_pnl_r'] / self.metrics['total_trades'] if self.metrics['total_trades'] > 0 else 0
        
        logger.info(f"📊 Performance Update:")
        logger.info(f"   Trades: {self.metrics['total_trades']}")
        logger.info(f"   Win Rate: {win_rate:.1%}")
        logger.info(f"   Expectancy: {expectancy:.3f}R")
        logger.info(f"   Total P&L: {self.metrics['total_pnl_r']:.2f}R")
        logger.info(f"   Account: ${self.metrics['account_balance']:.2f}")
        logger.info(f"   Drawdown: {self.metrics['current_drawdown']:.2f}R")
        logger.info(f"   Consecutive Losses: {self.metrics['consecutive_losses']}")
    
    def check_emergency_conditions(self) -> bool:
        """Check if emergency stop conditions are met"""
        # Check emergency stop flag
        if self.emergency_stop:
            return True
        
        # Check environment variable
        if os.getenv('EMERGENCY_STOP', '').lower() == 'true':
            return True
        
        # Check drawdown limits
        if self.metrics['current_drawdown'] >= 0.10:  # 10% drawdown
            logger.critical("🚨 EMERGENCY STOP: Drawdown limit exceeded")
            return True
        
        # Check consecutive losses
        if self.metrics['consecutive_losses'] >= 10:
            logger.critical("🚨 EMERGENCY STOP: Too many consecutive losses")
            return True
        
        return False
    
    async def run(self, phase: int = 1):
        """Main trading loop"""
        self.current_phase = phase
        self.running = True
        
        logger.info(f"🚀 Starting Elite Double-Up Trader - Phase {phase}")
        logger.info(f"💰 Starting balance: ${self.metrics['account_balance']:.2f}")
        logger.info(f"🎯 Target: ${self.metrics['account_balance'] * 2:.2f}")
        
        try:
            while self.running:
                # Check emergency conditions
                if self.check_emergency_conditions():
                    logger.critical("🚨 EMERGENCY STOP TRIGGERED")
                    break
                
                # Run trading cycle
                self.run_trading_cycle()
                
                # Wait before next cycle
                await asyncio.sleep(60)  # 1 minute between cycles
                
        except KeyboardInterrupt:
            logger.info("👋 Keyboard interrupt received")
        except Exception as e:
            logger.error(f"❌ Fatal error in main loop: {e}")
        finally:
            logger.info("🛑 Shutting down trader...")
            # Close all positions
            for coin in list(self.positions.keys()):
                if self.positions[coin]['status'] == 'open':
                    current_price = self.get_current_price(coin)
                    if current_price > 0:
                        self.close_position(coin, current_price, 'shutdown')
            
            # Final performance log
            self.log_performance()
            logger.info("✅ Elite Double-Up Trader shutdown complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Elite Double-Up Trading System')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                       help='Deployment phase (1, 2, or 3)')
    parser.add_argument('--config', type=str, default='deployment_config_double_up.yaml',
                       help='Configuration file path')
    parser.add_argument('--paper', action='store_true',
                       help='Run in paper trading mode')
    
    args = parser.parse_args()
    
    # Initialize trader
    trader = EliteDoubleUpTrader(args.config)
    
    if args.paper:
        logger.info("📄 Running in PAPER TRADING mode")
    
    # Run trader
    try:
        asyncio.run(trader.run(args.phase))
    except Exception as e:
        logger.error(f"❌ Failed to start trader: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 