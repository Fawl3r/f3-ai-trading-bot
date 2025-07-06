#!/usr/bin/env python3
"""
Elite 100%/5% Trading System
Targets +100% monthly returns with maximum 5% drawdown
Maintains current edge (PF≈2.3, WR≈40%) with enhanced risk management
"""

import asyncio
import json
import logging
import time
import webbrowser
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
import sqlite3
import numpy as np
from dataclasses import dataclass
from prometheus_client import start_http_server

# Import our enhanced components
from risk_manager_enhanced import EnhancedRiskManager, RiskConfig, load_config_from_yaml
from monitoring.exporter import ElitePrometheusExporter
from elite_dashboard_launcher import launch_elite_dashboards

# AI Learning imports
from elite_ai_learning_system import EliteAILearningSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    side: str
    signal_type: str
    signal_strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    volume_ratio: float = 1.0
    rsi_value: float = 50.0
    has_volume_climax: bool = False
    has_rsi_divergence: bool = False

@dataclass
class Trade:
    """Trade execution record"""
    trade_id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    r_multiple: float = 0.0
    signal_type: str = ""
    signal_strength: float = 0.0
    risk_pct: float = 0.0
    status: str = "open"  # open, closed, stopped
    exit_reason: str = ""

class AssetSelector:
    """Asset selection based on PF ranking"""
    
    def __init__(self, asset_universe: List[str]):
        self.asset_universe = asset_universe
        self.performance_history = {}
        
    def update_performance(self, symbol: str, pf: float, trades_count: int):
        """Update performance metrics for asset"""
        self.performance_history[symbol] = {
            'profit_factor': pf,
            'trade_count': trades_count,
            'last_update': datetime.now(),
            'score': pf * min(trades_count / 10, 1.0)  # Weighted by trade count
        }
    
    def select_top_assets(self, max_assets: int = 2) -> List[str]:
        """Select top assets based on PF ranking"""
        if not self.performance_history:
            return self.asset_universe[:max_assets]
        
        # Sort by score descending
        sorted_assets = sorted(
            self.performance_history.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        return [asset for asset, _ in sorted_assets[:max_assets]]

class SignalGenerator:
    """Enhanced signal generation with edge-preservation tweaks"""
    
    def __init__(self, config: dict):
        self.config = config
        self.signal_filters = config.get('signals', {})
        
    def generate_signals(self, market_data: Dict) -> List[TradingSignal]:
        """Generate trading signals with enhanced filtering"""
        signals = []
        
        for symbol, data in market_data.items():
            # Check for parabolic burst signals
            if self._is_burst_time() and self.signal_filters.get('parabolic_burst', {}).get('enabled', True):
                burst_signal = self._check_parabolic_burst(symbol, data)
                if burst_signal:
                    signals.append(burst_signal)
            
            # Check for fade signals with 2-of-2 rule
            if self.signal_filters.get('fade_signals', {}).get('enabled', True):
                fade_signal = self._check_fade_signal(symbol, data)
                if fade_signal:
                    signals.append(fade_signal)
            
            # Check for breakout signals
            if self.signal_filters.get('breakout_signals', {}).get('enabled', True):
                breakout_signal = self._check_breakout_signal(symbol, data)
                if breakout_signal:
                    signals.append(breakout_signal)
        
        return signals
    
    def _is_burst_time(self) -> bool:
        """Check if current time is within burst trading hours"""
        current_hour = datetime.utcnow().hour
        start_hour = self.config.get('edge_preservation', {}).get('burst_trade_start_hour', 13)
        end_hour = self.config.get('edge_preservation', {}).get('burst_trade_end_hour', 18)
        
        return start_hour <= current_hour <= end_hour
    
    def _check_parabolic_burst(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Check for parabolic burst signal"""
        # Simplified parabolic burst detection
        price = data.get('price', 0)
        volume_ratio = data.get('volume_ratio', 1.0)
        rsi = data.get('rsi', 50)
        
        min_volume = self.signal_filters.get('parabolic_burst', {}).get('min_volume_ratio', 1.5)
        
        if volume_ratio > min_volume and rsi > 70:
            return TradingSignal(
                symbol=symbol,
                side='long',
                signal_type='parabolic_burst_long',
                signal_strength=min(volume_ratio, 4.0),
                entry_price=price,
                stop_loss=price * 0.98,
                take_profit=price * 1.08,
                timestamp=datetime.now(),
                volume_ratio=volume_ratio,
                rsi_value=rsi
            )
        
        return None
    
    def _check_fade_signal(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Check for fade signal with 2-of-2 rule"""
        price = data.get('price', 0)
        volume_ratio = data.get('volume_ratio', 1.0)
        rsi = data.get('rsi', 50)
        has_volume_climax = data.get('volume_climax', False)
        has_rsi_divergence = data.get('rsi_divergence', False)
        
        fade_config = self.signal_filters.get('fade_signals', {})
        
        # 2-of-2 rule: both conditions must be met
        if (fade_config.get('require_volume_climax', True) and not has_volume_climax):
            return None
        if (fade_config.get('require_rsi_divergence', True) and not has_rsi_divergence):
            return None
        
        # Check for overbought fade
        if rsi > 80 and has_volume_climax and has_rsi_divergence:
            return TradingSignal(
                symbol=symbol,
                side='short',
                signal_type='overbought_fade',
                signal_strength=3.0,
                entry_price=price,
                stop_loss=price * 1.02,
                take_profit=price * 0.92,
                timestamp=datetime.now(),
                volume_ratio=volume_ratio,
                rsi_value=rsi,
                has_volume_climax=has_volume_climax,
                has_rsi_divergence=has_rsi_divergence
            )
        
        return None
    
    def _check_breakout_signal(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Check for breakout signal"""
        price = data.get('price', 0)
        volume_ratio = data.get('volume_ratio', 1.0)
        bb_upper = data.get('bb_upper', price * 1.02)
        
        min_volume = self.signal_filters.get('breakout_signals', {}).get('min_volume_confirmation', 2.0)
        
        if price > bb_upper and volume_ratio > min_volume:
            return TradingSignal(
                symbol=symbol,
                side='long',
                signal_type='bb_breakout_long',
                signal_strength=min(volume_ratio, 4.0),
                entry_price=price,
                stop_loss=price * 0.98,
                take_profit=price * 1.08,
                timestamp=datetime.now(),
                volume_ratio=volume_ratio
            )
        
        return None

class Elite100_5TradingSystem:
    """Main trading system implementing 100%/5% operating manual"""
    
    def __init__(self, config_path: str = "deployment_config_100_5.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.risk_manager = EnhancedRiskManager(
            load_config_from_yaml(config_path),
            db_path="elite_100_5_risk.db"
        )
        
        self.asset_selector = AssetSelector(
            self.config['trade_frequency']['asset_universe']
        )
        
        self.signal_generator = SignalGenerator(self.config)
        
        # Initialize AI Learning System (optional)
        self.ai_learning_enabled = self.config.get('ai_learning', {}).get('enabled', True)
        if self.ai_learning_enabled:
            try:
                from elite_ai_learning_system import EliteAILearningSystem
                model_path = self.config.get('ai_learning', {}).get('model_path', 'elite_ai_learning.db')
                self.ai_system = EliteAILearningSystem(model_path)
                logger.info("🧠 AI Learning System initialized")
            except ImportError:
                logger.warning("AI Learning System not available - continuing without AI")
                self.ai_learning_enabled = False
                self.ai_system = None
        else:
            self.ai_system = None
            logger.info("AI Learning disabled in configuration")
        
        # Initialize monitoring
        self.prometheus_exporter = ElitePrometheusExporter(
            db_path="elite_100_5_trades.db",
            port=self.config['monitoring']['prometheus_port']
        )
        
        # Trading state
        self.active_trades: Dict[str, Trade] = {}
        self.daily_stats = {
            'trades_count': 0,
            'pnl_r': 0.0,
            'start_balance': 10000.0,
            'current_balance': 10000.0,
            'peak_balance': 10000.0
        }
        
        # Initialize database
        self._init_database()
        
        # Load existing trades
        self._load_trades()
        
        logger.info("Elite 100%/5% Trading System initialized")
        logger.info(f"Target: +100% monthly return with ≤5% drawdown")
        logger.info(f"Base risk: {self.risk_manager.config.base_risk_pct}%")
        logger.info(f"Max positions: {self.risk_manager.config.max_concurrent_positions}")
    
    def _init_database(self):
        """Initialize trading database"""
        conn = sqlite3.connect("elite_100_5_trades.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                size REAL,
                entry_price REAL,
                exit_price REAL,
                entry_time TEXT,
                exit_time TEXT,
                pnl REAL,
                r_multiple REAL,
                signal_type TEXT,
                signal_strength REAL,
                risk_pct REAL,
                status TEXT,
                exit_reason TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                trades_count INTEGER,
                pnl_r REAL,
                start_balance REAL,
                end_balance REAL,
                peak_balance REAL,
                drawdown_pct REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_trades(self):
        """Load active trades from database"""
        conn = sqlite3.connect("elite_100_5_trades.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trades WHERE status = 'open'")
        for row in cursor.fetchall():
            trade = Trade(
                trade_id=row[0],
                symbol=row[1],
                side=row[2],
                size=row[3],
                entry_price=row[4],
                exit_price=row[5],
                entry_time=datetime.fromisoformat(row[6]) if row[6] else None,
                exit_time=datetime.fromisoformat(row[7]) if row[7] else None,
                pnl=row[8],
                r_multiple=row[9],
                signal_type=row[10],
                signal_strength=row[11],
                risk_pct=row[12],
                status=row[13],
                exit_reason=row[14]
            )
            self.active_trades[trade.symbol] = trade
        
        conn.close()
        logger.info(f"Loaded {len(self.active_trades)} active trades")
    
    def _save_trade(self, trade: Trade):
        """Save trade to database"""
        conn = sqlite3.connect("elite_100_5_trades.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.trade_id,
            trade.symbol,
            trade.side,
            trade.size,
            trade.entry_price,
            trade.exit_price,
            trade.entry_time.isoformat() if trade.entry_time else None,
            trade.exit_time.isoformat() if trade.exit_time else None,
            trade.pnl,
            trade.r_multiple,
            trade.signal_type,
            trade.signal_strength,
            trade.risk_pct,
            trade.status,
            trade.exit_reason
        ))
        
        conn.commit()
        conn.close()
    
    def _update_daily_stats(self):
        """Update daily statistics"""
        today = datetime.now().date().isoformat()
        
        # Calculate current drawdown
        dd_pct = (self.daily_stats['peak_balance'] - self.daily_stats['current_balance']) / self.daily_stats['peak_balance'] * 100
        
        conn = sqlite3.connect("elite_100_5_trades.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO daily_stats VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            today,
            self.daily_stats['trades_count'],
            self.daily_stats['pnl_r'],
            self.daily_stats['start_balance'],
            self.daily_stats['current_balance'],
            self.daily_stats['peak_balance'],
            dd_pct
        ))
        
        conn.commit()
        conn.close()
        
        # Update risk manager
        self.risk_manager.update_equity_metrics(
            self.daily_stats['current_balance'],
            self.daily_stats['peak_balance'],
            self.daily_stats['pnl_r']
        )
    
    async def run_trading_cycle(self):
        """Main trading cycle"""
        logger.info("Starting trading cycle...")
        
        while True:
            try:
                # Update active assets
                active_assets = self.asset_selector.select_top_assets(
                    self.config['trade_frequency']['max_active_assets']
                )
                
                # Get market data (simulated)
                market_data = await self._get_market_data(active_assets)
                
                # Generate signals
                signals = self.signal_generator.generate_signals(market_data)
                
                # Process signals
                for signal in signals:
                    await self._process_signal(signal)
                
                # Update existing positions
                await self._update_positions(market_data)
                
                # Update statistics
                self._update_daily_stats()
                
                # Update risk manager
                self.risk_manager.update_metrics()
                
                # Check for asset rotation (daily)
                if datetime.now().hour == 0:  # Run at midnight
                    await self._run_asset_selection()
                
                await asyncio.sleep(60)  # Wait 1 minute
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(60)
    
    async def _get_market_data(self, symbols: List[str]) -> Dict:
        """Get market data (simulated for demo)"""
        # In real implementation, this would fetch from exchange API
        market_data = {}
        
        for symbol in symbols:
            # Simulate market data
            base_price = 100.0  # Simplified
            market_data[symbol] = {
                'price': base_price * (1 + np.random.normal(0, 0.01)),
                'volume_ratio': np.random.uniform(0.5, 3.0),
                'rsi': np.random.uniform(20, 80),
                'bb_upper': base_price * 1.02,
                'bb_lower': base_price * 0.98,
                'volume_climax': np.random.random() > 0.8,
                'rsi_divergence': np.random.random() > 0.7,
                'atr_current': base_price * 0.02,
                'atr_reference': base_price * 0.015
            }
        
        return market_data
    
    async def _process_signal(self, signal: TradingSignal):
        """Process trading signal"""
        # Check if we can trade
        if self.risk_manager.check_dynamic_halt_conditions():
            logger.info(f"Trading halted: {self.risk_manager.state.halt_reason}")
            return
        
        # Check position limits
        can_trade, reason = self.risk_manager.check_position_limits(
            signal.symbol, signal.side, signal.signal_type
        )
        
        if not can_trade:
            logger.debug(f"Cannot trade {signal.symbol}: {reason}")
            return
        
        # Calculate position size
        position_size, risk_pct = self.risk_manager.calculate_position_size(
            signal.symbol, signal.side, signal.signal_type,
            signal.signal_strength, signal.entry_price
        )
        
        # Execute trade (simulated)
        trade = await self._execute_trade(signal, position_size, risk_pct)
        
        if trade:
            self.active_trades[signal.symbol] = trade
            self._save_trade(trade)
            
            # Add to risk manager
            self.risk_manager.add_position(
                signal.symbol, signal.side, position_size,
                signal.entry_price, signal.signal_type, risk_pct
            )
            
            logger.info(f"Opened {signal.side} {signal.symbol} @ {signal.entry_price:.2f}, Size: {position_size:.4f}, Risk: {risk_pct:.3f}%")
    
    async def _execute_trade(self, signal: TradingSignal, size: float, risk_pct: float) -> Optional[Trade]:
        """Execute trade (simulated)"""
        # In real implementation, this would place orders on exchange
        
        trade_id = f"{signal.symbol}_{int(time.time())}"
        
        trade = Trade(
            trade_id=trade_id,
            symbol=signal.symbol,
            side=signal.side,
            size=size,
            entry_price=signal.entry_price,
            entry_time=datetime.now(),
            signal_type=signal.signal_type,
            signal_strength=signal.signal_strength,
            risk_pct=risk_pct,
            status='open'
        )
        
        return trade
    
    async def _update_positions(self, market_data: Dict):
        """Update existing positions"""
        for symbol, trade in list(self.active_trades.items()):
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]['price']
            
            # Calculate current P&L
            if trade.side == 'long':
                pnl = (current_price - trade.entry_price) * trade.size
            else:
                pnl = (trade.entry_price - current_price) * trade.size
            
            # Calculate R multiple
            risk_amount = self.daily_stats['current_balance'] * (trade.risk_pct / 100)
            r_multiple = pnl / risk_amount if risk_amount > 0 else 0
            
            trade.pnl = pnl
            trade.r_multiple = r_multiple
            
            # Check for exit conditions
            exit_reason = self._check_exit_conditions(trade, current_price)
            
            if exit_reason:
                await self._close_position(trade, current_price, exit_reason)
    
    def _check_exit_conditions(self, trade: Trade, current_price: float) -> Optional[str]:
        """Check if position should be closed"""
        # Stop loss
        if trade.side == 'long' and current_price <= trade.entry_price * 0.98:
            return "Stop Loss"
        elif trade.side == 'short' and current_price >= trade.entry_price * 1.02:
            return "Stop Loss"
        
        # Take profit
        if trade.side == 'long' and current_price >= trade.entry_price * 1.08:
            return "Take Profit"
        elif trade.side == 'short' and current_price <= trade.entry_price * 0.92:
            return "Take Profit"
        
        # Time-based exit (simplified)
        if trade.entry_time and (datetime.now() - trade.entry_time).total_seconds() > 14400:  # 4 hours
            return "Time Exit"
        
        return None
    
    async def _close_position(self, trade: Trade, exit_price: float, exit_reason: str):
        """Close position"""
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.status = 'closed'
        trade.exit_reason = exit_reason
        
        # Update daily stats
        self.daily_stats['trades_count'] += 1
        self.daily_stats['pnl_r'] += trade.r_multiple
        self.daily_stats['current_balance'] += trade.pnl
        
        if self.daily_stats['current_balance'] > self.daily_stats['peak_balance']:
            self.daily_stats['peak_balance'] = self.daily_stats['current_balance']
        
        # Remove from active trades
        del self.active_trades[trade.symbol]
        
        # Save trade
        self._save_trade(trade)
        
        # Remove from risk manager
        self.risk_manager.remove_position(trade.symbol, exit_price, trade.r_multiple)
        
        logger.info(f"Closed {trade.side} {trade.symbol} @ {exit_price:.2f}, P&L: {trade.pnl:.2f} ({trade.r_multiple:.2f}R)")
    
    async def _run_asset_selection(self):
        """Run nightly asset selection"""
        logger.info("Running asset selection...")
        
        # Update performance metrics for each asset
        for symbol in self.config['trade_frequency']['asset_universe']:
            # Get recent performance (simplified)
            pf = np.random.uniform(1.5, 3.0)  # Simulated
            trades_count = np.random.randint(5, 25)
            
            self.asset_selector.update_performance(symbol, pf, trades_count)
        
        # Select new top assets
        new_assets = self.asset_selector.select_top_assets(
            self.config['trade_frequency']['max_active_assets']
        )
        
        logger.info(f"Selected assets for tomorrow: {new_assets}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'name': self.config['system']['name'],
                'version': self.config['system']['version'],
                'target_monthly_return': self.config['system']['target_monthly_return'],
                'max_monthly_drawdown': self.config['system']['max_monthly_drawdown']
            },
            'daily_stats': self.daily_stats,
            'active_trades': len(self.active_trades),
            'risk_status': self.risk_manager.get_risk_status(),
            'active_assets': self.asset_selector.select_top_assets(2)
        }

async def main():
    """Main entry point"""
    try:
        print("=" * 80)
        print("🏆 ELITE 100%/5% TRADING SYSTEM")
        print("💰 Target: +100% Monthly Returns | 🛡️ Max DD: 5%")
        print("=" * 80)
        
        # Initialize system
        system = Elite100_5TradingSystem()
        
        # Start Prometheus server
        prometheus_port = system.config['monitoring']['prometheus_port']
        start_http_server(prometheus_port)
        logger.info(f"Prometheus server started on port {prometheus_port}")
        
        # Launch dashboard in browser
        launch_elite_dashboards(prometheus_port)
        
        # Display dashboard URLs
        print("\n📊 MONITORING DASHBOARDS:")
        print(f"   🔍 Prometheus Metrics: http://localhost:{prometheus_port}/metrics")
        print(f"   📈 Grafana Dashboard: http://localhost:3000 (if available)")
        print(f"   🎯 System Status: Check logs above")
        print("\n💡 TIP: Keep the Prometheus dashboard open on a spare monitor!")
        print("=" * 80)
        
        # Run trading system
        await system.run_trading_cycle()
        
    except KeyboardInterrupt:
        logger.info("System shutdown requested")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 