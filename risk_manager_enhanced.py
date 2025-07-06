#!/usr/bin/env python3
"""
Enhanced Risk Manager - 100%/5% Operating Manual Implementation
Targets +100% monthly returns while capping worst-case drawdown at 5%
Maintains current edge (PF≈2.3, WR≈40%) with sophisticated position sizing
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import numpy as np
from dataclasses import dataclass
from prometheus_client import Counter, Gauge, Histogram

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
RISK_EVENTS = Counter('risk_events_total', 'Risk management events', ['event_type'])
RISK_PCT_LIVE = Gauge('risk_pct_live', 'Current live risk percentage')
EQUITY_DD_PCT = Gauge('equity_dd_pct', 'Current equity drawdown percentage')
PROFIT_FACTOR_30 = Gauge('profit_factor_30', '30-day rolling profit factor')
TRADE_COUNT_24H = Gauge('trade_count_24h', '24-hour trade count')
POSITION_COUNT = Gauge('position_count', 'Current open positions')
ATR_THROTTLE_ACTIVE = Gauge('atr_throttle_active', 'ATR throttle reduction factor')
EQUITY_DD_SCALER_ACTIVE = Gauge('equity_dd_scaler_active', 'Equity DD scaler active (1=active, 0=inactive)')
MODEL_SHA_GAUGE = Gauge('model_sha_deployed', 'Current model SHA hash as numeric')

@dataclass
class RiskConfig:
    """Risk management configuration"""
    base_risk_pct: float = 0.50  # Base 0.5% risk per trade
    max_risk_pct: float = 0.65   # ATR throttle cap
    atr_throttle_enabled: bool = True
    equity_dd_threshold: float = 3.5  # DD threshold for scaling
    equity_dd_scaler: float = 0.7     # Risk reduction factor
    max_concurrent_positions: int = 2
    correlation_threshold: float = 0.6
    pyramid_add_on: float = 0.3  # R multiple for add-ons (was 0.5)
    
    # Dynamic loss-halt matrix
    daily_r_halt_threshold: float = -3.0
    rolling_pf_halt_threshold: float = 1.0
    max_dd_emergency: float = 5.0
    max_dd_warning: float = 4.0
    latency_p95_threshold: float = 250.0  # ms
    
    # Edge preservation
    burst_trade_start_hour: int = 13  # UTC
    burst_trade_end_hour: int = 18    # UTC
    burst_size_reduction: float = 0.7  # Outside burst hours
    hedge_size_reduction: float = 0.7  # For correlated positions

@dataclass
class Position:
    """Position tracking"""
    symbol: str
    side: str
    size: float
    entry_price: float
    entry_time: datetime
    risk_pct: float
    signal_type: str
    pyramid_level: int = 0

@dataclass
class RiskState:
    """Current risk state"""
    current_dd_pct: float = 0.0
    daily_r_pnl: float = 0.0
    rolling_pf_20: float = 2.0
    trade_count_24h: int = 0
    latency_p95: float = 0.0
    atr_throttle_factor: float = 1.0
    equity_dd_scaler_factor: float = 1.0
    halt_new_entries: bool = False
    halt_reason: str = ""
    halt_until: Optional[datetime] = None

class EnhancedRiskManager:
    """Enhanced risk manager implementing 100%/5% operating manual"""
    
    def __init__(self, config: RiskConfig, db_path: str = "risk_manager.db"):
        self.config = config
        self.db_path = db_path
        self.state = RiskState()
        self.positions: Dict[str, Position] = {}
        self.atr_cache: Dict[str, float] = {}
        self.atr_annual_median: Dict[str, float] = {}  # Annual median ATR reference
        self.correlation_cache: Dict[Tuple[str, str], float] = {}
        self.cancels_per_second = 0
        self.last_cancel_time = time.time()
        self.cancel_timestamps = []
        
        # Initialize database
        self._init_database()
        
        # Load state from database
        self._load_state()
        
        # Load ATR annual medians
        self._load_atr_annual_medians()
        
        logger.info("Enhanced Risk Manager initialized with 100%/5% operating manual")
        logger.info(f"Base risk: {config.base_risk_pct}%, Max concurrent: {config.max_concurrent_positions}")
    
    def _init_database(self):
        """Initialize risk management database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Risk events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_events (
                timestamp TEXT,
                event_type TEXT,
                symbol TEXT,
                details TEXT,
                risk_pct REAL,
                dd_pct REAL,
                action_taken TEXT
            )
        ''')
        
        # Risk state table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_state (
                timestamp TEXT,
                current_dd_pct REAL,
                daily_r_pnl REAL,
                rolling_pf_20 REAL,
                trade_count_24h INTEGER,
                risk_pct_live REAL,
                atr_throttle_factor REAL,
                equity_dd_scaler_factor REAL,
                halt_new_entries INTEGER,
                halt_reason TEXT
            )
        ''')
        
        # Positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                side TEXT,
                size REAL,
                entry_price REAL,
                entry_time TEXT,
                risk_pct REAL,
                signal_type TEXT,
                pyramid_level INTEGER
            )
        ''')
        
        # ATR annual medians table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS atr_annual_medians (
                symbol TEXT PRIMARY KEY,
                annual_median_atr REAL,
                last_updated TEXT,
                sample_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_state(self):
        """Load current state from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load latest risk state
        cursor.execute('''
            SELECT * FROM risk_state 
            ORDER BY timestamp DESC LIMIT 1
        ''')
        row = cursor.fetchone()
        if row:
            self.state.current_dd_pct = row[1]
            self.state.daily_r_pnl = row[2]
            self.state.rolling_pf_20 = row[3]
            self.state.trade_count_24h = row[4]
            self.state.atr_throttle_factor = row[6]
            self.state.equity_dd_scaler_factor = row[7]
            self.state.halt_new_entries = bool(row[8])
            self.state.halt_reason = row[9] or ""
        
        # Load positions
        cursor.execute('SELECT * FROM positions')
        for row in cursor.fetchall():
            pos = Position(
                symbol=row[0],
                side=row[1],
                size=row[2],
                entry_price=row[3],
                entry_time=datetime.fromisoformat(row[4]),
                risk_pct=row[5],
                signal_type=row[6],
                pyramid_level=row[7]
            )
            self.positions[row[0]] = pos
        
        conn.close()
    
    def _save_state(self):
        """Save current state to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save risk state
        cursor.execute('''
            INSERT INTO risk_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            self.state.current_dd_pct,
            self.state.daily_r_pnl,
            self.state.rolling_pf_20,
            self.state.trade_count_24h,
            self.get_live_risk_pct(),
            self.state.atr_throttle_factor,
            self.state.equity_dd_scaler_factor,
            int(self.state.halt_new_entries),
            self.state.halt_reason
        ))
        
        conn.commit()
        conn.close()
    
    def _log_risk_event(self, event_type: str, symbol: str = "", details: str = "", action_taken: str = ""):
        """Log risk management event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO risk_events VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            event_type,
            symbol,
            details,
            self.get_live_risk_pct(),
            self.state.current_dd_pct,
            action_taken
        ))
        
        conn.commit()
        conn.close()
        
        # Update Prometheus
        RISK_EVENTS.labels(event_type=event_type).inc()
        
        logger.info(f"Risk event: {event_type} - {details} - Action: {action_taken}")
    
    def _load_atr_annual_medians(self):
        """Load annual median ATR values for proper throttling"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create ATR reference table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS atr_annual_medians (
                    symbol TEXT PRIMARY KEY,
                    annual_median_atr REAL,
                    last_updated TEXT,
                    sample_count INTEGER
                )
            ''')
            
            # Load existing values
            cursor.execute('SELECT symbol, annual_median_atr FROM atr_annual_medians')
            for row in cursor.fetchall():
                self.atr_annual_median[row[0]] = row[1]
            
            conn.commit()
            conn.close()
            
            logger.info(f"Loaded annual ATR medians for {len(self.atr_annual_median)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading ATR annual medians: {e}")
    
    def update_atr_annual_median(self, symbol: str, atr_values: List[float]):
        """Update annual median ATR for proper throttling reference"""
        if len(atr_values) < 252:  # Need at least 1 year of data
            logger.warning(f"Insufficient ATR data for {symbol}: {len(atr_values)} values")
            return
        
        annual_median = np.median(atr_values)
        self.atr_annual_median[symbol] = annual_median
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO atr_annual_medians VALUES (?, ?, ?, ?)
        ''', (symbol, annual_median, datetime.now().isoformat(), len(atr_values)))
        conn.commit()
        conn.close()
        
        logger.info(f"Updated annual ATR median for {symbol}: {annual_median:.6f}")
    
    def update_atr_data(self, symbol: str, atr_current: float):
        """Update ATR data for throttling calculations using annual median reference"""
        self.atr_cache[symbol] = atr_current
        
        # Calculate ATR throttle factor using annual median
        if symbol in self.atr_annual_median and self.atr_annual_median[symbol] > 0:
            atr_ratio = atr_current / self.atr_annual_median[symbol]
            self.state.atr_throttle_factor = min(atr_ratio, self.config.max_risk_pct / self.config.base_risk_pct)
            
            ATR_THROTTLE_ACTIVE.set(self.state.atr_throttle_factor)
            
            logger.debug(f"ATR throttle for {symbol}: {atr_ratio:.3f} (current: {atr_current:.6f}, annual median: {self.atr_annual_median[symbol]:.6f})")
        else:
            logger.warning(f"No annual ATR median available for {symbol}, using default throttle")
            self.state.atr_throttle_factor = 1.0
            ATR_THROTTLE_ACTIVE.set(1.0)
    
    def update_correlation_matrix(self, symbol1: str, symbol2: str, returns_data: Dict[str, List[float]]):
        """Update correlation using 5-minute returns over last 300 bars"""
        try:
            if symbol1 not in returns_data or symbol2 not in returns_data:
                return
            
            returns1 = returns_data[symbol1][-300:]  # Last 300 bars
            returns2 = returns_data[symbol2][-300:]
            
            if len(returns1) < 100 or len(returns2) < 100:
                logger.warning(f"Insufficient return data for correlation: {symbol1}={len(returns1)}, {symbol2}={len(returns2)}")
                return
            
            # Calculate correlation
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            
            # Store both directions
            self.correlation_cache[(symbol1, symbol2)] = correlation
            self.correlation_cache[(symbol2, symbol1)] = correlation
            
            logger.debug(f"Updated correlation {symbol1}-{symbol2}: {correlation:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
    
    def update_equity_metrics(self, current_balance: float, peak_balance: float, daily_pnl_r: float):
        """Update equity and drawdown metrics with proper gauge updates"""
        # Calculate current drawdown
        if peak_balance > 0:
            self.state.current_dd_pct = (peak_balance - current_balance) / peak_balance * 100
        
        # Update daily R P&L
        self.state.daily_r_pnl = daily_pnl_r
        
        # Update Prometheus metrics
        EQUITY_DD_PCT.set(self.state.current_dd_pct)
        
        # Check for equity DD scaling and update gauge properly
        if self.state.current_dd_pct > self.config.equity_dd_threshold:
            old_scaler = self.state.equity_dd_scaler_factor
            self.state.equity_dd_scaler_factor = self.config.equity_dd_scaler
            
            # Update Prometheus gauge to show scaler is active
            EQUITY_DD_SCALER_ACTIVE.set(1.0)
            
            # Update live risk percentage gauge immediately
            live_risk = self.get_live_risk_pct()
            RISK_PCT_LIVE.set(live_risk)
            
            if old_scaler != self.state.equity_dd_scaler_factor:
                self._log_risk_event(
                    "equity_dd_scaling_activated",
                    details=f"DD {self.state.current_dd_pct:.2f}% > {self.config.equity_dd_threshold}%",
                    action_taken=f"Risk scaled to {self.state.equity_dd_scaler_factor:.2f}x, live risk now {live_risk:.3f}%"
                )
        else:
            old_scaler = self.state.equity_dd_scaler_factor
            self.state.equity_dd_scaler_factor = 1.0
            
            # Update Prometheus gauge to show scaler is inactive
            EQUITY_DD_SCALER_ACTIVE.set(0.0)
            
            # Update live risk percentage gauge
            live_risk = self.get_live_risk_pct()
            RISK_PCT_LIVE.set(live_risk)
            
            if old_scaler != 1.0:
                self._log_risk_event(
                    "equity_dd_scaling_deactivated",
                    details=f"DD {self.state.current_dd_pct:.2f}% <= {self.config.equity_dd_threshold}%",
                    action_taken=f"Risk scaling removed, live risk now {live_risk:.3f}%"
                )
    
    def update_trade_metrics(self, trade_count_24h: int, rolling_pf_20: float):
        """Update trade count and performance metrics"""
        self.state.trade_count_24h = trade_count_24h
        self.state.rolling_pf_20 = rolling_pf_20
        
        # Update Prometheus
        TRADE_COUNT_24H.set(trade_count_24h)
        PROFIT_FACTOR_30.set(rolling_pf_20)
    
    def get_live_risk_pct(self, symbol: str = "") -> float:
        """Calculate live risk percentage with all adjustments"""
        base_risk = self.config.base_risk_pct
        
        # Apply ATR throttle
        if symbol and symbol in self.atr_cache and self.config.atr_throttle_enabled:
            atr_adjusted = base_risk * self.state.atr_throttle_factor
            risk_pct = min(atr_adjusted, self.config.max_risk_pct)
        else:
            risk_pct = base_risk
        
        # Apply equity DD scaler
        risk_pct *= self.state.equity_dd_scaler_factor
        
        # Time-based adjustment for burst trades
        current_hour = datetime.utcnow().hour
        if not (self.config.burst_trade_start_hour <= current_hour <= self.config.burst_trade_end_hour):
            risk_pct *= self.config.burst_size_reduction
        
        # Update Prometheus
        RISK_PCT_LIVE.set(risk_pct)
        
        return risk_pct
    
    def check_position_limits(self, symbol: str, side: str, signal_type: str) -> Tuple[bool, str]:
        """Check position limits and correlation with enhanced correlation matrix"""
        # Check concurrent position limit
        if len(self.positions) >= self.config.max_concurrent_positions:
            return False, f"Max concurrent positions ({self.config.max_concurrent_positions}) reached"
        
        # Check for existing position in same symbol
        if symbol in self.positions:
            existing = self.positions[symbol]
            if existing.side != side:
                return False, f"Conflicting position in {symbol}: {existing.side} vs {side}"
        
        # Enhanced correlation check using actual correlation matrix
        for pos_symbol, pos in self.positions.items():
            if pos_symbol != symbol:
                correlation_key = (symbol, pos_symbol)
                if correlation_key in self.correlation_cache:
                    correlation = abs(self.correlation_cache[correlation_key])
                    if correlation > self.config.correlation_threshold:
                        return False, f"High correlation with {pos_symbol}: {correlation:.3f} > {self.config.correlation_threshold}"
        
        return True, ""
    
    def _are_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are correlated (simplified)"""
        # Major crypto correlations
        major_cryptos = ['BTC', 'ETH', 'SOL', 'AVAX', 'ARB']
        
        base1 = symbol1.split('-')[0]
        base2 = symbol2.split('-')[0]
        
        # All major cryptos are somewhat correlated
        if base1 in major_cryptos and base2 in major_cryptos:
            return True
        
        return False
    
    def calculate_position_size(self, symbol: str, side: str, signal_type: str, 
                              signal_strength: float, current_price: float) -> Tuple[float, float]:
        """Calculate position size with all risk adjustments"""
        # Get base risk percentage
        risk_pct = self.get_live_risk_pct(symbol)
        
        # Check for hedge micro-basket adjustment
        if self._is_hedge_position(symbol, side):
            risk_pct *= self.config.hedge_size_reduction
            self._log_risk_event(
                "hedge_adjustment",
                symbol=symbol,
                details=f"Hedge position detected, size reduced to {self.config.hedge_size_reduction:.2f}x",
                action_taken=f"Risk: {risk_pct:.3f}%"
            )
        
        # Calculate position size (assuming $10,000 account)
        account_balance = 10000  # Would be dynamic in live system
        risk_amount = account_balance * (risk_pct / 100)
        
        # Position size calculation would depend on stop loss distance
        # For now, using simplified calculation
        stop_distance_pct = 0.02  # 2% stop loss
        position_size = risk_amount / (current_price * stop_distance_pct)
        
        return position_size, risk_pct
    
    def _is_hedge_position(self, symbol: str, side: str) -> bool:
        """Check if this would create a hedge position"""
        # Example: SOL long + BTC short simultaneously
        base_symbol = symbol.split('-')[0]
        
        for pos_symbol, pos in self.positions.items():
            pos_base = pos_symbol.split('-')[0]
            
            # Check for specific hedge pairs
            if ((base_symbol == 'SOL' and pos_base == 'BTC') or 
                (base_symbol == 'BTC' and pos_base == 'SOL')):
                if pos.side != side:  # Opposite sides
                    return True
        
        return False
    
    def check_order_cancel_burst_guard(self) -> bool:
        """Order-cancel burst guard to prevent API throttling"""
        current_time = time.time()
        
        # Clean old timestamps (keep only last second)
        self.cancel_timestamps = [ts for ts in self.cancel_timestamps if current_time - ts < 1.0]
        
        # Add current cancel
        self.cancel_timestamps.append(current_time)
        
        # Check if we're over the limit
        if len(self.cancel_timestamps) > 40:  # 40 cancels per second limit
            logger.warning(f"Order cancel burst detected: {len(self.cancel_timestamps)} cancels in last second")
            time.sleep(0.05)  # 50ms delay
            return True
        
        return False
    
    def check_dynamic_halt_conditions(self) -> bool:
        """Check dynamic loss-halt matrix conditions - allows existing positions to trail out"""
        now = datetime.now()
        
        # Check if currently halted and if halt period has expired
        if self.state.halt_new_entries and self.state.halt_until:
            if now < self.state.halt_until:
                # Still halted for new entries, but existing positions can trail out
                logger.debug(f"New entries halted until {self.state.halt_until}, existing positions can trail out")
                return True  # Halt new entries only
            else:
                # Halt period expired, clear halt
                self.state.halt_new_entries = False
                self.state.halt_reason = ""
                self.state.halt_until = None
                self._log_risk_event(
                    "halt_cleared",
                    details="Halt period expired, resuming new entries",
                    action_taken="Normal operations resumed"
                )
        
        # Check -3R day condition
        if self.state.daily_r_pnl <= self.config.daily_r_halt_threshold:
            self._trigger_halt("daily_r_limit", f"Daily R P&L: {self.state.daily_r_pnl:.2f}R", 12)
            return True
        
        # Check rolling PF condition
        if self.state.rolling_pf_20 < self.config.rolling_pf_halt_threshold:
            self._trigger_halt("rolling_pf_low", f"20-trade PF: {self.state.rolling_pf_20:.2f}", 12)
            return True
        
        # Check emergency DD condition
        if self.state.current_dd_pct > self.config.max_dd_emergency:
            self._trigger_halt("emergency_dd", f"DD: {self.state.current_dd_pct:.2f}%", 24)
            return True
        
        # Check warning DD condition
        if self.state.current_dd_pct > self.config.max_dd_warning:
            self._log_risk_event(
                "dd_warning",
                details=f"DD {self.state.current_dd_pct:.2f}% > {self.config.max_dd_warning}%",
                action_taken="Warning issued, monitoring closely"
            )
        
        return False
    
    def _trigger_halt(self, reason: str, details: str, hours: int):
        """Trigger trading halt for new entries only - existing positions can trail out"""
        self.state.halt_new_entries = True
        self.state.halt_reason = reason
        self.state.halt_until = datetime.now() + timedelta(hours=hours)
        
        self._log_risk_event(
            "trading_halt_new_entries",
            details=f"{reason}: {details}",
            action_taken=f"New entries halted for {hours} hours until {self.state.halt_until}. Existing positions can trail out."
        )
        
        logger.warning(f"🚨 NEW ENTRIES HALTED: {reason} - {details}")
        logger.warning(f"Existing positions can continue to trail out normally")
    
    def update_model_sha(self, model_sha: str):
        """Update deployed model SHA for monitoring"""
        try:
            # Convert SHA to numeric for Prometheus (use first 8 chars as hex)
            sha_numeric = int(model_sha[:8], 16)
            MODEL_SHA_GAUGE.set(sha_numeric)
            
            self._log_risk_event(
                "model_updated",
                details=f"Model SHA: {model_sha}",
                action_taken=f"Deployed model updated, numeric gauge: {sha_numeric}"
            )
            
        except Exception as e:
            logger.error(f"Error updating model SHA: {e}")
    
    def add_position(self, symbol: str, side: str, size: float, entry_price: float, 
                    signal_type: str, risk_pct: float):
        """Add new position to tracking"""
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=datetime.now(),
            risk_pct=risk_pct,
            signal_type=signal_type
        )
        
        self.positions[symbol] = position
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, side, size, entry_price,
            position.entry_time.isoformat(),
            risk_pct, signal_type, 0
        ))
        conn.commit()
        conn.close()
        
        # Update metrics
        POSITION_COUNT.set(len(self.positions))
        
        self._log_risk_event(
            "position_opened",
            symbol=symbol,
            details=f"{side} {size:.4f} @ {entry_price:.2f}",
            action_taken=f"Risk: {risk_pct:.3f}%"
        )
    
    def remove_position(self, symbol: str, exit_price: float, pnl_r: float):
        """Remove position from tracking"""
        if symbol in self.positions:
            position = self.positions[symbol]
            del self.positions[symbol]
            
            # Remove from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
            conn.commit()
            conn.close()
            
            # Update metrics
            POSITION_COUNT.set(len(self.positions))
            
            self._log_risk_event(
                "position_closed",
                symbol=symbol,
                details=f"{position.side} exit @ {exit_price:.2f}, P&L: {pnl_r:.2f}R",
                action_taken="Position removed from tracking"
            )
    
    def can_add_pyramid(self, symbol: str, current_r: float) -> bool:
        """Check if pyramid add-on is allowed"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        # Check if we're at the pyramid threshold
        if current_r >= self.config.pyramid_add_on:
            # Check max pyramid levels (limit to 2 add-ons)
            if position.pyramid_level < 2:
                return True
        
        return False
    
    def get_risk_status(self) -> Dict:
        """Get current risk status for monitoring"""
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_pct_live': self.get_live_risk_pct(),
            'current_dd_pct': self.state.current_dd_pct,
            'daily_r_pnl': self.state.daily_r_pnl,
            'rolling_pf_20': self.state.rolling_pf_20,
            'trade_count_24h': self.state.trade_count_24h,
            'positions_count': len(self.positions),
            'halt_new_entries': self.state.halt_new_entries,
            'halt_reason': self.state.halt_reason,
            'halt_until': self.state.halt_until.isoformat() if self.state.halt_until else None,
            'atr_throttle_factor': self.state.atr_throttle_factor,
            'equity_dd_scaler_factor': self.state.equity_dd_scaler_factor,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'risk_pct': pos.risk_pct,
                    'signal_type': pos.signal_type,
                    'pyramid_level': pos.pyramid_level
                }
                for pos in self.positions.values()
            ]
        }
    
    def update_metrics(self):
        """Update all Prometheus metrics"""
        self.update_trade_metrics(self.state.trade_count_24h, self.state.rolling_pf_20)
        RISK_PCT_LIVE.set(self.get_live_risk_pct())
        EQUITY_DD_PCT.set(self.state.current_dd_pct)
        POSITION_COUNT.set(len(self.positions))
        ATR_THROTTLE_ACTIVE.set(self.state.atr_throttle_factor)
        EQUITY_DD_SCALER_ACTIVE.set(1.0 if self.state.equity_dd_scaler_factor < 1.0 else 0.0)
        
        # Save state
        self._save_state()

def load_config_from_yaml(config_path: str) -> RiskConfig:
    """Load risk configuration from YAML file"""
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        risk_section = config_data.get('risk_management', {})
        
        return RiskConfig(
            base_risk_pct=risk_section.get('base_risk_pct', 0.50),
            max_risk_pct=risk_section.get('max_risk_pct', 0.65),
            atr_throttle_enabled=risk_section.get('atr_throttle_enabled', True),
            equity_dd_threshold=risk_section.get('equity_dd_threshold', 3.5),
            equity_dd_scaler=risk_section.get('equity_dd_scaler', 0.7),
            max_concurrent_positions=risk_section.get('max_concurrent_positions', 2),
            correlation_threshold=risk_section.get('correlation_threshold', 0.6),
            pyramid_add_on=risk_section.get('pyramid_add_on', 0.3),
            daily_r_halt_threshold=risk_section.get('daily_r_halt_threshold', -3.0),
            rolling_pf_halt_threshold=risk_section.get('rolling_pf_halt_threshold', 1.0),
            max_dd_emergency=risk_section.get('max_dd_emergency', 5.0),
            max_dd_warning=risk_section.get('max_dd_warning', 4.0),
            burst_trade_start_hour=risk_section.get('burst_trade_start_hour', 13),
            burst_trade_end_hour=risk_section.get('burst_trade_end_hour', 18),
            burst_size_reduction=risk_section.get('burst_size_reduction', 0.7),
            hedge_size_reduction=risk_section.get('hedge_size_reduction', 0.7)
        )
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return RiskConfig()

if __name__ == "__main__":
    # Example usage
    config = RiskConfig()
    risk_manager = EnhancedRiskManager(config)
    
    # Simulate some operations
    risk_manager.update_equity_metrics(9500, 10000, -1.5)  # 5% DD, -1.5R day
    risk_manager.update_trade_metrics(15, 2.1)  # 15 trades, 2.1 PF
    
    # Check if we can open a position
    can_trade, reason = risk_manager.check_position_limits('BTC-USD', 'long', 'bb_breakout')
    print(f"Can trade: {can_trade}, Reason: {reason}")
    
    # Get risk status
    status = risk_manager.get_risk_status()
    print(json.dumps(status, indent=2)) 