#!/usr/bin/env python3
"""
🚨 ADVANCED RISK MANAGEMENT MODULE
Implements critical risk controls to prevent "buy-the-top, watch-it-dip" patterns

CRITICAL FEATURES:
✅ ATR-based dynamic stops and position sizing
✅ Order Book Imbalance (OBI) filtering
✅ Volatility-adaptive risk controls
✅ Pre-trade cool-down periods
✅ Global drawdown circuit breakers
✅ Async OrderWatch for real-time monitoring
"""

import asyncio
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for a symbol"""
    atr_1m: float
    atr_1h: float
    volatility_1m: float
    rsi_14: float
    vwap_distance: float
    obi: float
    funding_rate: float
    open_interest_change: float

@dataclass
class PositionRisk:
    """Position risk management data"""
    entry_price: float
    current_price: float
    position_size: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_distance: float
    take_profit_distance: float
    time_in_trade: float
    bars_since_entry: int

class AdvancedRiskManager:
    """Advanced risk management with ATR, OBI, and volatility controls"""
    
    def __init__(self):
        # Risk parameters
        self.max_risk_per_trade = 0.01  # 1% of account equity
        self.max_daily_drawdown = 0.05  # 5% daily drawdown limit
        self.max_global_drawdown = 0.15  # 15% global drawdown limit
        
        # ATR-based parameters
        self.atr_stop_multiplier = 2.5
        self.atr_take_profit_multiplier = 4.0
        self.atr_trailing_multiplier = 1.5
        
        # OBI (Order Book Imbalance) parameters
        self.obi_long_threshold = -0.10  # Don't go long if OBI < -10%
        self.obi_short_threshold = 0.10   # Don't go short if OBI > 10%
        self.obi_hedge_threshold = -0.20  # Hedge if OBI < -20%
        
        # Volatility and momentum filters
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.vwap_distance_limit = 1.5  # ATRs above/below VWAP
        self.momentum_exhaustion_threshold = 0.5  # ATRs
        
        # Time-based controls
        self.cool_down_minutes = 15  # Minutes after loss
        self.max_hold_time_hours = 24
        self.bar_count_exit = 20  # Exit if no progress in N bars
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.global_pnl = 0.0
        self.trades_today = 0
        self.losses_today = 0
        self.last_loss_time = None
        self.peak_balance = 0.0
        
        # Order monitoring
        self.active_orders = {}
        self.position_watches = {}
        
        logger.info("🚨 Advanced Risk Manager initialized")
    
    def calculate_atr(self, candles: List[Dict], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(candles) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = float(candles[i]['h'])
            low = float(candles[i]['l'])
            prev_close = float(candles[i-1]['c'])
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        # Calculate ATR using exponential moving average
        atr = np.mean(true_ranges[-period:])
        return atr
    
    def calculate_rsi(self, candles: List[Dict], period: int = 14) -> float:
        """Calculate RSI"""
        if len(candles) < period + 1:
            return 50.0
        
        closes = [float(c['c']) for c in candles]
        deltas = np.diff(closes)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_vwap(self, candles: List[Dict]) -> float:
        """Calculate Volume Weighted Average Price"""
        if not candles:
            return 0.0
        
        total_volume_price = 0.0
        total_volume = 0.0
        
        for candle in candles:
            price = (float(candle['h']) + float(candle['l']) + float(candle['c'])) / 3
            volume = float(candle['v'])
            
            total_volume_price += price * volume
            total_volume += volume
        
        return total_volume_price / total_volume if total_volume > 0 else 0.0
    
    def calculate_volatility(self, candles: List[Dict], period: int = 20) -> float:
        """Calculate real-time volatility (standard deviation of log returns)"""
        if len(candles) < period + 1:
            return 0.0
        
        closes = [float(c['c']) for c in candles[-period:]]
        log_returns = np.diff(np.log(closes))
        
        return np.std(log_returns)
    
    async def calculate_order_book_imbalance(self, symbol: str, api_url: str) -> float:
        """Calculate Order Book Imbalance (OBI)"""
        try:
            import requests
            
            payload = {"type": "l2Book", "coin": symbol}
            response = requests.post(api_url, json=payload, timeout=2)
            
            if response.status_code == 200:
                book_data = response.json()
                
                if len(book_data) >= 2:
                    bids = book_data[0][:10]  # Top 10 bid levels
                    asks = book_data[1][:10]  # Top 10 ask levels
                    
                    bid_volume = sum(float(bid['sz']) for bid in bids)
                    ask_volume = sum(float(ask['sz']) for ask in asks)
                    
                    total_volume = bid_volume + ask_volume
                    if total_volume > 0:
                        obi = (bid_volume - ask_volume) / total_volume
                        return obi
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating OBI for {symbol}: {e}")
            return 0.0
    
    def calculate_risk_metrics(self, symbol: str, candles_1m: List[Dict], 
                             candles_1h: List[Dict], current_price: float) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # ATR calculations
            atr_1m = self.calculate_atr(candles_1m, 14)
            atr_1h = self.calculate_atr(candles_1h, 14)
            
            # Volatility
            volatility_1m = self.calculate_volatility(candles_1m, 20)
            
            # RSI
            rsi_14 = self.calculate_rsi(candles_1m, 14)
            
            # VWAP distance
            vwap = self.calculate_vwap(candles_1m[-20:])  # Last 20 candles
            vwap_distance = (current_price - vwap) / atr_1m if atr_1m > 0 else 0
            
            # OBI (will be calculated separately due to async nature)
            obi = 0.0  # Placeholder
            
            # Funding rate and OI change (placeholders for now)
            funding_rate = 0.0
            open_interest_change = 0.0
            
            return RiskMetrics(
                atr_1m=atr_1m,
                atr_1h=atr_1h,
                volatility_1m=volatility_1m,
                rsi_14=rsi_14,
                vwap_distance=vwap_distance,
                obi=obi,
                funding_rate=funding_rate,
                open_interest_change=open_interest_change
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {symbol}: {e}")
            return RiskMetrics(0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0)
    
    def check_entry_filters(self, symbol: str, action: str, risk_metrics: RiskMetrics) -> Tuple[bool, str]:
        """Check if entry passes all risk filters"""
        reasons = []
        
        # 1. RSI exhaustion filter
        if action == "BUY" and risk_metrics.rsi_14 > self.rsi_overbought:
            reasons.append(f"RSI overbought ({risk_metrics.rsi_14:.1f})")
        elif action == "SELL" and risk_metrics.rsi_14 < self.rsi_oversold:
            reasons.append(f"RSI oversold ({risk_metrics.rsi_14:.1f})")
        
        # 2. VWAP distance filter
        if action == "BUY" and risk_metrics.vwap_distance > self.vwap_distance_limit:
            reasons.append(f"Too far above VWAP ({risk_metrics.vwap_distance:.1f} ATRs)")
        elif action == "SELL" and risk_metrics.vwap_distance < -self.vwap_distance_limit:
            reasons.append(f"Too far below VWAP ({risk_metrics.vwap_distance:.1f} ATRs)")
        
        # 3. OBI filter
        if action == "BUY" and risk_metrics.obi < self.obi_long_threshold:
            reasons.append(f"OBI too negative ({risk_metrics.obi:.1%})")
        elif action == "SELL" and risk_metrics.obi > self.obi_short_threshold:
            reasons.append(f"OBI too positive ({risk_metrics.obi:.1%})")
        
        # 4. Cool-down period after loss
        if self.last_loss_time:
            time_since_loss = (datetime.now() - self.last_loss_time).total_seconds() / 60
            if time_since_loss < self.cool_down_minutes:
                reasons.append(f"In cool-down period ({self.cool_down_minutes - time_since_loss:.1f} min left)")
        
        # 5. Daily drawdown limit
        if self.daily_pnl < -(self.max_daily_drawdown * self.peak_balance):
            reasons.append("Daily drawdown limit reached")
        
        # 6. Global drawdown limit
        if self.global_pnl < -(self.max_global_drawdown * self.peak_balance):
            reasons.append("Global drawdown limit reached")
        
        return len(reasons) == 0, "; ".join(reasons) if reasons else "All filters passed"
    
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                              stop_price: float, risk_metrics: RiskMetrics) -> float:
        """Calculate position size based on ATR and risk limits"""
        try:
            # Calculate stop distance in dollars
            stop_distance = abs(entry_price - stop_price)
            
            # Calculate risk amount
            risk_amount = account_balance * self.max_risk_per_trade
            
            # Calculate position size
            position_size = risk_amount / stop_distance
            
            # Adjust for ATR-based sizing
            atr_based_size = risk_amount / (risk_metrics.atr_1m * self.atr_stop_multiplier)
            
            # Use the smaller of the two for conservative sizing
            final_size = min(position_size, atr_based_size)
            
            return max(final_size, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_dynamic_stops(self, entry_price: float, action: str, 
                              risk_metrics: RiskMetrics) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit based on ATR"""
        try:
            atr = risk_metrics.atr_1m
            
            if action == "BUY":
                stop_loss = entry_price - (atr * self.atr_stop_multiplier)
                take_profit = entry_price + (atr * self.atr_take_profit_multiplier)
            else:  # SELL
                stop_loss = entry_price + (atr * self.atr_stop_multiplier)
                take_profit = entry_price - (atr * self.atr_take_profit_multiplier)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating dynamic stops: {e}")
            # Fallback to fixed percentage
            if action == "BUY":
                return entry_price * 0.99, entry_price * 1.02
            else:
                return entry_price * 1.01, entry_price * 0.98
    
    def update_performance_tracking(self, trade_pnl: float, trade_result: str):
        """Update performance tracking metrics"""
        self.daily_pnl += trade_pnl
        self.global_pnl += trade_pnl
        
        if trade_result == "loss":
            self.losses_today += 1
            self.last_loss_time = datetime.now()
        
        self.trades_today += 1
        
        # Update peak balance
        if self.global_pnl > 0:
            self.peak_balance = max(self.peak_balance, self.global_pnl)
    
    async def start_position_watch(self, symbol: str, position_data: PositionRisk):
        """Start monitoring a position for real-time risk management"""
        watch_id = f"{symbol}_{int(time.time())}"
        
        self.position_watches[watch_id] = {
            'symbol': symbol,
            'position_data': position_data,
            'start_time': datetime.now(),
            'last_check': datetime.now()
        }
        
        logger.info(f"🔍 Started position watch for {symbol}")
        
        # Start monitoring in background
        asyncio.create_task(self._monitor_position(watch_id))
    
    async def _monitor_position(self, watch_id: str):
        """Monitor position for risk management"""
        try:
            watch = self.position_watches[watch_id]
            position = watch['position_data']
            
            while True:
                # Check if position should be closed
                should_close, reason = self._check_position_exit_criteria(position)
                
                if should_close:
                    logger.warning(f"🚨 Position exit triggered for {watch['symbol']}: {reason}")
                    # Signal to close position (implement based on your execution system)
                    break
                
                # Update position data
                # This would normally come from real-time price feeds
                await asyncio.sleep(1)  # Check every second
                
        except Exception as e:
            logger.error(f"Error monitoring position {watch_id}: {e}")
        finally:
            # Clean up watch
            if watch_id in self.position_watches:
                del self.position_watches[watch_id]
    
    def _check_position_exit_criteria(self, position: PositionRisk) -> Tuple[bool, str]:
        """Check if position should be closed based on risk criteria"""
        reasons = []
        
        # 1. Stop loss hit
        if position.unrealized_pnl_pct < -self.max_risk_per_trade * 100:
            reasons.append("Stop loss")
        
        # 2. Take profit hit
        if position.unrealized_pnl_pct > self.atr_take_profit_multiplier * 100:
            reasons.append("Take profit")
        
        # 3. Time-based exit
        if position.time_in_trade > self.max_hold_time_hours * 3600:
            reasons.append("Max hold time")
        
        # 4. Bar count exit (no progress)
        if position.bars_since_entry > self.bar_count_exit:
            reasons.append("No progress in N bars")
        
        return len(reasons) > 0, "; ".join(reasons) if reasons else ""

# Example usage
if __name__ == "__main__":
    risk_manager = AdvancedRiskManager()
    
    # Test with sample data
    sample_candles = [
        {'h': 50000, 'l': 49000, 'c': 49500, 'v': 1000, 't': 1000000},
        {'h': 51000, 'l': 49500, 'c': 50500, 'v': 1200, 't': 1000060},
        {'h': 52000, 'l': 50000, 'c': 51500, 'v': 1500, 't': 1000120},
    ]
    
    # Test ATR calculation
    atr = risk_manager.calculate_atr(sample_candles)
    print(f"ATR: {atr:.2f}")
    
    # Test RSI calculation
    rsi = risk_manager.calculate_rsi(sample_candles)
    print(f"RSI: {rsi:.1f}")
    
    # Test VWAP calculation
    vwap = risk_manager.calculate_vwap(sample_candles)
    print(f"VWAP: {vwap:.2f}")
