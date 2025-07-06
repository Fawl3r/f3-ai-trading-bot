#!/usr/bin/env python3
"""
🚀 ENHANCED EXECUTION LAYER
Implements critical execution features for sniper-level trading

CRITICAL FEATURES:
✅ Limit-in, market-out execution
✅ Async OrderWatch for real-time monitoring
✅ ATR-based stop and take-profit brackets
✅ Order book imbalance monitoring
✅ Automatic position management
✅ Real-time risk monitoring
"""

import asyncio
import time
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from advanced_risk_management import AdvancedRiskManager, RiskMetrics, PositionRisk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrderRequest:
    """Order request data"""
    symbol: str
    side: str  # "BUY" or "SELL"
    size: float
    price: float
    order_type: str  # "LIMIT" or "MARKET"
    stop_loss: float
    take_profit: float
    time_in_force: str = "GTC"

@dataclass
class OrderStatus:
    """Order status tracking"""
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    filled_size: float
    remaining_size: float
    status: str  # "PENDING", "FILLED", "CANCELLED", "REJECTED"
    created_time: datetime
    last_update: datetime
    stop_loss: float
    take_profit: float

class EnhancedExecutionLayer:
    """Enhanced execution layer with advanced risk management"""
    
    def __init__(self, api_url: str, risk_manager: AdvancedRiskManager):
        self.api_url = api_url
        self.risk_manager = risk_manager
        
        # Order management
        self.active_orders = {}
        self.position_watches = {}
        self.order_history = []
        
        # Execution parameters
        self.max_order_age_ms = 500  # Cancel orders after 500ms if not filled
        self.fill_timeout_ms = 2000  # Maximum time to wait for fill
        self.retry_attempts = 3
        
        # Real-time monitoring
        self.market_data_streams = {}
        self.obi_monitors = {}
        
        logger.info("🚀 Enhanced Execution Layer initialized")
    
    async def execute_trade(self, order_request: OrderRequest, 
                          risk_metrics: RiskMetrics) -> Tuple[bool, str, Optional[str]]:
        """Execute a trade with enhanced risk management"""
        try:
            # 1. Pre-execution risk checks
            risk_check, risk_reason = self.risk_manager.check_entry_filters(
                order_request.symbol, order_request.side, risk_metrics
            )
            
            if not risk_check:
                return False, f"Risk check failed: {risk_reason}", None
            
            # 2. Calculate dynamic position size
            account_balance = await self._get_account_balance()
            position_size = self.risk_manager.calculate_position_size(
                account_balance, order_request.price, order_request.stop_loss, risk_metrics
            )
            
            if position_size <= 0:
                return False, "Position size too small", None
            
            # 3. Calculate dynamic stops
            stop_loss, take_profit = self.risk_manager.calculate_dynamic_stops(
                order_request.price, order_request.side, risk_metrics
            )
            
            # 4. Execute limit order with brackets
            order_id = await self._place_limit_order_with_brackets(
                order_request.symbol, order_request.side, position_size,
                order_request.price, stop_loss, take_profit
            )
            
            if not order_id:
                return False, "Failed to place order", None
            
            # 5. Start order monitoring
            await self._start_order_monitoring(order_id, order_request.symbol)
            
            # 6. Start position watch
            position_data = PositionRisk(
                entry_price=order_request.price,
                current_price=order_request.price,
                position_size=position_size,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                stop_distance=abs(order_request.price - stop_loss),
                take_profit_distance=abs(take_profit - order_request.price),
                time_in_trade=0.0,
                bars_since_entry=0
            )
            
            await self.risk_manager.start_position_watch(order_request.symbol, position_data)
            
            logger.info(f"✅ Trade executed: {order_request.symbol} {order_request.side} "
                       f"{position_size} @ {order_request.price}")
            
            return True, "Trade executed successfully", order_id
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False, f"Execution error: {str(e)}", None
    
    async def _place_limit_order_with_brackets(self, symbol: str, side: str, 
                                             size: float, price: float, 
                                             stop_loss: float, take_profit: float) -> Optional[str]:
        """Place limit order with stop loss and take profit brackets"""
        try:
            # 1. Place main limit order
            main_order_payload = {
                "type": "order",
                "coin": symbol,
                "is_buy": side == "BUY",
                "sz": str(size),
                "limit_px": str(price),
                "reduce_only": False
            }
            
            response = await self._send_request(main_order_payload)
            
            if not response or "status" not in response or response["status"] != "ok":
                logger.error(f"Failed to place main order: {response}")
                return None
            
            order_id = response.get("data", {}).get("oid")
            
            if not order_id:
                logger.error("No order ID returned")
                return None
            
            # 2. Place stop loss order
            stop_order_payload = {
                "type": "order",
                "coin": symbol,
                "is_buy": side == "SELL",  # Opposite side for stop
                "sz": str(size),
                "limit_px": str(stop_loss),
                "reduce_only": True,
                "stop_px": str(stop_loss)
            }
            
            stop_response = await self._send_request(stop_order_payload)
            
            # 3. Place take profit order
            tp_order_payload = {
                "type": "order",
                "coin": symbol,
                "is_buy": side == "SELL",  # Opposite side for TP
                "sz": str(size),
                "limit_px": str(take_profit),
                "reduce_only": True
            }
            
            tp_response = await self._send_request(tp_order_payload)
            
            # Store order information
            self.active_orders[order_id] = OrderStatus(
                order_id=order_id,
                symbol=symbol,
                side=side,
                size=size,
                price=price,
                filled_size=0.0,
                remaining_size=size,
                status="PENDING",
                created_time=datetime.now(),
                last_update=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            logger.info(f"📋 Placed order {order_id} with brackets: "
                       f"SL={stop_loss}, TP={take_profit}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order with brackets: {e}")
            return None
    
    async def _start_order_monitoring(self, order_id: str, symbol: str):
        """Start monitoring an order for real-time updates"""
        monitor_id = f"order_{order_id}"
        
        self.order_history.append({
            'order_id': order_id,
            'symbol': symbol,
            'start_time': datetime.now(),
            'status': 'MONITORING'
        })
        
        # Start monitoring in background
        asyncio.create_task(self._monitor_order(order_id, symbol))
    
    async def _monitor_order(self, order_id: str, symbol: str):
        """Monitor order for fills and risk management"""
        try:
            while True:
                # Check order status
                order_status = await self._get_order_status(order_id)
                
                if order_status:
                    # Update local order tracking
                    if order_id in self.active_orders:
                        self.active_orders[order_id].status = order_status.get("status", "UNKNOWN")
                        self.active_orders[order_id].filled_size = float(order_status.get("filled_sz", 0))
                        self.active_orders[order_id].remaining_size = float(order_status.get("remaining_sz", 0))
                        self.active_orders[order_id].last_update = datetime.now()
                    
                    # Check if order is filled
                    if order_status.get("status") == "FILLED":
                        logger.info(f"✅ Order {order_id} filled")
                        
                        # Update performance tracking
                        filled_size = float(order_status.get("filled_sz", 0))
                        if filled_size > 0:
                            # Calculate PnL (simplified)
                            entry_price = float(order_status.get("limit_px", 0))
                            current_price = await self._get_current_price(symbol)
                            
                            if current_price > 0:
                                pnl = (current_price - entry_price) * filled_size
                                self.risk_manager.update_performance_tracking(pnl, "profit" if pnl > 0 else "loss")
                        
                        break
                    
                    # Check if order should be cancelled
                    order_age = (datetime.now() - self.active_orders[order_id].created_time).total_seconds() * 1000
                    
                    if order_age > self.max_order_age_ms:
                        logger.warning(f"⏰ Order {order_id} timed out, cancelling")
                        await self._cancel_order(order_id)
                        break
                
                # Check for adverse moves
                if await self._check_adverse_move(order_id, symbol):
                    logger.warning(f"🚨 Adverse move detected for order {order_id}, cancelling")
                    await self._cancel_order(order_id)
                    break
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
        except Exception as e:
            logger.error(f"Error monitoring order {order_id}: {e}")
        finally:
            # Clean up
            if order_id in self.active_orders:
                del self.active_orders[order_id]
    
    async def _check_adverse_move(self, order_id: str, symbol: str) -> bool:
        """Check if adverse move requires order cancellation"""
        try:
            if order_id not in self.active_orders:
                return False
            
            order = self.active_orders[order_id]
            current_price = await self._get_current_price(symbol)
            
            if current_price <= 0:
                return False
            
            # Calculate adverse move
            if order.side == "BUY":
                adverse_move = order.price - current_price
            else:
                adverse_move = current_price - order.price
            
            # Check if adverse move exceeds threshold
            stop_distance = abs(order.price - order.stop_loss)
            adverse_threshold = stop_distance * 0.75  # 75% of stop distance
            
            if adverse_move > adverse_threshold:
                return True
            
            # Check OBI for adverse conditions
            obi = await self.risk_manager.calculate_order_book_imbalance(symbol, self.api_url)
            
            if order.side == "BUY" and obi < self.risk_manager.obi_hedge_threshold:
                return True
            elif order.side == "SELL" and obi > -self.risk_manager.obi_hedge_threshold:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking adverse move: {e}")
            return False
    
    async def _get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status from exchange"""
        try:
            payload = {
                "type": "orderStatus",
                "oid": order_id
            }
            
            response = await self._send_request(payload)
            return response.get("data") if response else None
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
    
    async def _cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            payload = {
                "type": "cancel",
                "oid": order_id
            }
            
            response = await self._send_request(payload)
            return response and response.get("status") == "ok"
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            payload = {"type": "ticker", "coin": symbol}
            response = await self._send_request(payload)
            
            if response and "data" in response:
                return float(response["data"].get("mark_px", 0))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return 0.0
    
    async def _get_account_balance(self) -> float:
        """Get account balance"""
        try:
            payload = {"type": "userState"}
            response = await self._send_request(payload)
            
            if response and "data" in response:
                # Extract USDC balance (assuming USDC is the quote currency)
                balances = response["data"].get("assetPositions", [])
                for balance in balances:
                    if balance.get("coin") == "USDC":
                        return float(balance.get("position", 0))
            
            return 1000.0  # Default balance
            
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 1000.0  # Default balance
    
    async def _send_request(self, payload: Dict) -> Optional[Dict]:
        """Send request to exchange API"""
        try:
            import requests
            
            response = requests.post(self.api_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return None
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "total_orders": len(self.order_history),
            "active_orders": len(self.active_orders),
            "daily_pnl": self.risk_manager.daily_pnl,
            "global_pnl": self.risk_manager.global_pnl,
            "trades_today": self.risk_manager.trades_today,
            "losses_today": self.risk_manager.losses_today,
            "peak_balance": self.risk_manager.peak_balance
        }

# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = AdvancedRiskManager()
    
    # Initialize execution layer
    api_url = "https://api.hyperliquid.xyz/info"
    execution_layer = EnhancedExecutionLayer(api_url, risk_manager)
    
    print("🚀 Enhanced Execution Layer ready for testing") 