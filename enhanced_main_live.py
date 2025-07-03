#!/usr/bin/env python3
"""
Enhanced Trading Bot with LIVE OKX Data for Simulation
Uses real market data from OKX but executes trades in simulation mode
"""

import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Dashboard and metrics
from dashboard_app import run_dashboard, init_dashboard
from metrics_collector import MetricsCollector
from trend_analyzer import TrendAnalyzer, TrendDirection, TrendStrength

# Live data components
from live_data_simulator import LiveSimulationTradingEngine, LiveDataSimulator

# Configuration
from config import Config

class LiveSimulationTradingBot:
    """Enhanced trading bot with live OKX data for simulation"""
    
    def __init__(self):
        self.config = Config()
        
        # Core live simulation components
        self.simulation_engine = LiveSimulationTradingEngine(initial_balance=10000.0)
        self.metrics_collector = MetricsCollector()
        self.trend_analyzer = TrendAnalyzer()
        
        # Bot state
        self.is_running = False
        self.start_time = None
        
        # Performance tracking
        self.last_metrics_update = None
        self.last_trend_update = None
        
        print("🌐 Live Simulation Trading Bot initialized")
        print("📡 Using REAL OKX market data")
        print("💹 Trading in SIMULATION mode (no real money)")
        print("📊 Dashboard will be available at: http://127.0.0.1:5000")
    
    def start(self):
        """Start the complete live simulation trading system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        print("🚀 Starting Live Simulation Trading Bot")
        print("=" * 60)
        print(f"⏰ Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("📈 Mode: LIVE DATA + SIMULATION TRADING")
        print("💹 Symbol: SOL-USD-SWAP")
        print("📡 Data Source: OKX WebSocket (REAL-TIME)")
        print("💰 Trading: SIMULATION ONLY")
        print("=" * 60)
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Start live simulation
            self._start_live_simulation()
            
            # Start dashboard
            self._start_dashboard()
            
            # Start metrics collection
            self._start_metrics_collection()
            
            # Start main monitoring loop
            self._run_main_loop()
            
        except Exception as e:
            print(f"Error starting bot: {e}")
            self.stop()
    
    def _initialize_components(self):
        """Initialize all components"""
        print("🔧 Initializing components...")
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Set up simulation callbacks
        self.simulation_engine.set_callback(self._on_trade_event)
        
        # Initialize dashboard
        init_dashboard(self.metrics_collector, self.trend_analyzer)
        
        print("✅ All components initialized")
    
    def _start_live_simulation(self):
        """Start the live simulation trading engine"""
        print("🌐 Starting live simulation trading engine...")
        print("📡 Connecting to OKX for real market data...")
        self.simulation_engine.start_simulation()
        print("✅ Live simulation engine started")
    
    def _start_dashboard(self):
        """Start the web dashboard in a separate thread"""
        def run_dashboard_thread():
            try:
                print("🌐 Starting web dashboard...")
                run_dashboard(host='127.0.0.1', port=5000, debug=False)
            except Exception as e:
                print(f"Dashboard error: {e}")
        
        dashboard_thread = threading.Thread(target=run_dashboard_thread)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Give dashboard time to start
        time.sleep(2)
        print("✅ Dashboard started on http://127.0.0.1:5000")
    
    def _start_metrics_collection(self):
        """Start metrics collection and updates"""
        def metrics_loop():
            while self.is_running:
                try:
                    self._update_metrics()
                    time.sleep(3)  # Update every 3 seconds for live data
                except Exception as e:
                    print(f"Metrics update error: {e}")
                    time.sleep(10)
        
        metrics_thread = threading.Thread(target=metrics_loop)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        print("✅ Live metrics collection started")
    
    def _update_metrics(self):
        """Update all metrics with live data"""
        try:
            # Get performance metrics from simulation engine
            performance = self.simulation_engine.get_performance_metrics()
            
            # Get live market data
            market_data = self.simulation_engine.get_market_data()
            
            # Update trading metrics
            self.metrics_collector.update_trading_metrics({
                'total_pnl': performance['total_pnl'],
                'win_rate': performance['win_rate'] / 100,  # Convert to decimal
                'profit_factor': performance['profit_factor'],
                'max_drawdown': performance['max_drawdown'] / 100,  # Convert to decimal
                'total_trades': performance['total_trades'],
                'active_positions': performance['active_positions'],
                'balance': performance['balance'],
                'return_pct': performance['return_pct'],
                'current_price': market_data.get('current_price', 0),
                'price_change_24h': market_data.get('price_change_24h', 0),
                'high_24h': market_data.get('high_24h', 0),
                'low_24h': market_data.get('low_24h', 0),
                'volume_24h': market_data.get('volume_24h', 0)
            })
            
            # Update trend analysis with live data
            self._update_live_trend_analysis(market_data)
            
            self.last_metrics_update = datetime.now()
            
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    def _update_live_trend_analysis(self, market_data: Dict):
        """Update trend analysis based on live market data"""
        try:
            if not market_data:
                return
            
            current_price = market_data.get('current_price', 0)
            price_change_24h = market_data.get('price_change_24h', 0)
            
            # Determine trend direction based on price change
            if price_change_24h > 2.0:
                direction = TrendDirection.STRONG_UPTREND
            elif price_change_24h > 0.5:
                direction = TrendDirection.UPTREND
            elif price_change_24h < -2.0:
                direction = TrendDirection.STRONG_DOWNTREND
            elif price_change_24h < -0.5:
                direction = TrendDirection.DOWNTREND
            else:
                direction = TrendDirection.SIDEWAYS
            
            # Determine strength based on price movement magnitude
            change_magnitude = abs(price_change_24h)
            if change_magnitude > 5.0:
                strength = TrendStrength.VERY_STRONG
            elif change_magnitude > 3.0:
                strength = TrendStrength.STRONG
            elif change_magnitude > 1.0:
                strength = TrendStrength.MODERATE
            else:
                strength = TrendStrength.WEAK
            
            # Calculate confidence based on trend consistency
            confidence = min(85.0, 50.0 + (change_magnitude * 10))
            
            # Calculate support and resistance levels
            high_24h = market_data.get('high_24h', current_price)
            low_24h = market_data.get('low_24h', current_price)
            
            # Support levels (below current price)
            support_levels = [
                current_price * 0.98,  # 2% below
                current_price * 0.95,  # 5% below
                low_24h                # 24h low
            ]
            
            # Resistance levels (above current price)
            resistance_levels = [
                current_price * 1.02,  # 2% above
                current_price * 1.05,  # 5% above
                high_24h               # 24h high
            ]
            
            # Update trend metrics
            trend_metrics = {
                'direction': direction.value,
                'strength': strength.value,
                'confidence': confidence,
                'current_price': current_price,
                'price_change_24h': price_change_24h,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'volume_24h': market_data.get('volume_24h', 0),
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'is_live_data': True
            }
            
            self.metrics_collector.update_trend_metrics(trend_metrics)
            self.last_trend_update = datetime.now()
            
        except Exception as e:
            print(f"Error updating live trend analysis: {e}")
    
    def _on_trade_event(self, trade_data: Dict):
        """Handle trade events from simulation engine"""
        try:
            if trade_data['event'] == 'trade_close':
                # Log completed trade
                self.metrics_collector.log_trade({
                    'symbol': trade_data['symbol'],
                    'side': trade_data['side'],
                    'size': trade_data['size'],
                    'price': trade_data['price'],
                    'pnl': trade_data['pnl'],
                    'status': 'closed',
                    'metadata': {
                        'close_reason': trade_data['close_reason'],
                        'timestamp': trade_data['timestamp'],
                        'live_data': True
                    }
                })
                
                # Print trade notification
                pnl_sign = "+" if trade_data['pnl'] >= 0 else ""
                print(f"💰 TRADE CLOSED: {trade_data['side'].upper()} "
                      f"@ ${trade_data['price']:.4f} "
                      f"P&L: {pnl_sign}${trade_data['pnl']:.2f}")
        
        except Exception as e:
            print(f"Error processing trade event: {e}")
    
    def _run_main_loop(self):
        """Main monitoring and status loop"""
        print("🔄 Starting main monitoring loop...")
        print("📡 Live data feed active - watching real market movements")
        print("\nPress Ctrl+C to stop gracefully...\n")
        
        try:
            while self.is_running:
                # Print periodic status
                self._print_live_status()
                
                # Sleep for status interval
                time.sleep(30)  # Print status every 30 seconds for live data
                
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received...")
            self.stop()
        except Exception as e:
            print(f"Error in main loop: {e}")
            self.stop()
    
    def _print_live_status(self):
        """Print current bot status with live data"""
        try:
            uptime = datetime.now() - self.start_time
            performance = self.simulation_engine.get_performance_metrics()
            market_data = self.simulation_engine.get_market_data()
            
            print("=" * 70)
            print(f"📊 Live Trading Bot Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            print(f"⏰ Uptime: {str(uptime).split('.')[0]}")
            print(f"📡 Data Source: OKX WebSocket (LIVE)")
            print(f"💹 Current Price: ${market_data.get('current_price', 0):.4f}")
            print(f"📈 24h Change: {market_data.get('price_change_24h', 0):+.2f}%")
            print(f"📊 24h High: ${market_data.get('high_24h', 0):.4f}")
            print(f"📉 24h Low: ${market_data.get('low_24h', 0):.4f}")
            print(f"💰 Simulation P&L: ${performance['total_pnl']:.2f}")
            print(f"📈 Return: {performance['return_pct']:.2f}%")
            print(f"🎯 Win Rate: {performance['win_rate']:.1f}%")
            print(f"🔄 Total Trades: {performance['total_trades']}")
            print(f"📊 Active Positions: {performance['active_positions']}")
            print("=" * 70)
            
        except Exception as e:
            print(f"Error printing status: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get complete performance summary with live data"""
        performance = self.simulation_engine.get_performance_metrics()
        market_data = self.simulation_engine.get_market_data()
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        return {
            'performance': performance,
            'market_data': market_data,
            'uptime': str(uptime).split('.')[0],
            'trades': self.simulation_engine.get_recent_trades(10),
            'positions': self.simulation_engine.get_active_positions(),
            'data_source': 'OKX Live WebSocket',
            'is_live': True
        }
    
    def stop(self):
        """Stop the bot gracefully"""
        if not self.is_running:
            return
        
        print("\n🛑 Stopping live simulation trading bot...")
        self.is_running = False
        
        try:
            # Stop simulation engine
            self.simulation_engine.stop_simulation()
            
            # Stop metrics collection
            self.metrics_collector.stop_collection()
            
            # Print final summary
            self._print_final_summary()
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
        
        print("✅ Bot stopped successfully")
    
    def _print_final_summary(self):
        """Print final performance summary"""
        try:
            summary = self.get_performance_summary()
            performance = summary['performance']
            market_data = summary['market_data']
            
            print("\n" + "=" * 70)
            print("📊 FINAL LIVE SIMULATION SUMMARY")
            print("=" * 70)
            print(f"⏰ Total Runtime: {summary['uptime']}")
            print(f"📡 Data Source: {summary['data_source']}")
            print(f"💹 Final Price: ${market_data.get('current_price', 0):.4f}")
            print(f"📈 24h Price Change: {market_data.get('price_change_24h', 0):+.2f}%")
            print(f"💰 Simulation P&L: ${performance['total_pnl']:.2f}")
            print(f"📈 Total Return: {performance['return_pct']:.2f}%")
            print(f"🎯 Win Rate: {performance['win_rate']:.1f}%")
            print(f"📊 Profit Factor: {performance['profit_factor']:.2f}")
            print(f"📉 Max Drawdown: {performance['max_drawdown']:.2f}%")
            print(f"🔄 Total Trades: {performance['total_trades']}")
            print(f"🏆 Winning Trades: {performance['winning_trades']}")
            print("=" * 70)
            
        except Exception as e:
            print(f"Error printing final summary: {e}")

def main():
    """Main entry point"""
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print(f"\n🛑 Received signal {sig}")
        if 'bot' in locals():
            bot.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start bot
    bot = LiveSimulationTradingBot()
    
    try:
        bot.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        bot.stop()
        sys.exit(1)

if __name__ == "__main__":
    main() 