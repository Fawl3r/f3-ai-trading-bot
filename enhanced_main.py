"""
Enhanced Main Application with Dashboard Integration
Comprehensive trading bot with real-time monitoring and trend analysis
"""

import asyncio
import websocket
import json
import threading
import time
import signal
import sys
from datetime import datetime
import pandas as pd
import traceback

# Import bot components
from config import Config
from okx_client import OKXClient
from strategy import AdvancedTradingStrategy
from trader import TradingEngine
from optimized_strategy import OptimizedTradingStrategy
from indicators import TechnicalIndicators

# Import dashboard components  
from metrics_collector import MetricsCollector
from trend_analyzer import TrendAnalyzer, TrendDirection, TrendStrength
from dashboard_app import init_dashboard, run_dashboard

# Import backtest components
from advanced_backtest import AdvancedBacktester
from parameter_optimizer import ParameterOptimizer

class EnhancedTradingBot:
    """Enhanced trading bot with integrated dashboard and trend analysis"""
    
    def __init__(self, use_simulation=True):
        self.config = Config()
        self.client = OKXClient()
        
        # Core components
        self.indicators = TechnicalIndicators()
        self.strategy = OptimizedTradingStrategy()  # Use optimized strategy
        self.trader = TradingEngine()
        
        # Analytics components
        self.metrics_collector = MetricsCollector()
        self.trend_analyzer = TrendAnalyzer()
        
        # Data management
        self.price_data = []
        self.current_data = pd.DataFrame()
        self.is_running = False
        self.ws = None
        
        # Threading
        self.data_thread = None
        self.trading_thread = None
        self.dashboard_thread = None
        
        # Performance tracking
        self.last_optimization = None
        self.optimization_interval = 24 * 60 * 60  # 24 hours
        
    def start(self):
        """Start the enhanced trading bot"""
        print("🚀 Starting Enhanced OKX Perpetual Trading Bot")
        print("=" * 60)
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Start metrics collection
            self.metrics_collector.start_collection()
            print("✅ Metrics collection started")
            
            # Start dashboard
            self._start_dashboard()
            print("✅ Dashboard started on http://127.0.0.1:5000")
            
            # Start data feed
            self._start_data_feed()
            print("✅ Real-time data feed started")
            
            # Start trading engine
            self._start_trading()
            print("✅ Trading engine started")
            
            # Start optimization scheduler
            self._start_optimization_scheduler()
            print("✅ Optimization scheduler started")
            
            print("\n🎯 Bot is now running with full dashboard integration!")
            print("📊 Dashboard: http://127.0.0.1:5000")
            print("💹 Symbol: SOL-USD-SWAP")
            print("🔄 Mode: SIMULATION")
            print("\nPress Ctrl+C to stop gracefully...")
            
            # Keep main thread alive
            self._run_main_loop()
            
        except KeyboardInterrupt:
            print("\n⏹️  Stopping bot gracefully...")
            self.stop()
        except Exception as e:
            print(f"❌ Error starting bot: {e}")
            traceback.print_exc()
            self.stop()
    
    def _initialize_components(self):
        """Initialize all bot components"""
        # Initialize dashboard with collectors
        init_dashboard(self.metrics_collector, self.trend_analyzer)
        
        # Load optimized parameters if available
        try:
            self.strategy.load_optimized_parameters()
            print("✅ Loaded optimized strategy parameters")
        except:
            print("⚠️  Using default strategy parameters")
    
    def _start_dashboard(self):
        """Start dashboard in separate thread"""
        def run_dashboard_thread():
            try:
                run_dashboard(host='127.0.0.1', port=5000, debug=False)
            except Exception as e:
                print(f"Dashboard error: {e}")
        
        self.dashboard_thread = threading.Thread(target=run_dashboard_thread)
        self.dashboard_thread.daemon = True
        self.dashboard_thread.start()
        
        # Give dashboard time to start
        time.sleep(2)
    
    def _start_data_feed(self):
        """Start WebSocket data feed"""
        def run_data_feed():
            while self.is_running:
                try:
                    self._connect_websocket()
                    if self.ws:
                        self.ws.run_forever()
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    time.sleep(5)  # Reconnect after 5 seconds
        
        self.is_running = True
        self.data_thread = threading.Thread(target=run_data_feed)
        self.data_thread.daemon = True
        self.data_thread.start()
    
    def _connect_websocket(self):
        """Connect to OKX WebSocket"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'data' in data:
                    self._process_market_data(data['data'])
            except Exception as e:
                print(f"Message processing error: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
        
        def on_open(ws):
            print("WebSocket connection opened")
            # Subscribe to SOL-USD-SWAP kline data
            subscribe_msg = {
                "op": "subscribe",
                "args": [
                    {
                        "channel": "candle1m",
                        "instId": "SOL-USD-SWAP"
                    }
                ]
            }
            ws.send(json.dumps(subscribe_msg))
        
        self.ws = websocket.WebSocketApp(
            "wss://ws.okx.com:8443/ws/v5/public",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
    
    def _process_market_data(self, data):
        """Process incoming market data"""
        try:
            for item in data:
                if len(item) >= 6:
                    timestamp = int(item[0])
                    open_price = float(item[1])
                    high_price = float(item[2])
                    low_price = float(item[3])
                    close_price = float(item[4])
                    volume = float(item[5])
                    
                    # Create data point
                    data_point = {
                        'timestamp': pd.to_datetime(timestamp, unit='ms'),
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume
                    }
                    
                    self.price_data.append(data_point)
                    
                    # Keep only last 1000 data points
                    if len(self.price_data) > 1000:
                        self.price_data = self.price_data[-1000:]
                    
                    # Update current data DataFrame
                    self.current_data = pd.DataFrame(self.price_data)
                    
                    # Analyze trends
                    self._analyze_market_trends()
                    
        except Exception as e:
            print(f"Error processing market data: {e}")
    
    def _analyze_market_trends(self):
        """Analyze current market trends"""
        if len(self.current_data) < 50:
            return
        
        try:
            # Perform trend analysis
            trend_analysis = self.trend_analyzer.analyze_trend_direction(self.current_data)
            
            # Update metrics collector with trend data
            trend_metrics = {
                'direction': trend_analysis['direction'].value,
                'strength': trend_analysis['strength'].value,
                'confidence': trend_analysis['confidence'],
                'timeframe_alignment': trend_analysis['timeframe_alignment'],
                'momentum': trend_analysis['momentum'],
                'volume_confirmation': trend_analysis['volume_confirmation']
            }
            
            self.metrics_collector.update_trend_metrics(trend_metrics)
            
            # Update strategy with trend information
            self.strategy.update_trend_context(trend_analysis)
            
        except Exception as e:
            print(f"Error in trend analysis: {e}")
    
    def _start_trading(self):
        """Start trading engine"""
        def run_trading():
            while self.is_running:
                try:
                    if len(self.current_data) >= 50:
                        self._execute_trading_logic()
                    time.sleep(10)  # Check for signals every 10 seconds
                except Exception as e:
                    print(f"Trading error: {e}")
                    time.sleep(30)
        
        self.trading_thread = threading.Thread(target=run_trading)
        self.trading_thread.daemon = True
        self.trading_thread.start()
    
    def _execute_trading_logic(self):
        """Execute main trading logic"""
        try:
            # Generate trading signals
            signals = self.strategy.generate_signals(self.current_data)
            
            if not signals:
                return
            
            # Get current positions
            positions = self.trader.get_current_positions()
            
            for signal in signals:
                if signal.confidence >= self.config.MIN_SIGNAL_CONFIDENCE:
                    # Execute trade
                    if signal.action == 'buy' and not any(pos.side == 'long' for pos in positions):
                        self._execute_long_trade(signal)
                    elif signal.action == 'sell' and not any(pos.side == 'short' for pos in positions):
                        self._execute_short_trade(signal)
                    elif signal.action == 'close':
                        self._close_positions(signal)
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            print(f"Error in trading logic: {e}")
    
    def _execute_long_trade(self, signal):
        """Execute long trade"""
        try:
            result = self.trader.open_position(
                symbol=self.config.SYMBOL,
                side='long',
                size=self.config.POSITION_SIZE,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if result:
                print(f"🟢 LONG position opened at ${signal.price:.4f}")
                self._log_trade(result, signal)
            
        except Exception as e:
            print(f"Error executing long trade: {e}")
    
    def _execute_short_trade(self, signal):
        """Execute short trade"""
        try:
            result = self.trader.open_position(
                symbol=self.config.SYMBOL,
                side='short',
                size=self.config.POSITION_SIZE,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if result:
                print(f"🔴 SHORT position opened at ${signal.price:.4f}")
                self._log_trade(result, signal)
            
        except Exception as e:
            print(f"Error executing short trade: {e}")
    
    def _close_positions(self, signal):
        """Close existing positions"""
        try:
            positions = self.trader.get_current_positions()
            for position in positions:
                result = self.trader.close_position(position.id)
                if result:
                    print(f"✅ Position closed: {position.side} at ${signal.price:.4f}")
                    self._log_trade(result, signal)
        
        except Exception as e:
            print(f"Error closing positions: {e}")
    
    def _log_trade(self, trade_result, signal):
        """Log trade to metrics collector"""
        trade_data = {
            'symbol': self.config.SYMBOL,
            'side': trade_result.get('side', 'unknown'),
            'size': trade_result.get('size', 0),
            'price': trade_result.get('price', signal.price),
            'pnl': trade_result.get('pnl', 0),
            'status': trade_result.get('status', 'executed'),
            'metadata': {
                'signal_confidence': signal.confidence,
                'signal_type': signal.signal_type,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        self.metrics_collector.log_trade(trade_data)
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            performance = self.trader.get_performance_metrics()
            strategy_metrics = self.strategy.get_performance_metrics()
            
            # Combine metrics
            combined_metrics = {
                **performance,
                **strategy_metrics,
                'last_update': datetime.now().isoformat()
            }
            
            self.metrics_collector.update_trading_metrics(combined_metrics)
            self.metrics_collector.update_strategy_metrics(strategy_metrics)
            
        except Exception as e:
            print(f"Error updating performance metrics: {e}")
    
    def _start_optimization_scheduler(self):
        """Start automatic optimization scheduler"""
        def optimization_loop():
            while self.is_running:
                try:
                    # Check if optimization is due
                    current_time = time.time()
                    if (self.last_optimization is None or 
                        current_time - self.last_optimization > self.optimization_interval):
                        
                        print("🔧 Starting automatic optimization...")
                        self._run_optimization()
                        self.last_optimization = current_time
                    
                    time.sleep(3600)  # Check every hour
                    
                except Exception as e:
                    print(f"Optimization scheduler error: {e}")
                    time.sleep(3600)
        
        optimization_thread = threading.Thread(target=optimization_loop)
        optimization_thread.daemon = True
        optimization_thread.start()
    
    def _run_optimization(self):
        """Run parameter optimization"""
        try:
            # Get historical data for optimization
            end_time = datetime.now()
            backtester = AdvancedBacktester()
            
            # Fetch recent data
            df = backtester.get_extended_historical_data(days=7)
            
            if len(df) > 1000:
                # Run optimization
                optimizer = ParameterOptimizer()
                results = optimizer.optimize_for_accuracy(
                    target_accuracy=85.0,  # 85% target
                    days=7
                )
                
                if results['optimization_successful']:
                    print(f"✅ Optimization complete. New accuracy: {results['optimized_accuracy']:.2%}")
                    
                    # Update optimization results in metrics
                    self.metrics_collector.update_backtest_metrics({
                        'optimization_date': datetime.now().isoformat(),
                        'optimized_accuracy': results['optimized_accuracy'],
                        'optimization_method': 'differential_evolution'
                    })
                else:
                    print("⚠️  Optimization did not improve performance")
            
        except Exception as e:
            print(f"Error in optimization: {e}")
    
    def _run_main_loop(self):
        """Main application loop"""
        try:
            while self.is_running:
                # Print status every 60 seconds
                time.sleep(60)
                self._print_status()
                
        except KeyboardInterrupt:
            pass
    
    def _print_status(self):
        """Print current bot status"""
        try:
            metrics = self.metrics_collector.get_real_time_metrics()
            
            print("\n" + "="*60)
            print(f"📊 Bot Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
            
            # Trading metrics
            if 'trading' in metrics:
                trading = metrics['trading']
                print(f"💰 Total PnL: ${trading.get('total_pnl', 0):.2f}")
                print(f"📈 Win Rate: {trading.get('win_rate', 0):.1f}%")
                print(f"🔄 Total Trades: {trading.get('total_trades', 0)}")
                print(f"📊 Active Positions: {trading.get('active_positions', 0)}")
            
            # Trend metrics
            if 'trend' in metrics:
                trend = metrics['trend']
                print(f"📈 Trend Direction: {trend.get('direction', 'UNKNOWN')}")
                print(f"💪 Trend Strength: {trend.get('strength', 'UNKNOWN')}")
                print(f"🎯 Confidence: {trend.get('confidence', 0):.1f}%")
            
            # System metrics
            if 'system' in metrics:
                system = metrics['system']
                print(f"🖥️  CPU: {system.get('cpu_usage', 0):.1f}%")
                print(f"💾 Memory: {system.get('memory_usage', 0):.1f}%")
            
            print("="*60)
            
        except Exception as e:
            print(f"Error printing status: {e}")
    
    def stop(self):
        """Stop the trading bot gracefully"""
        print("\n🛑 Shutting down Enhanced Trading Bot...")
        
        self.is_running = False
        
        # Close WebSocket
        if self.ws:
            self.ws.close()
        
        # Stop metrics collection
        if self.metrics_collector:
            self.metrics_collector.stop_collection()
        
        # Close any open positions in simulation mode
        try:
            if hasattr(self, 'trader') and self.trader:
                positions = self.trader.get_current_positions()
                for position in positions:
                    self.trader.close_position(position.id)
                    print(f"✅ Closed position: {position.side}")
        except:
            pass
        
        print("✅ Bot stopped successfully")
        sys.exit(0)

def main():
    """Main entry point"""
    print("🤖 Enhanced OKX Perpetual Trading Bot")
    print("=====================================")
    
    # Check if we should run in live mode
    simulation_mode = True
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'live':
        simulation_mode = False
        print("⚠️  LIVE TRADING MODE ENABLED!")
        response = input("Are you sure you want to trade with real money? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting...")
            return
    
    # Create and start bot
    bot = EnhancedTradingBot(use_simulation=simulation_mode)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        bot.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the bot
    bot.start()

if __name__ == "__main__":
    main() 