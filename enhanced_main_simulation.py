#!/usr/bin/env python3
"""
Enhanced Trading Bot with Integrated Simulation Mode
Provides realistic trading simulation with dashboard integration
"""

import asyncio
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

# Simulation components
from simulation_data_generator import SimulationDataGenerator
from simulation_trader import SimulationTradingEngine

# Configuration
from config import Config

import requests
import pandas as pd

class SimulationTradingBot:
    """Enhanced trading bot with simulation mode"""
    
    def __init__(self):
        self.config = Config()
        
        # Core simulation components
        self.simulation_engine = SimulationTradingEngine(initial_balance=10000.0)
        self.metrics_collector = MetricsCollector()
        self.trend_analyzer = TrendAnalyzer()
        
        # Bot state
        self.is_running = False
        self.start_time = None
        
        # Performance tracking
        self.last_metrics_update = None
        self.last_trend_update = None
        
        print("🎮 Simulation Trading Bot initialized")
        print("📊 Dashboard will be available at: http://127.0.0.1:5000")
    
    def start(self):
        """Start the complete simulation trading system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        print("🚀 Starting Enhanced Simulation Trading Bot")
        print("=" * 60)
        print(f"⏰ Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("📈 Mode: SIMULATION")
        print("💹 Symbol: SOL-USD-SWAP")
        print("=" * 60)
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Start simulation engine
            self._start_simulation()
            
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
    
    def _start_simulation(self):
        """Start the simulation trading engine"""
        print("🎮 Starting simulation trading engine...")
        self.simulation_engine.start_simulation()
        print("✅ Simulation engine started")
    
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
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    print(f"Metrics update error: {e}")
                    time.sleep(10)
        
        metrics_thread = threading.Thread(target=metrics_loop)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        print("✅ Metrics collection started")
    
    def _update_metrics(self):
        """Update all metrics"""
        try:
            # Get performance metrics from simulation engine
            performance = self.simulation_engine.get_performance_metrics()
            
            # Update trading metrics
            self.metrics_collector.update_trading_metrics({
                'total_pnl': performance['total_pnl'],
                'win_rate': performance['win_rate'] / 100,  # Convert to decimal
                'profit_factor': performance['profit_factor'],
                'max_drawdown': performance['max_drawdown'] / 100,  # Convert to decimal
                'total_trades': performance['total_trades'],
                'active_positions': performance['active_positions'],
                'balance': performance['balance'],
                'return_pct': performance['return_pct']
            })
            
            # Update trend analysis
            self._update_trend_analysis()
            
            self.last_metrics_update = datetime.now()
            
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    def _update_trend_analysis(self):
        """Update trend analysis based on current market data"""
        try:
            # Get market state from simulation
            market_state = self.simulation_engine.data_generator.get_market_state()
            
            # Convert simulation trend to our trend format
            if market_state['trend_direction'] > 0.3:
                direction = TrendDirection.STRONG_UPTREND
            elif market_state['trend_direction'] < -0.3:
                direction = TrendDirection.STRONG_DOWNTREND
            else:
                direction = TrendDirection.SIDEWAYS
            
            # Convert strength
            if market_state['trend_strength'] > 0.7:
                strength = TrendStrength.VERY_STRONG
            elif market_state['trend_strength'] > 0.5:
                strength = TrendStrength.STRONG
            elif market_state['trend_strength'] > 0.3:
                strength = TrendStrength.MODERATE
            else:
                strength = TrendStrength.WEAK
            
            # Calculate confidence based on trend strength and volatility
            confidence = (market_state['trend_strength'] * 70) + (30 if market_state['volatility'] < 0.002 else 10)
            confidence = min(confidence, 95)  # Cap at 95%
            
            # Update trend metrics
            trend_metrics = {
                'direction': direction.value,
                'strength': strength.value,
                'confidence': confidence,
                'current_price': market_state['current_price'],
                'volatility': market_state['volatility'],
                'support_levels': market_state['support_levels'],
                'resistance_levels': market_state['resistance_levels']
            }
            
            self.metrics_collector.update_trend_metrics(trend_metrics)
            self.last_trend_update = datetime.now()
            
        except Exception as e:
            print(f"Error updating trend analysis: {e}")
    
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
                        'timestamp': trade_data['timestamp']
                    }
                })
        
        except Exception as e:
            print(f"Error processing trade event: {e}")
    
    def _run_main_loop(self):
        """Main monitoring and status loop"""
        print("🔄 Starting main monitoring loop...")
        print("\nPress Ctrl+C to stop gracefully...\n")
        
        try:
            while self.is_running:
                # Print periodic status
                self._print_status()
                
                # Sleep for status interval
                time.sleep(60)  # Print status every minute
                
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received...")
            self.stop()
        except Exception as e:
            print(f"Error in main loop: {e}")
            self.stop()
    
    def _print_status(self):
        """Print current bot status"""
        try:
            uptime = datetime.now() - self.start_time
            performance = self.simulation_engine.get_performance_metrics()
            market_state = self.simulation_engine.data_generator.get_market_state()
            
            print("=" * 60)
            print(f"📊 Bot Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            print(f"⏰ Uptime: {str(uptime).split('.')[0]}")
            print(f"💰 Total P&L: ${performance['total_pnl']:.2f}")
            print(f"📈 Return: {performance['return_pct']:.2f}%")
            print(f"🎯 Win Rate: {performance['win_rate']:.1f}%")
            print(f"🔄 Total Trades: {performance['total_trades']}")
            print(f"📊 Active Positions: {performance['active_positions']}")
            print(f"💹 Current Price: ${market_state['current_price']:.4f}")
            print(f"📈 Market Trend: {self._format_trend_direction(market_state['trend_direction'])}")
            print(f"⚡ Volatility: {self._format_volatility(market_state['volatility'])}")
            print("=" * 60)
            
        except Exception as e:
            print(f"Error printing status: {e}")
    
    def _format_trend_direction(self, trend_direction: float) -> str:
        """Format trend direction for display"""
        if trend_direction > 0.3:
            return "📈 Bullish"
        elif trend_direction < -0.3:
            return "📉 Bearish"
        else:
            return "↔️  Sideways"
    
    def _format_volatility(self, volatility: float) -> str:
        """Format volatility for display"""
        if volatility > 0.003:
            return "🔥 High"
        elif volatility > 0.002:
            return "⚡ Medium"
        else:
            return "😴 Low"
    
    def inject_market_event(self, event_type: str):
        """Inject market events for testing"""
        print(f"💥 Injecting market event: {event_type}")
        self.simulation_engine.inject_trading_event(event_type)
    
    def get_performance_summary(self) -> Dict:
        """Get complete performance summary"""
        performance = self.simulation_engine.get_performance_metrics()
        market_state = self.simulation_engine.data_generator.get_market_state()
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        return {
            'performance': performance,
            'market_state': market_state,
            'uptime': str(uptime).split('.')[0],
            'trades': self.simulation_engine.get_recent_trades(10),
            'positions': self.simulation_engine.get_active_positions()
        }
    
    def stop(self):
        """Stop the bot gracefully"""
        if not self.is_running:
            return
        
        print("\n🛑 Stopping simulation trading bot...")
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
            
            print("\n" + "=" * 60)
            print("📊 FINAL PERFORMANCE SUMMARY")
            print("=" * 60)
            print(f"⏰ Total Runtime: {summary['uptime']}")
            print(f"💰 Final P&L: ${performance['total_pnl']:.2f}")
            print(f"📈 Total Return: {performance['return_pct']:.2f}%")
            print(f"🎯 Win Rate: {performance['win_rate']:.1f}%")
            print(f"📊 Profit Factor: {performance['profit_factor']:.2f}")
            print(f"📉 Max Drawdown: {performance['max_drawdown']:.2f}%")
            print(f"🔄 Total Trades: {performance['total_trades']}")
            print(f"🏆 Winning Trades: {performance['winning_trades']}")
            print("=" * 60)
            
        except Exception as e:
            print(f"Error printing final summary: {e}")

    def _fetch_historical_data(self):
        """Fetch initial historical data from OKX public API"""
        try:
            print("📚 Fetching historical data from OKX...")
            
            # Use public API endpoint (no auth required)
            url = f"{self.okx_client.base_url}/api/v5/market/candles"
            params = {
                'instId': INSTRUMENT_ID,
                'bar': '1m',
                'limit': '200'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == '0' and data.get('data'):
                    candles = []
                    
                    # Convert OKX data format to our format
                    for candle in reversed(data['data']):  # OKX returns newest first
                        formatted_candle = {
                            'timestamp': int(candle[0]),
                            'datetime': datetime.fromtimestamp(int(candle[0]) / 1000),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5])
                        }
                        candles.append(formatted_candle)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(candles)
                    df = calculate_all_indicators(df)
                    
                    self.data_buffer = df
                    self.latest_price = float(candles[-1]['close'])
                    
                    print(f"✅ Loaded {len(candles)} candles from OKX")
                    print(f"💰 Current SOL price: ${self.latest_price:.4f}")
                    
                    return True
                    
            print(f"❌ Failed to fetch data from OKX: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False

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
    bot = SimulationTradingBot()
    
    try:
        bot.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        bot.stop()
        sys.exit(1)

if __name__ == "__main__":
    main() 