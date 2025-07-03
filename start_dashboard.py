"""
Startup Script for Enhanced Trading Bot with Dashboard
Comprehensive trading bot with real-time monitoring and analytics
"""

import os
import sys
import time
import signal
import threading
from datetime import datetime

# Import bot components
from enhanced_main import EnhancedTradingBot
from dashboard import DashboardController
from metrics_collector import MetricsCollector
from trend_analyzer import TrendAnalyzer
from dashboard_app import init_dashboard, run_dashboard

def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🚀 OKX PERPETUAL TRADING BOT WITH DASHBOARD 🚀           ║
    ║                                                              ║
    ║    📊 Real-time Performance Monitoring                      ║
    ║    📈 Advanced Trend Analysis                               ║
    ║    🔧 Automated Parameter Optimization                      ║
    ║    💹 SOL-USD-SWAP Perpetual Futures                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment configuration...")
    
    required_files = [
        '.env',
        'config.py',
        'okx_client.py',
        'strategy.py',
        'trader.py',
        'indicators.py',
        'trend_analyzer.py',
        'metrics_collector.py',
        'dashboard_app.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    # Check if .env exists
    if not os.path.exists('.env'):
        if os.path.exists('env_example.txt'):
            print("⚠️  .env file not found. Please copy env_example.txt to .env and configure your API keys.")
        else:
            print("❌ No .env file found. Please create one with your OKX API credentials.")
        return False
    
    print("✅ Environment configuration check passed")
    return True

def print_features():
    """Print bot features"""
    print("\n🎯 BOT FEATURES:")
    print("=" * 50)
    print("📊 Real-time Dashboard      - http://127.0.0.1:5000")
    print("📈 Trend Analysis          - Multi-timeframe direction detection")
    print("🔧 Auto Optimization       - Daily parameter tuning")
    print("💹 Risk Management         - Dynamic stop-loss/take-profit")
    print("📱 WebSocket Data Feed      - Real-time market data")
    print("💾 Performance Tracking    - SQLite database storage")
    print("🚨 Alert System           - Performance and system alerts")
    print("📋 Backtest Integration    - Historical strategy validation")
    print("🎨 Beautiful UI            - Modern web dashboard")
    print("🔐 Simulation Mode         - Safe testing environment")

def print_dashboard_info():
    """Print dashboard information"""
    print("\n📊 DASHBOARD FEATURES:")
    print("=" * 50)
    print("📈 Performance Metrics     - PnL, Win Rate, Profit Factor")
    print("📊 Real-time Charts        - Live trading performance")
    print("🎯 Trend Analysis          - Direction, Strength, Confidence") 
    print("🖥️  System Health          - CPU, Memory, Disk usage")
    print("📋 Trade History           - Recent trades with analysis")
    print("🔧 Optimization Control    - Run backtests and optimization")
    print("🚨 Alert System           - Real-time notifications")
    print("📱 Mobile Responsive       - Works on all devices")

def start_complete_system(simulation_mode=True):
    """Start the complete trading system with dashboard"""
    
    print(f"\n🚀 Starting Complete Trading System...")
    print(f"🔄 Mode: {'SIMULATION' if simulation_mode else '🔥 LIVE TRADING'}")
    print("=" * 60)
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        
        # Create core instances
        metrics_collector = MetricsCollector()
        trend_analyzer = TrendAnalyzer()
        dashboard_controller = DashboardController(metrics_collector, trend_analyzer)
        
        # Initialize dashboard
        init_dashboard(metrics_collector, trend_analyzer)
        
        # Start metrics collection
        metrics_collector.start_collection()
        print("✅ Metrics collection started")
        
        # Start dashboard monitoring
        dashboard_controller.start_monitoring()
        print("✅ Dashboard monitoring started")
        
        # Start web dashboard in separate thread
        def run_web_dashboard():
            try:
                print("✅ Web dashboard starting on http://127.0.0.1:5000")
                run_dashboard(host='127.0.0.1', port=5000, debug=False)
            except Exception as e:
                print(f"❌ Dashboard error: {e}")
        
        dashboard_thread = threading.Thread(target=run_web_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Give dashboard time to start
        time.sleep(3)
        
        # Create and start trading bot
        print("🤖 Starting trading bot...")
        bot = EnhancedTradingBot(use_simulation=simulation_mode)
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            print("\n⏹️  Shutdown signal received...")
            bot.stop()
            dashboard_controller.stop_monitoring()
            metrics_collector.stop_collection()
            print("✅ System stopped gracefully")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Print startup complete message
        print("\n" + "=" * 60)
        print("🎉 SYSTEM STARTUP COMPLETE!")
        print("=" * 60)
        print(f"📊 Dashboard URL: http://127.0.0.1:5000")
        print(f"💹 Trading Symbol: SOL-USD-SWAP")
        print(f"🔄 Mode: {'SIMULATION' if simulation_mode else 'LIVE TRADING'}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n🎯 The bot is now running with full dashboard integration!")
        print("📱 Open the dashboard in your browser to monitor performance")
        print("\nPress Ctrl+C to stop gracefully...")
        print("=" * 60)
        
        # Start the bot (this will block)
        bot.start()
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutdown requested by user")
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🛑 System shutdown complete")

def run_backtest_only():
    """Run backtest mode only"""
    print("\n📊 BACKTEST MODE")
    print("=" * 50)
    
    try:
        from advanced_backtest import AdvancedBacktester
        from parameter_optimizer import ParameterOptimizer
        
        # Initialize
        backtester = AdvancedBacktester()
        optimizer = ParameterOptimizer()
        
        print("📈 Fetching historical data...")
        df = backtester.fetch_historical_data(
            symbol="SOL-USD-SWAP",
            timeframe="1m",
            days=7
        )
        
        if len(df) < 100:
            print("❌ Insufficient data for backtest")
            return
        
        print(f"✅ Loaded {len(df)} data points")
        
        # Run backtest
        print("🔄 Running backtest...")
        results = backtester.run_backtest(df)
        
        # Print results
        print("\n📊 BACKTEST RESULTS:")
        print("=" * 50)
        print(f"💰 Total Return: {results.get('total_return', 0):.2%}")
        print(f"📈 Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"💹 Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"📉 Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"📊 Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"🔄 Total Trades: {results.get('total_trades', 0)}")
        
        # Ask if user wants to run optimization
        response = input("\n🔧 Run parameter optimization? (y/n): ")
        if response.lower() == 'y':
            print("🔄 Running optimization...")
            opt_results = optimizer.optimize_for_accuracy(df, target_accuracy=0.85)
            
            if opt_results.get('optimization_successful'):
                print(f"✅ Optimization complete!")
                print(f"📈 New accuracy: {opt_results['optimized_accuracy']:.1%}")
                print("🔧 Parameters saved to optimized_strategy_config.json")
            else:
                print("⚠️  Optimization did not improve performance")
        
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    print_banner()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    # Print features
    print_features()
    print_dashboard_info()
    
    # Get user choice
    print("\n🎯 SELECT MODE:")
    print("=" * 50)
    print("1. 🔐 Simulation Mode (Recommended)")
    print("2. 🔥 Live Trading Mode")
    print("3. 📊 Backtest Only")
    print("4. 🎨 Dashboard Only")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            start_complete_system(simulation_mode=True)
        elif choice == '2':
            print("\n⚠️  LIVE TRADING MODE SELECTED!")
            print("🚨 This will trade with real money!")
            confirm = input("Type 'CONFIRM' to proceed with live trading: ")
            if confirm == 'CONFIRM':
                start_complete_system(simulation_mode=False)
            else:
                print("❌ Live trading cancelled")
        elif choice == '3':
            run_backtest_only()
        elif choice == '4':
            # Dashboard only mode
            print("\n📊 Starting dashboard only...")
            metrics_collector = MetricsCollector()
            trend_analyzer = TrendAnalyzer()
            init_dashboard(metrics_collector, trend_analyzer)
            metrics_collector.start_collection()
            
            print("✅ Dashboard started on http://127.0.0.1:5000")
            print("Press Ctrl+C to stop...")
            
            try:
                run_dashboard(host='127.0.0.1', port=5000, debug=False)
            except KeyboardInterrupt:
                print("\n⏹️  Dashboard stopped")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n⏹️  Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 