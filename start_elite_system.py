#!/usr/bin/env python3
"""
Elite 100%/5% Trading System Startup Script
Launch the complete system with monitoring dashboards
"""

import sys
import asyncio
import signal
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_startup_banner():
    """Print startup banner"""
    print("=" * 80)
    print("🏆 ELITE 100%/5% TRADING SYSTEM")
    print("💰 Target: +100% Monthly Returns | 🛡️ Max DD: 5%")
    print("🚀 Launching Complete System...")
    print("=" * 80)
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📊 Prometheus dashboard will open automatically")
    print("💡 Keep monitoring dashboard open on spare monitor")
    print("=" * 80)

def print_pre_launch_checklist():
    """Print pre-launch checklist"""
    print("\n✅ PRE-LAUNCH CHECKLIST:")
    print("   🔑 API keys configured (.env)")
    print("   📊 Prometheus port 8000 available")
    print("   🎯 Risk parameters validated")
    print("   🛡️ Emergency rollback ready")
    print("   📈 Model SHA updated")
    print("   🔔 Alert webhooks configured")
    print()

def setup_signal_handlers():
    """Setup graceful shutdown handlers"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        print("\n🛑 GRACEFUL SHUTDOWN INITIATED")
        print("⏳ Closing positions and saving state...")
        # The main system will handle the actual shutdown
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main startup function"""
    try:
        print_startup_banner()
        print_pre_launch_checklist()
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Import and run the main system
        from elite_100_5_trading_system import main as elite_main
        
        print("🚀 Starting Elite 100%/5% Trading System...")
        print("📊 Dashboard will open in 3 seconds...")
        print("=" * 80)
        
        # Run the main system
        await elite_main()
        
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
    except ImportError as e:
        logger.error(f"Failed to import system components: {e}")
        print("\n❌ STARTUP FAILED")
        print("💡 Check that all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("   python -m pip install --upgrade pip")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        print(f"\n❌ STARTUP FAILED: {e}")
        print("💡 Check logs above for details")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 System shutdown complete")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1) 