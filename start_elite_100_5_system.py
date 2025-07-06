#!/usr/bin/env python3
"""
Elite 100%/5% Trading System Startup Script
Windows-compatible launcher for the complete system
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('elite_100_5_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Elite100_5Launcher:
    """System launcher and health monitor"""
    
    def __init__(self):
        self.processes = {}
        self.config_file = "deployment_config_100_5.yaml"
        self.emergency_status_file = "emergency_status.json"
        
    def check_prerequisites(self):
        """Check if all required files exist"""
        required_files = [
            "deployment_config_100_5.yaml",
            "risk_manager_enhanced.py",
            "elite_100_5_trading_system.py",
            "monitoring/exporter.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        logger.info("All required files present")
        return True
    
    def check_emergency_status(self):
        """Check if system is in emergency mode"""
        if Path(self.emergency_status_file).exists():
            with open(self.emergency_status_file, 'r') as f:
                emergency_data = json.load(f)
            
            logger.warning("🚨 SYSTEM IN EMERGENCY MODE 🚨")
            logger.warning(f"Reason: {emergency_data.get('reason', 'Unknown')}")
            logger.warning(f"Emergency risk: {emergency_data.get('emergency_risk_pct', 'Unknown')}%")
            logger.warning(f"Fallback model: {emergency_data.get('fallback_model', 'Unknown')}")
            logger.warning(f"Reverted at: {emergency_data.get('revert_timestamp', 'Unknown')}")
            
            return True
        
        return False
    
    def start_prometheus_monitoring(self):
        """Start Prometheus monitoring"""
        logger.info("Starting Prometheus monitoring...")
        
        try:
            # Start main system monitoring
            cmd = [
                sys.executable, "monitoring/exporter.py",
                "--db", "elite_100_5_trades.db",
                "--port", "8000"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            self.processes['prometheus'] = process
            logger.info("Prometheus monitoring started on port 8000")
            
            # Give it time to start
            time.sleep(3)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Prometheus monitoring: {e}")
            return False
    
    def start_trading_system(self):
        """Start the main trading system"""
        logger.info("Starting Elite 100%/5% Trading System...")
        
        try:
            cmd = [sys.executable, "elite_100_5_trading_system.py"]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            self.processes['trading_system'] = process
            logger.info("Trading system started")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor running processes"""
        logger.info("Starting process monitoring...")
        
        while True:
            try:
                # Check each process
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        # Process has terminated
                        logger.error(f"Process {name} has terminated unexpectedly")
                        
                        # Get exit code and output
                        stdout, stderr = process.communicate()
                        logger.error(f"Exit code: {process.returncode}")
                        if stderr:
                            logger.error(f"Error output: {stderr.decode()}")
                        
                        # Restart process
                        logger.info(f"Restarting {name}...")
                        if name == 'prometheus':
                            self.start_prometheus_monitoring()
                        elif name == 'trading_system':
                            self.start_trading_system()
                
                # Wait before next check
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Error in process monitoring: {e}")
                time.sleep(60)
    
    def shutdown(self):
        """Graceful shutdown of all processes"""
        logger.info("Shutting down Elite 100%/5% Trading System...")
        
        for name, process in self.processes.items():
            try:
                logger.info(f"Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}...")
                    process.kill()
                    process.wait()
                
                logger.info(f"{name} stopped")
                
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        logger.info("System shutdown complete")
    
    def run(self):
        """Main run loop"""
        logger.info("=== Elite 100%/5% Trading System Startup ===")
        logger.info("Target: +100% monthly return with ≤5% drawdown")
        logger.info(f"Startup time: {datetime.now()}")
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                return False
            
            # Check emergency status
            is_emergency = self.check_emergency_status()
            if is_emergency:
                logger.warning("System starting in EMERGENCY MODE")
            
            # Start monitoring
            if not self.start_prometheus_monitoring():
                return False
            
            # Start trading system
            if not self.start_trading_system():
                return False
            
            logger.info("=== System startup complete ===")
            logger.info("Monitoring endpoints:")
            logger.info("  - Prometheus: http://localhost:8000/metrics")
            logger.info("  - System logs: elite_100_5_startup.log")
            
            if is_emergency:
                logger.warning("⚠️  EMERGENCY MODE ACTIVE - Review settings before full deployment")
            
            # Start monitoring
            self.monitor_processes()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Startup interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            return False
        finally:
            self.shutdown()

def main():
    """Main entry point"""
    launcher = Elite100_5Launcher()
    
    try:
        success = launcher.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 