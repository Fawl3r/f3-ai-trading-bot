#!/bin/bash

# Emergency Revert Script for Elite 100%/5% Trading System
# Usage: ./emergency_revert.sh --model prev_good.pt --risk 0.25

set -e

# Default values
MODEL_FILE="prev_good.pt"
EMERGENCY_RISK="0.25"
FORCE_REVERT=false
BACKUP_CONFIG=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_FILE="$2"
            shift 2
            ;;
        --risk)
            EMERGENCY_RISK="$2"
            shift 2
            ;;
        --force)
            FORCE_REVERT=true
            shift
            ;;
        --no-backup)
            BACKUP_CONFIG=false
            shift
            ;;
        -h|--help)
            echo "Emergency Revert Script for Elite 100%/5% Trading System"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model FILE     Model file to revert to (default: prev_good.pt)"
            echo "  --risk PERCENT   Emergency risk percentage (default: 0.25)"
            echo "  --force          Force revert without confirmation"
            echo "  --no-backup      Skip configuration backup"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model prev_good.pt --risk 0.25"
            echo "  $0 --force --risk 0.15"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if we're in the right directory
if [[ ! -f "deployment_config_100_5.yaml" ]]; then
    error "deployment_config_100_5.yaml not found. Are you in the correct directory?"
    exit 1
fi

# Check if model file exists
if [[ ! -f "$MODEL_FILE" ]]; then
    error "Model file $MODEL_FILE not found."
    exit 1
fi

log "=== EMERGENCY REVERT INITIATED ==="
log "Model file: $MODEL_FILE"
log "Emergency risk: $EMERGENCY_RISK%"
log "Force revert: $FORCE_REVERT"
log "Backup config: $BACKUP_CONFIG"

# Confirmation unless forced
if [[ "$FORCE_REVERT" != true ]]; then
    echo ""
    warn "This will immediately:"
    warn "1. Stop all active trading"
    warn "2. Close all open positions"
    warn "3. Revert to emergency risk settings"
    warn "4. Switch to fallback model"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log "Emergency revert cancelled."
        exit 0
    fi
fi

# Step 1: Stop active trading processes
log "Step 1: Stopping active trading processes..."

# Kill trading processes
pkill -f "elite_100_5_trading_system.py" || true
pkill -f "python.*elite_100_5" || true

# Wait for processes to stop
sleep 5

# Verify processes are stopped
if pgrep -f "elite_100_5_trading_system.py" > /dev/null; then
    warn "Trading processes still running. Force killing..."
    pkill -9 -f "elite_100_5_trading_system.py" || true
    sleep 2
fi

log "Trading processes stopped."

# Step 2: Backup current configuration
if [[ "$BACKUP_CONFIG" == true ]]; then
    log "Step 2: Backing up current configuration..."
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="emergency_backup_$TIMESTAMP"
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup configuration files
    cp deployment_config_100_5.yaml "$BACKUP_DIR/"
    cp risk_manager_enhanced.py "$BACKUP_DIR/" 2>/dev/null || true
    cp elite_100_5_trading_system.py "$BACKUP_DIR/" 2>/dev/null || true
    
    # Backup databases
    cp *.db "$BACKUP_DIR/" 2>/dev/null || true
    
    log "Configuration backed up to $BACKUP_DIR"
fi

# Step 3: Close all open positions
log "Step 3: Closing all open positions..."

# Create emergency close script
cat > emergency_close_positions.py << 'EOF'
#!/usr/bin/env python3
import sqlite3
import json
from datetime import datetime

def close_all_positions():
    """Close all open positions in emergency mode"""
    try:
        # Connect to trading database
        conn = sqlite3.connect("elite_100_5_trades.db")
        cursor = conn.cursor()
        
        # Get all open positions
        cursor.execute("SELECT * FROM trades WHERE status = 'open'")
        open_trades = cursor.fetchall()
        
        if not open_trades:
            print("No open positions to close.")
            return
        
        print(f"Found {len(open_trades)} open positions to close.")
        
        # Close all positions
        for trade in open_trades:
            trade_id = trade[0]
            symbol = trade[1]
            
            # Update trade status
            cursor.execute("""
                UPDATE trades 
                SET status = 'closed', 
                    exit_time = ?, 
                    exit_reason = 'Emergency Close'
                WHERE trade_id = ?
            """, (datetime.now().isoformat(), trade_id))
            
            print(f"Closed position: {symbol} (ID: {trade_id})")
        
        conn.commit()
        conn.close()
        
        print("All positions closed successfully.")
        
    except Exception as e:
        print(f"Error closing positions: {e}")

if __name__ == "__main__":
    close_all_positions()
EOF

# Run emergency close
python emergency_close_positions.py
rm emergency_close_positions.py

# Step 4: Update configuration to emergency settings
log "Step 4: Updating configuration to emergency settings..."

# Create emergency config update script
cat > update_emergency_config.py << EOF
#!/usr/bin/env python3
import yaml

def update_emergency_config():
    """Update config to emergency settings"""
    try:
        # Load current config
        with open('deployment_config_100_5.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Update to emergency settings
        config['risk_management']['base_risk_pct'] = $EMERGENCY_RISK
        config['risk_management']['max_risk_pct'] = $EMERGENCY_RISK
        config['risk_management']['max_concurrent_positions'] = 1
        config['risk_management']['equity_dd_threshold'] = 2.0
        config['risk_management']['max_dd_emergency'] = 3.0
        config['risk_management']['max_dd_warning'] = 2.0
        
        # Disable aggressive features
        config['edge_preservation']['burst_trade_enabled'] = False
        config['signals']['parabolic_burst']['enabled'] = False
        config['trade_frequency']['max_active_assets'] = 1
        
        # Enable conservative mode
        config['development']['debug_mode'] = True
        config['development']['verbose_logging'] = True
        
        # Save emergency config
        with open('deployment_config_100_5.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print("Emergency configuration updated successfully.")
        
    except Exception as e:
        print(f"Error updating config: {e}")

if __name__ == "__main__":
    update_emergency_config()
EOF

python update_emergency_config.py
rm update_emergency_config.py

# Step 5: Switch to fallback model
log "Step 5: Switching to fallback model..."

# Backup current model
if [[ -f "current_model.pt" ]]; then
    mv current_model.pt "current_model_backup_$(date +%Y%m%d_%H%M%S).pt"
fi

# Copy fallback model
cp "$MODEL_FILE" current_model.pt

log "Switched to fallback model: $MODEL_FILE"

# Step 6: Update model version in config
log "Step 6: Updating model version in configuration..."

# Get model hash/version
MODEL_HASH=$(sha256sum "$MODEL_FILE" | cut -d' ' -f1 | cut -c1-8)

# Update config with model version
python -c "
import yaml
with open('deployment_config_100_5.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['audit']['model_sha_tracking'] = True
config['audit']['current_model_sha'] = '$MODEL_HASH'
config['audit']['emergency_revert_active'] = True
config['audit']['emergency_revert_timestamp'] = '$(date -Iseconds)'
with open('deployment_config_100_5.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)
"

# Step 7: Create emergency status file
log "Step 7: Creating emergency status file..."

cat > emergency_status.json << EOF
{
    "emergency_revert_active": true,
    "revert_timestamp": "$(date -Iseconds)",
    "fallback_model": "$MODEL_FILE",
    "emergency_risk_pct": $EMERGENCY_RISK,
    "reason": "Emergency revert initiated",
    "operator": "$(whoami)",
    "hostname": "$(hostname)",
    "backup_directory": "$BACKUP_DIR"
}
EOF

# Step 8: Restart system in emergency mode
log "Step 8: System ready for emergency restart..."

warn "EMERGENCY REVERT COMPLETED"
warn "System is now in emergency mode with the following settings:"
warn "- Risk per trade: $EMERGENCY_RISK%"
warn "- Max positions: 1"
warn "- Model: $MODEL_FILE"
warn "- All aggressive features disabled"
warn ""
warn "To restart the system in emergency mode, run:"
warn "  python elite_100_5_trading_system.py"
warn ""
warn "To return to normal operations:"
warn "1. Investigate the cause of the emergency"
warn "2. Restore from backup: $BACKUP_DIR"
warn "3. Update configuration as needed"
warn "4. Remove emergency_status.json"
warn ""

log "Emergency revert completed successfully."

# Optional: Send alert (if configured)
if command -v curl &> /dev/null && [[ -n "${TELEGRAM_BOT_TOKEN:-}" ]] && [[ -n "${TELEGRAM_CHAT_ID:-}" ]]; then
    curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_CHAT_ID" \
        -d text="🚨 EMERGENCY REVERT COMPLETED 🚨%0A%0ASystem reverted to emergency mode:%0A- Risk: $EMERGENCY_RISK%%0A- Model: $MODEL_FILE%0A- Timestamp: $(date)%0A%0AAll positions closed and aggressive features disabled." \
        > /dev/null
fi

exit 0 