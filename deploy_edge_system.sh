#!/bin/bash
# Edge System Deployment Pipeline
# Automated validation, shadow trading, and progressive rollout

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="${SCRIPT_DIR}/deployment_config.yaml"
LOG_DIR="${SCRIPT_DIR}/logs"
METRICS_DIR="${SCRIPT_DIR}/backtests"

# Create directories
mkdir -p "$LOG_DIR" "$METRICS_DIR"

# Logging function
log() {
    echo -e "${1}[$(date +'%Y-%m-%d %H:%M:%S')] $2${NC}" | tee -a "$LOG_DIR/deployment.log"
}

# Check prerequisites
check_prerequisites() {
    log "$YELLOW" "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log "$RED" "Python 3 not found!"
        exit 1
    fi
    
    # Check required files
    for file in "improved_edge_system.py" "edge_validation_suite.py" "deployment_config.yaml"; do
        if [ ! -f "$file" ]; then
            log "$RED" "Required file $file not found!"
            exit 1
        fi
    done
    
    # Check environment variables
    if [ -z "$HYPERLIQUID_PRIVATE_KEY" ]; then
        log "$RED" "HYPERLIQUID_PRIVATE_KEY not set!"
        exit 1
    fi
    
    log "$GREEN" "Prerequisites check passed ✓"
}

# Run validation suite
run_validation() {
    log "$YELLOW" "Running walk-forward validation..."
    
    python edge_validation_suite.py \
        --assets SOL,BTC,ETH \
        --window 5000 \
        --shift 3000 \
        --min_trades 100 \
        --metrics_out "$METRICS_DIR/edge_results.csv"
    
    # Check validation status
    if [ ! -f "validation_status.txt" ]; then
        log "$RED" "Validation status file not found!"
        exit 1
    fi
    
    STATUS=$(cat validation_status.txt)
    if [ "$STATUS" != "PASSED" ]; then
        log "$RED" "Validation FAILED! Check logs for details."
        exit 1
    fi
    
    log "$GREEN" "Validation PASSED ✓"
}

# Run shadow trading
run_shadow_trading() {
    log "$YELLOW" "Starting shadow trading (200 trades)..."
    
    # Start metrics collection in background
    if command -v prometheus &> /dev/null; then
        prometheus --config.file=prometheus.yml &
        PROM_PID=$!
        log "$GREEN" "Prometheus started (PID: $PROM_PID)"
    fi
    
    # Run shadow trading
    python edge_validation_suite.py --paper
    
    # Check shadow status
    if [ ! -f "shadow_status.txt" ]; then
        log "$RED" "Shadow status file not found!"
        exit 1
    fi
    
    STATUS=$(cat shadow_status.txt)
    if [ "$STATUS" != "PASSED" ]; then
        log "$RED" "Shadow trading FAILED! Check logs for details."
        [ ! -z "$PROM_PID" ] && kill $PROM_PID
        exit 1
    fi
    
    log "$GREEN" "Shadow trading PASSED ✓"
}

# Deploy phase
deploy_phase() {
    PHASE=$1
    RISK=$2
    DAYS=$3
    
    log "$YELLOW" "Deploying Phase: $PHASE (Risk: ${RISK}%, Duration: ${DAYS} days)"
    
    # Create phase config
    cat > "phase_config.json" <<EOF
{
    "phase": "$PHASE",
    "risk_per_trade": $RISK,
    "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "duration_days": $DAYS
}
EOF
    
    # Start the bot with phase config
    python improved_edge_system.py \
        --config phase_config.json \
        --mode live \
        --log-level INFO &
    
    BOT_PID=$!
    log "$GREEN" "Bot started for $PHASE (PID: $BOT_PID)"
    
    # Monitor for duration
    if [ "$DAYS" != "null" ]; then
        log "$YELLOW" "Monitoring for $DAYS days..."
        sleep $((DAYS * 86400))
        
        # Check metrics before proceeding
        METRICS=$(curl -s http://localhost:8080/metrics)
        EXPECTANCY=$(echo "$METRICS" | jq -r '.expectancy')
        PF=$(echo "$METRICS" | jq -r '.profit_factor')
        
        if (( $(echo "$EXPECTANCY < 0.0025" | bc -l) )); then
            log "$RED" "Expectancy below threshold: $EXPECTANCY"
            kill $BOT_PID
            exit 1
        fi
        
        if (( $(echo "$PF < 1.1" | bc -l) )); then
            log "$RED" "Profit factor below threshold: $PF"
            kill $BOT_PID
            exit 1
        fi
        
        log "$GREEN" "Phase $PHASE completed successfully!"
        kill $BOT_PID
    fi
}

# Main deployment flow
main() {
    log "$GREEN" "🚀 EDGE SYSTEM DEPLOYMENT PIPELINE"
    log "$GREEN" "=================================="
    
    # Step 1: Prerequisites
    check_prerequisites
    
    # Step 2: Validation
    run_validation
    
    # Step 3: Shadow Trading
    run_shadow_trading
    
    # Step 4: Progressive Deployment
    log "$YELLOW" "Starting progressive deployment..."
    
    # Phase 1: Verification (0.25% risk, 2 days)
    deploy_phase "Verification" 0.0025 2
    
    # Phase 2: Confidence Building (0.5% risk, 2 days)
    deploy_phase "Confidence" 0.005 2
    
    # Phase 3: Full Production (1% risk, ongoing)
    log "$GREEN" "🎯 DEPLOYING TO FULL PRODUCTION!"
    deploy_phase "Production" 0.01 null
    
    log "$GREEN" "✅ Deployment complete!"
}

# Cleanup function
cleanup() {
    log "$YELLOW" "Cleaning up..."
    [ ! -z "$PROM_PID" ] && kill $PROM_PID 2>/dev/null
    [ ! -z "$BOT_PID" ] && kill $BOT_PID 2>/dev/null
    rm -f validation_status.txt shadow_status.txt phase_config.json
}

# Set trap for cleanup
trap cleanup EXIT

# Run main
main "$@" 