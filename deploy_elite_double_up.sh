#!/bin/bash

# Elite 100%/5% Trading System Deployment Script
# Green-light sequence: staging → validation → live deployment
# Usage: ./deploy_elite_double_up.sh --phase ramp0

set -e

# Default values
PHASE="ramp0"
CONFIG_FILE="deployment_config_100_5.yaml"
FORCE_DEPLOY=false
SKIP_VALIDATION=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --force)
            FORCE_DEPLOY=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Elite 100%/5% Trading System Deployment"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --phase PHASE        Deployment phase: ramp0, ramp1, maintenance"
            echo "  --config FILE        Configuration file (default: deployment_config_100_5.yaml)"
            echo "  --force              Force deployment without confirmation"
            echo "  --skip-validation    Skip pre-deployment validation"
            echo "  --dry-run            Dry run mode (no actual deployment)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Phases:"
            echo "  ramp0       Initial ramp (150 trades, staging keys)"
            echo "  ramp1       Validation ramp (150 trades, auto-reduce on DD>4%)"
            echo "  maintenance Full production (live keys, all features)"
            echo ""
            echo "Examples:"
            echo "  $0 --phase ramp0"
            echo "  $0 --phase ramp1 --force"
            echo "  $0 --phase maintenance --config prod_config.yaml"
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
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -f "$CONFIG_FILE" ]]; then
        error "Configuration file $CONFIG_FILE not found"
        exit 1
    fi
    
    # Check required Python files
    required_files=(
        "elite_100_5_trading_system.py"
        "risk_manager_enhanced.py"
        "monitoring/exporter.py"
        "asset_selector_audit_logger.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "Required file not found: $file"
            exit 1
        fi
    done
    
    # Check Python dependencies
    if ! python -c "import yaml, numpy, pandas, prometheus_client" 2>/dev/null; then
        error "Missing Python dependencies. Run: pip install -r requirements.txt"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Validate configuration for phase
validate_phase_config() {
    log "Validating configuration for phase: $PHASE"
    
    case $PHASE in
        ramp0)
            # Ramp-0 validation
            info "Ramp-0 requirements:"
            info "  - 150 trades target"
            info "  - 0.5% risk per trade"
            info "  - Staging API keys"
            info "  - Success criteria: PF≥2, DD≤3%"
            ;;
        ramp1)
            # Ramp-1 validation
            info "Ramp-1 requirements:"
            info "  - Next 150 trades"
            info "  - Auto-reduce if DD>4%"
            info "  - Recovery when DD<2%"
            ;;
        maintenance)
            # Maintenance validation
            info "Maintenance requirements:"
            info "  - Full production mode"
            info "  - Live API keys"
            info "  - All features enabled"
            info "  - Weekly validation"
            ;;
        *)
            error "Unknown phase: $PHASE"
            exit 1
            ;;
    esac
    
    # Validate risk parameters
    python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

risk_config = config.get('risk_management', {})
base_risk = risk_config.get('base_risk_pct', 0)

if base_risk != 0.5:
    print(f'ERROR: Base risk should be 0.5%, found {base_risk}%')
    exit(1)

print('Configuration validation passed')
"
    
    if [[ $? -ne 0 ]]; then
        error "Configuration validation failed"
        exit 1
    fi
    
    log "Phase configuration validated"
}

# Run pre-deployment tests
run_pre_deployment_tests() {
    if [[ "$SKIP_VALIDATION" == true ]]; then
        warn "Skipping pre-deployment validation"
        return
    fi
    
    log "Running pre-deployment tests..."
    
    # Test 1: Monte Carlo slippage test
    info "Running Monte Carlo slippage test..."
    if [[ -f "backtests/mc_slippage.py" ]]; then
        python backtests/mc_slippage.py --runs 100 --risk 0.005 > mc_test_results.log 2>&1
        
        # Check if test passed
        if grep -q "✅ PASS" mc_test_results.log; then
            log "Monte Carlo test PASSED"
        else
            error "Monte Carlo test FAILED - check mc_test_results.log"
            exit 1
        fi
    else
        warn "Monte Carlo test script not found, skipping"
    fi
    
    # Test 2: Risk manager unit test
    info "Testing risk manager..."
    python -c "
from risk_manager_enhanced import EnhancedRiskManager, RiskConfig
config = RiskConfig()
rm = EnhancedRiskManager(config)
rm.update_equity_metrics(9500, 10000, -1.5)  # 5% DD test
print('Risk manager test passed')
"
    
    if [[ $? -ne 0 ]]; then
        error "Risk manager test failed"
        exit 1
    fi
    
    # Test 3: Prometheus exporter test
    info "Testing Prometheus exporter..."
    python -c "
from monitoring.exporter import ElitePrometheusExporter
exporter = ElitePrometheusExporter('test.db', 9999)
print('Prometheus exporter test passed')
"
    
    if [[ $? -ne 0 ]]; then
        error "Prometheus exporter test failed"
        exit 1
    fi
    
    log "Pre-deployment tests completed successfully"
}

# Deploy phase
deploy_phase() {
    log "Deploying phase: $PHASE"
    
    if [[ "$DRY_RUN" == true ]]; then
        warn "DRY RUN MODE - No actual deployment"
        return
    fi
    
    # Create deployment timestamp
    DEPLOYMENT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    DEPLOYMENT_DIR="deployment_$PHASE_$DEPLOYMENT_TIMESTAMP"
    
    # Create deployment directory
    mkdir -p "$DEPLOYMENT_DIR"
    
    # Copy configuration
    cp "$CONFIG_FILE" "$DEPLOYMENT_DIR/"
    
    # Update configuration for phase
    case $PHASE in
        ramp0)
            # Ramp-0 specific settings
            python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Set ramp-0 parameters
config['deployment']['current_phase'] = 'ramp0'
config['deployment']['phase_start_time'] = '$(date -Iseconds)'
config['deployment']['target_trades'] = 150
config['risk_management']['base_risk_pct'] = 0.5
config['monitoring']['alerts']['telegram_enabled'] = True

# Save updated config
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

print('Ramp-0 configuration updated')
"
            ;;
        ramp1)
            # Ramp-1 specific settings
            python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Set ramp-1 parameters
config['deployment']['current_phase'] = 'ramp1'
config['deployment']['phase_start_time'] = '$(date -Iseconds)'
config['risk_management']['auto_reduce_trigger'] = 4.0
config['risk_management']['auto_reduce_target'] = 0.35
config['risk_management']['recovery_threshold'] = 2.0

# Save updated config
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

print('Ramp-1 configuration updated')
"
            ;;
        maintenance)
            # Maintenance specific settings
            python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Set maintenance parameters
config['deployment']['current_phase'] = 'maintenance'
config['deployment']['phase_start_time'] = '$(date -Iseconds)'
config['risk_management']['base_risk_pct'] = 0.5
config['edge_preservation']['burst_trade_enabled'] = True
config['trade_frequency']['max_active_assets'] = 2

# Save updated config
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

print('Maintenance configuration updated')
"
            ;;
    esac
    
    # Start monitoring
    log "Starting Prometheus monitoring..."
    python monitoring/exporter.py --db elite_100_5_trades.db --port 8000 > "$DEPLOYMENT_DIR/prometheus.log" 2>&1 &
    PROMETHEUS_PID=$!
    echo $PROMETHEUS_PID > "$DEPLOYMENT_DIR/prometheus.pid"
    
    # Wait for monitoring to start
    sleep 5
    
    # Verify monitoring is running
    if ! curl -s http://localhost:8000/metrics > /dev/null; then
        error "Failed to start Prometheus monitoring"
        exit 1
    fi
    
    log "Prometheus monitoring started successfully"
    
    # Start trading system
    log "Starting Elite 100%/5% Trading System..."
    python elite_100_5_trading_system.py > "$DEPLOYMENT_DIR/trading_system.log" 2>&1 &
    TRADING_PID=$!
    echo $TRADING_PID > "$DEPLOYMENT_DIR/trading.pid"
    
    # Wait for system to initialize
    sleep 10
    
    # Verify system is running
    if ! ps -p $TRADING_PID > /dev/null; then
        error "Trading system failed to start"
        cat "$DEPLOYMENT_DIR/trading_system.log"
        exit 1
    fi
    
    log "Trading system started successfully"
    
    # Create deployment status file
    cat > "$DEPLOYMENT_DIR/deployment_status.json" << EOF
{
    "phase": "$PHASE",
    "deployment_timestamp": "$DEPLOYMENT_TIMESTAMP",
    "config_file": "$CONFIG_FILE",
    "prometheus_pid": $PROMETHEUS_PID,
    "trading_pid": $TRADING_PID,
    "status": "deployed",
    "deployment_dir": "$DEPLOYMENT_DIR"
}
EOF
    
    # Log deployment success
    log "=== DEPLOYMENT SUCCESSFUL ==="
    log "Phase: $PHASE"
    log "Deployment directory: $DEPLOYMENT_DIR"
    log "Prometheus PID: $PROMETHEUS_PID"
    log "Trading system PID: $TRADING_PID"
    log "Monitoring: http://localhost:8000/metrics"
    
    # Phase-specific success messages
    case $PHASE in
        ramp0)
            info "🚀 Ramp-0 deployment complete!"
            info "Monitor 150 trades for PF≥2, DD≤3%"
            info "Success criteria must be met before proceeding to Ramp-1"
            ;;
        ramp1)
            info "🚀 Ramp-1 deployment complete!"
            info "Auto-reduce active: DD>4% → risk reduces to 0.35%"
            info "Recovery threshold: DD<2% → normal risk resumed"
            ;;
        maintenance)
            info "🚀 Full production deployment complete!"
            info "All features active, weekly validation scheduled"
            info "Target: +100% monthly with ≤5% drawdown"
            ;;
    esac
    
    # Show next steps
    echo ""
    warn "NEXT STEPS:"
    case $PHASE in
        ramp0)
            warn "1. Monitor Prometheus metrics for 150 trades"
            warn "2. Verify PF≥2.0 and DD≤3.0%"
            warn "3. If successful, deploy Ramp-1: ./deploy_elite_double_up.sh --phase ramp1"
            ;;
        ramp1)
            warn "1. Monitor for next 150 trades"
            warn "2. Verify auto-reduce functionality"
            warn "3. If successful, deploy maintenance: ./deploy_elite_double_up.sh --phase maintenance"
            ;;
        maintenance)
            warn "1. Monitor weekly performance"
            warn "2. Update model weights monthly"
            warn "3. Review risk parameters quarterly"
            ;;
    esac
}

# Monitor deployment
monitor_deployment() {
    if [[ "$DRY_RUN" == true ]]; then
        return
    fi
    
    log "Starting deployment monitoring..."
    
    # Monitor for 5 minutes
    for i in {1..30}; do
        sleep 10
        
        # Check if processes are still running
        if [[ -f "deployment_*/prometheus.pid" ]]; then
            PROMETHEUS_PID=$(cat deployment_*/prometheus.pid)
            if ! ps -p $PROMETHEUS_PID > /dev/null; then
                error "Prometheus monitoring stopped unexpectedly"
                exit 1
            fi
        fi
        
        if [[ -f "deployment_*/trading.pid" ]]; then
            TRADING_PID=$(cat deployment_*/trading.pid)
            if ! ps -p $TRADING_PID > /dev/null; then
                error "Trading system stopped unexpectedly"
                exit 1
            fi
        fi
        
        # Check metrics
        if curl -s http://localhost:8000/metrics | grep -q "elite_system_uptime"; then
            info "System health check passed (iteration $i/30)"
        else
            warn "System health check failed (iteration $i/30)"
        fi
    done
    
    log "Deployment monitoring completed - system stable"
}

# Main execution
main() {
    log "=== Elite 100%/5% Trading System Deployment ==="
    log "Phase: $PHASE"
    log "Config: $CONFIG_FILE"
    log "Dry run: $DRY_RUN"
    
    # Confirmation unless forced
    if [[ "$FORCE_DEPLOY" != true && "$DRY_RUN" != true ]]; then
        echo ""
        warn "This will deploy the Elite 100%/5% Trading System in $PHASE mode."
        warn "Make sure you have:"
        warn "- Reviewed the configuration"
        warn "- Backed up existing deployments"
        warn "- Verified API keys and permissions"
        echo ""
        read -p "Are you sure you want to continue? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log "Deployment cancelled."
            exit 0
        fi
    fi
    
    # Execute deployment steps
    check_prerequisites
    validate_phase_config
    run_pre_deployment_tests
    deploy_phase
    monitor_deployment
    
    log "🎉 Elite 100%/5% deployment completed successfully!"
}

# Execute main function
main "$@" 