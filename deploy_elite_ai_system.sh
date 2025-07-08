#!/bin/bash

# Elite AI System Deployment Script
# Side-by-side deployment for maximum safety

echo "🚀 ELITE AI SYSTEM DEPLOYMENT"
echo "==============================================="
echo "🧠 Integrating trained AI models with Hyperliquid"
echo "🛡️ Side-by-side deployment for safety"
echo "==============================================="

# Check if we're in the correct directory
if [ ! -f "final_production_hyperliquid_bot.py" ]; then
    echo "❌ Error: Please run this script from the OKX PERP BOT directory"
    exit 1
fi

# Check required environment variables
echo "🔍 Checking environment variables..."
if [ -z "$HYPERLIQUID_PRIVATE_KEY" ]; then
    echo "⚠️  Warning: HYPERLIQUID_PRIVATE_KEY not set"
    echo "   Please set it in your .env file or environment"
fi

# Validate AI models exist
echo "🧠 Validating AI models..."
models_ok=true

if [ ! -f "models/encoder_20250707_153740_b59c66da.pt" ]; then
    echo "❌ Missing: TSA-MAE Encoder"
    models_ok=false
else
    echo "✅ TSA-MAE Encoder found"
fi

if [ ! -f "models/lgbm_SOL_20250707_191855_0a65ca5b.pkl" ]; then
    echo "❌ Missing: LightGBM Model"
    models_ok=false
else
    echo "✅ LightGBM Model found"
fi

if [ ! -f "models/timesnet_SOL_20250707_204629_93387ccf.pt" ]; then
    echo "❌ Missing: TimesNet Model"
    models_ok=false
else
    echo "✅ TimesNet Model found"
fi

if [ ! -f "models/ppo_strict_20250707_161252.pt" ]; then
    echo "❌ Missing: PPO Model"
    models_ok=false
else
    echo "✅ PPO Model found"
fi

if [ ! -f "enhanced_register_policy.py" ]; then
    echo "❌ Missing: Thompson Sampling Bandit"
    models_ok=false
else
    echo "✅ Thompson Sampling Bandit found"
fi

if [ "$models_ok" = false ]; then
    echo "❌ Some required AI models are missing!"
    echo "   Please ensure all models are trained and available"
    exit 1
fi

echo "✅ All AI models validated!"

# Create backup of original bot
echo "💾 Creating backup..."
if [ ! -f "final_production_hyperliquid_bot.py.backup" ]; then
    cp final_production_hyperliquid_bot.py final_production_hyperliquid_bot.py.backup
    echo "✅ Backup created: final_production_hyperliquid_bot.py.backup"
fi

# Install additional dependencies if needed
echo "📦 Checking dependencies..."
python3 -c "import torch; import sklearn; import pandas; import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing missing dependencies..."
    pip install torch scikit-learn pandas numpy joblib
fi

# Run deployment validation
echo "🧪 Running deployment validation..."
python3 -c "
import sys
sys.path.append('.')
try:
    from elite_ai_hyperliquid_integration import EliteAIHyperliquidBot
    from enhanced_register_policy import ThompsonSamplingBandit
    print('✅ Elite AI system imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Deployment validation failed!"
    exit 1
fi

echo "✅ Deployment validation passed!"

# Deployment options
echo ""
echo "🎯 DEPLOYMENT OPTIONS:"
echo "=================================="
echo "1. Paper Trading (48h validation) - RECOMMENDED"
echo "2. Live Trading (real money) - ADVANCED USERS"
echo "3. Cancel deployment"
echo ""

read -p "Select option (1-3): " choice

case $choice in
    1)
        echo "📝 Starting Paper Trading Validation..."
        echo "⏱️  Duration: 48 hours"
        echo "🛡️  No real money at risk"
        echo ""
        echo "🚀 Launching Elite AI system in paper mode..."
        python3 elite_ai_hyperliquid_integration.py --mode paper --duration 48
        ;;
    2)
        echo "🔴 LIVE TRADING SELECTED"
        echo "💰 REAL MONEY WILL BE AT RISK"
        echo ""
        read -p "⚠️  Are you absolutely sure? Type 'I UNDERSTAND THE RISKS': " confirm
        if [ "$confirm" = "I UNDERSTAND THE RISKS" ]; then
            echo "🚀 Launching Elite AI system in live mode..."
            python3 elite_ai_hyperliquid_integration.py --mode live
        else
            echo "❌ Live trading cancelled"
        fi
        ;;
    3)
        echo "❌ Deployment cancelled"
        exit 0
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "========================"
echo "📊 Monitor performance in the logs"
echo "🔧 Adjust parameters as needed"
echo "💰 Your Elite AI system is now active!" 