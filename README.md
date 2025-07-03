# OKX SOL-USD Perpetual Trading Bot

A high-frequency automated trading bot for OKX SOL-USD perpetual futures with advanced technical analysis and risk management.

## 🚀 Features

### Advanced Technical Analysis
- **Chaikin Money Flow (CMF)** - Money flow analysis for volume-price relationship
- **On Balance Volume (OBV)** - Volume-based momentum indicator
- **Bollinger Bands** - Dynamic support/resistance levels
- **RSI** - Relative Strength Index for overbought/oversold conditions
- **EMA Cross** - Exponential Moving Average trend analysis
- **ATR** - Average True Range for volatility measurement

### Pattern Detection
- **Divergence Detection** - Bullish/bearish divergences between price and indicators
- **Range Breakouts** - Automated detection of range breaks with volume confirmation
- **Pullback Trading** - Trend continuation entries on pullbacks
- **Reversal Signals** - Multi-factor reversal pattern recognition
- **Parabolic Moves** - Detection of unsustainable price movements for exits

### Risk Management
- **Dynamic Stop Loss/Take Profit** - ATR-based position sizing
- **Position Sizing** - Automated position sizing based on account balance
- **Daily Loss Limits** - Maximum daily loss protection
- **Consecutive Loss Protection** - Cooling-off periods after losses
- **Emergency Stop** - Immediate position closure on extreme moves

### Performance Features
- **Target Accuracy: 80-90%+** - Multi-factor signal confirmation
- **Real-time WebSocket Data** - Live OKX market data feed
- **Comprehensive Logging** - Detailed trade and performance logs
- **Performance Tracking** - Win rate, P&L, drawdown monitoring

## 📋 Prerequisites

- Python 3.8+
- OKX Account with API access
- Minimum $100 account balance (recommended)

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup environment variables:**
```bash
# Copy the example environment file
cp env_example.txt .env

# Edit .env with your OKX API credentials
OKX_API_KEY=your_api_key_here
OKX_API_SECRET=your_api_secret_here
OKX_API_PASSPHRASE=your_passphrase_here
SIMULATED_TRADING=1  # Start with simulation mode
```

4. **Get OKX API Credentials:**
   - Go to [OKX API Management](https://www.okx.com/account/my-api)
   - Create new API key with trading permissions
   - Save the key, secret, and passphrase securely

## ⚡ Quick Start

### 1. Simulation Mode (Recommended First)
```bash
python main.py
```
The bot starts in simulation mode by default. Monitor logs to understand behavior.

### 2. Live Trading (After Testing)
```bash
# Edit .env file
SIMULATED_TRADING=0

# Run with live trading
python main.py
```

## 📊 Configuration

### Trading Parameters (`config.py`)
```python
# Position sizing
POSITION_SIZE_USD = 100      # USD per trade
LEVERAGE = 10                # Leverage multiplier

# Risk management
STOP_LOSS_PCT = 2.0         # 2% stop loss
TAKE_PROFIT_PCT = 3.0       # 3% take profit
MAX_DRAWDOWN_PCT = 5.0      # 5% emergency exit

# Strategy parameters
CMF_PERIOD = 20             # Chaikin Money Flow period
OBV_SMA_PERIOD = 14         # OBV smoothing period
RSI_PERIOD = 14             # RSI calculation period
```

### Signal Confidence Weights
The bot uses weighted scoring for signal confidence:
- **Divergence**: 25% weight
- **Range Breaks**: 20% weight  
- **Reversals**: 20% weight
- **Pullbacks**: 15% weight
- **Volume Confirmation**: 10% weight
- **Parabolic Exits**: 10% weight

Minimum 60% confidence required for trade execution.

## 📈 Strategy Logic

### Entry Signals

**Long (Buy) Conditions:**
- Bullish divergence between price and CMF/OBV/RSI
- Range breakout above Bollinger Band upper with volume
- Bullish reversal: Oversold RSI + positive CMF + rising OBV
- Pullback to EMA in uptrend
- Price at lower Bollinger Band in non-bearish trend

**Short (Sell) Conditions:**
- Bearish divergence between price and indicators
- Range breakdown below Bollinger Band lower with volume
- Bearish reversal: Overbought RSI + negative CMF + falling OBV
- Pullback to EMA in downtrend
- Price at upper Bollinger Band in non-bullish trend

### Exit Signals
- Take profit/stop loss levels hit
- Parabolic move detection (>2.5x ATR movement)
- Reversal signal opposite to position
- 4%+ profit taking
- 5%+ emergency stop loss

## 📋 Monitoring & Logs

### Real-time Monitoring
The bot provides comprehensive real-time monitoring:

```
2024-01-15 10:30:45 - INFO - Signal Generated: BUY at $142.3456
2024-01-15 10:30:45 - INFO - Confidence: 78.5%
2024-01-15 10:30:45 - INFO - Reason: Bullish divergence detected | Oversold with positive CMF
2024-01-15 10:30:45 - INFO - Position opened successfully: long 67.2 @ 142.3456
```

### Performance Summary (Every 30 seconds)
```
============================================================
PERFORMANCE SUMMARY
Uptime: 2:45:30
Signals Generated: 23
Signals Executed: 18
Execution Rate: 78.3%
Data Feed Connected: True
Buffer Size: 200 candles
============================================================
Heartbeat - Trades: 12, Win Rate: 83.3%, Total P&L: $156.78, Daily P&L: $156.78
Position: long P&L: $12.45 (2.1%)
============================================================
```

## ⚠️ Risk Warnings

### Important Disclaimers
- **Trading involves substantial risk of loss**
- **Past performance does not guarantee future results**
- **Start with small position sizes**
- **Always use simulation mode first**
- **Monitor the bot actively, especially initially**

### Risk Controls Built-in
- Maximum 10 trades per day
- 5-minute cooldown between signals
- 3 consecutive loss limit with cooling period
- 5% maximum drawdown emergency stop
- Dynamic position sizing based on volatility

## 🔧 Troubleshooting

### Common Issues

**WebSocket Connection Issues:**
```bash
# Check internet connection and OKX API status
# Bot will automatically reconnect with exponential backoff
```

**API Authentication Errors:**
```bash
# Verify API credentials in .env file
# Ensure API key has trading permissions
# Check if IP is whitelisted (if applicable)
```

**Insufficient Balance:**
```bash
# Ensure account has sufficient balance for position size + margin
# Reduce POSITION_SIZE_USD in config.py
```

### Debug Mode
Enable detailed logging by modifying `main.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 File Structure

```
okx-sol-perp-bot/
├── requirements.txt      # Python dependencies
├── env_example.txt       # Environment variables template
├── config.py            # Trading configuration
├── okx_client.py        # OKX API client
├── indicators.py        # Technical indicators & pattern detection
├── strategy.py          # Trading strategy logic
├── trader.py           # Position management & execution
├── main.py             # Main application entry point
├── README.md           # This file
└── trading_bot.log     # Generated log file
```

## 🔄 Backtesting

To backtest the strategy on historical data:

```python
# Create backtest.py
from strategy import AdvancedTradingStrategy
from indicators import TechnicalIndicators
import pandas as pd

# Load historical data
df = pd.read_csv('historical_sol_data.csv')

# Run backtest
strategy = AdvancedTradingStrategy()
signals = []

for i in range(50, len(df)):
    signal = strategy.generate_signal(df.iloc[:i+1])
    if signal:
        signals.append({
            'timestamp': df.iloc[i]['datetime'],
            'signal': signal.signal_type,
            'confidence': signal.confidence,
            'price': df.iloc[i]['close']
        })

print(f"Generated {len(signals)} signals")
```

## 🤝 Support & Contributing

### Getting Help
- Review logs in `trading_bot.log`
- Check OKX API documentation
- Verify network connectivity
- Test with smaller position sizes

### Customization
The bot is designed to be easily customizable:
- Modify indicator parameters in `config.py`
- Adjust signal weights in `strategy.py`
- Add new technical indicators in `indicators.py`
- Implement custom risk rules in `trader.py`

## 📜 License & Disclaimer

This software is provided "as is" without warranty. Trading cryptocurrencies involves substantial risk of loss. Users are responsible for their own trading decisions and risk management.

**Use at your own risk. Start with paper trading and small positions.**

---

*Last updated: January 2024*
*Bot version: 1.0*
*Tested on: OKX API v5* 