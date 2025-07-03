import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the trading bot"""
    
    # API Configuration
    OKX_API_KEY = os.getenv("OKX_API_KEY")
    OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY") 
    OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE")
    
    # Trading Configuration
    SYMBOL = "SOL-USD-SWAP"
    TIMEFRAME = "1m"
    LEVERAGE = int(os.getenv("LEVERAGE", 10))
    POSITION_SIZE = float(os.getenv("POSITION_SIZE_USD", 100))
    
    # Risk Management
    STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 2.0)) / 100
    TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", 3.0)) / 100
    MAX_DRAWDOWN_PCT = 0.05
    MAX_DAILY_TRADES = 10
    EMERGENCY_STOP_LOSS = 0.10
    MAX_POSITION_SIZE = 1000
    CONSECUTIVE_LOSS_LIMIT = 5
    
    # Technical Analysis Parameters
    CMF_PERIOD = 20
    OBV_SMA_PERIOD = 14
    RSI_PERIOD = 14
    BB_PERIOD = 20
    BB_STD = 2.0
    ATR_PERIOD = 14
    EMA_FAST = 9
    EMA_SLOW = 21
    
    # Strategy Parameters
    DIVERGENCE_LOOKBACK = 10
    PARABOLIC_THRESHOLD = 2.5
    RANGE_BREAK_CONFIRMATION = 3
    MIN_VOLUME_FACTOR = 1.5
    MIN_SIGNAL_CONFIDENCE = 60
    
    # Trend Analysis Parameters
    TREND_CONFIDENCE_THRESHOLD = 60.0
    TREND_STRENGTH_THRESHOLD = 3
    TIMEFRAME_ALIGNMENT_REQUIRED = True
    
    # Dashboard Configuration
    DASHBOARD_UPDATE_INTERVAL = 5
    METRICS_RETENTION_DAYS = 30
    DASHBOARD_HOST = "127.0.0.1"
    DASHBOARD_PORT = 5000
    
    # WebSocket Configuration
    RECONNECT_DELAY = 5
    HEARTBEAT_INTERVAL = 30
    DATA_BUFFER_SIZE = 500
    
    # Alert Thresholds
    ALERT_THRESHOLDS = {
        'high_drawdown': 0.03,
        'low_win_rate': 0.60,
        'high_cpu': 80.0,
        'high_memory': 80.0,
        'low_trend_confidence': 50.0
    }

# Legacy variables for backward compatibility
API_KEY = os.getenv("OKX_API_KEY")
API_SECRET = os.getenv("OKX_SECRET_KEY") 
API_PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# OKX Endpoints
BASE_URL = "https://www.okx.com"
WS_PUBLIC_URL = "wss://ws.okx.com:8443/ws/v5/public"
WS_PRIVATE_URL = "wss://ws.okx.com:8443/ws/v5/private"

# Trading Configuration
INSTRUMENT_ID = "SOL-USD-SWAP"  # Solana perpetual contract
TIMEFRAME = "1m"  # 1-minute candles for high-frequency analysis
LEVERAGE = int(os.getenv("LEVERAGE", 10))
POSITION_SIZE_USD = float(os.getenv("POSITION_SIZE_USD", 100))
SIMULATED_TRADING = bool(int(os.getenv("SIMULATED_TRADING", 1)))

# Risk Management
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 2.0))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", 3.0))
MAX_DRAWDOWN_PCT = 5.0  # Emergency flatten at 5% drawdown
MAX_DAILY_TRADES = 10  # Limit daily trades to prevent over-trading

# Technical Analysis Parameters
CMF_PERIOD = 20  # Chaikin Money Flow period
OBV_SMA_PERIOD = 14  # OBV smoothing period
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21

# Strategy Parameters
DIVERGENCE_LOOKBACK = 10  # Bars to look back for divergence
PARABOLIC_THRESHOLD = 2.5  # Multiple of ATR for parabolic detection
RANGE_BREAK_CONFIRMATION = 3  # Bars to confirm range break
MIN_VOLUME_FACTOR = 1.5  # Minimum volume factor for valid signals

# WebSocket Configuration
RECONNECT_DELAY = 5  # Seconds to wait before reconnecting
HEARTBEAT_INTERVAL = 30  # Ping interval for websocket
DATA_BUFFER_SIZE = 500  # Number of candles to keep in memory

# Trend Analysis Parameters
TREND_CONFIDENCE_THRESHOLD = 60.0  # Minimum confidence for trend signals
TREND_STRENGTH_THRESHOLD = 3  # Minimum strength (1-5 scale)
TIMEFRAME_ALIGNMENT_REQUIRED = True  # Require timeframe alignment
MIN_SIGNAL_CONFIDENCE = 60  # Minimum signal confidence percentage

# Dashboard Configuration
DASHBOARD_UPDATE_INTERVAL = 5  # seconds
METRICS_RETENTION_DAYS = 30  # days to keep historical data
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5000

# Alert Thresholds
ALERT_THRESHOLDS = {
    'high_drawdown': 0.03,  # 3%
    'low_win_rate': 0.60,   # 60%
    'high_cpu': 80.0,       # 80%
    'high_memory': 80.0,    # 80%
    'low_trend_confidence': 50.0  # 50%
}

# Additional Risk Management
EMERGENCY_STOP_LOSS = 0.10  # 10% emergency stop
MAX_POSITION_SIZE = 1000  # USD maximum position size
CONSECUTIVE_LOSS_LIMIT = 5  # Max consecutive losses before pause 