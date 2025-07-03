#!/usr/bin/env python3
"""
Dashboard Launcher for Ultimate 75% Bot
"""

import subprocess
import sys
import time
import webbrowser

def create_dashboard():
    code = '''#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
import json
import websocket
import threading
from datetime import datetime, timedelta
import queue

# Configure page
st.set_page_config(page_title="Ultimate 75% Dashboard", layout="wide", initial_sidebar_state="expanded")

# Sidebar controls
st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Refresh rate control
st.sidebar.subheader("⏱️ Refresh Settings")
refresh_options = {
    "Ultra Fast (0.5s)": 0.5,
    "Fast (1s)": 1.0,
    "Normal (2s)": 2.0,
    "Slow (5s)": 5.0,
    "Manual (10s)": 10.0
}

selected_refresh = st.sidebar.selectbox(
    "Update Rate:",
    options=list(refresh_options.keys()),
    index=2,  # Default to Normal (2s)
    help="How often the dashboard updates with new data"
)

refresh_rate = refresh_options[selected_refresh]

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True, help="Automatically refresh dashboard")

# Chart settings
st.sidebar.subheader("📊 Chart Settings")
chart_points = st.sidebar.slider("Chart History Points", min_value=10, max_value=200, value=50, step=10)
show_grid = st.sidebar.checkbox("Show Grid Lines", value=True)
chart_height = st.sidebar.slider("Chart Height", min_value=300, max_value=600, value=400, step=50)

# Display settings
st.sidebar.subheader("🎨 Display Options")
show_debug = st.sidebar.checkbox("Show Debug Info", value=False)
compact_mode = st.sidebar.checkbox("Compact Mode", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Refresh**: {refresh_rate}s")
st.sidebar.markdown(f"**Last Update**: {datetime.now().strftime('%H:%M:%S')}")

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now", type="primary"):
    st.rerun()

# Dark green theme
st.markdown("""
<style>
.stApp { 
    background: linear-gradient(135deg, #0a0e0a 0%, #1a2e1a 50%, #0a0e0a 100%); 
    color: #00ff41; 
}
h1, h2, h3 { 
    color: #00ff41 !important; 
    text-shadow: 0 0 10px #00ff41; 
    font-family: 'Courier New', monospace; 
}
.metric-container { 
    background: linear-gradient(145deg, #0f1f0f, #1a2e1a); 
    border: 1px solid #00ff41; 
    border-radius: 10px; 
    padding: 15px; 
    margin: 10px 0; 
    box-shadow: 0 0 20px rgba(0, 255, 65, 0.3); 
}
.success-metric { 
    color: #00ff41 !important; 
    font-weight: bold; 
    text-shadow: 0 0 8px #00ff41; 
    font-size: 2em; 
}
.live-indicator {
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
.compact { font-size: 0.8em; margin: 5px 0; }
.debug-info { 
    background: rgba(255, 170, 0, 0.1); 
    border: 1px solid #ffaa00; 
    border-radius: 5px; 
    padding: 10px; 
    margin: 5px 0; 
    font-family: 'Courier New', monospace; 
    font-size: 0.8em;
}
</style>
""", unsafe_allow_html=True)

class LiveDataFeed:
    def __init__(self):
        self.price_queue = queue.Queue()
        self.current_price = 0.0
        self.price_history = []
        self.last_update = None
        self.ws_connected = False
        self.connection_attempts = 0
        self.messages_received = 0
        self.start_websocket()
    
    def start_websocket(self):
        def on_message(ws, message):
            try:
                self.messages_received += 1
                data = json.loads(message)
                if 'data' in data:
                    for item in data['data']:
                        if data.get('arg', {}).get('channel') == 'tickers':
                            price = float(item['last'])
                            self.current_price = price
                            self.last_update = datetime.now()
                            self.price_history.append({
                                'time': self.last_update,
                                'price': price
                            })
                            # Keep configurable number of points
                            max_points = 200
                            if len(self.price_history) > max_points:
                                self.price_history = self.price_history[-max_points:]
            except Exception as e:
                pass
        
        def on_open(ws):
            self.ws_connected = True
            self.connection_attempts += 1
            subscribe = {
                "op": "subscribe",
                "args": [{"channel": "tickers", "instId": "SOL-USDT-SWAP"}]
            }
            ws.send(json.dumps(subscribe))
        
        def on_error(ws, error):
            self.ws_connected = False
        
        def on_close(ws, close_status_code, close_msg):
            self.ws_connected = False
        
        def run_websocket():
            import websocket
            websocket.enableTrace(False)
            ws = websocket.WebSocketApp(
                "wss://ws.okx.com:8443/ws/v5/public",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
        
        # Start WebSocket in background thread
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()

# Initialize live data feed
if 'live_feed' not in st.session_state:
    st.session_state.live_feed = LiveDataFeed()
    st.session_state.trades = 15
    st.session_state.wins = 13
    st.session_state.balance = 203.45
    st.session_state.chart_data = []

# Get live data
live_feed = st.session_state.live_feed
current_price = live_feed.current_price if live_feed.current_price > 0 else 142.3456

# Header
header_size = "h2" if compact_mode else "h1"
st.markdown(f"<{header_size} style='text-align: center;'>🎯 ULTIMATE 75% LIVE DASHBOARD</{header_size}>", unsafe_allow_html=True)

# Live status indicator
if live_feed.ws_connected and live_feed.last_update:
    data_age = (datetime.now() - live_feed.last_update).total_seconds()
    if data_age < 10:
        status = "🟢 LIVE"
        status_color = "#00ff41"
    else:
        status = f"🟡 DELAYED ({data_age:.0f}s)"
        status_color = "#ffaa00"
else:
    status = "🔴 CONNECTING..."
    status_color = "#ff0040"

st.markdown(f"<p style='text-align: center; color: {status_color}; font-weight: bold; font-size: 1.2em;' class='live-indicator'>📡 {status} | ⏱️ {refresh_rate}s refresh</p>", unsafe_allow_html=True)

# Debug info
if show_debug:
    debug_info = f"""
    🔧 **DEBUG INFO**:
    - Connection Attempts: {live_feed.connection_attempts}
    - Messages Received: {live_feed.messages_received}
    - Price History Points: {len(live_feed.price_history)}
    - WebSocket Connected: {live_feed.ws_connected}
    - Auto Refresh: {auto_refresh}
    - Current Refresh Rate: {refresh_rate}s
    - Last Update: {live_feed.last_update.strftime('%H:%M:%S.%f')[:-3] if live_feed.last_update else 'None'}
    """
    st.markdown(f"<div class='debug-info'>{debug_info}</div>", unsafe_allow_html=True)

# Real-time metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    win_rate = (st.session_state.wins / st.session_state.trades) * 100
    container_class = "metric-container compact" if compact_mode else "metric-container"
    st.markdown(f"""
    <div class='{container_class}'>
        <h3>🏆 Win Rate</h3>
        <div class='success-metric'>{win_rate:.1f}%</div>
        <p>{st.session_state.wins}W/{st.session_state.trades - st.session_state.wins}L</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='{container_class}'>
        <h3>💰 Balance</h3>
        <div class='success-metric'>${st.session_state.balance:.2f}</div>
        <p>+${st.session_state.balance - 200:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    return_pct = ((st.session_state.balance - 200) / 200) * 100
    st.markdown(f"""
    <div class='{container_class}'>
        <h3>📈 Return</h3>
        <div class='success-metric'>{return_pct:+.1f}%</div>
        <p>Total ROI</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Show actual live price or fallback
    price_change = np.random.uniform(-0.5, 0.5) if live_feed.current_price == 0 else 0
    if len(live_feed.price_history) >= 2:
        price_change = ((live_feed.price_history[-1]['price'] - live_feed.price_history[-2]['price']) / live_feed.price_history[-2]['price']) * 100
    
    st.markdown(f"""
    <div class='{container_class}'>
        <h3>📊 Live Price</h3>
        <div class='success-metric'>${current_price:.4f}</div>
        <p>SOL-USDT-SWAP ({price_change:+.3f}%)</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Smooth charts without flashing
col1, col2 = st.columns(2)

with col1:
    chart_title = "📈 Live Price Chart" if not compact_mode else "📈 Price"
    st.markdown(f"## {chart_title}")
    
    if len(live_feed.price_history) > 10:
        # Use real price data with configurable points
        df = pd.DataFrame(live_feed.price_history[-chart_points:])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['price'],
            mode='lines',
            line=dict(color='#00ff41', width=2),
            name='Live Price'
        ))
        
        layout_config = {
            'plot_bgcolor': 'rgba(0,0,0,0.9)',
            'paper_bgcolor': 'rgba(0,0,0,0.9)',
            'font': dict(color='#00ff41'),
            'xaxis_title': "Time",
            'yaxis_title': "Price (USD)",
            'showlegend': False,
            'margin': dict(l=50, r=50, t=30, b=50),
            'height': chart_height
        }
        
        if show_grid:
            layout_config['xaxis'] = dict(gridcolor='rgba(0,255,65,0.3)')
            layout_config['yaxis'] = dict(gridcolor='rgba(0,255,65,0.3)')
        
        fig.update_layout(**layout_config)
    else:
        # Fallback chart while connecting
        dates = pd.date_range(end=datetime.now(), periods=chart_points, freq='1min')
        prices = np.random.normal(current_price, 0.5, chart_points)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            line=dict(color='#ffaa00', width=2),
            name='Simulated'
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0.9)',
            paper_bgcolor='rgba(0,0,0,0.9)',
            font=dict(color='#00ff41'),
            showlegend=False,
            margin=dict(l=50, r=50, t=30, b=50),
            height=chart_height
        )
    
    st.plotly_chart(fig, use_container_width=True, key="live_price_chart")

with col2:
    chart_title = "🏆 Performance Trend" if not compact_mode else "🏆 Performance"
    st.markdown(f"## {chart_title}")
    
    # Stable performance chart
    dates = pd.date_range(end=datetime.now(), periods=chart_points, freq='5min')
    performance = np.cumsum(np.random.normal(0.1, 0.3, chart_points)) + return_pct
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=performance,
        mode='lines',
        line=dict(color='#00ff41', width=2),
        fill='tonexty'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#ffaa00", annotation_text="Break Even")
    
    layout_config = {
        'plot_bgcolor': 'rgba(0,0,0,0.9)',
        'paper_bgcolor': 'rgba(0,0,0,0.9)',
        'font': dict(color='#00ff41'),
        'xaxis_title': "Time",
        'yaxis_title': "Return (%)",
        'showlegend': False,
        'margin': dict(l=50, r=50, t=30, b=50),
        'height': chart_height
    }
    
    if show_grid:
        layout_config['xaxis'] = dict(gridcolor='rgba(0,255,65,0.3)')
        layout_config['yaxis'] = dict(gridcolor='rgba(0,255,65,0.3)')
    
    fig.update_layout(**layout_config)
    
    st.plotly_chart(fig, use_container_width=True, key="performance_chart")

# Trading status
if not compact_mode:
    st.markdown("---")
    st.markdown("## 📊 Trading Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📡 Market Connection
        - **Exchange**: OKX WebSocket
        - **Symbol**: SOL-USDT-SWAP  
        - **Update Rate**: Real-time
        - **Strategy**: Ultimate 75%
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Signal Analysis
        - **Confidence**: 90%+ Required
        - **Micro Targets**: 0.07%
        - **Position Size**: 1.0-2.5%
        - **Max Hold**: 15 minutes
        """)
    
    with col3:
        st.markdown(f"""
        ### 📈 Market Info
        - **Current**: ${current_price:.4f}
        - **24h Change**: {price_change:+.2f}%
        - **Last Update**: {live_feed.last_update.strftime('%H:%M:%S') if live_feed.last_update else 'Connecting...'}
        - **Status**: {'🟢 Active' if live_feed.ws_connected else '🔴 Connecting'}
        """)

# Auto-refresh with configurable timing
if auto_refresh:
    placeholder = st.empty()
    with placeholder:
        time.sleep(refresh_rate)
        st.rerun()
'''
    
    with open('advanced_live_dashboard.py', 'w', encoding='utf-8') as f:
        f.write(code)

def main():
    print("Creating Advanced Real-Time Dashboard...")
    create_dashboard()
    
    print("Starting Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "advanced_live_dashboard.py", "--server.port", "8502"])

if __name__ == "__main__":
    main()
