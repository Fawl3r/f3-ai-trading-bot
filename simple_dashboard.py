#!/usr/bin/env python3
"""
Simple Hyperliquid Trading Dashboard - FIXED REFRESH
Direct connection to real account data
"""

import streamlit as st
import time
import asyncio
import json
from datetime import datetime
from hyperliquid.info import Info
from hyperliquid.utils import constants
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title="🚀 Live Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .metric-card { 
        background: #262730; 
        padding: 1rem; 
        border-radius: 10px; 
        border-left: 4px solid #00FF88;
        margin: 0.5rem 0;
    }
    .trading-status { 
        background: linear-gradient(90deg, #1f4037 0%, #99f2c8 100%);
        padding: 1rem; 
        border-radius: 10px; 
        text-align: center;
        font-weight: bold;
        color: #000;
    }
    .price-card {
        background: #1E1E1E;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 3px solid #00D4FF;
    }
</style>
""", unsafe_allow_html=True)

class DashboardData:
    def __init__(self):
        self.testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
        self.base_url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
        self.info = Info(self.base_url, skip_ws=True)
        self.account_address = os.getenv('HYPERLIQUID_WALLET_ADDRESS', '')
        
    def get_account_balance(self):
        """Get account balance"""
        try:
            user_state = self.info.user_state(self.account_address)
            if user_state and 'marginSummary' in user_state:
                return float(user_state['marginSummary']['accountValue'])
            return 0.0
        except Exception as e:
            return 0.0
    
    def get_network_status(self):
        """Get network and connection status"""
        try:
            self.info.all_mids()  # Test connection
            return "CONNECTED"
        except:
            return "DISCONNECTED"
    
    def get_live_prices(self):
        """Get live crypto prices"""
        try:
            all_mids = self.info.all_mids()
            symbols = ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX']
            prices = {}
            
            for symbol in symbols:
                if symbol in all_mids:
                    price = float(all_mids[symbol])
                    
                    # Get 24h stats
                    meta = self.info.meta()
                    universe = meta.get('universe', [])
                    
                    change_24h = 0.0
                    for coin_info in universe:
                        if coin_info.get('name') == symbol:
                            change_24h = float(coin_info.get('prevDayPx', price))
                            if change_24h > 0:
                                change_24h = (price - change_24h) / change_24h * 100
                            break
                    
                    prices[symbol] = {
                        'price': price,
                        'change_24h': change_24h
                    }
            
            return prices
        except Exception as e:
            return {}
    
    def get_positions(self):
        """Get current positions"""
        try:
            user_state = self.info.user_state(self.account_address)
            if not user_state or 'assetPositions' not in user_state:
                return 0
            
            active_positions = 0
            for pos in user_state['assetPositions']:
                if abs(float(pos['position']['szi'])) > 0:
                    active_positions += 1
            
            return active_positions
        except Exception as e:
            return 0
    
    def get_trade_history(self, limit=5):
        """Get recent trade history"""
        try:
            user_fills = self.info.user_fills(self.account_address)
            if not user_fills:
                return []
            
            trades = []
            for fill in user_fills[:limit]:
                trade_time = datetime.fromtimestamp(int(fill['time']) / 1000)
                trades.append({
                    'time': trade_time.strftime('%m/%d %H:%M:%S'),
                    'side': 'BUY' if float(fill['sz']) > 0 else 'SELL',
                    'symbol': fill['coin'],
                    'size': abs(float(fill['sz'])),
                    'price': f"${float(fill['px']):,.2f}"
                })
            
            return trades
        except Exception as e:
            return []

# Initialize dashboard data
@st.cache_resource
def get_dashboard():
    return DashboardData()

def main():
    # Header
    st.markdown("# 🚀 Live Trading Dashboard")
    
    # Real-time data container
    placeholder = st.empty()
    
    # Get dashboard instance
    dashboard = get_dashboard()
    
    with placeholder.container():
        # Get fresh data each time
        balance = dashboard.get_account_balance()
        network = "MAINNET" if not dashboard.testnet else "TESTNET"
        wallet = dashboard.account_address[:10] + "..." if dashboard.account_address else "Not Set"
        status = dashboard.get_network_status()
        positions = dashboard.get_positions()
        prices = dashboard.get_live_prices()
        trades = dashboard.get_trade_history()
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💰 Balance</h4>
                <h2>${balance:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🌐 Network</h4>
                <h3>{network}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔗 Wallet</h4>
                <h4>{wallet}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            color = "🟢" if status == "CONNECTED" else "🔴"
            st.markdown(f"""
            <div class="metric-card">
                <h4>📡 Network Status</h4>
                <h3>{color} {status}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Time and positions
        col5, col6 = st.columns(2)
        with col5:
            current_time = datetime.now().strftime('%H:%M:%S')
            st.markdown(f"""
            <div class="metric-card">
                <h4>⏰ Time</h4>
                <h3>{current_time}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Positions</h4>
                <h3>{positions}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Live Prices Section
        st.markdown("## 📊 Live Prices")
        if prices:
            for symbol, data in prices.items():
                price = data['price']
                change = data['change_24h']
                change_color = "🟢" if change >= 0 else "🔴"
                status_dot = "🟢"  # All connected
                
                st.markdown(f"""
                <div class="price-card">
                    <strong>{symbol}</strong> &nbsp;&nbsp;&nbsp; 
                    <strong>${price:,.2f}</strong> &nbsp;&nbsp;&nbsp;
                    {change_color} {change:+.2f}% &nbsp;&nbsp;&nbsp;
                    Status: {status_dot}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Unable to fetch live prices")
        
        # Recent Trading Activity
        st.markdown("## 📈 Recent Trading Activity")
        if trades:
            for trade in trades:
                side_color = "🟢" if trade['side'] == 'BUY' else "🔴"
                st.markdown(f"""
                <div style="background: #2D2D2D; padding: 0.5rem; margin: 0.3rem 0; border-radius: 5px;">
                    <strong>{trade['time']}</strong> &nbsp;&nbsp;
                    {side_color} <strong>{trade['side']}</strong> &nbsp;&nbsp;
                    <strong>{trade['symbol']}</strong> &nbsp;&nbsp;
                    Size: {trade['size']:.4f} &nbsp;&nbsp;
                    Price: {trade['price']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📝 No recent trading activity")
        
        # Trading Bot Status
        st.markdown("## 🤖 Trading Bot Status")
        st.markdown(f"""
        <div class="trading-status">
            🟢 LIVE TRADING - Real Money at Risk!
        </div>
        """, unsafe_allow_html=True)
        
        bot_info = f"""
        **Current Status:** Active  
        **Balance:** ${balance:.2f}  
        **Network:** {network}  
        **Last Update:** {datetime.now().strftime('%H:%M:%S')}  
        """
        st.markdown(bot_info)
    
    # Auto-refresh every 5 seconds
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main() 