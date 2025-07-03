#!/usr/bin/env python3
"""
Simple Hyperliquid Dashboard
Real-time data from your live trading bot
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Hyperliquid imports
from hyperliquid.info import Info
from hyperliquid.utils import constants

load_dotenv()

class HyperliquidDashboard:
    def __init__(self):
        self.setup_page()
        self.init_hyperliquid()
    
    def setup_page(self):
        st.set_page_config(
            page_title="Hyperliquid Live Dashboard",
            page_icon="🚀",
            layout="wide"
        )
    
    def init_hyperliquid(self):
        """Initialize Hyperliquid connection"""
        testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
        base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.info = Info(base_url, skip_ws=True)
        self.wallet = os.getenv('HYPERLIQUID_ACCOUNT_ADDRESS')
        self.network = "TESTNET" if testnet else "MAINNET"
    
    def run(self):
        """Main dashboard"""
        
        # Header
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h1>🚀 Hyperliquid Live Dashboard</h1>
            <p>Real-time Trading Data | Network: {self.network} | Wallet: {self.wallet[:8]}...{self.wallet[-4:]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Auto refresh
        auto_refresh = st.sidebar.checkbox("Auto Refresh (10s)", value=True)
        if auto_refresh:
            time.sleep(1)
            st.rerun()
        
        # Main content
        col1, col2, col3 = st.columns(3)
        
        # Account Info
        with col1:
            st.subheader("💰 Account Status")
            try:
                user_state = self.info.user_state(self.wallet)
                balance = float(user_state.get('marginSummary', {}).get('accountValue', 0))
                pnl = float(user_state.get('marginSummary', {}).get('totalNtlPos', 0))
                
                st.metric("Balance", f"${balance:,.2f}")
                st.metric("Total PnL", f"${pnl:,.2f}", delta=pnl)
                
                # Positions
                positions = user_state.get('assetPositions', [])
                active_positions = len([p for p in positions if float(p.get('position', {}).get('szi', 0)) != 0])
                st.metric("Active Positions", active_positions)
                
            except Exception as e:
                st.error(f"Error loading account: {e}")
        
        # Live Price Data
        with col2:
            st.subheader("📈 BTC Live Price")
            try:
                all_mids = self.info.all_mids()
                btc_price = float(all_mids.get('BTC', 0))
                
                st.metric("BTC Price", f"${btc_price:,.2f}")
                
                # Price change (simple calculation)
                if 'prev_price' not in st.session_state:
                    st.session_state.prev_price = btc_price
                
                price_change = btc_price - st.session_state.prev_price
                change_pct = (price_change / st.session_state.prev_price) * 100 if st.session_state.prev_price > 0 else 0
                
                st.metric("5s Change", f"${price_change:,.2f}", delta=f"{change_pct:.3f}%")
                st.session_state.prev_price = btc_price
                
            except Exception as e:
                st.error(f"Error loading price: {e}")
        
        # Trading Status
        with col3:
            st.subheader("🤖 Bot Status")
            if self.network == "TESTNET":
                st.success("🧪 TESTNET - Safe Testing")
            else:
                st.warning("💰 MAINNET - Real Money!")
            
            current_time = datetime.now().strftime("%H:%M:%S")
            st.info(f"⏰ Last Update: {current_time}")
        
        # Recent Trades
        st.subheader("📊 Recent Trades")
        try:
            fills = self.info.user_fills(self.wallet)
            
            if fills:
                # Convert to DataFrame
                trades_data = []
                for fill in fills[:10]:  # Last 10 trades
                    fill_time = datetime.fromtimestamp(fill.get('time', 0) / 1000)
                    trades_data.append({
                        'Time': fill_time.strftime('%H:%M:%S'),
                        'Side': '🟢 BUY' if fill.get('side') == 'B' else '🔴 SELL',
                        'Size': f"{fill.get('sz', 0)} {fill.get('coin', 'N/A')}",
                        'Price': f"${fill.get('px', 0):,.2f}",
                        'ID': fill.get('oid', 'N/A')
                    })
                
                if trades_data:
                    df = pd.DataFrame(trades_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No recent trades found")
            else:
                st.info("No trade history available")
                
        except Exception as e:
            st.error(f"Error loading trades: {e}")
        
        # Position Details
        st.subheader("📈 Open Positions")
        try:
            user_state = self.info.user_state(self.wallet)
            positions = user_state.get('assetPositions', [])
            
            active_positions = []
            for pos in positions:
                position_size = float(pos.get('position', {}).get('szi', 0))
                if abs(position_size) > 0.000001:  # Active position
                    coin = pos.get('position', {}).get('coin', 'N/A')
                    pnl = float(pos.get('unrealizedPnl', 0))
                    entry_px = float(pos.get('position', {}).get('entryPx', 0))
                    
                    active_positions.append({
                        'Coin': coin,
                        'Size': f"{position_size:.6f}",
                        'Side': '🟢 LONG' if position_size > 0 else '🔴 SHORT',
                        'Entry Price': f"${entry_px:,.2f}",
                        'PnL': f"${pnl:,.2f}",
                        'PnL %': f"{(pnl/abs(position_size*entry_px)*100):,.2f}%" if entry_px > 0 else "0%"
                    })
            
            if active_positions:
                df_positions = pd.DataFrame(active_positions)
                st.dataframe(df_positions, use_container_width=True, hide_index=True)
            else:
                st.info("No open positions")
                
        except Exception as e:
            st.error(f"Error loading positions: {e}")

def main():
    dashboard = HyperliquidDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 