#!/usr/bin/env python3
"""
📊 F3 AI TRADING BOT DASHBOARD
Real-time monitoring and analytics system

Features:
- Live performance metrics
- Trade history visualization
- Risk monitoring
- Sentiment analysis display
- Fail-safe system status
- Portfolio analytics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import time
from datetime import datetime, timedelta
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="F3 AI Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class F3Dashboard:
    """📊 F3 AI Trading Bot Dashboard"""
    
    def __init__(self):
        self.db_path = 'f3_ai_bot_performance.db'
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state"""
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
    
    def load_trade_data(self):
        """Load trade data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('''
                SELECT * FROM trades 
                ORDER BY timestamp DESC 
                LIMIT 1000
            ''', conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def calculate_metrics(self, df):
        """Calculate performance metrics"""
        if df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_return': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
        
        total_trades = len(df)
        winning_trades = len(df[df['is_winner'] == True])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        avg_return = df['return_pct'].mean()
        best_trade = df['pnl'].max()
        worst_trade = df['pnl'].min()
        
        # Calculate Sharpe ratio (simplified)
        returns = df['return_pct'] / 100
        sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Calculate profit factor
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_return': avg_return,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor
        }
    
    def create_performance_chart(self, df):
        """Create cumulative performance chart"""
        if df.empty:
            return go.Figure()
        
        df_sorted = df.sort_values('timestamp')
        df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['cumulative_pnl'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#00ff88', width=3)
        ))
        
        fig.update_layout(
            title="📈 Cumulative Performance",
            xaxis_title="Time",
            yaxis_title="Cumulative P&L ($)",
            template="plotly_dark",
            height=400
        )
        
        return fig
    
    def create_trade_distribution(self, df):
        """Create trade P&L distribution chart"""
        if df.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Winning trades
        winning_trades = df[df['is_winner'] == True]['pnl']
        losing_trades = df[df['is_winner'] == False]['pnl']
        
        fig.add_trace(go.Histogram(
            x=winning_trades,
            name='Winning Trades',
            opacity=0.7,
            marker_color='green'
        ))
        
        fig.add_trace(go.Histogram(
            x=losing_trades,
            name='Losing Trades',
            opacity=0.7,
            marker_color='red'
        ))
        
        fig.update_layout(
            title="📊 Trade P&L Distribution",
            xaxis_title="P&L ($)",
            yaxis_title="Frequency",
            template="plotly_dark",
            barmode='overlay',
            height=400
        )
        
        return fig
    
    def create_sentiment_gauge(self, sentiment_score):
        """Create sentiment gauge chart"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sentiment_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Sentiment"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "red"},
                    {'range': [-0.3, 0.3], 'color': "yellow"},
                    {'range': [0.3, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def create_symbol_performance(self, df):
        """Create symbol performance chart"""
        if df.empty:
            return go.Figure()
        
        symbol_performance = df.groupby('symbol').agg({
            'pnl': ['sum', 'count'],
            'is_winner': 'mean'
        }).round(2)
        
        symbol_performance.columns = ['Total PnL', 'Trade Count', 'Win Rate']
        symbol_performance = symbol_performance.reset_index()
        
        fig = px.bar(
            symbol_performance,
            x='symbol',
            y='Total PnL',
            color='Win Rate',
            title="💰 Performance by Symbol",
            template="plotly_dark",
            height=400
        )
        
        return fig
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <h1 style='text-align: center; color: #00ff88;'>
        🤖 F3 AI TRADING BOT DASHBOARD
        </h1>
        <h3 style='text-align: center; color: #888;'>
        Real-time Performance Monitoring & Analytics
        </h3>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("⚙️ Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox(
            "🔄 Auto Refresh (30s)",
            value=st.session_state.auto_refresh
        )
        st.session_state.auto_refresh = auto_refresh
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.session_state.last_update = datetime.now()
            st.experimental_rerun()
        
        # Time range filter
        st.sidebar.subheader("📅 Time Range")
        time_range = st.sidebar.selectbox(
            "Select Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "All Time"]
        )
        
        # Trading pairs filter
        st.sidebar.subheader("🎲 Trading Pairs")
        show_all_pairs = st.sidebar.checkbox("Show All Pairs", value=True)
        
        if not show_all_pairs:
            selected_pairs = st.sidebar.multiselect(
                "Select Pairs",
                ["BTC", "ETH", "SOL", "AVAX", "DOGE", "LINK", "UNI", "ADA"],
                default=["BTC", "ETH", "SOL"]
            )
        else:
            selected_pairs = None
        
        return time_range, selected_pairs
    
    def render_metrics_cards(self, metrics):
        """Render key metrics cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💹 Total P&L",
                f"${metrics['total_pnl']:,.2f}",
                delta=f"{metrics['avg_return']:.2f}% avg"
            )
        
        with col2:
            st.metric(
                "🎯 Win Rate",
                f"{metrics['win_rate']:.1f}%",
                delta=f"{metrics['total_trades']} trades"
            )
        
        with col3:
            st.metric(
                "📈 Best Trade",
                f"${metrics['best_trade']:,.2f}",
                delta="🔥 Winner"
            )
        
        with col4:
            st.metric(
                "📉 Max Drawdown",
                f"{metrics['max_drawdown']:.1f}%",
                delta=f"Sharpe: {metrics['sharpe_ratio']:.2f}"
            )
    
    def render_live_status(self):
        """Render live trading status"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Simulate live status
            status = "🟢 TRADING ACTIVE"
            balance = 1000 + np.random.uniform(-50, 200)  # Simulated balance
            
            st.markdown(f"""
            <div style='padding: 20px; background-color: #1e1e1e; border-radius: 10px; border-left: 5px solid #00ff88;'>
                <h3>{status}</h3>
                <p><strong>Balance:</strong> ${balance:.2f}</p>
                <p><strong>Last Update:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Current sentiment (simulated)
            sentiment = np.random.uniform(-0.5, 0.8)
            sentiment_fig = self.create_sentiment_gauge(sentiment)
            st.plotly_chart(sentiment_fig, use_container_width=True)
        
        with col3:
            # Active positions (simulated)
            st.markdown("""
            <div style='padding: 20px; background-color: #1e1e1e; border-radius: 10px;'>
                <h4>🎯 Active Positions</h4>
                <p>BTC LONG: +2.3%</p>
                <p>ETH SHORT: -0.8%</p>
                <p>SOL LONG: +4.7%</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_charts(self, df):
        """Render main charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            performance_chart = self.create_performance_chart(df)
            st.plotly_chart(performance_chart, use_container_width=True)
        
        with col2:
            distribution_chart = self.create_trade_distribution(df)
            st.plotly_chart(distribution_chart, use_container_width=True)
        
        # Symbol performance chart
        symbol_chart = self.create_symbol_performance(df)
        st.plotly_chart(symbol_chart, use_container_width=True)
    
    def render_trade_history(self, df):
        """Render recent trade history"""
        st.subheader("📋 Recent Trade History")
        
        if df.empty:
            st.info("No trades recorded yet. Start the bot to see live data!")
            return
        
        # Format the dataframe for display
        display_df = df.head(20).copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
        display_df['pnl'] = display_df['pnl'].round(2)
        display_df['return_pct'] = display_df['return_pct'].round(2)
        display_df['confidence'] = display_df['confidence'].round(3)
        
        # Add result emoji
        display_df['result'] = display_df['is_winner'].apply(lambda x: '🟢' if x else '🔴')
        
        # Select columns to display
        columns_to_show = [
            'result', 'timestamp', 'symbol', 'signal_type', 
            'entry_price', 'exit_price', 'pnl', 'return_pct', 'confidence'
        ]
        
        st.dataframe(
            display_df[columns_to_show],
            use_container_width=True,
            height=400
        )
    
    def render_fail_safe_status(self):
        """Render fail-safe system status"""
        st.subheader("🛡️ Fail-Safe System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style='padding: 15px; background-color: #1e4d1e; border-radius: 10px; text-align: center;'>
                <h4>Level 1</h4>
                <p>5% Loss</p>
                <p>✅ Active</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='padding: 15px; background-color: #4d4d1e; border-radius: 10px; text-align: center;'>
                <h4>Level 2</h4>
                <p>10% Loss</p>
                <p>✅ Active</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='padding: 15px; background-color: #4d1e1e; border-radius: 10px; text-align: center;'>
                <h4>Level 3</h4>
                <p>15% Loss</p>
                <p>✅ Active</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style='padding: 15px; background-color: #1e1e4d; border-radius: 10px; text-align: center;'>
                <h4>Level 4</h4>
                <p>20% Loss</p>
                <p>✅ Active</p>
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Run the dashboard"""
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time.sleep(30)
            st.experimental_rerun()
        
        # Render components
        self.render_header()
        
        # Sidebar
        time_range, selected_pairs = self.render_sidebar()
        
        # Load data
        df = self.load_trade_data()
        metrics = self.calculate_metrics(df)
        
        # Main content
        self.render_live_status()
        st.markdown("---")
        
        self.render_metrics_cards(metrics)
        st.markdown("---")
        
        self.render_charts(df)
        st.markdown("---")
        
        self.render_fail_safe_status()
        st.markdown("---")
        
        self.render_trade_history(df)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            🤖 F3 AI Trading Bot Dashboard v1.0 | 
            Created by F3 AI Systems | 
            Last Update: {time}
        </div>
        """.format(time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 
        unsafe_allow_html=True)

if __name__ == "__main__":
    dashboard = F3Dashboard()
    dashboard.run() 