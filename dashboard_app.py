"""
Flask Web Application for Trading Bot Dashboard
Real-time monitoring and control interface
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import json
import threading
import time
from datetime import datetime
import os
from metrics_collector import MetricsCollector
from trend_analyzer import TrendAnalyzer, TrendDirection, TrendStrength
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
import sqlite3

app = Flask(__name__)
app.config['SECRET_KEY'] = 'okx_perp_bot_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global instances
metrics_collector = None
trend_analyzer = None
dashboard_data = {}
DATABASE_PATH = "bot_metrics.db"  # Database path for direct queries

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def init_dashboard(collector: MetricsCollector, analyzer: TrendAnalyzer = None):
    """Initialize dashboard with metrics collector and trend analyzer"""
    global metrics_collector, trend_analyzer
    metrics_collector = collector
    trend_analyzer = analyzer or TrendAnalyzer()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics data"""
    if not metrics_collector:
        return jsonify({'error': 'Metrics collector not initialized'})
    
    data = metrics_collector.get_trading_dashboard_data()
    return jsonify(data)

@app.route('/api/performance')
def get_performance():
    """Get performance metrics"""
    if not metrics_collector:
        return jsonify({'error': 'Metrics collector not initialized'})
    
    performance = metrics_collector.get_performance_summary()
    return jsonify(performance)

@app.route('/api/system')
def get_system_health():
    """Get system health metrics"""
    if not metrics_collector:
        return jsonify({'error': 'Metrics collector not initialized'})
    
    health = metrics_collector.get_system_health()
    return jsonify(health)

@app.route('/api/trends')
def get_trend_analysis():
    """Get trend analysis data"""
    if not metrics_collector:
        return jsonify({'error': 'Metrics collector not initialized'})
    
    trend_data = metrics_collector.trend_metrics
    return jsonify(trend_data)

@app.route('/api/charts/performance')
def get_performance_chart():
    """Get performance chart data"""
    if not metrics_collector:
        return jsonify({'error': 'Metrics collector not initialized'})
    
    # Get historical performance data
    history = metrics_collector.get_historical_metrics('performance', 24)
    
    if not history:
        return jsonify({'data': [], 'layout': {}})
    
    df = pd.DataFrame(history)
    
    # Create performance chart
    fig = go.Figure()
    
    # Total PnL line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['total_pnl'],
        mode='lines+markers',
        name='Total PnL',
        line=dict(color='#3fb950', width=3),
        marker=dict(color='#3fb950', size=6)
    ))
    
    # Win rate line (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['win_rate'],
        mode='lines+markers',
        name='Win Rate %',
        yaxis='y2',
        line=dict(color='#58a6ff', width=3),
        marker=dict(color='#58a6ff', size=6)
    ))
    
    fig.update_layout(
        title='Trading Performance',
        xaxis_title='Time',
        yaxis_title='PnL ($)',
        yaxis2=dict(
            title='Win Rate (%)',
            overlaying='y',
            side='right',
            gridcolor='#30363d',
            color='#f0f6fc',
            zerolinecolor='#30363d'
        ),
        template='plotly_dark',
        height=400,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#f0f6fc'),
        title_font=dict(color='#f0f6fc', size=16),
        xaxis=dict(
            gridcolor='#30363d',
            color='#f0f6fc',
            zerolinecolor='#30363d'
        ),
        yaxis=dict(
            gridcolor='#30363d',
            color='#f0f6fc',
            zerolinecolor='#30363d'
        ),
        legend=dict(
            font=dict(color='#f0f6fc'),
            bgcolor='rgba(13, 17, 23, 0.8)',
            bordercolor='#30363d'
        )
    )
    
    return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))

@app.route('/api/charts/system')
def get_system_chart():
    """Get system metrics chart data"""
    if not metrics_collector:
        return jsonify({'error': 'Metrics collector not initialized'})
    
    history = metrics_collector.get_historical_metrics('system', 6)
    
    if not history:
        return jsonify({'data': [], 'layout': {}})
    
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    
    # CPU usage
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cpu_usage'],
        mode='lines+markers',
        name='CPU %',
        line=dict(color='#f85149', width=3),
        marker=dict(color='#f85149', size=5)
    ))
    
    # Memory usage
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['memory_usage'],
        mode='lines+markers',
        name='Memory %',
        line=dict(color='#a5a5f0', width=3),
        marker=dict(color='#a5a5f0', size=5)
    ))
    
    fig.update_layout(
        title='System Health',
        xaxis_title='Time',
        yaxis_title='Usage (%)',
        template='plotly_dark',
        height=300,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#f0f6fc'),
        title_font=dict(color='#f0f6fc', size=16),
        xaxis=dict(
            gridcolor='#30363d',
            color='#f0f6fc',
            zerolinecolor='#30363d'
        ),
        yaxis=dict(
            gridcolor='#30363d',
            color='#f0f6fc',
            zerolinecolor='#30363d'
        ),
        legend=dict(
            font=dict(color='#f0f6fc'),
            bgcolor='rgba(13, 17, 23, 0.8)',
            bordercolor='#30363d'
        )
    )
    
    return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))

@app.route('/api/charts/trades')
def get_trades_chart():
    """Get recent trades chart"""
    if not metrics_collector:
        return jsonify({'error': 'Metrics collector not initialized'})
    
    trades = metrics_collector.get_historical_metrics('trades', 6)
    
    if not trades:
        return jsonify({'data': [], 'layout': {}})
    
    df = pd.DataFrame(trades)
    
    # Create candlestick-like chart for trades
    colors = ['#3fb950' if pnl > 0 else '#f85149' for pnl in df['pnl']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['pnl'],
        marker_color=colors,
        name='Trade PnL'
    ))
    
    fig.update_layout(
        title='Recent Trades',
        xaxis_title='Time',
        yaxis_title='PnL ($)',
        template='plotly_dark',
        height=300,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#f0f6fc'),
        title_font=dict(color='#f0f6fc', size=16),
        xaxis=dict(
            gridcolor='#30363d',
            color='#f0f6fc',
            zerolinecolor='#30363d'
        ),
        yaxis=dict(
            gridcolor='#30363d',
            color='#f0f6fc',
            zerolinecolor='#30363d'
        ),
        legend=dict(
            font=dict(color='#f0f6fc'),
            bgcolor='rgba(13, 17, 23, 0.8)',
            bordercolor='#30363d'
        )
    )
    
    return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))

@app.route('/api/export/<format>')
def export_metrics(format):
    """Export metrics data"""
    if not metrics_collector:
        return jsonify({'error': 'Metrics collector not initialized'})
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'bot_metrics_{timestamp}.{format}'
    
    try:
        metrics_collector.export_metrics(filename, format)
        return send_from_directory('.', filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected to dashboard')
    emit('status', {'msg': 'Connected to trading bot dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected from dashboard')

@socketio.on('request_update')
def handle_request_update():
    """Handle request for data update"""
    if metrics_collector:
        data = metrics_collector.get_real_time_metrics()
        emit('metrics_update', data)

def start_real_time_updates():
    """Start real-time data updates via WebSocket"""
    def update_loop():
        while True:
            if metrics_collector:
                try:
                    data = metrics_collector.get_real_time_metrics()
                    socketio.emit('metrics_update', data)
                    time.sleep(2)  # Update every 2 seconds
                except Exception as e:
                    print(f"Error in real-time update: {e}")
                    time.sleep(5)
            else:
                time.sleep(1)
    
    thread = threading.Thread(target=update_loop)
    thread.daemon = True
    thread.start()

@app.route('/api/trades')
def get_trades():
    """Get recent trades data"""
    try:
        # Get recent trades from database
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, side, size, price, pnl, status, timestamp, metadata
                FROM trades 
                ORDER BY timestamp DESC 
                LIMIT 50
            """)
            
            trades = []
            for row in cursor.fetchall():
                trades.append({
                    'symbol': row[0],
                    'side': row[1],
                    'size': safe_float(row[2]),
                    'price': safe_float(row[3]),
                    'pnl': safe_float(row[4]),
                    'status': row[5],
                    'timestamp': row[6],
                    'metadata': row[7]
                })
            
            return jsonify(trades)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/real-time-metrics')
def get_real_time_metrics():
    """Get real-time metrics data"""
    try:
        if metrics_collector:
            return jsonify(metrics_collector.get_real_time_metrics())
        else:
            return jsonify({'error': 'Metrics collector not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    """Get current alerts"""
    try:
        alerts = []
        
        if metrics_collector:
            # Get latest trading metrics
            latest_metrics = metrics_collector.get_latest_trading_metrics()
            trend_summary = metrics_collector.get_trend_summary()
            
            # Check for alerts
            if latest_metrics:
                # High drawdown alert
                if latest_metrics.get('max_drawdown', 0) > 0.1:  # 10%
                    alerts.append({
                        'type': 'warning',
                        'message': f"High drawdown detected: {latest_metrics['max_drawdown']*100:.1f}%",
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Low win rate alert
                if latest_metrics.get('win_rate', 1) < 0.5 and latest_metrics.get('total_trades', 0) > 5:
                    alerts.append({
                        'type': 'warning',
                        'message': f"Low win rate: {latest_metrics['win_rate']*100:.1f}%",
                        'timestamp': datetime.now().isoformat()
                    })
                
                # High profit alert
                if latest_metrics.get('total_pnl', 0) > 500:
                    alerts.append({
                        'type': 'success',
                        'message': f"High profits achieved: ${latest_metrics['total_pnl']:.2f}",
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Trend alerts
            if trend_summary.get('confidence', 0) > 80:
                direction = trend_summary.get('current_direction', 'unknown')
                alerts.append({
                    'type': 'info',
                    'message': f"Strong {direction} trend detected (confidence: {trend_summary['confidence']:.0f}%)",
                    'timestamp': datetime.now().isoformat()
                })
        
        return jsonify(alerts)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance-chart-data')
def get_performance_chart_data():
    """Get performance chart data"""
    try:
        # Get performance data from database
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, total_pnl, total_trades, win_rate, max_drawdown
                FROM performance 
                ORDER BY timestamp ASC
                LIMIT 100
            """)
            
            chart_data = {
                'timestamps': [],
                'pnl': [],
                'trades': [],
                'win_rate': [],
                'drawdown': []
            }
            
            for row in cursor.fetchall():
                chart_data['timestamps'].append(row[0])
                chart_data['pnl'].append(safe_float(row[1]))
                chart_data['trades'].append(safe_float(row[2]))
                chart_data['win_rate'].append(safe_float(row[3]))
                chart_data['drawdown'].append(safe_float(row[4]))
            
            return jsonify(chart_data)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Template directory setup
if not os.path.exists('templates'):
    os.makedirs('templates')

# Create the HTML template
dashboard_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OKX Perpetual Trading Bot Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { 
            background: #0d1117; 
            color: #ffffff; 
            font-family: 'Arial', sans-serif; 
            min-height: 100vh;
        }
        .dashboard-card { 
            background: #161b22; 
            border: 1px solid #30363d; 
            border-radius: 10px; 
            margin-bottom: 20px; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .metric-card { 
            background: linear-gradient(135deg, #238636 0%, #2ea043 100%); 
            color: white;
        }
        .status-good { color: #3fb950; }
        .status-warning { color: #d29922; }
        .status-danger { color: #f85149; }
        .trend-indicator { font-size: 1.2em; font-weight: bold; }
        .navbar-brand { font-weight: bold; color: #3fb950 !important; }
        .chart-container { 
            height: 400px; 
            background: #0d1117;
            border-radius: 8px;
            padding: 10px;
        }
        .metric-value { font-size: 2em; font-weight: bold; color: #3fb950; }
        .metric-label { font-size: 0.9em; color: #8b949e; }
        .connection-status { position: fixed; top: 10px; right: 10px; z-index: 1000; }
        .navbar-dark { background-color: #161b22 !important; }
        .card { background-color: #161b22; border-color: #30363d; }
        .card-header { background-color: #21262d; border-color: #30363d; }
        .btn-primary { background-color: #238636; border-color: #238636; }
        .btn-primary:hover { background-color: #2ea043; border-color: #2ea043; }
        .text-muted { color: #8b949e !important; }
        
        /* Plotly chart dark mode enhancements */
        .js-plotly-plot .plotly .main-svg {
            background: #0d1117 !important;
        }
        .js-plotly-plot .plotly .bg {
            fill: #0d1117 !important;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">
                <i class="fas fa-robot"></i> OKX Perpetual Trading Bot Dashboard
            </span>
            <div class="connection-status">
                <span id="connectionStatus" class="badge bg-secondary">Connecting...</span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- Key Metrics Row -->
        <div class="row">
            <div class="col-md-3">
                <div class="card dashboard-card metric-card text-center p-3">
                    <div class="metric-value" id="totalPnl">$0.00</div>
                    <div class="metric-label">Total PnL</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card dashboard-card metric-card text-center p-3">
                    <div class="metric-value" id="winRate">0%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card dashboard-card metric-card text-center p-3">
                    <div class="metric-value" id="totalTrades">0</div>
                    <div class="metric-label">Total Trades</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card dashboard-card metric-card text-center p-3">
                    <div class="metric-value" id="activePositions">0</div>
                    <div class="metric-label">Active Positions</div>
                </div>
            </div>
        </div>

        <!-- Trend Analysis Row -->
        <div class="row">
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-line"></i> Trend Analysis</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6">
                                <div class="trend-indicator" id="trendDirection">SIDEWAYS</div>
                                <small class="text-muted">Direction</small>
                            </div>
                            <div class="col-6">
                                <div class="trend-indicator" id="trendStrength">MODERATE</div>
                                <small class="text-muted">Strength</small>
                            </div>
                        </div>
                        <div class="mt-3">
                            <div class="progress">
                                <div id="trendConfidence" class="progress-bar bg-info" style="width: 50%">50%</div>
                            </div>
                            <small class="text-muted">Confidence</small>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-header">
                        <h5><i class="fas fa-heartbeat"></i> System Health</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-4">
                                <div id="cpuUsage">0%</div>
                                <small class="text-muted">CPU</small>
                            </div>
                            <div class="col-4">
                                <div id="memoryUsage">0%</div>
                                <small class="text-muted">Memory</small>
                            </div>
                            <div class="col-4">
                                <div id="systemStatus" class="status-good">Healthy</div>
                                <small class="text-muted">Status</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row">
            <div class="col-md-8">
                <div class="card dashboard-card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-area"></i> Performance Chart</h5>
                    </div>
                    <div class="card-body">
                        <div id="performanceChart" class="chart-container"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card dashboard-card">
                    <div class="card-header">
                        <h5><i class="fas fa-exchange-alt"></i> Recent Trades</h5>
                    </div>
                    <div class="card-body">
                        <div id="tradesChart" style="height: 300px;"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- System Metrics Row -->
        <div class="row">
            <div class="col-md-12">
                <div class="card dashboard-card">
                    <div class="card-header">
                        <h5><i class="fas fa-server"></i> System Metrics</h5>
                    </div>
                    <div class="card-body">
                        <div id="systemChart" style="height: 300px;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        socket.on('connect', function() {
            document.getElementById('connectionStatus').textContent = 'Connected';
            document.getElementById('connectionStatus').className = 'badge bg-success';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            document.getElementById('connectionStatus').className = 'badge bg-danger';
        });
        
        socket.on('metrics_update', function(data) {
            updateDashboard(data);
        });
        
        function updateDashboard(data) {
            // Update key metrics
            if (data.trading) {
                document.getElementById('totalPnl').textContent = `$${(data.trading.total_pnl || 0).toFixed(2)}`;
                document.getElementById('winRate').textContent = `${(data.trading.win_rate || 0).toFixed(1)}%`;
                document.getElementById('totalTrades').textContent = data.trading.total_trades || 0;
                document.getElementById('activePositions').textContent = data.trading.active_positions || 0;
            }
            
            // Update trend analysis
            if (data.trend) {
                const direction = data.trend.direction || 'SIDEWAYS';
                const strength = data.trend.strength || 'MODERATE';
                const confidence = data.trend.confidence || 50;
                
                document.getElementById('trendDirection').textContent = direction.replace('_', ' ');
                document.getElementById('trendStrength').textContent = strength;
                document.getElementById('trendConfidence').style.width = `${confidence}%`;
                document.getElementById('trendConfidence').textContent = `${confidence.toFixed(0)}%`;
                
                // Set trend direction color
                const directionElement = document.getElementById('trendDirection');
                if (direction.includes('UPTREND')) {
                    directionElement.className = 'trend-indicator status-good';
                } else if (direction.includes('DOWNTREND')) {
                    directionElement.className = 'trend-indicator status-danger';
                } else {
                    directionElement.className = 'trend-indicator status-warning';
                }
            }
            
            // Update system health
            if (data.system) {
                document.getElementById('cpuUsage').textContent = `${(data.system.cpu_usage || 0).toFixed(1)}%`;
                document.getElementById('memoryUsage').textContent = `${(data.system.memory_usage || 0).toFixed(1)}%`;
                
                const systemStatus = document.getElementById('systemStatus');
                const cpu = data.system.cpu_usage || 0;
                const memory = data.system.memory_usage || 0;
                
                if (cpu > 80 || memory > 80) {
                    systemStatus.textContent = 'Warning';
                    systemStatus.className = 'status-danger';
                } else if (cpu > 60 || memory > 60) {
                    systemStatus.textContent = 'Medium';
                    systemStatus.className = 'status-warning';
                } else {
                    systemStatus.textContent = 'Healthy';
                    systemStatus.className = 'status-good';
                }
            }
        }
        
        // Load charts
        function loadCharts() {
            // Load performance chart
            fetch('/api/charts/performance')
                .then(response => response.json())
                .then(data => {
                    if (data.data) {
                        Plotly.newPlot('performanceChart', data.data, data.layout);
                    }
                });
            
            // Load system chart
            fetch('/api/charts/system')
                .then(response => response.json())
                .then(data => {
                    if (data.data) {
                        Plotly.newPlot('systemChart', data.data, data.layout);
                    }
                });
            
            // Load trades chart
            fetch('/api/charts/trades')
                .then(response => response.json())
                .then(data => {
                    if (data.data) {
                        Plotly.newPlot('tradesChart', data.data, data.layout);
                    }
                });
        }
        
        // Initial load
        loadCharts();
        
        // Refresh charts every 30 seconds
        setInterval(loadCharts, 30000);
        
        // Request immediate update
        socket.emit('request_update');
    </script>
</body>
</html>
'''

# Save the HTML template
with open('templates/dashboard.html', 'w') as f:
    f.write(dashboard_html)

def run_dashboard(host='127.0.0.1', port=5000, debug=False):
    """Run the dashboard application"""
    start_real_time_updates()
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    # For testing purposes
    from metrics_collector import MetricsCollector
    from trend_analyzer import TrendAnalyzer
    
    collector = MetricsCollector()
    analyzer = TrendAnalyzer()
    
    init_dashboard(collector, analyzer)
    collector.start_collection()
    
    print("Starting Trading Bot Dashboard...")
    print("Open http://127.0.0.1:5000 in your browser")
    
    run_dashboard(debug=True) 