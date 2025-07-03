# 🎛️ Dashboard Controls Guide

## User-Configurable Settings ⏱️

The Ultimate 75% Dashboard now includes a **comprehensive control panel** in the sidebar allowing full customization of refresh rates and display options.

## ⏱️ Refresh Settings

### Available Refresh Rates
- **Ultra Fast (0.5s)**: Maximum responsiveness for active monitoring
- **Fast (1s)**: High-frequency updates for scalping strategies  
- **Normal (2s)**: Default balanced refresh rate ✅
- **Slow (5s)**: Reduced frequency for lower bandwidth
- **Manual (10s)**: Minimal auto-refresh, use manual button

### Auto-Refresh Control
- **🔄 Auto Refresh**: Toggle automatic updates on/off
- **🔄 Refresh Now**: Manual refresh button for instant updates
- **Current Rate Display**: Shows selected refresh rate in status bar

## 📊 Chart Settings

### Chart Customization
- **Chart History Points**: 10-200 data points (slider control)
  - More points = longer history
  - Fewer points = faster rendering
- **Chart Height**: 300-600 pixels (adjustable)
- **Show Grid Lines**: Toggle chart grid visibility

### Performance Optimization
- **Chart Points**: Affects rendering speed and memory usage
- **Height**: Customize for screen size and preferences
- **Grid Lines**: Can be disabled for cleaner appearance

## 🎨 Display Options

### Layout Modes
- **Compact Mode**: Smaller headers and metrics for dense layouts
- **Standard Mode**: Full-size display for main monitoring ✅

### Debug Information
- **Show Debug Info**: Technical connection details
  - Connection attempts count
  - Messages received counter
  - WebSocket status
  - Price history buffer size
  - Refresh rate confirmation
  - Precise update timestamps

## 🎯 Usage Examples

### Day Trading Setup
```
Refresh Rate: Ultra Fast (0.5s)
Chart Points: 50-100
Auto Refresh: ON
Compact Mode: OFF
Debug Info: OFF
```

### Swing Trading Setup  
```
Refresh Rate: Normal (2s)
Chart Points: 100-200
Auto Refresh: ON
Compact Mode: OFF
Debug Info: OFF
```

### Resource Conservation
```
Refresh Rate: Slow (5s)
Chart Points: 20-30
Auto Refresh: ON
Compact Mode: ON
Debug Info: OFF
```

### Debugging/Development
```
Refresh Rate: Fast (1s)
Chart Points: 50
Auto Refresh: ON
Compact Mode: OFF
Debug Info: ON
```

## 📡 Real-Time Indicators

### Status Display
- **📡 Status**: Live connection indicator with refresh rate
- **⏱️ Refresh Rate**: Shows current setting in status bar
- **🕒 Last Update**: Precise timestamp in sidebar

### Color Coding
- **🟢 LIVE**: Fresh data (< 10 seconds old)
- **🟡 DELAYED**: Older data (10+ seconds old)  
- **🔴 CONNECTING**: No connection/startup

## 🔧 Technical Details

### Performance Impact
- **Ultra Fast (0.5s)**: High CPU/bandwidth usage
- **Fast (1s)**: Moderate resource usage
- **Normal (2s)**: Balanced performance ✅
- **Slow (5s)**: Low resource usage
- **Manual (10s)**: Minimal background activity

### Memory Management
- Price history automatically limited to prevent memory leaks
- Chart rendering optimized for selected point count
- WebSocket reconnection handled automatically

### Browser Compatibility
- All refresh rates work in modern browsers
- Mobile devices may benefit from slower refresh rates
- Auto-refresh can be disabled for manual control

## 🚀 Getting Started

1. **Open Dashboard**: `python dashboard_launcher.py`
2. **Sidebar**: Controls appear on the left side
3. **Select Rate**: Choose from dropdown menu
4. **Customize**: Adjust charts and display options
5. **Monitor**: Watch real-time updates at your preferred speed

## 💡 Pro Tips

- **Start with Normal (2s)** for balanced performance
- **Use Ultra Fast** only during active trading periods
- **Enable Debug Info** when troubleshooting connections
- **Compact Mode** is perfect for multi-monitor setups
- **Manual Refresh** saves bandwidth on slow connections

The dashboard now adapts to your exact monitoring needs! 🎯 