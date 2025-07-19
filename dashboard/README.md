# Trading Bot Dashboard

A comprehensive, modern web interface for controlling and monitoring your trading bot with real-time features.

## 🚀 Features

### ✅ **Bot Process Control**
- **Start/Stop Bot**: One-click bot process management
- **Real-time Status**: Live status indicator with visual feedback
- **Process Monitoring**: Automatic detection of bot state changes

### ✅ **Live Log Streaming**
- **Real-time Logs**: Live streaming of bot output with timestamps
- **Resizable Panel**: Adjustable log window size
- **Log Management**: Clear logs and scroll through history
- **Auto-scroll**: Automatically scrolls to latest entries

### ✅ **Performance Analytics**
- **Live Metrics**: Real-time performance indicators
  - Total Return
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
- **Interactive Charts**: Chart.js powered equity curve visualization
- **Auto-refresh**: Automatic analytics updates
- **Responsive Design**: Charts adapt to panel size

### ✅ **Symbol Management**
- **Popular Stock Picker**: Quick access to 100+ popular stocks
- **Custom Symbols**: Add any stock symbol manually
- **Symbol Categories**: Organized by sector (Tech, Finance, Healthcare, etc.)
- **Real-time Updates**: Instant symbol list updates
- **Persistent Storage**: Symbols saved to configuration

### ✅ **Recent Trades**
- **Trade History**: Display recent bot trades
- **Trade Details**: Symbol, side, quantity, price, timestamp
- **Visual Indicators**: Color-coded buy/sell badges

### ✅ **Modern UI/UX**
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Resizable Panels**: Drag to resize any panel
- **Toast Notifications**: Real-time feedback for actions
- **Loading States**: Visual feedback during operations
- **Professional Styling**: Modern gradient design with glassmorphism

## 🛠️ Installation

### Prerequisites
```bash
# Install required Python packages
pip install flask flask-socketio psutil
```

### Quick Start
```bash
# Navigate to dashboard directory
cd dashboard

# Start the dashboard
python start_dashboard.py
```

### Alternative Start Methods
```bash
# Method 1: Direct Flask run
python app.py

# Method 2: Using the main project
cd ..
python -m dashboard.app
```

## 📱 Usage Guide

### 1. **Starting the Dashboard**
1. Open terminal and navigate to `dashboard/` directory
2. Run `python start_dashboard.py`
3. Open browser to `http://localhost:5000`

### 2. **Bot Control**
- **Start Bot**: Click the green "Start Bot" button
- **Stop Bot**: Click the red "Stop Bot" button
- **Status**: Real-time status indicator shows current state
- **Refresh**: Click refresh button to update all data

### 3. **Managing Symbols**
- **Add Symbol**: Type symbol in input field and press Enter or click "+"
- **Popular Symbols**: Click any symbol from the grid to add instantly
- **Remove Symbol**: Click the "×" button next to any symbol
- **Save Changes**: Click the save button to persist changes

### 4. **Viewing Logs**
- **Live Stream**: Logs appear in real-time as bot runs
- **Clear Logs**: Click trash icon to clear log history
- **Resize Panel**: Drag panel corners to resize log window

### 5. **Analytics Dashboard**
- **Metrics Cards**: View key performance indicators
- **Equity Chart**: Interactive chart showing portfolio value over time
- **Refresh Data**: Click refresh button to update analytics
- **Recent Trades**: View latest bot trades in table format

## 🎨 UI Features

### **Resizable Panels**
- Drag any panel corner to resize
- Panels maintain aspect ratio
- Minimum size constraints prevent unusable panels

### **Real-time Updates**
- WebSocket connection for live data
- Automatic reconnection on disconnect
- Toast notifications for user feedback

### **Responsive Design**
- Works on all screen sizes
- Mobile-friendly interface
- Adaptive layout for different devices

### **Professional Styling**
- Modern gradient background
- Glassmorphism effects
- Smooth animations and transitions
- Color-coded status indicators

## 🔧 Configuration

### **Environment Variables**
The dashboard reads from your existing trading bot configuration:
- `SYMBOLS`: Comma-separated list of trading symbols
- Bot configuration from `.env` file

### **Customization**
- **Port**: Change port in `app.py` (default: 5000)
- **Host**: Modify host binding in `app.py`
- **Styling**: Edit CSS variables in `templates/index.html`

## 📊 API Endpoints

### **Bot Control**
- `GET /api/status` - Get bot status
- `POST /api/start` - Start bot with symbols
- `POST /api/stop` - Stop bot process

### **Symbol Management**
- `GET /api/symbols` - Get current symbols and popular symbols
- `POST /api/symbols` - Update trading symbols

### **Analytics**
- `GET /api/analytics` - Get performance analytics data
- `GET /api/logs` - Get recent log entries

### **WebSocket Events**
- `log_update` - New log entry
- `status_update` - Bot status change
- `symbols_update` - Symbols list update

## 🚨 Troubleshooting

### **Common Issues**

#### **Dashboard won't start**
```bash
# Check dependencies
pip install flask flask-socketio psutil

# Check if port 5000 is in use
lsof -i :5000
```

#### **Log stream disconnected**
- Bot process may have stopped
- Check bot logs for errors
- Restart bot from dashboard

#### **Symbols not saving**
- Check file permissions
- Verify environment variable access
- Check browser console for errors

#### **Charts not loading**
- Check internet connection (Chart.js CDN)
- Verify browser supports Canvas
- Check browser console for errors

### **Debug Mode**
```bash
# Enable debug logging
export FLASK_ENV=development
python app.py
```

## 🔒 Security Notes

- Dashboard runs on localhost by default
- No authentication implemented (for development)
- Consider adding authentication for production use
- WebSocket connections are local only

## 🚀 Production Deployment

### **Using Gunicorn**
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

### **Using Docker**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 📈 Future Enhancements

- [ ] User authentication and authorization
- [ ] Multiple bot instance management
- [ ] Advanced charting with more indicators
- [ ] Trade execution interface
- [ ] Risk management controls
- [ ] Strategy parameter tuning
- [ ] Email/SMS alerts
- [ ] Mobile app version

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This dashboard is part of the Trading Bot project and follows the same license terms. 