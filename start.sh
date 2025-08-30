#!/bin/bash
echo "Starting start.sh script"

# Trading Bot Start Script
# This script activates the virtual environment and starts the trading bot

set -e  # Exit on any error

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if arguments are provided
if [ $# -eq 0 ]; then
    echo "🤖 Trading Bot Menu"
    echo "=================="
    echo "Usage: ./start.sh [command] [options]"
    echo ""
    echo "Available commands:"
    echo "  run          Start live trading"
    echo "  backtest     Run backtesting"
    echo "  config       Show configuration"
    echo "  test         Run tests"
    echo "  help         Show detailed help"
    echo "  web          Start the web dashboard"
    echo "  research     🤖 Run AI-powered market research & symbol selection"
    echo ""
    echo "Examples:"
    echo "  ./start.sh run --strategy momentum"
    echo "  ./start.sh backtest --strategy momentum_crossover --symbol AAPL --days 30"
    echo '  ./start.sh run --symbols "AAPL,GOOGL,MSFT,TSLA,AMZN,NVDA,META,NFLX,AMD,INTC,CSCO,QCOM,PYPL,ADBE,CMCSA"'
    echo "  ./start.sh config"
    echo "  ./start.sh test"
    echo "  ./start.sh web"
    echo "  ./start.sh research  # 🚀 Use LangGraph AI to find hot trading stocks"
    echo ""
    echo "For detailed help: ./start.sh help"
    exit 0
fi

# Run the trading bot with provided arguments
if [ "$1" == "web" ]; then
    echo "🌐 Starting Web Dashboard..."
    echo "📊 Dashboard will be available at: http://localhost:5000"
    echo "🔄 Press Ctrl+C to stop the dashboard"
    echo ""
    echo "🎯 Dashboard Features:"
    echo "   • Real-time log streaming"
    echo "   • Live performance analytics"
    echo "   • Symbol management with popular stock picker"
    echo "   • Bot process control (start/stop)"
    echo "   • Responsive web interface"
    echo "   • Resizable panels"
    echo ""
    
    # Check if dashboard dependencies are installed
    python -c "import flask_socketio" 2>/dev/null || {
        echo "📦 Installing dashboard dependencies..."
        pip install flask-socketio eventlet
    }
    
    python dashboard/app.py
    exit 0
fi

if [ "$1" == "research" ]; then
    echo "🤖 Starting AI-Powered Market Research..."
    echo "🔗 Connecting to LangGraph cluster for intelligent analysis"
    echo ""
    echo "🚀 Features:"
    echo "   • Fetch current Alpaca positions (top 14 by value)"
    echo "   • AI-powered web search for trending stocks"
    echo "   • LLM analysis of market trends and opportunities"
    echo "   • Intelligent symbol recommendations"
    echo "   • Automatic trading bot configuration update"
    echo ""
    echo "⚡ Powered by your local LangGraph infrastructure:"
    echo "   • Jetson Orin (fast LLM inference)"
    echo "   • CPU node (heavy analysis)"
    echo "   • Tools server (web search & scraping)"
    echo "   • Embeddings server (semantic intelligence)"
    echo ""
    
    # Check if research dependencies are installed
    python -c "import aiohttp, langgraph" 2>/dev/null || {
        echo "📦 Installing research dependencies..."
        pip install aiohttp langgraph
    }
    
    # Check if LangGraph cluster is accessible
    echo "🔍 Checking LangGraph cluster connectivity..."
    python -c "
import requests
try:
    # Test LLM endpoint
    r = requests.get('http://192.168.1.177:11434/api/tags', timeout=5)
    print('✅ Jetson LLM: Connected')
except:
    print('❌ Jetson LLM: Not accessible at 192.168.1.177:11434')

try:
    # Test tools server
    r = requests.get('http://192.168.1.190:8082/', timeout=5)
    print('✅ Tools Server: Connected')
except:
    print('❌ Tools Server: Not accessible at 192.168.1.190:8082')

try:
    # Test embeddings server  
    r = requests.get('http://192.168.1.81:9002/', timeout=5)
    print('✅ Embeddings: Connected')
except:
    print('❌ Embeddings: Not accessible at 192.168.1.81:9002')
    "
    echo ""
    
    python update_symbols_with_research.py
    exit 0
fi

echo "🚀 Starting Trading Bot..."
python3 main.py "$@"
