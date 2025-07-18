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
    echo "  run     Start live trading"
    echo "  backtest       Run backtesting"
    echo "  config         Show configuration"
    echo "  test           Run tests"
    echo "  help           Show detailed help"
    echo "  web           Start the web dashboard"
    echo ""
    echo "Examples:"
    echo "  ./start.sh run --strategy momentum"
    echo "  ./start.sh backtest --strategy momentum_crossover --symbol AAPL --days 30"
    echo '  ./start.sh run --symbols "AAPL,GOOGL,MSFT,TSLA,AMZN,NVDA,META,NFLX,AMD,INTC,CSCO,QCOM,PYPL,ADBE,CMCSA"'
    echo "  ./start.sh config"
    echo "  ./start.sh test"
    echo "  ./start.sh web"
    echo ""
    echo "For detailed help: ./start.sh help"
    exit 0
fi

# Run the trading bot with provided arguments
if [ "$1" == "web" ]; then
    echo "🌐 Starting Web Dashboard..."
    python dashboard/app.py
    exit 0
fi

echo "🚀 Starting Trading Bot..."
python3 main.py "$@"
