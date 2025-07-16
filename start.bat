@echo off
REM Trading Bot Start Script for Windows
REM This script activates the virtual environment and starts the trading bot

REM Check if virtual environment exists
if not exist venv (
    echo ❌ Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if arguments are provided
if "%1"=="" (
    echo 🤖 Trading Bot Menu
    echo ==================
    echo Usage: start.bat [command] [options]
    echo.
    echo Available commands:
    echo   live-trade     Start live trading
    echo   backtest       Run backtesting
    echo   config         Show configuration
    echo   test           Run tests
    echo   help           Show detailed help
    echo.
    echo Examples:
    echo   start.bat live-trade --strategy momentum
    echo   start.bat backtest --start-date 2023-01-01 --end-date 2023-12-31
    echo   start.bat config
    echo   start.bat test
    echo.
    echo For detailed help: start.bat help
    pause
    exit /b 0
)

REM Run the trading bot with provided arguments
echo 🚀 Starting Trading Bot...
python main.py %* 