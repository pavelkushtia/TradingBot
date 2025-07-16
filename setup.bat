@echo off
REM Trading Bot Setup Script for Windows
REM This script sets up the trading bot environment using Python's built-in venv

echo 🚀 Setting up Trading Bot Environment...

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%v in ('python --version') do set PYTHON_VERSION=%%v
echo ✅ Python %PYTHON_VERSION% detected

REM Create virtual environment
echo 📦 Creating virtual environment...
if exist venv (
    echo ⚠️  Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
echo ✅ Virtual environment created

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📈 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing dependencies...
pip install -r requirements.txt

echo ✅ Setup complete!
echo.
echo To start using the trading bot:
echo 1. Run: venv\Scripts\activate.bat
echo 2. Run: python main.py --help
echo.
echo Or use the start script: start.bat
pause 