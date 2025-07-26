#!/bin/bash

# Trading Bot Setup Script
# This script sets up the trading bot environment automatically

set -e  # Exit on any error

echo "🚀 Setting up Trading Bot Environment..."

# Function to install system packages
install_system_packages() {
    echo "📦 Installing required system packages..."
    if command -v apt &> /dev/null; then
        sudo apt update
        sudo apt install python3-venv python3-pip -y
    elif command -v yum &> /dev/null; then
        sudo yum install python3-venv python3-pip -y
    elif command -v dnf &> /dev/null; then
        sudo dnf install python3-venv python3-pip -y
    else
        echo "❌ Could not install packages automatically. Please install python3-venv and python3-pip manually."
        exit 1
    fi
}

# Function to check if venv module is available
check_venv() {
    python3 -c "import venv" 2>/dev/null
}

# Function to check if pip is available
check_pip() {
    python3 -c "import pip" 2>/dev/null
}

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MIN_VERSION="3.8"

if [ "$(printf '%s\n' "$MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
    echo "❌ Python $MIN_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check and install missing system packages
if ! check_venv || ! check_pip; then
    echo "⚠️  Missing required packages. Installing automatically..."
    install_system_packages
fi

# Clean up any existing corrupted venv and fix PATH
echo "🧹 Cleaning up environment..."
if [ -d "venv" ]; then
    echo "⚠️  Removing old virtual environment..."
    rm -rf venv
fi

# Reset PATH to avoid old venv interference
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"

# Create virtual environment
echo "📦 Creating virtual environment..."
if python3 -m venv venv; then
    echo "✅ Virtual environment created successfully"
    USE_VENV=true
else
    echo "⚠️  Failed to create virtual environment. Using system Python instead."
    echo "⚠️  This is not recommended for production use."
    USE_VENV=false
fi

# Install dependencies
if [ "$USE_VENV" = true ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    echo "📈 Upgrading pip..."
    pip install --upgrade pip
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "📈 Upgrading pip (system-wide)..."
    python3 -m pip install --upgrade pip --user
    echo "📦 Installing dependencies..."
    python3 -m pip install -r requirements.txt --user
fi

echo "✅ Setup complete!"
echo ""
if [ "$USE_VENV" = true ]; then
    echo "To start using the trading bot:"
    echo "1. Run: source venv/bin/activate"
    echo "2. Run: python main.py --help"
    echo ""
    echo "Or use the start script: ./start.sh"
else
    echo "⚠️  Using system Python installation."
    echo "To start using the trading bot:"
    echo "1. Run: python3 main.py --help"
    echo ""
    echo "Or use the start script: ./start.sh"
fi
