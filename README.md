# High-Performance Trading Bot

A professional-grade trading bot built with Python, featuring real-time market data processing, multiple trading strategies, comprehensive risk management, and advanced backtesting capabilities.

## 🚀 Features

### Core Features
- **Real-time Market Data**: WebSocket-based market data feeds with automatic reconnection
- **Multiple Trading Strategies**: Momentum crossover, mean reversion, and breakout strategies
- **Risk Management**: Position sizing, daily loss limits, and portfolio concentration controls
- **Order Execution**: Mock and live order execution with retry logic
- **Backtesting Engine**: Comprehensive backtesting with performance metrics
- **Performance Monitoring**: Real-time performance tracking and logging

### Technical Features
- **Async Architecture**: High-performance async/await design for concurrent processing
- **Modular Design**: Clean separation of concerns with pluggable components
- **Database Integration**: SQLite-based persistence for trades, orders, and performance data
- **Configuration Management**: Environment-based configuration with validation
- **Comprehensive Testing**: Unit tests, integration tests, and strategy tests
- **Rich CLI Interface**: Beautiful command-line interface with progress bars and tables

## 📋 Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`
- Optional: API keys for live trading (Alpaca, etc.)

## 📁 Project Structure

```
trading/
├── main.py                      # Main CLI application
├── setup.sh / setup.bat         # Setup scripts for different platforms
├── start.sh / start.bat         # Start scripts for different platforms
├── format.sh / format.bat       # Code formatting scripts
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── config.env.template         # Configuration template
├── trading_bot/                 # Core trading bot package
│   ├── core/                   # Core components (bot, config, models)
│   ├── market_data/            # Market data management
│   ├── strategy/               # Trading strategies
│   ├── risk/                   # Risk management
│   ├── execution/              # Order execution
│   ├── backtesting/            # Backtesting engine
│   └── database/               # Database operations
└── tests/                       # Test suite
```

## 🛠️ Installation

### Quick Setup (Recommended)

**Linux/macOS:**
```bash
git clone <repository-url>
cd trading
./setup.sh
```

**Windows:**
```cmd
git clone <repository-url>
cd trading
setup.bat
```

### Manual Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd trading
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **For development (optional)**:
```bash
pip install -r requirements-dev.txt
```

5. **Set up configuration**:
```bash
cp config.env.template .env
# Edit .env with your API keys
```

## 🏃 Quick Start

### Method 1: Using Start Scripts (Recommended)

**Linux/macOS:**
```bash
# Run configuration check
./start.sh config

# Run tests
./start.sh test

# Run backtest
./start.sh backtest --strategy momentum_crossover --symbol AAPL --days 30

# Run live trading (paper trading)
./start.sh run
```

**Windows:**
```cmd
# Run configuration check
start.bat config

# Run tests
start.bat test

# Run backtest
start.bat backtest --strategy momentum_crossover --symbol AAPL --days 30

# Run live trading (paper trading)
start.bat run
```

### Method 2: Direct Python Execution

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run configuration check
python main.py config

# Run tests
python main.py test

# Run backtest
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 30

# Run live trading (paper trading)
python main.py run
```

## ⚙️ Configuration

### Basic Configuration

Create a `.env` file based on `config.env.template`:

```bash
# Trading Configuration
DEFAULT_PORTFOLIO_VALUE=100000
MAX_POSITION_SIZE=0.05
STOP_LOSS_PERCENTAGE=0.02
TAKE_PROFIT_PERCENTAGE=0.04

# Risk Management
MAX_DAILY_LOSS=0.02
MAX_OPEN_POSITIONS=10

# Strategy Configuration
DEFAULT_STRATEGY=momentum_crossover
STRATEGY_PARAMETERS={"short_window": 10, "long_window": 30, "min_strength_threshold": 0.0001}

# Exchange Configuration (Paper Trading - Safe)
ALPACA_API_KEY=your_paper_trading_key
ALPACA_SECRET_KEY=your_paper_trading_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ENVIRONMENT=sandbox
```

### Strategy Parameters

#### Momentum Crossover Strategy
```json
{
  "short_window": 10,
  "long_window": 30,
  "min_strength_threshold": 0.0001
}
```

#### Mean Reversion Strategy
```json
{
  "bollinger_window": 20,
  "rsi_window": 14,
  "rsi_oversold": 30,
  "rsi_overbought": 70
}
```

#### Breakout Strategy
```json
{
  "lookback_window": 20,
  "breakout_threshold": 0.02,
  "volume_multiplier": 1.5
}
```

## 📊 Usage Examples

### Backtesting

```bash
# Basic backtest
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 30

# Multiple symbols
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 30
python main.py backtest --strategy momentum_crossover --symbol GOOGL --days 30
python main.py backtest --strategy momentum_crossover --symbol MSFT --days 30

# Different strategies
python main.py backtest --strategy mean_reversion --symbol AAPL --days 30
python main.py backtest --strategy breakout --symbol TSLA --days 30
```

### Live Trading (Paper Trading)

```bash
# Start live trading in paper trading mode
python main.py run

# Or using start script
./start.sh run
```

### Testing

```bash
# Run all tests
python main.py test

# Run tests with coverage
python -m pytest tests/ -v --cov=trading_bot --cov-report=html
```

## 🎨 Code Formatting

### Automatic Formatting

```bash
# Linux/macOS
./format.sh

# Windows
format.bat
```

### Manual Formatting

```bash
# Format with Black
black .

# Sort imports
isort .

# Check for issues
flake8 .
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_strategies.py -v

# Run with coverage
python -m pytest tests/ -v --cov=trading_bot --cov-report=html
```

### Code Quality

```bash
# Type checking
mypy trading_bot/

# Linting
flake8 trading_bot/

# Format checking
black --check .
isort --check-only .
```

## 📈 Trading vs Backtesting Alignment

### Critical Feature: Aligned Execution Logic

The trading bot uses **identical execution logic** for both real trading and backtesting, ensuring:

- ✅ **Consistent Commission**: Both use $0.005 per share, $1 minimum
- ✅ **Identical Slippage**: 0.1% base + volume-based adjustment
- ✅ **Same Risk Management**: Identical position sizing and risk checks
- ✅ **Unified Order Processing**: Same signal-to-order conversion logic

### Shared Execution Logic

All execution logic is centralized in `trading_bot/core/shared_execution.py`:

```python
class SharedExecutionLogic:
    def signal_to_order(self, signal: StrategySignal, position_size: Decimal) -> Optional[Order]
    def calculate_commission(self, quantity: Decimal, price: Decimal) -> Decimal
    def simulate_execution_price(self, market_price: Decimal, signal_type: str, quantity: Decimal) -> Decimal
    def create_trade_from_order(self, order: Order, fill_price: Decimal) -> Trade
```

This ensures **backtest results accurately predict real trading performance**.

## 🛡️ Safety Features

### Risk Management
- **Position Sizing**: Based on portfolio value and volatility
- **Daily Loss Limits**: 2% maximum daily loss
- **Position Limits**: 5% maximum position size
- **Stop Loss**: 2% stop loss per position

### Paper Trading Mode
- **Environment**: `sandbox` (no real money at risk)
- **Mock Execution**: Simulated order fills
- **Realistic Slippage**: 0.1% base + volume impact
- **Commission Modeling**: $0.005 per share, $1 minimum

## 📊 Performance Metrics

### Backtest Results Example
```
Performance Summary
┌─────────────────┬──────────────────┐
│ Metric          │ Value            │
├─────────────────┼──────────────────┤
│ Total Return    │ 21.68%           │
│ Sharpe Ratio    │ 1.38             │
│ Max Drawdown    │ 13.76%           │
│ Win Rate        │ 0.00%            │
│ Total Trades    │ 2                │
│ Profit Factor   │ 0.00             │
└─────────────────┴──────────────────┘
```

### Interpreting Results
- **Total Return**: >10% annually is good
- **Sharpe Ratio**: >1.0 indicates good risk-adjusted returns
- **Max Drawdown**: <15% is acceptable
- **Win Rate**: >50% is good
- **Profit Factor**: >1.2 indicates profitable strategy

## 🚨 Important Notes

### Paper Trading Only
- The bot runs in **paper trading mode** by default
- No real money is at risk
- Perfect for testing strategies safely

### Configuration Required
- Copy `config.env.template` to `.env`
- Add your Alpaca API keys for paper trading
- Adjust risk parameters as needed

### Testing Before Live Trading
- Always run backtests before live trading
- Start with paper trading to validate strategies
- Monitor performance closely in live mode

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/ -v`
5. Format code: `./format.sh`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This trading bot is for educational and research purposes. Trading involves risk, and past performance does not guarantee future results. Always test thoroughly in paper trading mode before using real money.

---

**Happy Trading! 🚀**
