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

- Python 3.8+
- Dependencies listed in `requirements.txt`
- Optional: API keys for live trading (Alpaca, etc.)

## 📁 Project Structure

```
trading/
├── setup.sh / setup.bat         # Setup scripts for different platforms
├── start.sh / start.bat         # Start scripts for different platforms
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── main.py                      # Main CLI application
├── trading_bot/                 # Core trading bot package
│   ├── market_data/            # Market data management
│   ├── strategy/               # Trading strategies
│   ├── risk/                   # Risk management
│   ├── execution/              # Order execution
│   ├── backtesting/            # Backtesting engine
│   └── database/               # Database operations
├── tests/                       # Test suite
├── config/                      # Configuration files
└── docs/                        # Documentation
```

## 🛠️ Installation

### Using Setup Scripts (Recommended)

The easiest way to get started is using the provided setup scripts that handle virtual environment creation and dependency installation:

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
# Create .env file with your API keys
echo "ALPACA_API_KEY=your_api_key_here" > .env
echo "ALPACA_SECRET_KEY=your_secret_key_here" >> .env
echo "ALPACA_BASE_URL=https://paper-api.alpaca.markets" >> .env
```

## 🏃 Quick Start

### Automated Setup (Recommended)

**For Linux/macOS:**
```bash
# Clone the repository
git clone <repository-url>
cd trading

# Run setup script (creates venv and installs dependencies)
./setup.sh

# Start the bot
./start.sh --help
```

**For Windows:**
```cmd
# Clone the repository
git clone <repository-url>
cd trading

# Run setup script (creates venv and installs dependencies)
setup.bat

# Start the bot
start.bat --help
```

### Manual Setup

1. **Create Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   ```bash
   # Create .env file with your API keys
   echo "ALPACA_API_KEY=your_api_key_here" > .env
   echo "ALPACA_SECRET_KEY=your_secret_key_here" >> .env
   echo "ALPACA_BASE_URL=https://paper-api.alpaca.markets" >> .env
   ```

### Usage Examples

1. **Run Configuration Check:**
   ```bash
   ./start.sh config
   ```

2. **Run Tests:**
   ```bash
   ./start.sh test
   ```

3. **Run Backtest:**
   ```bash
   ./start.sh backtest --strategy momentum_crossover --symbol AAPL --days 30
   ```

4. **Run Live Trading (Paper Trading):**
   ```bash
   ./start.sh live-trade
   ```

## 📊 Trading Strategies

### Momentum Crossover Strategy
Generates buy/sell signals based on moving average crossovers:
- **Buy Signal**: Short MA crosses above Long MA
- **Sell Signal**: Short MA crosses below Long MA
- **Parameters**: `short_window`, `long_window`, `min_strength_threshold`

### Mean Reversion Strategy
Uses Bollinger Bands and RSI for mean reversion trading:
- **Buy Signal**: Price near lower Bollinger Band + RSI oversold
- **Sell Signal**: Price near upper Bollinger Band + RSI overbought
- **Parameters**: `bollinger_window`, `rsi_window`, `rsi_oversold`, `rsi_overbought`

### Breakout Strategy
Identifies price breakouts from consolidation ranges:
- **Buy Signal**: Price breaks above resistance with volume confirmation
- **Sell Signal**: Price breaks below support with volume confirmation
- **Parameters**: `lookback_window`, `breakout_threshold`, `volume_multiplier`

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# Trading Configuration
DEFAULT_PORTFOLIO_VALUE=100000
MAX_POSITION_SIZE=0.05
STOP_LOSS_PERCENTAGE=0.02
TAKE_PROFIT_PERCENTAGE=0.04

# Risk Management
MAX_DAILY_LOSS=0.02
MAX_OPEN_POSITIONS=10
RISK_FREE_RATE=0.02

# Strategy Configuration
DEFAULT_STRATEGY=momentum_crossover
STRATEGY_PARAMETERS={"short_window": 10, "long_window": 30}
```

### Configuration Sections

1. **Exchange Config**: API credentials and endpoints
2. **Trading Config**: Portfolio size, position sizing, stop losses
3. **Risk Config**: Daily loss limits, position limits
4. **Strategy Config**: Strategy selection and parameters
5. **Database Config**: Database connection settings
6. **Logging Config**: Log levels and formatting

## 📈 Backtesting

### Run a Backtest
```bash
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 30
```

### Backtest Results
The backtesting engine provides comprehensive performance metrics:
- **Total Return**: Overall portfolio return
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough loss
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Example Output
```
Performance Summary
┌─────────────────┬──────────────────┐
│ Metric          │ Value            │
├─────────────────┼──────────────────┤
│ Strategy        │ momentum_crossover│
│ Total Return    │ 15.23%           │
│ Sharpe Ratio    │ 1.45             │
│ Max Drawdown    │ 8.12%            │
│ Win Rate        │ 62.50%           │
│ Total Trades    │ 24               │
└─────────────────┴──────────────────┘
```

## 🚨 Risk Management

### Built-in Risk Controls
- **Position Sizing**: Automatic position sizing based on portfolio value
- **Daily Loss Limits**: Stop trading if daily loss exceeds threshold
- **Portfolio Concentration**: Limit exposure to any single position
- **Maximum Positions**: Limit number of concurrent positions
- **Volatility Adjustment**: Adjust position sizes based on volatility

### Risk Parameters
```python
MAX_POSITION_SIZE = 0.05        # 5% max position size
MAX_DAILY_LOSS = 0.02           # 2% daily loss limit
MAX_OPEN_POSITIONS = 10         # Maximum concurrent positions
STOP_LOSS_PERCENTAGE = 0.02     # 2% stop loss
TAKE_PROFIT_PERCENTAGE = 0.04   # 4% take profit
```

## 📊 Performance Monitoring

### Real-time Monitoring
- Portfolio value tracking
- Position P&L monitoring
- Trade execution logging
- Risk metrics calculation

### Logging Features
- Structured logging with JSON format
- Rich console output with colors
- Error tracking and alerting
- Performance metrics logging

## 🧪 Testing

### Test Suite
```bash
# Run all tests
python main.py test

# Run specific test file
python -m pytest tests/test_strategies.py -v

# Run with coverage
python -m pytest tests/ --cov=trading_bot --cov-report=html
```

### Test Coverage
- Unit tests for all core components
- Integration tests for full workflow
- Strategy-specific tests
- Risk management tests
- Database tests

## 🏗️ Architecture

### Core Components

```
trading_bot/
├── core/           # Core framework components
│   ├── bot.py      # Main TradingBot class
│   ├── config.py   # Configuration management
│   ├── models.py   # Data models
│   └── exceptions.py
├── strategy/       # Trading strategies
│   ├── base.py     # Base strategy class
│   ├── momentum_crossover.py
│   ├── mean_reversion.py
│   └── breakout.py
├── market_data/    # Market data handling
│   └── manager.py  # Real-time data feeds
├── risk/          # Risk management
│   └── manager.py  # Risk controls
├── execution/     # Order execution
│   └── manager.py  # Order management
├── backtesting/   # Backtesting engine
│   └── engine.py   # Backtest execution
└── database/      # Data persistence
    └── manager.py  # Database operations
```

### Design Patterns
- **Strategy Pattern**: Pluggable trading strategies
- **Observer Pattern**: Event-driven architecture
- **Factory Pattern**: Component creation
- **Async/Await**: Concurrent processing

## 📝 API Reference

### Main Classes

#### TradingBot
```python
bot = TradingBot(config)
await bot.start()           # Start the bot
await bot.stop()            # Stop the bot
status = bot.get_status()   # Get current status
```

#### Strategy Development
```python
class CustomStrategy(BaseStrategy):
    async def generate_signals(self) -> List[StrategySignal]:
        # Implementation here
        pass
```

#### Configuration
```python
config = Config.from_env()  # Load from environment
config.trading.portfolio_value  # Access settings
```

## 🔒 Security Considerations

### API Security
- Store API keys in environment variables
- Use paper trading endpoints for testing
- Implement proper error handling
- Log security events

### Risk Controls
- Multiple layers of risk checks
- Position size limits
- Daily loss limits
- Emergency stop functionality

## 🚀 Deployment

### Local Development
```bash
# Set up development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Production Deployment
1. Set up proper logging and monitoring
2. Configure production API endpoints
3. Set up database backups
4. Implement monitoring and alerting
5. Use process managers (systemd, supervisor)

## 📊 Performance Optimization

### Optimization Features
- Async I/O for market data
- Efficient data structures
- Database indexing
- Connection pooling
- Memory management

### Monitoring
- Real-time performance metrics
- System resource monitoring
- Trade execution latency
- Market data throughput

## 🤝 Contributing

### Development Guidelines
1. Follow PEP 8 coding standards
2. Write comprehensive tests
3. Document new features
4. Use type hints
5. Follow async patterns

### Testing Requirements
- Unit tests for new features
- Integration tests for workflows
- Performance tests for critical paths
- Documentation updates

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This trading bot is for educational and research purposes only. Trading involves substantial risk and is not suitable for every investor. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🆘 Support

For questions and support:
- Check the documentation
- Review the test examples
- Create an issue on GitHub
- Review the configuration guide

---

**Happy Trading! 🚀📈**
