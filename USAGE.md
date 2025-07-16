# Trading Bot Usage Guide

This guide provides detailed examples and usage patterns for the trading bot.

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Configuration](#configuration)
3. [Running Strategies](#running-strategies)
4. [Backtesting](#backtesting)
5. [Live Trading](#live-trading)
6. [Custom Strategies](#custom-strategies)
7. [Monitoring & Logging](#monitoring--logging)
8. [Troubleshooting](#troubleshooting)

## 🚀 Getting Started

### Initial Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Copy configuration template**:
```bash
cp config.env.template .env
```

3. **Edit configuration**:
```bash
nano .env  # or your preferred editor
```

4. **Test configuration**:
```bash
python main.py config
```

### First Run

1. **Run tests to verify setup**:
```bash
python main.py test
```

2. **Run a simple backtest**:
```bash
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 7
```

## ⚙️ Configuration

### Basic Configuration

```bash
# Basic trading configuration
DEFAULT_PORTFOLIO_VALUE=100000
MAX_POSITION_SIZE=0.05
STOP_LOSS_PERCENTAGE=0.02
TAKE_PROFIT_PERCENTAGE=0.04

# Risk management
MAX_DAILY_LOSS=0.02
MAX_OPEN_POSITIONS=10
```

### Strategy Configuration

```bash
# Momentum Crossover Strategy
DEFAULT_STRATEGY=momentum_crossover
STRATEGY_PARAMETERS={"short_window": 10, "long_window": 30, "min_strength_threshold": 0.01}

# Mean Reversion Strategy
DEFAULT_STRATEGY=mean_reversion
STRATEGY_PARAMETERS={"bollinger_window": 20, "rsi_window": 14, "rsi_oversold": 30, "rsi_overbought": 70}

# Breakout Strategy
DEFAULT_STRATEGY=breakout
STRATEGY_PARAMETERS={"lookback_window": 20, "breakout_threshold": 0.02, "volume_multiplier": 1.5}
```

### Exchange Configuration

```bash
# Paper Trading (Safe for testing)
ALPACA_API_KEY=your_paper_trading_key
ALPACA_SECRET_KEY=your_paper_trading_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ENVIRONMENT=sandbox

# Live Trading (Use with caution)
ALPACA_API_KEY=your_live_trading_key
ALPACA_SECRET_KEY=your_live_trading_secret
ALPACA_BASE_URL=https://api.alpaca.markets
ENVIRONMENT=live
```

## 📊 Running Strategies

### Command Line Interface

```bash
# Show available commands
python main.py --help

# Show configuration
python main.py config

# Run backtest
python main.py backtest --help
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 30

# Run live trading
python main.py run
```

### Strategy Parameters

#### Momentum Crossover
```bash
python main.py backtest \
  --strategy momentum_crossover \
  --symbol AAPL \
  --days 30

# Configuration in .env:
STRATEGY_PARAMETERS={"short_window": 10, "long_window": 30}
```

#### Mean Reversion
```bash
python main.py backtest \
  --strategy mean_reversion \
  --symbol AAPL \
  --days 30

# Configuration in .env:
STRATEGY_PARAMETERS={"bollinger_window": 20, "rsi_window": 14, "rsi_oversold": 30, "rsi_overbought": 70}
```

#### Breakout Strategy
```bash
python main.py backtest \
  --strategy breakout \
  --symbol TSLA \
  --days 30

# Configuration in .env:
STRATEGY_PARAMETERS={"lookback_window": 20, "breakout_threshold": 0.02, "volume_multiplier": 1.5}
```

## 🔄 Backtesting

### Basic Backtesting

```bash
# Simple backtest
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 30

# Multiple symbols (run separately)
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 30
python main.py backtest --strategy momentum_crossover --symbol GOOGL --days 30
python main.py backtest --strategy momentum_crossover --symbol MSFT --days 30
```

### Backtesting Results

The backtest will show:
- **Performance Summary**: Returns, Sharpe ratio, max drawdown
- **Trade Analysis**: Win rate, profit factor, average win/loss
- **Portfolio Info**: Initial/final capital, positions
- **Recent Trades**: Last 5 trades with details

### Interpreting Results

```
Performance Summary
┌─────────────────┬──────────────────┐
│ Metric          │ Value            │
├─────────────────┼──────────────────┤
│ Total Return    │ 15.23%           │  # Good: >10% annually
│ Sharpe Ratio    │ 1.45             │  # Good: >1.0
│ Max Drawdown    │ 8.12%            │  # Good: <15%
│ Win Rate        │ 62.50%           │  # Good: >50%
│ Profit Factor   │ 1.8              │  # Good: >1.2
└─────────────────┴──────────────────┘
```

### Performance Optimization

```bash
# Test different parameters
STRATEGY_PARAMETERS={"short_window": 5, "long_window": 15}
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 60

STRATEGY_PARAMETERS={"short_window": 10, "long_window": 30}
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 60

STRATEGY_PARAMETERS={"short_window": 20, "long_window": 50}
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 60
```

## 🔴 Live Trading

### Paper Trading Setup

```bash
# Use paper trading for testing
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ENVIRONMENT=sandbox
```

### Running Live Trading

```bash
# Start the bot
python main.py run

# Monitor output
# Bot will display real-time status every 5 seconds
```

### Live Trading Output

```
🚀 Trading Bot
┌─────────────────────────────────────────────┐
│ Trading Bot Starting                        │
│ Environment: sandbox                        │
│ Exchange: alpaca                            │
│ Strategy: momentum_crossover                │
│ Portfolio Value: $100,000.00                │
└─────────────────────────────────────────────┘

Trading Bot Status
┌─────────────────┬─────────────────┐
│ Metric          │ Value           │
├─────────────────┼─────────────────┤
│ Running         │ ✅ Yes          │
│ Portfolio Value │ $100,000.00     │
│ Active Orders   │ 0               │
│ Open Positions  │ 0               │
│ Daily P&L       │ $0.00           │
│ Runtime         │ 0:02:15         │
└─────────────────┴─────────────────┘
```

### Stopping the Bot

```bash
# Graceful shutdown with Ctrl+C
# Bot will:
# 1. Cancel all pending orders
# 2. Save current state
# 3. Close connections
# 4. Generate final report
```

## 🛠️ Custom Strategies

### Creating a Custom Strategy

```python
# trading_bot/strategy/custom_strategy.py
from typing import List
from decimal import Decimal
from .base import BaseStrategy
from ..core.models import StrategySignal

class CustomStrategy(BaseStrategy):
    """Custom trading strategy example."""
    
    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)
        self.threshold = parameters.get("threshold", 0.02)
    
    async def generate_signals(self) -> List[StrategySignal]:
        """Generate custom signals."""
        signals = []
        
        for symbol in self.symbols:
            # Get latest price
            current_price = self.get_latest_price(symbol)
            if not current_price:
                continue
            
            # Custom logic here
            if self._should_buy(symbol, current_price):
                signal = self.create_signal(
                    symbol=symbol,
                    signal_type="buy",
                    strength=0.7,
                    metadata={"reason": "custom_buy_condition"}
                )
                signals.append(signal)
            
            elif self._should_sell(symbol, current_price):
                signal = self.create_signal(
                    symbol=symbol,
                    signal_type="sell",
                    strength=0.6,
                    metadata={"reason": "custom_sell_condition"}
                )
                signals.append(signal)
        
        return signals
    
    def _should_buy(self, symbol: str, price: Decimal) -> bool:
        """Custom buy logic."""
        # Example: Buy if price increased by threshold
        bars = self.get_bars(symbol, 2)
        if len(bars) < 2:
            return False
        
        price_change = (bars[-1].close - bars[-2].close) / bars[-2].close
        return price_change > self.threshold
    
    def _should_sell(self, symbol: str, price: Decimal) -> bool:
        """Custom sell logic."""
        # Example: Sell if price decreased by threshold
        bars = self.get_bars(symbol, 2)
        if len(bars) < 2:
            return False
        
        price_change = (bars[-1].close - bars[-2].close) / bars[-2].close
        return price_change < -self.threshold
```

### Registering Custom Strategy

```python
# trading_bot/strategy/manager.py
from .custom_strategy import CustomStrategy

class StrategyManager:
    def __init__(self, config: Config):
        # ... existing code ...
        
        # Add custom strategy
        self.strategy_classes["custom"] = CustomStrategy
```

### Using Custom Strategy

```bash
# Configuration
DEFAULT_STRATEGY=custom
STRATEGY_PARAMETERS={"threshold": 0.02}

# Run backtest
python main.py backtest --strategy custom --symbol AAPL --days 30
```

## 📊 Monitoring & Logging

### Log Files

```bash
# Check log files (if file logging is enabled)
tail -f trading_bot.log

# Monitor specific events
grep "Order event" trading_bot.log
grep "Trade executed" trading_bot.log
grep "Risk event" trading_bot.log
```

### Real-time Monitoring

```python
# Monitor via Python
from trading_bot import TradingBot, Config

async def monitor_bot():
    config = Config.from_env()
    bot = TradingBot(config)
    
    # Get status periodically
    while True:
        status = bot.get_status()
        print(f"Portfolio: ${status['portfolio_value']}")
        print(f"Positions: {status['open_positions']}")
        await asyncio.sleep(10)
```

### Performance Metrics

```python
# Access performance data
from trading_bot.database.manager import DatabaseManager

async def get_performance():
    db = DatabaseManager(config)
    await db.initialize()
    
    # Get trades
    trades = await db.get_trades(limit=100)
    
    # Get performance metrics
    metrics = await db.get_performance_metrics()
    
    print(f"Total trades: {len(trades)}")
    print(f"Win rate: {metrics.win_rate:.2%}")
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Configuration Errors
```bash
# Check configuration
python main.py config

# Common issues:
# - Missing API keys
# - Invalid parameter values
# - Wrong data types
```

#### 2. Connection Issues
```bash
# Check network connectivity
# Verify API endpoints
# Check API key permissions
```

#### 3. Strategy Not Generating Signals
```bash
# Check strategy parameters
# Verify market data is being received
# Check symbol availability
```

#### 4. Database Issues
```bash
# Check database file permissions
# Verify SQLite installation
# Check disk space
```

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python main.py run

# Run with verbose output
python main.py backtest --strategy momentum_crossover --symbol AAPL --days 7 -v
```

### Testing Individual Components

```bash
# Test specific components
python -m pytest tests/test_strategies.py::TestMomentumCrossoverStrategy -v
python -m pytest tests/test_trading_bot.py::TestRiskManagement -v
python -m pytest tests/test_trading_bot.py::TestOrderExecution -v
```

## 📈 Advanced Usage

### Multiple Strategy Backtesting

```bash
# Test multiple strategies
for strategy in momentum_crossover mean_reversion breakout; do
    echo "Testing $strategy..."
    python main.py backtest --strategy $strategy --symbol AAPL --days 30
done
```

### Parameter Optimization

```python
# Parameter sweep example
parameters = [
    {"short_window": 5, "long_window": 15},
    {"short_window": 10, "long_window": 30},
    {"short_window": 20, "long_window": 50}
]

for params in parameters:
    # Update configuration
    # Run backtest
    # Compare results
```

### Portfolio Backtesting

```bash
# Test multiple symbols
symbols=("AAPL" "GOOGL" "MSFT" "TSLA" "AMZN")

for symbol in "${symbols[@]}"; do
    echo "Backtesting $symbol..."
    python main.py backtest --strategy momentum_crossover --symbol $symbol --days 30
done
```

## 🎯 Best Practices

### 1. Start with Paper Trading
- Always test strategies in paper trading first
- Verify all components work correctly
- Monitor for several days before going live

### 2. Risk Management
- Never risk more than you can afford to lose
- Set appropriate position sizes
- Use stop losses
- Monitor daily loss limits

### 3. Strategy Development
- Backtest thoroughly before deployment
- Test on multiple symbols and time periods
- Consider market conditions
- Document strategy logic

### 4. Monitoring
- Monitor bot performance regularly
- Set up alerts for important events
- Keep logs for analysis
- Review trades and performance metrics

### 5. Maintenance
- Update API keys regularly
- Monitor system resources
- Keep dependencies updated
- Regular backups of important data

---

This guide should help you effectively use the trading bot. Remember to always test thoroughly and understand the risks involved in automated trading. 