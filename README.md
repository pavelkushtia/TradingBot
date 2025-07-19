# 🤖 Super Intelligent Trading Bot

A **revolutionary AI-powered trading bot** that combines cutting-edge machine learning with traditional algorithmic trading for intelligent, adaptive, and profitable automated trading. `(Note: This is the project's vision. The AI/ML features are currently under development.)`

## 🧠 What Makes This Bot Super Intelligent?

This isn't just another trading bot - it's a **Super Intelligent Trading System** that leverages:

- **🤖 Advanced AI/ML Models**: Linear Regression, Random Forest, XGBoost, and ensemble predictions `(⏳ Pending)`
- **🧬 Intelligent Feature Engineering**: 17+ technical features with market microstructure analysis `(⏳ Pending)`
- **📊 Predictive Analytics**: ML-driven signal generation and risk assessment `(⏳ Pending)`
- **🔄 Adaptive Learning**: Models that continuously learn and adapt to market conditions `(⏳ Pending)`
- **⚡ Real-time Intelligence**: AI-enhanced decision making in milliseconds `(✅ Implemented at a basic level)`
- **🛡️ Risk-Aware AI**: ML-driven position sizing and dynamic risk management `(⏳ Pending)`

## 📚 Documentation

- **[📖 Architecture Documentation](ARCHITECTURE.md)** - Comprehensive technical overview including AI/ML architecture
- **[📋 API Documentation](docs/)** - Trading and Alpaca API references

## 🚀 Super Intelligent Features

### 🤖 AI/ML Capabilities
- **Advanced Machine Learning**: Linear, Random Forest, XGBoost models with ensemble predictions `(⏳ Pending)`
- **Intelligent Feature Engineering**: 17+ technical features including returns, volatility, momentum, volume analysis `(⏳ Pending)`
- **Predictive Modeling**: ML-driven signal generation with confidence scoring `(⏳ Pending)`
- **Adaptive Learning**: Models that retrain and adapt to changing market conditions `(⏳ Pending)`
- **Feature Importance Analysis**: Model interpretability and feature selection `(⏳ Pending)`
- **Time Series Cross-Validation**: Proper temporal validation for trading models `(⏳ Pending)`
- **Hyperparameter Optimization**: Grid search with cross-validation for optimal performance `(⏳ Pending)`

### 📊 Multi-Source Data Intelligence
- **Multi-Source Market Data**: Real-time data from Alpaca, Alpha Vantage, Yahoo Finance with intelligent fallback `(✅ Implemented)`
- **Advanced Technical Indicators**: Comprehensive library with SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic `(⏳ In Development: Basic indicators implemented)`
- **Multiple Timeframes**: Aggregation and synchronization across different time periods (1m, 515, 1h, 1d) `(✅ Implemented)`
- **Data Quality Monitoring**: Intelligent validation and quality checks across all data sources `(⏳ Pending)`

### 🎯 Professional Trading Capabilities
- **Advanced Backtesting Engine**: Professional metrics including Sharpe ratio, Sortino ratio, Calmar ratio, VaR, drawdown analysis `(⏳ In Development: Basic backtesting available)`
- **Portfolio Optimization**: Modern portfolio theory algorithms (Mean-Variance, Risk Parity, Kelly Criterion) `(⏳ Pending)`
- **Advanced Order Types**: Stop-loss, take-profit, trailing stops, OCO, bracket orders with sophisticated execution `(⏳ In Development: Basic order types available)`
- **Enhanced Risk Management**: Position sizing algorithms, volatility stops, correlation analysis, risk dashboards `(⏳ In Development: Basic risk management implemented)`
- **Strategy Templates**: Professional strategies (mean reversion, momentum, pairs trading, arbitrage, market making) `(⏳ Pending)`

### ⚡ Technical Excellence
- **Async Architecture**: High-performance async/await design for concurrent processing `(✅ Implemented)`
- **Modular Design**: Clean separation of concerns with pluggable components `(✅ Implemented)`
- **Database Integration**: SQLite-based persistence for trades, orders, and performance data `(✅ Implemented)`
- **Configuration Management**: Environment-based configuration with validation `(✅ Implemented)`
- **Comprehensive Testing**: Unit tests, integration tests, and strategy tests `(✅ Implemented)`
- **Rich CLI Interface**: Beautiful command-line interface with progress bars and tables `(✅ Implemented)`

## 🧠 AI/ML Architecture Deep Dive
`(Note: This section describes the future vision for the AI/ML architecture. The `trading_bot/ml` directory and basic structure exist, but the features are not yet implemented.)`

### Intelligent Feature Engineering
The bot extracts **17+ sophisticated features** from market data:

```python
# Returns and Volatility Features
returns_1d, returns_5d, returns_10d, returns_20ity_5d, volatility_10, volatility_20d

# Momentum Indicators
rsi, price_to_sma20, price_percentile_20d

# Volume Analysis
volume_ratio_10 price_volume_corr_5d, volume_roc_5d

# Advanced Technical Features
bollinger_position, vwap_ratio, atr_normalized

# Market Microstructure
spread_ratio, order_imbalance, market_impact

# Cross-Asset Features
sector_correlation, market_beta, currency_impact
```

### Multi-Model Ensemble Intelligence
The system employs multiple ML models for robust predictions:

```python
# Available Models
models = {
  linear': LinearModel('linear'),
 ridge': LinearModel('ridge'),
 lasso': LinearModel('lasso'),
  random_forest': RandomForestModel(n_estimators=100max_depth=10,
    xgboost: XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1# Ensemble Predictions
ensemble_prediction = weighted_average(predictions)
confidence_score = calculate_confidence(predictions)
```

### AI-Enhanced Strategy Example
```python
class AIEnhancedStrategy(BaseStrategy):
    def __init__(self, name: str, parameters: Dict[str, Any]):
        super().__init__(name, parameters)
        self.ml_predictor = MLPredictor()
        self.feature_engineer = FeatureEngineer()
    
    async def generate_signals(self) -> List[StrategySignal]:
        signals = []
        
        for symbol in self.symbols:
            # Engineer intelligent features
            features, _ = self.feature_engineer.engineer_features(symbol, bars)
            
            # Get AI predictions
            predictions = self.ml_predictor.predict(features[-1      
            # Combine traditional and AI signals
            traditional_signal = self._generate_traditional_signal(bars)
            ai_signal = self._interpret_ml_prediction(predictions)
            
            # Intelligent signal combination
            final_signal = self._combine_signals(traditional_signal, ai_signal)
            
            if final_signal:
                signals.append(final_signal)
        
        return signals
```

## 📋 Requirements

- Python 3.9+ and dependencies listed in `requirements.txt`
- Optional: API keys for live trading (Alpaca, etc.)
- **AI/ML Libraries**: scikit-learn, XGBoost (optional, for future AI capabilities)

## 📁 Project Structure

```
trading/
├── main.py                      # Main CLI application
├── setup.sh                     # Setup script (Linux/macOS)
├── start.sh                     # Start script (Linux/macOS)
├── format.sh                    # Format script (Linux/macOS)
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── config.env.template         # Configuration template
├── docs/                        # Documentation and resources
│   ├── trading.pdf             # Trading documentation
│   └── alpaca.pdf              # Alpaca API documentation
├── windows/                     # Windows-specific files
│   ├── setup.bat               # Setup script (Windows)
│   ├── start.bat               # Start script (Windows)
│   └── format.bat              # Format script (Windows)
├── trading_bot/                 # Core trading bot package
│   ├── core/                   # Core components (bot, config, models)
│   ├── market_data/            # Market data management
│   ├── strategy/               # Trading strategies
│   ├── risk/                   # Risk management
│   ├── execution/              # Order execution
│   ├── backtesting/            # Backtesting engine
│   ├── database/               # Database operations
│   └── ml/                     # 🤖 AI/ML Integration
│       ├── models.py           # ML models (Linear, Random Forest, XGBoost)
│       ├── features.py         # Feature engineering (17+ features)
│       └── training.py         # Model training and validation
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
windows\setup.bat
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
windows\start.bat config

# Run tests
windows\start.bat test

# Run backtest
windows\start.bat backtest --strategy momentum_crossover --symbol AAPL --days 30

# Run live trading (paper trading)
windows\start.bat run
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

## 🤖 AI/ML Usage Examples

### Training AI Models

```bash
# Train ML models with feature engineering
python main.py ml train --symbol AAPL --days 90 --models linear,random_forest,xgboost

# Train with cross-validation
python main.py ml train --symbol AAPL --days 90 --splits 5

# Train ensemble model
python main.py ml train --symbol AAPL --days 90 --ensemble
```

### AI-Enhanced Backtesting

```bash
# Backtest with AI-enhanced strategy
python main.py backtest --strategy ai_enhanced --symbol AAPL --days 30
# Backtest with ML predictions
python main.py backtest --strategy ml_momentum --symbol AAPL --days 30

# Compare traditional vs AI strategies
python main.py backtest --compare traditional,ai_enhanced --symbol AAPL --days 30
```

### Feature Engineering and Analysis

```bash
# Analyze feature importance
python main.py ml features --symbol AAPL --days 30

# Generate feature report
python main.py ml features --symbol AAPL --days 30 --report

# Cross-asset feature analysis
python main.py ml features --symbols AAPL,GOOGL,MSFT --days 30
```

### Model Performance Analysis

```bash
# Get model performance metrics
python main.py ml performance --symbol AAPL

# Compare model predictions
python main.py ml compare --symbol AAPL --models linear,random_forest,xgboost

# Generate ML performance report
python main.py ml report --symbol AAPL --days 90
```

## 📊 Traditional Usage Examples

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
windows\format.bat
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
