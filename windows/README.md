# Windows Scripts

This directory contains Windows batch files for the trading bot.

## 📁 Files

### `setup.bat`
- **Purpose**: Sets up the trading bot environment on Windows
- **Usage**: Run from the project root directory
- **Actions**: Creates virtual environment, installs dependencies, sets up configuration

### `start.bat`
- **Purpose**: Main entry point for running the trading bot on Windows
- **Usage**: `docs\windows\start.bat [command] [options]`
- **Commands**:
  - `config` - Show current configuration
  - `test` - Run test suite
  - `backtest` - Run backtesting
  - `run` - Start live trading (paper trading)

### `format.bat`
- **Purpose**: Format Python code using Black, isort, and other tools
- **Usage**: Run from the project root directory
- **Actions**: Removes unused imports, formats code, sorts imports, checks for issues

## 🚀 Quick Start

1. **Setup the environment**:
   ```cmd
   windows\setup.bat
   ```

2. **Check configuration**:
   ```cmd
   windows\start.bat config
   ```

3. **Run tests**:
   ```cmd
   windows\start.bat test
   ```

4. **Run backtest**:
   ```cmd
   windows\start.bat backtest --strategy momentum_crossover --symbol AAPL --days 30
   ```

5. **Start live trading**:
   ```cmd
   windows\start.bat run
   ```

6. **Format code**:
   ```cmd
   windows\format.bat
   ```

## ⚠️ Important Notes

- These scripts assume you're running from the project root directory
- Make sure Python 3.9+ is installed and in your PATH
- The scripts will create a virtual environment in the `venv/` directory
- All trading is done in paper trading mode by default (no real money at risk)

---

*For comprehensive documentation, see the main README.md in the project root* 