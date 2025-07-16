"""
High-Performance Trading Bot

A professional-grade trading bot with advanced features:
- Real-time market data processing
- Multiple trading strategies
- Risk management
- Order execution
- Backtesting capabilities
- Performance monitoring
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Team"

print("trading_bot/__init__.py: START")
from .core.bot import TradingBot
from .core.config import Config
from .core.exceptions import TradingBotError
from .core.logging import setup_logging
from .core.version import __version__

__all__ = ["TradingBot", "Config", "TradingBotError", "setup_logging", "__version__"]
