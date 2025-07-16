"""Custom exceptions for the trading bot."""

from typing import Optional, Any


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(TradingBotError):
    """Raised when there's a configuration error."""
    pass


class StrategyError(TradingBotError):
    """Raised when there's an error in strategy execution."""
    pass


class RiskManagementError(TradingBotError):
    """Raised when risk limits are exceeded."""
    pass


class MarketDataError(TradingBotError):
    """Raised when there's an error with market data."""
    pass


class OrderExecutionError(TradingBotError):
    """Raised when there's an error executing orders."""
    pass


class BacktestError(TradingBotError):
    """Raised when there's an error during backtesting."""
    pass


class DatabaseError(TradingBotError):
    """Raised when there's a database error."""
    pass 