"""Custom exceptions for the trading bot."""

from typing import Any, Optional


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(TradingBotError):
    """Raised when there's a configuration error."""


class StrategyError(TradingBotError):
    """Raised when there's an error in strategy execution."""


class RiskManagementError(TradingBotError):
    """Raised when risk limits are exceeded."""


class MarketDataError(TradingBotError):
    """Raised when there's an error with market data."""


class OrderExecutionError(TradingBotError):
    """Raised when there's an error executing orders."""


class BacktestError(TradingBotError):
    """Raised when there's an error during backtesting."""


class DatabaseError(TradingBotError):
    """Raised when there's a database error."""
