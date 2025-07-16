"""Logging configuration and setup."""

import sys
import logging
from typing import Any, Dict
import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """Setup structured logging with rich formatting."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(stderr=True), rich_tracebacks=True)],
        level=getattr(logging, config.level.upper()),
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer() if config.format == "console" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class TradingLogger:
    """Specialized logger for trading events."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def log_order(self, order: Dict[str, Any], action: str) -> None:
        """Log order events."""
        self.logger.info(
            "Order event",
            action=action,
            order_id=order.get("id"),
            symbol=order.get("symbol"),
            side=order.get("side"),
            quantity=str(order.get("quantity", 0)),
            price=str(order.get("price", 0)),
            status=order.get("status"),
        )
    
    def log_trade(self, trade: Dict[str, Any]) -> None:
        """Log trade executions."""
        self.logger.info(
            "Trade executed",
            trade_id=trade.get("id"),
            order_id=trade.get("order_id"),
            symbol=trade.get("symbol"),
            side=trade.get("side"),
            quantity=str(trade.get("quantity", 0)),
            price=str(trade.get("price", 0)),
            notional=str(trade.get("notional_value", 0)),
        )
    
    def log_position(self, position: Dict[str, Any], action: str) -> None:
        """Log position changes."""
        self.logger.info(
            "Position update",
            action=action,
            symbol=position.get("symbol"),
            side=position.get("side"),
            quantity=str(position.get("quantity", 0)),
            market_value=str(position.get("market_value", 0)),
            unrealized_pnl=str(position.get("unrealized_pnl", 0)),
        )
    
    def log_strategy_signal(self, signal: Dict[str, Any]) -> None:
        """Log strategy signals."""
        self.logger.info(
            "Strategy signal",
            strategy=signal.get("strategy_name"),
            symbol=signal.get("symbol"),
            signal_type=signal.get("signal_type"),
            strength=signal.get("strength"),
            price=str(signal.get("price", 0)),
        )
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        self.logger.info(
            "Performance update",
            total_return=str(metrics.get("total_return", 0)),
            sharpe_ratio=str(metrics.get("sharpe_ratio", 0)),
            max_drawdown=str(metrics.get("max_drawdown", 0)),
            win_rate=str(metrics.get("win_rate", 0)),
            total_trades=metrics.get("total_trades", 0),
        )
    
    def log_risk_event(self, event: str, details: Dict[str, Any]) -> None:
        """Log risk management events."""
        self.logger.warning(
            "Risk event",
            event=event,
            **details
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log errors with context."""
        self.logger.error(
            "Trading bot error",
            error_type=type(error).__name__,
            error_message=str(error),
            **context
        ) 