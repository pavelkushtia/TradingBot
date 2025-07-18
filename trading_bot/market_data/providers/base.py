"""Base data provider interface and utilities."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ...core.exceptions import TradingBotError
from ...core.logging import TradingLogger
from ...core.models import MarketData, Quote


class DataProviderError(TradingBotError):
    """Exception raised when data provider operations fail."""


class AsyncRateLimiter:
    """Simple async rate limiter."""

    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        async with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [call_time for call_time in self.calls if now - call_time < 60]

            # If we're at the limit, wait
            if len(self.calls) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    # Refresh the calls list after sleeping
                    now = time.time()
                    self.calls = [
                        call_time for call_time in self.calls if now - call_time < 60
                    ]

            # Record this call
            self.calls.append(now)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class CircuitBreaker:
    """Simple circuit breaker for provider resilience."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def is_open(self) -> bool:
        if self.state == "open" and self.last_failure_time is not None:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return False
            return True
        return False

    async def __aenter__(self):
        if self.is_open():
            raise DataProviderError("Circuit breaker is open")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            # Record failure
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
        else:
            # Reset on success
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0


@dataclass
class DataProviderConfig:
    """Configuration for data providers."""

    name: str
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: int = 60
    max_retries: int = 3
    timeout: float = 10.0
    priority: int = 1  # Lower number = higher priority
    cost_per_call: float = 0.0  # For cost optimization
    enabled: bool = True
    fallback_on_error: bool = True
    host: Optional[str] = None
    port: Optional[int] = None
    clientId: Optional[int] = None


class BaseDataProvider(ABC):
    """Abstract base class for all data providers."""

    def __init__(self, config: DataProviderConfig):
        self.config = config
        self.logger = TradingLogger(f"provider_{config.name}")
        self.rate_limiter = AsyncRateLimiter(config.rate_limit)
        self.circuit_breaker = CircuitBreaker()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the provider. Override if needed."""
        self._initialized = True
        self.logger.logger.info(f"Initialized {self.config.name} provider")

    async def cleanup(self) -> None:
        """Cleanup provider resources. Override if needed."""
        self._initialized = False
        self.logger.logger.info(f"Cleaned up {self.config.name} provider")

    @abstractmethod
    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> List[MarketData]:
        """Get historical OHLCV bars."""

    @abstractmethod
    async def get_real_time_quote(self, symbol: str) -> Quote:
        """Get real-time quote data."""

    async def get_company_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamental data. Default implementation returns empty dict."""
        self.logger.logger.warning(
            f"{self.config.name} provider does not support fundamentals"
        )
        return {}

    async def get_latest_price(self, symbol: str) -> Decimal:
        """Get latest price for a symbol."""
        try:
            quote = await self.get_real_time_quote(symbol)
            return quote.mid_price
        except Exception as e:
            self.logger.log_error(e, {"context": "get_latest_price", "symbol": symbol})
            raise DataProviderError(f"Failed to get latest price for {symbol}: {e}")

    async def is_available(self) -> bool:
        """Check if provider is available."""
        return self.config.enabled and not self.circuit_breaker.is_open()

    async def test_connection(self) -> bool:
        """Test connection to the provider."""
        try:
            # Simple test with a common symbol
            await self.get_latest_price("AAPL")
            return True
        except Exception as e:
            self.logger.logger.warning(
                f"Connection test failed for {self.config.name}: {e}"
            )
            return False

    def _validate_symbol(self, symbol: str) -> str:
        """Validate and normalize symbol format."""
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty")

        # Remove any spaces and convert to uppercase
        normalized = symbol.strip().upper()

        # Basic validation - alphanumeric and dots only
        if not all(c.isalnum() or c in ".,-_" for c in normalized):
            raise ValueError(f"Invalid symbol format: {symbol}")

        return normalized

    def _validate_timeframe(self, timeframe: str) -> str:
        """Validate timeframe format."""
        valid_timeframes = [
            "1min",
            "5min",
            "15min",
            "30min",
            "1hour",
            "1day",
            "1week",
            "1month",
        ]
        if timeframe not in valid_timeframes:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}"
            )
        return timeframe

    def _validate_date_range(self, start: datetime, end: datetime) -> None:
        """Validate date range."""
        if start >= end:
            raise ValueError("Start date must be before end date")

        if end > datetime.now():
            raise ValueError("End date cannot be in the future")

    async def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute operation with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.rate_limiter:
                    async with self.circuit_breaker:
                        return await operation(*args, **kwargs)

            except Exception as e:
                last_exception = e
                self.logger.logger.warning(
                    f"Attempt {attempt + 1} failed for {self.config.name}: {e}"
                )

                if attempt < self.config.max_retries:
                    # Exponential backoff
                    wait_time = (2**attempt) * 1.0
                    await asyncio.sleep(wait_time)
                else:
                    break

        raise DataProviderError(
            f"All {self.config.max_retries + 1} attempts failed. Last error: {last_exception}"
        )
