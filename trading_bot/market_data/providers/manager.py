"""Smart data provider manager with fallback and cost optimization."""

import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Type

from ...core.config import Config
from ...core.logging import TradingLogger
from ...core.models import MarketData, Quote
from .alpaca import AlpacaProvider
from .alpha_vantage import AlphaVantageProvider
from .base import BaseDataProvider, DataProviderConfig, DataProviderError
from .yahoo import YahooFinanceProvider


class CostTracker:
    """Track API usage costs across providers."""

    def __init__(self):
        self.usage_stats = {}
        self.daily_costs = {}
        self.logger = TradingLogger("cost_tracker")

    async def record_usage(self, provider_name: str, calls: int, cost: float = 0.0):
        """Record API usage for a provider."""
        today = datetime.now().date().isoformat()

        if provider_name not in self.usage_stats:
            self.usage_stats[provider_name] = {}

        if today not in self.usage_stats[provider_name]:
            self.usage_stats[provider_name][today] = {"calls": 0, "cost": 0.0}

        self.usage_stats[provider_name][today]["calls"] += calls
        self.usage_stats[provider_name][today]["cost"] += cost

        # Update daily totals
        if today not in self.daily_costs:
            self.daily_costs[today] = 0.0

        self.daily_costs[today] += cost

    async def get_daily_usage(self, provider_name: str) -> Dict[str, Any]:
        """Get today's usage for a provider."""
        today = datetime.now().date().isoformat()

        if (
            provider_name in self.usage_stats
            and today in self.usage_stats[provider_name]
        ):
            return self.usage_stats[provider_name][today]

        return {"calls": 0, "cost": 0.0}

    async def get_report(self) -> Dict[str, Any]:
        """Get comprehensive cost and usage report."""
        report = {
            "daily_costs": self.daily_costs,
            "usage_stats": self.usage_stats,
            "summary": {},
        }

        # Calculate summary statistics
        total_cost = sum(self.daily_costs.values())
        total_calls = 0

        for provider_stats in self.usage_stats.values():
            for day_stats in provider_stats.values():
                total_calls += day_stats["calls"]

        report["summary"] = {
            "total_cost": total_cost,
            "total_calls": total_calls,
            "providers": list(self.usage_stats.keys()),
        }

        return report


class DataProviderManager:
    """Smart manager for multiple data providers with fallback and cost optimization."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = TradingLogger("data_provider_manager")
        self.providers: Dict[str, BaseDataProvider] = {}
        self.cost_tracker = CostTracker()
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes

        # Provider classes registry
        self.provider_classes: Dict[str, Type[BaseDataProvider]] = {
            "alpaca": AlpacaProvider,
            "alpha_vantage": AlphaVantageProvider,
            "yahoo": YahooFinanceProvider,
        }

    async def initialize(self) -> None:
        """Initialize all configured data providers."""
        self.logger.logger.info("Initializing data provider manager...")

        # Define provider configurations
        provider_configs = await self._create_provider_configs()

        # Initialize providers
        for provider_config in provider_configs:
            if not provider_config.enabled:
                continue

            try:
                provider_class = self.provider_classes.get(provider_config.name)
                if not provider_class:
                    self.logger.logger.warning(
                        f"Unknown provider: {provider_config.name}"
                    )
                    continue

                provider = provider_class(provider_config)
                await provider.initialize()

                # Test connection
                if await provider.test_connection():
                    self.providers[provider_config.name] = provider
                    self.logger.logger.info(
                        f"Successfully initialized {provider_config.name} provider"
                    )
                else:
                    self.logger.logger.warning(
                        f"Connection test failed for {provider_config.name}"
                    )
                    await provider.cleanup()

            except Exception as e:
                self.logger.logger.error(
                    f"Failed to initialize {provider_config.name}: {e}"
                )

        if not self.providers:
            raise DataProviderError("No data providers could be initialized")

        self.logger.logger.info(
            f"Initialized {len(self.providers)} data providers: {list(self.providers.keys())}"
        )

    async def cleanup(self) -> None:
        """Cleanup all providers."""
        for provider in self.providers.values():
            try:
                await provider.cleanup()
            except Exception as e:
                self.logger.logger.error(f"Error cleaning up provider: {e}")

        self.providers.clear()
        self.logger.logger.info("All data providers cleaned up")

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "1day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
        fallback: bool = True,
    ) -> List[MarketData]:
        """Get historical market data with intelligent provider selection."""
        # Set default date range if not provided
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=30)

        # Check cache first
        cache_key = (
            f"hist:{symbol}:{timeframe}:{start.isoformat()}:{end.isoformat()}:{limit}"
        )
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            self.logger.logger.debug(f"Returning cached data for {symbol}")
            return cached_data

        # Get providers in priority order
        sorted_providers = await self._get_sorted_providers()

        last_exception = None
        for provider in sorted_providers:
            try:
                if not await provider.is_available():
                    self.logger.logger.debug(
                        f"Provider {provider.config.name} not available"
                    )
                    continue

                self.logger.logger.debug(
                    f"Trying {provider.config.name} for historical data: {symbol}"
                )

                data = await provider.get_historical_bars(
                    symbol, timeframe, start, end, limit
                )

                if data:
                    # Cache successful result
                    self._set_cache(cache_key, data)

                    # Record usage
                    await self.cost_tracker.record_usage(
                        provider.config.name, 1, provider.config.cost_per_call
                    )

                    self.logger.logger.info(
                        f"Successfully retrieved {len(data)} bars from {provider.config.name}"
                    )
                    return data
                else:
                    self.logger.logger.warning(
                        f"No data returned from {provider.config.name}"
                    )

            except Exception as e:
                last_exception = e
                self.logger.logger.warning(
                    f"Provider {provider.config.name} failed: {e}"
                )

                if not fallback:
                    break

        # If all providers failed
        error_msg = (
            f"All data providers failed for {symbol}. Last error: {last_exception}"
        )
        self.logger.logger.error(error_msg)
        raise DataProviderError(error_msg)

    async def get_real_time_quote(self, symbol: str, fallback: bool = True) -> Quote:
        """Get real-time quote with intelligent provider selection."""
        # Check cache first (shorter TTL for real-time data)
        cache_key = f"quote:{symbol}"
        cached_quote = self._get_from_cache(cache_key)
        if cached_quote:
            return cached_quote

        # Get providers in priority order
        sorted_providers = await self._get_sorted_providers()

        last_exception = None
        for provider in sorted_providers:
            try:
                if not await provider.is_available():
                    continue

                self.logger.logger.debug(
                    f"Trying {provider.config.name} for quote: {symbol}"
                )

                quote = await provider.get_real_time_quote(symbol)

                # Cache successful result
                self._set_cache(cache_key, quote)

                # Record usage
                await self.cost_tracker.record_usage(
                    provider.config.name, 1, provider.config.cost_per_call
                )

                self.logger.logger.debug(
                    f"Successfully retrieved quote from {provider.config.name}"
                )
                return quote

            except Exception as e:
                last_exception = e
                self.logger.logger.warning(
                    f"Provider {provider.config.name} failed: {e}"
                )

                if not fallback:
                    break

        # If all providers failed
        error_msg = f"All data providers failed for quote {symbol}. Last error: {last_exception}"
        self.logger.logger.error(error_msg)
        raise DataProviderError(error_msg)

    async def get_company_fundamentals(
        self, symbol: str, fallback: bool = True
    ) -> Dict[str, Any]:
        """Get company fundamentals with intelligent provider selection."""
        # Check cache first (longer TTL for fundamentals)
        cache_key = f"fundamentals:{symbol}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        # Get providers in priority order
        sorted_providers = await self._get_sorted_providers()

        for provider in sorted_providers:
            try:
                if not await provider.is_available():
                    continue

                self.logger.logger.debug(
                    f"Trying {provider.config.name} for fundamentals: {symbol}"
                )

                fundamentals = await provider.get_company_fundamentals(symbol)

                if fundamentals:
                    # Cache successful result
                    self._set_cache(cache_key, fundamentals)

                    # Record usage
                    await self.cost_tracker.record_usage(
                        provider.config.name, 1, provider.config.cost_per_call
                    )

                    self.logger.logger.info(
                        f"Successfully retrieved fundamentals from {provider.config.name}"
                    )
                    return fundamentals

            except Exception as e:
                self.logger.logger.warning(
                    f"Provider {provider.config.name} failed: {e}"
                )

                if not fallback:
                    break

        # Return empty dict if all providers failed (fundamentals are optional)
        self.logger.logger.warning(f"No fundamentals available for {symbol}")
        return {}

    async def get_latest_price(self, symbol: str, fallback: bool = True) -> Decimal:
        """Get latest price with intelligent provider selection."""
        quote = await self.get_real_time_quote(symbol, fallback)
        return quote.mid_price

    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers."""
        status = {}

        for name, provider in self.providers.items():
            try:
                is_available = await provider.is_available()
                connection_ok = (
                    await provider.test_connection() if is_available else False
                )

                status[name] = {
                    "available": is_available,
                    "connection_ok": connection_ok,
                    "priority": provider.config.priority,
                    "rate_limit": provider.config.rate_limit,
                    "cost_per_call": provider.config.cost_per_call,
                }
            except Exception as e:
                status[name] = {
                    "available": False,
                    "connection_ok": False,
                    "error": str(e),
                }

        return status

    async def get_cost_report(self) -> Dict[str, Any]:
        """Get cost and usage report."""
        return await self.cost_tracker.get_report()

    async def _create_provider_configs(self) -> List[DataProviderConfig]:
        """Create provider configurations from the main config."""
        provider_configs = []

        # Alpaca
        if self.config.exchange.name == "alpaca":
            provider_configs.append(
                DataProviderConfig(
                    name="alpaca",
                    enabled=True,
                    priority=1,
                    api_key=os.getenv("ALPACA_API_KEY", ""),
                    secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
                    base_url=os.getenv(
                        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
                    ),
                    cost_per_call=0.0,
                    rate_limit=200,
                )
            )

        # Alpha Vantage
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            provider_configs.append(
                DataProviderConfig(
                    name="alpha_vantage",
                    api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
                    priority=2,
                    cost_per_call=0.0,
                    enabled=True,
                    rate_limit=25,
                )
            )

        # Yahoo Finance (Last resort)
        provider_configs.append(
            DataProviderConfig(
                name="yahoo",
                rate_limit=60,  # Conservative rate limiting
                priority=3,
                cost_per_call=0.0,
                enabled=True,
                fallback_on_error=True,
            )
        )

        return provider_configs

    async def _get_sorted_providers(self) -> List[BaseDataProvider]:
        """Get providers sorted by priority and availability."""
        available_providers = []

        for provider in self.providers.values():
            if await provider.is_available():
                available_providers.append(provider)

        # Sort by priority (lower number = higher priority)
        return sorted(available_providers, key=lambda p: p.config.priority)

    def _get_cache_key_with_ttl(self, key: str) -> str:
        """Generate cache key with TTL."""
        return f"{key}:{int(datetime.now().timestamp())}"

    def _get_from_cache(self, key: str) -> Any:
        """Get data from cache if not expired."""
        if key in self._cache:
            cached_item = self._cache[key]
            cached_time = cached_item.get("timestamp", 0)

            if datetime.now().timestamp() - cached_time < self._cache_ttl:
                return cached_item.get("data")
            else:
                # Remove expired item
                del self._cache[key]

        return None

    def _set_cache(self, key: str, data: Any) -> None:
        """Set data in cache with timestamp."""
        self._cache[key] = {
            "data": data,
            "timestamp": datetime.now().timestamp(),
            "ttl": self._cache_ttl,
        }

        # Simple cache size management (keep last 1000 items)
        if len(self._cache) > 1000:
            # Remove oldest 100 items
            oldest_keys = sorted(
                self._cache.keys(), key=lambda k: self._cache[k]["timestamp"]
            )[:100]

            for old_key in oldest_keys:
                del self._cache[old_key]
