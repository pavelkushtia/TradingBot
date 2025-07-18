"""Alpaca data provider implementation."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.models import MarketData, Quote
from .base import BaseDataProvider, DataProviderError


class AlpacaProvider(BaseDataProvider):
    """Alpaca data provider wrapper for existing MarketDataManager."""

    def __init__(self, config):
        super().__init__(config)
        self.market_data_manager: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize the Alpaca provider using existing MarketDataManager."""
        await super().initialize()
        # We'll set this from the main manager
        self.logger.logger.info("Alpaca provider initialized (wrapper)")

    def set_market_data_manager(self, manager: Any) -> None:
        """Set the market data manager reference."""
        self.market_data_manager = manager

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> List[MarketData]:
        """Get historical OHLCV bars using existing Alpaca integration."""
        if not self.market_data_manager:
            raise DataProviderError("Market data manager not set")

        # Use existing implementation
        return await self.market_data_manager.get_historical_bars(
            symbol, timeframe, start, end, limit
        )

    async def get_real_time_quote(self, symbol: str) -> Quote:
        """Get real-time quote using existing Alpaca integration."""
        if not self.market_data_manager:
            raise DataProviderError("Market data manager not set")

        # Get latest price from existing system
        latest_prices = self.market_data_manager.latest_prices
        if symbol in latest_prices:
            price = latest_prices[symbol]
            return Quote(
                symbol=symbol,
                timestamp=datetime.now(),
                bid_price=price,
                ask_price=price,
                bid_size=0,
                ask_size=0,
            )

        raise DataProviderError(f"No real-time data available for {symbol}")

    async def get_company_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamentals - not available in current Alpaca integration."""
        # Alpaca doesn't provide fundamentals data
        return {}
