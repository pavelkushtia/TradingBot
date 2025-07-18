"""Yahoo Finance data provider implementation."""

import asyncio
import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import aiohttp

from ...core.models import MarketData, Quote
from .base import BaseDataProvider, DataProviderError


class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance data provider for market data (fallback option)."""

    def __init__(self, config):
        super().__init__(config)
        self.base_url = "https://query1.finance.yahoo.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum 1 second between requests

    async def initialize(self) -> None:
        """Initialize the Yahoo Finance provider."""
        await super().initialize()

        # Use transparent, honest identification instead of browser spoofing
        import platform

        headers = {
            "User-Agent": f"TradingBot/1.0 Python/{platform.python_version()}",
            "Accept": "application/json, */*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout), headers=headers
        )

        # Add disclaimer in logs about Yahoo Finance usage
        self.logger.logger.warning(
            "Using Yahoo Finance data - ensure compliance with their Terms of Service"
        )
        self.logger.logger.info(
            "Yahoo Finance provider initialized with transparent headers"
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        await super().cleanup()

    async def _throttle_requests(self):
        """Throttle requests to avoid overwhelming Yahoo Finance."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)

        self._last_request_time = time.time()

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> List[MarketData]:
        """Get historical OHLCV bars from Yahoo Finance."""
        symbol = self._validate_symbol(symbol)
        timeframe = self._validate_timeframe(timeframe)
        self._validate_date_range(start, end)

        return await self._execute_with_retry(
            self._fetch_historical_bars, symbol, timeframe, start, end, limit
        )

    async def _fetch_historical_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime, limit: int
    ) -> List[MarketData]:
        """Internal method to fetch historical bars."""
        if not self.session:
            raise DataProviderError("Session not initialized")

        await self._throttle_requests()

        # Convert timeframe to Yahoo Finance format
        interval = self._convert_timeframe(timeframe)

        # Convert dates to Unix timestamps
        period1 = int(start.timestamp())
        period2 = int(end.timestamp())

        url = f"{self.base_url}/v8/finance/chart/{symbol}"
        params = {
            "period1": period1,
            "period2": period2,
            "interval": interval,
            "includePrePost": "false",
            "events": "div,splits",
        }

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 429:
                    raise DataProviderError("Yahoo Finance rate limit exceeded")

                if response.status != 200:
                    raise DataProviderError(
                        f"Yahoo Finance API error: {response.status}"
                    )

                data = await response.json()

                # Check for errors
                if "error" in data:
                    raise DataProviderError(f"Yahoo Finance error: {data['error']}")

                return self._parse_chart_data(data, symbol, limit)

        except aiohttp.ClientError as e:
            raise DataProviderError(f"Yahoo Finance request failed: {e}")
        except json.JSONDecodeError as e:
            raise DataProviderError(f"Yahoo Finance response parsing failed: {e}")

    async def get_real_time_quote(self, symbol: str) -> Quote:
        """Get real-time quote from Yahoo Finance."""
        symbol = self._validate_symbol(symbol)

        return await self._execute_with_retry(self._fetch_real_time_quote, symbol)

    async def _fetch_real_time_quote(self, symbol: str) -> Quote:
        """Internal method to fetch real-time quote."""
        if not self.session:
            raise DataProviderError("Session not initialized")

        await self._throttle_requests()

        url = f"{self.base_url}/v6/finance/quote"
        params = {
            "symbols": symbol,
            "fields": "regularMarketPrice,bid,ask,bidSize,askSize",
        }

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 429:
                    raise DataProviderError("Yahoo Finance rate limit exceeded")

                if response.status != 200:
                    raise DataProviderError(
                        f"Yahoo Finance API error: {response.status}"
                    )

                data = await response.json()

                # Check for errors
                if "error" in data:
                    raise DataProviderError(f"Yahoo Finance error: {data['error']}")

                return self._parse_quote_data(data, symbol)

        except aiohttp.ClientError as e:
            raise DataProviderError(f"Yahoo Finance request failed: {e}")
        except json.JSONDecodeError as e:
            raise DataProviderError(f"Yahoo Finance response parsing failed: {e}")

    async def get_company_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamentals from Yahoo Finance."""
        symbol = self._validate_symbol(symbol)

        return await self._execute_with_retry(self._fetch_company_fundamentals, symbol)

    async def _fetch_company_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Internal method to fetch company fundamentals."""
        if not self.session:
            raise DataProviderError("Session not initialized")

        await self._throttle_requests()

        url = f"{self.base_url}/v10/finance/quoteSummary/{symbol}"
        params = {
            "modules": "summaryDetail,financialData,defaultKeyStatistics,assetProfile"
        }

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 429:
                    raise DataProviderError("Yahoo Finance rate limit exceeded")

                if response.status != 200:
                    raise DataProviderError(
                        f"Yahoo Finance API error: {response.status}"
                    )

                data = await response.json()

                # Check for errors
                if "error" in data:
                    raise DataProviderError(f"Yahoo Finance error: {data['error']}")

                return self._parse_fundamentals_data(data)

        except aiohttp.ClientError as e:
            raise DataProviderError(f"Yahoo Finance request failed: {e}")
        except json.JSONDecodeError as e:
            raise DataProviderError(f"Yahoo Finance response parsing failed: {e}")

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Yahoo Finance format."""
        mapping = {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1hour": "1h",
            "1day": "1d",
            "1week": "1wk",
            "1month": "1mo",
        }
        yahoo_interval = mapping.get(timeframe)
        if not yahoo_interval:
            raise ValueError(f"Unsupported timeframe for Yahoo Finance: {timeframe}")
        return yahoo_interval

    def _parse_chart_data(
        self, data: dict, symbol: str, limit: int
    ) -> List[MarketData]:
        """Parse Yahoo Finance chart response."""
        try:
            result = data.get("chart", {}).get("result", [])
            if not result:
                raise DataProviderError("No chart data found in Yahoo Finance response")

            chart_data = result[0]

            # Get timestamps
            timestamps = chart_data.get("timestamp", [])
            if not timestamps:
                return []

            # Get indicators
            indicators = chart_data.get("indicators", {})
            quote_data = indicators.get("quote", [])
            if not quote_data:
                raise DataProviderError("No quote data found in Yahoo Finance response")

            ohlcv = quote_data[0]
            opens = ohlcv.get("open", [])
            highs = ohlcv.get("high", [])
            lows = ohlcv.get("low", [])
            closes = ohlcv.get("close", [])
            volumes = ohlcv.get("volume", [])

            bars = []
            for i, timestamp in enumerate(timestamps[:limit]):
                try:
                    # Skip bars with missing data
                    if (
                        i >= len(opens)
                        or opens[i] is None
                        or i >= len(highs)
                        or highs[i] is None
                        or i >= len(lows)
                        or lows[i] is None
                        or i >= len(closes)
                        or closes[i] is None
                    ):
                        continue

                    bar = MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                        open=Decimal(str(opens[i])),
                        high=Decimal(str(highs[i])),
                        low=Decimal(str(lows[i])),
                        close=Decimal(str(closes[i])),
                        volume=(
                            int(volumes[i])
                            if i < len(volumes) and volumes[i] is not None
                            else 0
                        ),
                    )
                    bars.append(bar)

                except (ValueError, IndexError) as e:
                    self.logger.logger.warning(f"Error parsing bar at index {i}: {e}")
                    continue

            return bars

        except (KeyError, IndexError) as e:
            raise DataProviderError(f"Error parsing Yahoo Finance chart data: {e}")

    def _parse_quote_data(self, data: dict, symbol: str) -> Quote:
        """Parse Yahoo Finance quote response."""
        try:
            quote_summary = data.get("quoteResponse", {})
            result = quote_summary.get("result", [])

            if not result:
                raise DataProviderError("No quote data found in Yahoo Finance response")

            quote_data = result[0]

            # Extract price data
            price = quote_data.get("regularMarketPrice", 0)
            bid = quote_data.get("bid", price)
            ask = quote_data.get("ask", price)
            bid_size = quote_data.get("bidSize", 0)
            ask_size = quote_data.get("askSize", 0)

            return Quote(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bid_price=Decimal(str(bid)) if bid is not None else Decimal(str(price)),
                ask_price=Decimal(str(ask)) if ask is not None else Decimal(str(price)),
                bid_size=int(bid_size) if bid_size is not None else 0,
                ask_size=int(ask_size) if ask_size is not None else 0,
            )

        except (KeyError, ValueError) as e:
            raise DataProviderError(f"Error parsing Yahoo Finance quote data: {e}")

    def _parse_fundamentals_data(self, data: dict) -> Dict[str, Any]:
        """Parse Yahoo Finance fundamentals response."""
        try:
            quote_summary = data.get("quoteSummary", {})
            result = quote_summary.get("result", [])

            if not result:
                return {}

            fundamentals_data = result[0]
            fundamentals = {}

            # Extract data from different modules
            modules = [
                "summaryDetail",
                "financialData",
                "defaultKeyStatistics",
                "assetProfile",
            ]

            for module_name in modules:
                module_data = fundamentals_data.get(module_name, {})
                if not module_data:
                    continue

                # Extract and flatten the data
                for key, value in module_data.items():
                    if isinstance(value, dict) and "raw" in value:
                        # Yahoo Finance often wraps values in {'raw': value, 'fmt': formatted_value}
                        fundamentals[key] = value["raw"]
                    elif not isinstance(value, dict):
                        fundamentals[key] = value

            # Map common fields to standardized names
            field_mapping = {
                "marketCap": "market_cap",
                "bookValue": "book_value",
                "dividendRate": "dividend_per_share",
                "dividendYield": "dividend_yield",
                "earningsGrowth": "earnings_growth",
                "revenueGrowth": "revenue_growth",
                "profitMargins": "profit_margin",
                "operatingMargins": "operating_margin",
                "returnOnAssets": "roa",
                "returnOnEquity": "roe",
                "totalRevenue": "revenue",
                "grossProfits": "gross_profit",
                "trailingEps": "eps",
                "forwardEps": "forward_eps",
                "targetMeanPrice": "target_price",
                "trailingPE": "pe_ratio",
                "forwardPE": "forward_pe",
                "priceToSalesTrailing12Months": "ps_ratio",
                "priceToBook": "pb_ratio",
                "enterpriseToRevenue": "ev_revenue",
                "enterpriseToEbitda": "ev_ebitda",
                "beta": "beta",
                "fiftyTwoWeekHigh": "week_52_high",
                "fiftyTwoWeekLow": "week_52_low",
                "shortName": "company_name",
                "longBusinessSummary": "description",
                "sector": "sector",
                "industry": "industry",
            }

            # Apply mapping
            mapped_fundamentals = {}
            for yahoo_field, std_field in field_mapping.items():
                if yahoo_field in fundamentals:
                    mapped_fundamentals[std_field] = fundamentals[yahoo_field]

            return mapped_fundamentals

        except (KeyError, ValueError) as e:
            self.logger.logger.warning(f"Error parsing Yahoo Finance fundamentals: {e}")
            return {}
