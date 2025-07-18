"""Alpha Vantage data provider implementation."""

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import aiohttp

from ...core.models import MarketData, Quote
from .base import BaseDataProvider, DataProviderError


class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage data provider for market data."""

    def __init__(self, config):
        super().__init__(config)
        self.base_url = "https://www.alphavantage.co/query"
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize the Alpha Vantage provider."""
        await super().initialize()
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        self.logger.logger.info("Alpha Vantage provider initialized")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        await super().cleanup()

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> List[MarketData]:
        """Get historical OHLCV bars from Alpha Vantage."""
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

        # Determine the function based on timeframe
        function = self._get_function_for_timeframe(timeframe)

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.config.api_key,
            "outputsize": "full",
        }

        # Add interval for intraday data
        if function == "TIME_SERIES_INTRADAY":
            params["interval"] = self._convert_timeframe(timeframe)

        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    raise DataProviderError(
                        f"Alpha Vantage API error: {response.status}"
                    )

                data = await response.json()

                # Check for error messages
                if "Error Message" in data:
                    raise DataProviderError(
                        f"Alpha Vantage error: {data['Error Message']}"
                    )

                if "Note" in data:
                    # Rate limit hit
                    raise DataProviderError(f"Alpha Vantage rate limit: {data['Note']}")

                return self._parse_timeseries_data(data, symbol, start, end, limit)

        except aiohttp.ClientError as e:
            raise DataProviderError(f"Alpha Vantage request failed: {e}")
        except json.JSONDecodeError as e:
            raise DataProviderError(f"Alpha Vantage response parsing failed: {e}")

    async def get_real_time_quote(self, symbol: str) -> Quote:
        """Get real-time quote from Alpha Vantage."""
        symbol = self._validate_symbol(symbol)

        return await self._execute_with_retry(self._fetch_real_time_quote, symbol)

    async def _fetch_real_time_quote(self, symbol: str) -> Quote:
        """Internal method to fetch real-time quote."""
        if not self.session:
            raise DataProviderError("Session not initialized")

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.config.api_key,
        }

        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    raise DataProviderError(
                        f"Alpha Vantage API error: {response.status}"
                    )

                data = await response.json()

                # Check for error messages
                if "Error Message" in data:
                    raise DataProviderError(
                        f"Alpha Vantage error: {data['Error Message']}"
                    )

                if "Note" in data:
                    raise DataProviderError(f"Alpha Vantage rate limit: {data['Note']}")

                return self._parse_quote_data(data, symbol)

        except aiohttp.ClientError as e:
            raise DataProviderError(f"Alpha Vantage request failed: {e}")
        except json.JSONDecodeError as e:
            raise DataProviderError(f"Alpha Vantage response parsing failed: {e}")

    async def get_company_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamentals from Alpha Vantage."""
        symbol = self._validate_symbol(symbol)

        return await self._execute_with_retry(self._fetch_company_fundamentals, symbol)

    async def _fetch_company_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Internal method to fetch company fundamentals."""
        if not self.session:
            raise DataProviderError("Session not initialized")

        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.config.api_key,
        }

        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    raise DataProviderError(
                        f"Alpha Vantage API error: {response.status}"
                    )

                data = await response.json()

                # Check for error messages
                if "Error Message" in data:
                    raise DataProviderError(
                        f"Alpha Vantage error: {data['Error Message']}"
                    )

                if "Note" in data:
                    raise DataProviderError(f"Alpha Vantage rate limit: {data['Note']}")

                return self._parse_fundamentals_data(data)

        except aiohttp.ClientError as e:
            raise DataProviderError(f"Alpha Vantage request failed: {e}")
        except json.JSONDecodeError as e:
            raise DataProviderError(f"Alpha Vantage response parsing failed: {e}")

    def _get_function_for_timeframe(self, timeframe: str) -> str:
        """Get Alpha Vantage function based on timeframe."""
        if timeframe in ["1min", "5min", "15min", "30min", "1hour"]:
            return "TIME_SERIES_INTRADAY"
        elif timeframe == "1day":
            return "TIME_SERIES_DAILY"
        elif timeframe == "1week":
            return "TIME_SERIES_WEEKLY"
        elif timeframe == "1month":
            return "TIME_SERIES_MONTHLY"
        else:
            raise ValueError(f"Unsupported timeframe for Alpha Vantage: {timeframe}")

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Alpha Vantage format."""
        mapping = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "1hour": "60min",
        }
        return mapping.get(timeframe, "5min")

    def _parse_timeseries_data(
        self, data: dict, symbol: str, start: datetime, end: datetime, limit: int
    ) -> List[MarketData]:
        """Parse Alpha Vantage time series response."""
        # Find the time series key
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break

        if not time_series_key:
            raise DataProviderError(
                "No time series data found in Alpha Vantage response"
            )

        time_series = data[time_series_key]
        bars = []

        for timestamp_str, values in time_series.items():
            try:
                # Parse timestamp
                if "T" in timestamp_str:
                    # Intraday format: 2023-12-01 09:30:00
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                else:
                    # Daily format: 2023-12-01
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")

                # Add timezone info
                timestamp = timestamp.replace(tzinfo=timezone.utc)

                # Filter by date range
                if timestamp < start or timestamp > end:
                    continue

                # Parse OHLCV data
                bar = MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=Decimal(values.get("1. open", "0")),
                    high=Decimal(values.get("2. high", "0")),
                    low=Decimal(values.get("3. low", "0")),
                    close=Decimal(values.get("4. close", "0")),
                    volume=int(values.get("5. volume", "0")),
                )
                bars.append(bar)

                # Respect limit
                if len(bars) >= limit:
                    break

            except (ValueError, KeyError) as e:
                self.logger.logger.warning(f"Error parsing bar data: {e}")
                continue

        # Sort by timestamp (oldest first)
        bars.sort(key=lambda x: x.timestamp)
        return bars

    def _parse_quote_data(self, data: dict, symbol: str) -> Quote:
        """Parse Alpha Vantage quote response."""
        quote_data = data.get("Global Quote", {})

        if not quote_data:
            raise DataProviderError("No quote data found in Alpha Vantage response")

        try:
            # Alpha Vantage doesn't provide bid/ask, so we use the last price
            price = Decimal(quote_data.get("05. price", "0"))

            return Quote(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bid_price=price,  # Using last price as bid
                ask_price=price,  # Using last price as ask
                bid_size=0,  # Not available in Alpha Vantage
                ask_size=0,  # Not available in Alpha Vantage
            )

        except (ValueError, KeyError) as e:
            raise DataProviderError(f"Error parsing quote data: {e}")

    def _parse_fundamentals_data(self, data: dict) -> Dict[str, Any]:
        """Parse Alpha Vantage fundamentals response."""
        if not data:
            return {}

        # Map Alpha Vantage fields to standardized format
        fundamentals = {}

        # Basic company info
        if "Name" in data:
            fundamentals["company_name"] = data["Name"]
        if "Description" in data:
            fundamentals["description"] = data["Description"]
        if "Sector" in data:
            fundamentals["sector"] = data["Sector"]
        if "Industry" in data:
            fundamentals["industry"] = data["Industry"]

        # Financial metrics
        financial_fields = {
            "MarketCapitalization": "market_cap",
            "BookValue": "book_value",
            "DividendPerShare": "dividend_per_share",
            "DividendYield": "dividend_yield",
            "EarningsPerShare": "eps",
            "RevenuePerShareTTM": "revenue_per_share",
            "ProfitMargin": "profit_margin",
            "OperatingMarginTTM": "operating_margin",
            "ReturnOnAssetsTTM": "roa",
            "ReturnOnEquityTTM": "roe",
            "RevenueTTM": "revenue",
            "GrossProfitTTM": "gross_profit",
            "DilutedEPSTTM": "diluted_eps",
            "QuarterlyEarningsGrowthYOY": "earnings_growth",
            "QuarterlyRevenueGrowthYOY": "revenue_growth",
            "AnalystTargetPrice": "target_price",
            "TrailingPE": "pe_ratio",
            "ForwardPE": "forward_pe",
            "PriceToSalesRatioTTM": "ps_ratio",
            "PriceToBookRatio": "pb_ratio",
            "EVToRevenue": "ev_revenue",
            "EVToEBITDA": "ev_ebitda",
            "Beta": "beta",
            "52WeekHigh": "week_52_high",
            "52WeekLow": "week_52_low",
        }

        for av_field, std_field in financial_fields.items():
            if av_field in data and data[av_field] not in ["None", "", "-"]:
                try:
                    # Try to convert to float if possible
                    value = data[av_field]
                    if (
                        isinstance(value, str)
                        and value.replace(".", "").replace("-", "").isdigit()
                    ):
                        fundamentals[std_field] = float(value)
                    else:
                        fundamentals[std_field] = value
                except (ValueError, TypeError):
                    fundamentals[std_field] = data[av_field]

        return fundamentals
