"""Market data manager for real-time data feeds."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp
import websockets

from ..core.config import Config
from ..core.events import EventBus, MarketDataEvent
from ..core.exceptions import MarketDataError
from ..core.logging import TradingLogger
from ..core.models import MarketData, Quote


class MarketDataManager:
    """High-performance market data manager with real-time feeds."""

    def __init__(self, config: Config, event_bus: EventBus):
        """Initialize market data manager."""
        self.config = config
        self.event_bus = event_bus
        self.logger = TradingLogger("market_data")

        # Connection state
        self.connected = False
        self.authenticated = False  # Track authentication state
        self.websocket: Optional[Any] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.shutting_down = False  # Flag to prevent reconnection during shutdown

        # Data storage
        self.latest_quotes: Dict[str, Quote] = {}
        self.latest_prices: Dict[str, Decimal] = {}
        self.subscribed_symbols: Set[str] = set()

        # Event callbacks
        self.quote_callbacks: List[Callable[[Quote], None]] = []
        self.bar_callbacks: List[Callable[[MarketData], None]] = []

        # Performance tracking
        self.message_count = 0
        self.last_message_time: Optional[datetime] = None

        # Reconnection
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.market_data.max_reconnect_attempts
        self.reconnect_delay = config.market_data.websocket_reconnect_delay
        self.reconnect_task: Optional[asyncio.Task] = None  # Track reconnection task

    async def initialize(self) -> None:
        """Initialize market data connections."""
        try:
            self.logger.logger.info("Initializing market data manager...")

            # Create HTTP session
            self.session = aiohttp.ClientSession()

            # Connect to WebSocket feed
            await self._connect_websocket()

            self.logger.logger.info("Market data manager initialized")

        except Exception as e:
            raise MarketDataError(f"Failed to initialize market data manager: {e}")

    async def shutdown(self) -> None:
        """Shutdown market data connections."""
        self.logger.logger.info("Shutting down market data manager...")

        # Set shutdown flag to prevent reconnection
        self.shutting_down = True

        # Cancel any pending reconnection task
        if self.reconnect_task and not self.reconnect_task.done():
            self.reconnect_task.cancel()
            try:
                await self.reconnect_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket connection
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                self.logger.logger.warning(f"Error closing WebSocket: {e}")

        # Close HTTP session
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                self.logger.logger.warning(f"Error closing HTTP session: {e}")

        self.connected = False
        self.logger.logger.info("Market data manager shutdown")

    async def subscribe_symbols(self, symbols: List[str]) -> None:
        """Subscribe to market data for given symbols, enforcing a total limit."""
        new_symbols = set(symbols) - self.subscribed_symbols
        if not new_symbols:
            self.logger.logger.info("No new symbols to subscribe to.")
            return

        # Enforce a hard limit on the total number of subscribed symbols
        limit = 15
        remaining_capacity = limit - len(self.subscribed_symbols)

        if remaining_capacity <= 0:
            self.logger.logger.warning(
                f"Cannot subscribe to new symbols. Already at the limit of {limit} symbols."
            )
            return

        symbols_to_subscribe = list(new_symbols)[:remaining_capacity]
        self.logger.logger.info(
            f"Attempting to subscribe to {len(symbols_to_subscribe)} new symbols: {symbols_to_subscribe}"
        )

        try:
            await self._subscribe_symbols_batch(symbols_to_subscribe)
            self.subscribed_symbols.update(symbols_to_subscribe)
            self.logger.logger.info(
                f"Successfully sent subscription request for {len(symbols_to_subscribe)} symbols."
            )
        except Exception as e:
            self.logger.logger.error(
                f"Failed to subscribe to symbols {symbols_to_subscribe}: {e}",
                exc_info=True,
            )
            raise MarketDataError(f"Failed to subscribe to symbols: {e}") from e

        # Warn if some symbols were ignored due to the limit
        if len(new_symbols) > len(symbols_to_subscribe):
            ignored_symbols = list(new_symbols)[remaining_capacity:]
            self.logger.logger.warning(
                f"Reached symbol limit of {limit}. The following {len(ignored_symbols)} symbols were not subscribed: {ignored_symbols}"
            )

    async def unsubscribe_symbols(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for given symbols."""
        symbols_to_unsubscribe = [s for s in symbols if s in self.subscribed_symbols]

        if symbols_to_unsubscribe:
            # Send one unsubscription request with all symbols
            await self._unsubscribe_symbols_batch(symbols_to_unsubscribe)

            # Remove symbols from subscribed set and clean up data
            for symbol in symbols_to_unsubscribe:
                self.subscribed_symbols.discard(symbol)
                self.latest_quotes.pop(symbol, None)
                self.latest_prices.pop(symbol, None)

    async def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote for a symbol."""
        return self.latest_quotes.get(symbol)

    async def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """Get latest price for a symbol."""
        quote = self.latest_quotes.get(symbol)
        if quote:
            return quote.mid_price
        return self.latest_prices.get(symbol)

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1min",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[MarketData]:
        """Get historical market data bars."""
        if not self.session:
            raise MarketDataError("HTTP session not initialized")

        # Default to last 1000 minutes if no start/end provided
        if not end:
            end = datetime.now(timezone.utc)
        if not start:
            start = end - timedelta(minutes=limit)

        try:
            # Use the configured data API endpoint for market data
            data_api_url = self.config.market_data.data_api_url
            url = f"{data_api_url}/stocks/{symbol}/bars"
            params = {
                "timeframe": timeframe,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": limit,
                "adjustment": "raw",
            }

            headers = {
                "APCA-API-KEY-ID": self.config.exchange.api_key,
                "APCA-API-SECRET-KEY": self.config.exchange.secret_key,
            }

            async with self.session.get(
                url, params=params, headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_historical_data(data, symbol)
                else:
                    # If data API fails, try fallback or log gracefully
                    error_text = await response.text()
                    self.logger.logger.warning(
                        f"Data API failed for {symbol}: {error_text}. "
                        f"Trying Yahoo Finance fallback..."
                    )
                    # Try Yahoo Finance fallback
                    return await self._get_historical_bars_yahoo_fallback(
                        symbol, timeframe, start, end, limit
                    )

        except Exception as e:
            self.logger.log_error(
                e, {"context": "get_historical_bars", "symbol": symbol}
            )
            # Try Yahoo Finance fallback on any error
            return await self._get_historical_bars_yahoo_fallback(
                symbol, timeframe, start, end, limit
            )

    async def _get_historical_bars_yahoo_fallback(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> List[MarketData]:
        """Fallback to Yahoo Finance for historical data."""
        try:
            from .providers.yahoo import YahooFinanceProvider
            from .providers.base import DataProviderConfig

            # Create Yahoo Finance provider config
            yahoo_config = DataProviderConfig(
                name="yahoo_finance", timeout=10.0, rate_limit=60, max_retries=3
            )

            # Create Yahoo Finance provider
            yahoo_provider = YahooFinanceProvider(yahoo_config)
            await yahoo_provider.initialize()

            # Ensure timezone-aware datetimes for Yahoo Finance
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)

            # Get historical data from Yahoo Finance
            bars = await yahoo_provider.get_historical_bars(
                symbol, timeframe, start, end, limit
            )

            self.logger.logger.info(
                f"Successfully loaded {len(bars)} historical bars for {symbol} from Yahoo Finance"
            )

            await yahoo_provider.cleanup()
            return bars

        except Exception as e:
            self.logger.logger.error(
                f"Yahoo Finance fallback also failed for {symbol}: {e}"
            )
            return []

    def register_quote_callback(self, callback: Callable[[Quote], None]) -> None:
        """Register callback for real-time quotes."""
        self.logger.logger.warning(
            "register_quote_callback is deprecated, use event_bus.subscribe('market_data') instead"
        )
        self.quote_callbacks.append(callback)

    def register_bar_callback(self, callback: Callable[[MarketData], None]) -> None:
        """Register callback for real-time bars."""
        self.logger.logger.warning(
            "register_bar_callback is deprecated, use event_bus.subscribe('market_data') instead"
        )
        self.bar_callbacks.append(callback)

    async def _connect_websocket(self) -> None:
        """Connect to WebSocket market data feed."""
        try:
            # Validate API keys before connecting
            if not self.config.exchange.api_key or not self.config.exchange.secret_key:
                raise MarketDataError(
                    "API key and/or secret key are not configured. "
                    "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
                )

            # Use WebSocket URL from configuration
            ws_url = self.config.market_data.websocket_url

            self.logger.logger.info(f"Connecting to WebSocket: {ws_url}")

            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=self.config.market_data.websocket_ping_interval,
                ping_timeout=self.config.market_data.websocket_ping_timeout,
            )
            self.connected = True
            self.authenticated = False  # Reset authentication state
            self.reconnect_attempts = 0

            # Authenticate
            if self.websocket:
                auth_message = {
                    "action": "auth",
                    "key": self.config.exchange.api_key,
                    "secret": self.config.exchange.secret_key,
                }
                await self.websocket.send(json.dumps(auth_message))

                # Start message handling immediately
                asyncio.create_task(self._handle_websocket_messages())

            self.logger.logger.info("WebSocket connected and authentication sent")

        except Exception as e:
            self.connected = False
            self.authenticated = False
            await self._handle_reconnection(e)

    async def _handle_websocket_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        if not self.websocket:
            return
        try:
            async for message in self.websocket:
                self.message_count += 1
                self.last_message_time = datetime.now(timezone.utc)
                await self._process_message(message)

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.logger.warning(f"WebSocket connection closed: {e}")
            await self._handle_reconnection(e)
        except Exception as e:
            self.logger.log_error(e, {"context": "websocket_handling"})
            await self._handle_reconnection(e)

    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Handle different message types
            if isinstance(data, list):
                for item in data:
                    await self._process_data_item(item)
            else:
                await self._process_data_item(data)

        except Exception as e:
            self.logger.log_error(
                e, {"context": "message_processing", "message": message[:100]}
            )

    async def _process_data_item(self, item: Dict[str, Any]) -> None:
        """Process individual data item."""
        msg_type = item.get("T")
        self.logger.logger.debug(f"Processing message type: {msg_type}, data: {item}")

        if msg_type == "q":  # Quote
            await self._handle_quote(item)
        elif msg_type == "t":  # Trade
            await self._handle_trade(item)
        elif msg_type == "b":  # Bar
            await self._handle_bar(item)
        elif msg_type in ["subscription", "success", "error"]:
            await self._handle_status_message(item)

    async def _handle_quote(self, data: Dict[str, Any]) -> None:
        """Handle incoming quote data."""
        timestamp_str = data["t"]

        # Normalize timestamp to handle nanoseconds and 'Z' timezone
        if "." in timestamp_str:
            parts = timestamp_str.split(".")
            # Truncate fractional seconds to 6 digits (microseconds)
            parts[1] = parts[1][:6]
            timestamp_str = ".".join(parts)

        # Replace 'Z' with UTC offset for ISO 8601 compatibility
        timestamp_str = timestamp_str.replace("Z", "+00:00")

        try:
            quote = Quote(
                symbol=data["S"],
                bid_price=Decimal(str(data["bp"])),
                bid_size=int(data["bs"]),
                ask_price=Decimal(str(data["ap"])),
                ask_size=int(data["as"]),
                timestamp=datetime.fromisoformat(timestamp_str),
            )
            self.latest_quotes[quote.symbol] = quote

            # Convert quote to a MarketData-like object for the event
            market_data = MarketData(
                symbol=quote.symbol,
                timestamp=quote.timestamp,
                open=quote.mid_price,
                high=quote.mid_price,
                low=quote.mid_price,
                close=quote.mid_price,
                volume=0,  # Quotes don't have volume
            )
            await self.event_bus.publish("market_data", MarketDataEvent(market_data))

            # Legacy callbacks
            for callback in self.quote_callbacks:
                try:
                    callback(quote)
                except Exception as e:
                    self.logger.log_error(e, {"context": "quote_callback"})
        except ValueError as e:
            self.logger.logger.error(
                f"Failed to parse quote timestamp: {data['t']}. Error: {e}"
            )
        except Exception as e:
            self.logger.log_error(e, {"context": "handle_quote", "data": data})

    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        """Handle incoming trade data."""
        timestamp_str = data["t"]

        # Normalize timestamp to handle nanoseconds and 'Z' timezone
        if "." in timestamp_str:
            parts = timestamp_str.split(".")
            # Truncate fractional seconds to 6 digits (microseconds)
            parts[1] = parts[1][:6]
            timestamp_str = ".".join(parts)

        # Replace 'Z' with UTC offset for ISO 8601 compatibility
        timestamp_str = timestamp_str.replace("Z", "+00:00")

        try:
            # Although we don't create a Trade object, we can still parse the timestamp
            # to ensure its validity and for potential future use.
            datetime.fromisoformat(timestamp_str)

            price = Decimal(str(data["p"]))
            self.latest_prices[data["S"]] = price

            # Here you might want to create a Trade object and publish an event,
            # similar to _handle_quote, if you need to react to individual trades.

        except ValueError as e:
            self.logger.logger.error(
                f"Failed to parse trade timestamp: {data['t']}. Error: {e}"
            )
        except Exception as e:
            self.logger.log_error(e, {"context": "handle_trade", "data": data})

    async def _handle_bar(self, data: Dict[str, Any]) -> None:
        """Handle incoming bar data."""
        timestamp_str = data["t"]
        if "." in timestamp_str:
            parts = timestamp_str.split(".")
            parts[1] = parts[1][:6]
            timestamp_str = ".".join(parts)
        bar = MarketData(
            symbol=data["S"],
            timestamp=datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")),
            open=Decimal(str(data["o"])),
            high=Decimal(str(data["h"])),
            low=Decimal(str(data["l"])),
            close=Decimal(str(data["c"])),
            volume=int(data["v"]),
        )

        # Publish event
        await self.event_bus.publish("market_data", MarketDataEvent(bar))

        # Legacy callbacks are deprecated in favor of events
        # for callback in self.bar_callbacks:
        #     try:
        #         callback(bar)
        #     except Exception as e:
        #         self.logger.log_error(e, {"context": "bar_callback"})

    async def _handle_status_message(self, data: Dict[str, Any]) -> None:
        """Handle status messages."""
        msg_type = data.get("T")
        if msg_type == "error":
            error_msg = data.get("msg", "Unknown error")
            error_code = data.get("code", 0)
            self.logger.logger.error(f"WebSocket error: {error_msg}")

            # Handle connection limit errors specially
            if error_code == 406 or "connection limit exceeded" in error_msg.lower():
                # Don't raise exception, just log and disconnect
                self.logger.logger.error(
                    "Connection limit exceeded. Multiple bot instances may be running. "
                    "Please ensure only one instance is active."
                )
                self.connected = False
                return

            raise MarketDataError(f"WebSocket error: {error_msg}")
        elif msg_type == "success":
            if data.get("msg") == "authenticated":
                self.authenticated = True
                self.logger.logger.info("WebSocket authenticated successfully")
        elif msg_type == "subscription":
            # This message is typically sent after successful subscription
            # We can use it to confirm if the subscription was successful
            # For now, we'll just log it.
            self.logger.logger.info(
                f"WebSocket subscription successful: {data.get('msg')}"
            )

    async def _subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to a single symbol (legacy method for backward compatibility)."""
        await self._subscribe_symbols_batch([symbol])

    async def _subscribe_symbols_batch(self, symbols: List[str]) -> None:
        """Subscribe to multiple symbols in a single request."""
        if self.websocket and self.connected:
            subscribe_message = {
                "action": "subscribe",
                "quotes": symbols,
                "trades": symbols,
                "bars": symbols,
            }
            await self.websocket.send(json.dumps(subscribe_message))
        else:
            self.logger.logger.warning(
                f"Cannot subscribe to symbols: connected={self.connected}, authenticated={self.authenticated}"
            )

    async def _unsubscribe_symbols_batch(self, symbols: List[str]) -> None:
        """Unsubscribe from multiple symbols in a single request."""
        if self.websocket and self.connected:
            unsubscribe_message = {
                "action": "unsubscribe",
                "quotes": symbols,
                "trades": symbols,
                "bars": symbols,
            }
            await self.websocket.send(json.dumps(unsubscribe_message))

    async def _unsubscribe_symbol(self, symbol: str) -> None:
        """Unsubscribe from a single symbol (legacy method for backward compatibility)."""
        await self._unsubscribe_symbols_batch([symbol])

    async def _handle_reconnection(self, error: Exception) -> None:
        """Handle WebSocket reconnection."""
        # Don't reconnect if shutting down
        if self.shutting_down:
            self.logger.logger.info("Skipping reconnection: shutting down")
            return

        # Check if this is a connection limit error
        error_str = str(error).lower()
        if "connection limit exceeded" in error_str or "406" in error_str:
            self.logger.logger.error(
                "Connection limit exceeded. This may be due to multiple bot instances. "
                "Please ensure only one instance is running."
            )
            # Stop trying to reconnect for connection limit errors
            return

        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = min(
                self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 60
            )  # Cap at 60s

            self.logger.logger.warning(
                f"Reconnecting to WebSocket "
                f"(attempt {self.reconnect_attempts}/"
                f"{self.max_reconnect_attempts}) in {delay}s..."
            )

            # Create reconnection task so it can be cancelled during shutdown
            self.reconnect_task = asyncio.create_task(self._perform_reconnection(delay))

        else:
            self.logger.logger.error(
                "Max reconnection attempts reached. Market data disconnected."
            )

    async def _perform_reconnection(self, delay: float) -> None:
        """Perform the actual reconnection after delay."""
        try:
            await asyncio.sleep(delay)

            # Check again if we're shutting down after the delay
            if self.shutting_down:
                return

            await self._connect_websocket()

            # Re-subscribe to all symbols
            for symbol in self.subscribed_symbols:
                await self._subscribe_symbol(symbol)

        except asyncio.CancelledError:
            # Task was cancelled during shutdown
            return
        except Exception as reconnect_error:
            self.logger.log_error(reconnect_error, {"context": "reconnection"})
            if not self.shutting_down:
                await self._handle_reconnection(reconnect_error)

    def _parse_historical_data(
        self, data: Dict[str, Any], symbol: str
    ) -> List[MarketData]:
        """Parse historical data response."""
        bars = []

        if "bars" in data:
            for bar_data in data["bars"]:
                timestamp_str = bar_data["t"]
                if "." in timestamp_str:
                    parts = timestamp_str.split(".")
                    parts[1] = parts[1][:6]
                    timestamp_str = ".".join(parts)
                bar = MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    ),
                    open=Decimal(str(bar_data["o"])),
                    high=Decimal(str(bar_data["h"])),
                    low=Decimal(str(bar_data["l"])),
                    close=Decimal(str(bar_data["c"])),
                    volume=bar_data["v"],
                    vwap=Decimal(str(bar_data.get("vw", bar_data["c"]))),
                )
                bars.append(bar)

        return bars

    def get_stats(self) -> Dict[str, Any]:
        """Get market data statistics."""
        return {
            "connected": self.connected,
            "subscribed_symbols": len(self.subscribed_symbols),
            "message_count": self.message_count,
            "last_message_time": (
                self.last_message_time.isoformat() if self.last_message_time else None
            ),
            "reconnect_attempts": self.reconnect_attempts,
            "latest_quotes_count": len(self.latest_quotes),
        }
