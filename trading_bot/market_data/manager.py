"""Market data manager for real-time data feeds."""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp
import websockets

from ..core.config import Config
from ..core.exceptions import MarketDataError
from ..core.logging import TradingLogger
from ..core.models import MarketData, Quote


class MarketDataManager:
    """High-performance market data manager with real-time feeds."""

    def __init__(self, config: Config):
        """Initialize market data manager."""
        self.config = config
        self.logger = TradingLogger("market_data")

        # Connection state
        self.connected = False
        self.websocket: Optional[Any] = None
        self.session: Optional[aiohttp.ClientSession] = None

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

        if self.websocket:
            await self.websocket.close()

        if self.session:
            await self.session.close()

        self.connected = False
        self.logger.logger.info("Market data manager shutdown")

    async def subscribe_symbols(self, symbols: List[str]) -> None:
        """Subscribe to market data for given symbols."""
        new_symbols = set(symbols) - self.subscribed_symbols

        if new_symbols:
            self.logger.logger.info(
                f"Subscribing to {len(new_symbols)} new symbols: {list(new_symbols)}"
            )

            for symbol in new_symbols:
                await self._subscribe_symbol(symbol)
                self.subscribed_symbols.add(symbol)

    async def unsubscribe_symbols(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for given symbols."""
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                await self._unsubscribe_symbol(symbol)
                self.subscribed_symbols.discard(symbol)

                # Clean up stored data
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
            end = datetime.utcnow()
        if not start:
            start = end - timedelta(minutes=limit)

        try:
            # This is a simplified implementation for Alpaca
            # In practice, you'd implement different providers
            url = f"{self.config.exchange.base_url}/v2/stocks/{symbol}/bars"
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
                    error_text = await response.text()
                    raise MarketDataError(
                        f"Failed to fetch historical data: {error_text}"
                    )

        except Exception as e:
            self.logger.log_error(
                e, {"context": "get_historical_bars", "symbol": symbol}
            )
            raise MarketDataError(f"Error fetching historical data: {e}")

    def register_quote_callback(self, callback: Callable[[Quote], None]) -> None:
        """Register callback for real-time quotes."""
        self.quote_callbacks.append(callback)

    def register_bar_callback(self, callback: Callable[[MarketData], None]) -> None:
        """Register callback for real-time bars."""
        self.bar_callbacks.append(callback)

    async def _connect_websocket(self) -> None:
        """Connect to WebSocket market data feed."""
        try:
            # Alpaca WebSocket URL
            ws_url = "wss://stream.data.alpaca.markets/v2/stocks"

            self.websocket = await websockets.connect(ws_url)
            self.connected = True
            self.reconnect_attempts = 0

            # Authenticate
            auth_message = {
                "action": "auth",
                "key": self.config.exchange.api_key,
                "secret": self.config.exchange.secret_key,
            }
            await self.websocket.send(json.dumps(auth_message))

            # Start message handling
            asyncio.create_task(self._handle_websocket_messages())

            self.logger.logger.info("WebSocket connected and authenticated")

        except Exception as e:
            self.connected = False
            await self._handle_reconnection(e)

    async def _handle_websocket_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        try:
            while self.connected and self.websocket:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(), timeout=30.0
                    )
                    await self._process_message(message)

                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    if self.websocket:
                        await self.websocket.ping()

                except websockets.exceptions.ConnectionClosed:
                    self.logger.logger.warning("WebSocket connection closed")
                    self.connected = False
                    await self._handle_reconnection(Exception("Connection closed"))
                    break

        except Exception as e:
            self.logger.log_error(e, {"context": "websocket_message_handling"})
            self.connected = False
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

            self.message_count += 1
            self.last_message_time = datetime.utcnow()

        except Exception as e:
            self.logger.log_error(
                e, {"context": "message_processing", "message": message[:100]}
            )

    async def _process_data_item(self, item: Dict[str, Any]) -> None:
        """Process individual data item."""
        msg_type = item.get("T")

        if msg_type == "q":  # Quote
            await self._handle_quote(item)
        elif msg_type == "t":  # Trade
            await self._handle_trade(item)
        elif msg_type == "b":  # Bar
            await self._handle_bar(item)
        elif msg_type in ["subscription", "success", "error"]:
            await self._handle_status_message(item)

    async def _handle_quote(self, data: Dict[str, Any]) -> None:
        """Handle quote message."""
        try:
            quote = Quote(
                symbol=data["S"],
                timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00")),
                bid_price=Decimal(str(data["bp"])),
                ask_price=Decimal(str(data["ap"])),
                bid_size=data["bs"],
                ask_size=data["as"],
            )

            self.latest_quotes[quote.symbol] = quote
            self.latest_prices[quote.symbol] = quote.mid_price

            # Notify callbacks
            for callback in self.quote_callbacks:
                try:
                    callback(quote)
                except Exception as e:
                    self.logger.log_error(e, {"context": "quote_callback"})

        except Exception as e:
            self.logger.log_error(e, {"context": "quote_processing", "data": data})

    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        """Handle trade message."""
        try:
            symbol = data["S"]
            price = Decimal(str(data["p"]))

            # Update latest price
            self.latest_prices[symbol] = price

        except Exception as e:
            self.logger.log_error(e, {"context": "trade_processing", "data": data})

    async def _handle_bar(self, data: Dict[str, Any]) -> None:
        """Handle bar message."""
        try:
            bar = MarketData(
                symbol=data["S"],
                timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00")),
                open=Decimal(str(data["o"])),
                high=Decimal(str(data["h"])),
                low=Decimal(str(data["l"])),
                close=Decimal(str(data["c"])),
                volume=data["v"],
                vwap=Decimal(str(data.get("vw", data["c"]))),
            )

            # Update latest price
            self.latest_prices[bar.symbol] = bar.close

            # Notify callbacks
            for callback in self.bar_callbacks:
                try:
                    callback(bar)
                except Exception as e:
                    self.logger.log_error(e, {"context": "bar_callback"})

        except Exception as e:
            self.logger.log_error(e, {"context": "bar_processing", "data": data})

    async def _handle_status_message(self, data: Dict[str, Any]) -> None:
        """Handle status messages."""
        msg_type = data.get("T")
        if msg_type == "error":
            error_msg = data.get("msg", "Unknown error")
            self.logger.logger.error(f"WebSocket error: {error_msg}")
            raise MarketDataError(f"WebSocket error: {error_msg}")

    async def _subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to a symbol."""
        if self.websocket and self.connected:
            subscribe_message = {
                "action": "subscribe",
                "quotes": [symbol],
                "trades": [symbol],
                "bars": [symbol],
            }
            await self.websocket.send(json.dumps(subscribe_message))

    async def _unsubscribe_symbol(self, symbol: str) -> None:
        """Unsubscribe from a symbol."""
        if self.websocket and self.connected:
            unsubscribe_message = {
                "action": "unsubscribe",
                "quotes": [symbol],
                "trades": [symbol],
                "bars": [symbol],
            }
            await self.websocket.send(json.dumps(unsubscribe_message))

    async def _handle_reconnection(self, error: Exception) -> None:
        """Handle WebSocket reconnection."""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = self.reconnect_delay * (
                2 ** (self.reconnect_attempts - 1)
            )  # Exponential backoff

            self.logger.logger.warning(
                f"Reconnecting to WebSocket "
                f"(attempt {self.reconnect_attempts}/"
                f"{self.max_reconnect_attempts}) in {delay}s..."
            )

            await asyncio.sleep(delay)

            try:
                await self._connect_websocket()

                # Re-subscribe to all symbols
                for symbol in self.subscribed_symbols:
                    await self._subscribe_symbol(symbol)

            except Exception as reconnect_error:
                self.logger.log_error(reconnect_error, {"context": "reconnection"})
                await self._handle_reconnection(reconnect_error)
        else:
            self.logger.logger.error(
                "Max reconnection attempts reached. Market data disconnected."
            )

    def _parse_historical_data(
        self, data: Dict[str, Any], symbol: str
    ) -> List[MarketData]:
        """Parse historical data response."""
        bars = []

        if "bars" in data:
            for bar_data in data["bars"]:
                bar = MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromisoformat(
                        bar_data["t"].replace("Z", "+00:00")
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
