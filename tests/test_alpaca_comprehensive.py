#!/usr/bin/env python3
"""
Comprehensive Alpaca API Connection and Functionality Test Suite

This test suite verifies all aspects of Alpaca integration:
1. Configuration loading and validation
2. Account connectivity and authentication
3. Order placement and execution
4. WebSocket connectivity with correct endpoints
5. Market data retrieval
6. Position management
7. Trading bot functionality

Usage:
    python tests/test_alpaca_comprehensive.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import aiohttp
import pytest
import websockets
from dotenv import load_dotenv

# Add the trading_bot module to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.core.bot import TradingBot
from trading_bot.core.config import Config


@pytest.mark.asyncio
class TestAlpacaComprehensive:
    """Comprehensive test suite for Alpaca integration."""

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Initialize the test suite and cleanup after."""
        load_dotenv()

        # Get credentials from environment
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        ).rstrip("/")

        # Test configuration
        self.test_symbol = "AAPL"
        self.test_quantity = 1
        self.test_timeout = 30

        # Common headers
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

        # HTTP session for reuse
        self.session = aiohttp.ClientSession()

        yield

        await self.session.close()

    async def test_configuration(self) -> None:
        """Test configuration loading and validation."""
        # Test 1: Environment file existence
        assert os.path.exists(".env"), "No .env file"

        # Test 2: Critical environment variables
        required_vars = {
            "ALPACA_API_KEY": self.api_key,
            "ALPACA_SECRET_KEY": self.secret_key,
            "ALPACA_BASE_URL": self.base_url,
            "ALPACA_WEBSOCKET_URL": os.getenv("ALPACA_WEBSOCKET_URL", ""),
        }

        for var_name, var_value in required_vars.items():
            assert bool(var_value), f"Missing environment variable: {var_name}"

        # Test 3: Configuration object creation
        config = Config.from_env()
        assert config is not None, "Failed to create Config object"
        assert (
            config.exchange.name == "alpaca"
        ), f"Exchange is not alpaca: {config.exchange.name}"
        assert "v2/iex" in config.market_data.websocket_url
        assert config.trading.portfolio_value > 0

    async def test_account_connectivity(self) -> None:
        """Test account connectivity and authentication."""
        async with self.session.get(
            f"{self.base_url}/v2/account", headers=self.headers
        ) as response:
            assert response.status == 200, await response.text()
            account_data = await response.json()
            assert account_data.get("status") == "ACTIVE"
            assert float(account_data.get("portfolio_value", 0)) > 0

    async def test_market_data(self) -> None:
        """Test market data retrieval."""
        # Test 1: Latest trade data
        data_url = "https://data.alpaca.markets/v2"
        url = f"{data_url}/stocks/{self.test_symbol}/trades/latest"
        async with self.session.get(url, headers=self.headers) as response:
            if response.status == 200:
                trade_data = await response.json()
                assert "trade" in trade_data
                assert "p" in trade_data["trade"]
                assert "s" in trade_data["trade"]
            else:
                # Fallback to quote endpoint for free accounts
                url = f"{self.base_url}/v2/stocks/{self.test_symbol}/quotes/latest"
                async with self.session.get(
                    url, headers=self.headers
                ) as quote_response:
                    assert quote_response.status == 200, await quote_response.text()
                    quote_data = await quote_response.json()
                    assert "quote" in quote_data
                    assert "bp" in quote_data["quote"]
                    assert "ap" in quote_data["quote"]

        # Test 2: Historical bars
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=5)
        url = f"{data_url}/stocks/{self.test_symbol}/bars"
        params = {
            "timeframe": "1Day",
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "limit": 5,
            "adjustment": "raw",
        }
        async with self.session.get(
            url, params=params, headers=self.headers
        ) as response:
            if response.status == 200:
                bars_data = await response.json()
                assert "bars" in bars_data
                assert len(bars_data["bars"]) > 0
                latest_bar = bars_data["bars"][-1]
                assert all(key in latest_bar for key in ["o", "h", "l", "c", "v"])

    async def test_order_execution(self) -> None:
        """Test order placement and execution."""
        # Test 1: Place a buy order
        order_data = {
            "symbol": self.test_symbol,
            "qty": self.test_quantity,
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        }
        async with self.session.post(
            f"{self.base_url}/v2/orders", json=order_data, headers=self.headers
        ) as response:
            assert response.status in [200, 201], await response.text()
            order_response = await response.json()
            order_id = order_response.get("id")
            assert order_id is not None

            # Test 2: Check order status
            await asyncio.sleep(1)
            async with self.session.get(
                f"{self.base_url}/v2/orders/{order_id}", headers=self.headers
            ) as status_response:
                assert status_response.status == 200, await status_response.text()
                order_status = await status_response.json()
                status = order_status.get("status")

                # Test 3: Cancel the order if it's still pending
                if status in ["new", "pending_new", "accepted"]:
                    async with self.session.delete(
                        f"{self.base_url}/v2/orders/{order_id}", headers=self.headers
                    ) as cancel_response:
                        assert (
                            cancel_response.status == 204
                        ), await cancel_response.text()
                else:
                    assert status in ["filled", "partially_filled"]

    async def test_websocket_connectivity(self) -> None:
        """Test WebSocket connectivity with correct endpoints."""
        # Test 1: Test stream (always available)
        test_url = "wss://stream.data.alpaca.markets/v2/test"
        async with websockets.connect(test_url) as websocket:
            welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            welcome_data = json.loads(welcome_msg)
            assert welcome_data[0].get("T") == "success"

        # Test 2: IEX stream (for free plan)
        iex_url = os.getenv(
            "ALPACA_WEBSOCKET_URL", "wss://stream.data.alpaca.markets/v2/iex"
        )
        async with websockets.connect(iex_url) as websocket:
            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key,
            }
            await websocket.send(json.dumps(auth_message))

            subscribe_message = {
                "action": "subscribe",
                "trades": [self.test_symbol],
                "quotes": [self.test_symbol],
            }
            await websocket.send(json.dumps(subscribe_message))

            # Wait for any message, and check if it's the subscription confirmation
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=20.0)
                data = json.loads(response)
                if (
                    isinstance(data, list)
                    and len(data) > 0
                    and data[0].get("T") == "subscription"
                ):
                    break

    async def test_trading_bot_initialization(self) -> None:
        """Test trading bot initialization with new configuration."""
        config = Config.from_env()
        bot = TradingBot(config)
        assert bot is not None
        status = bot.get_status()
        assert "running" in status
        assert config.exchange.name == "alpaca"
        assert "v2/iex" in config.market_data.websocket_url
        assert config.trading.portfolio_value > 0

    async def test_position_management(self) -> None:
        """Test position management functionality."""
        async with self.session.get(
            f"{self.base_url}/v2/positions", headers=self.headers
        ) as response:
            assert response.status == 200, await response.text()
            positions = await response.json()
            if positions:
                position = positions[0]
                assert all(
                    field in position
                    for field in ["symbol", "qty", "market_value", "side"]
                )

        async with self.session.get(
            f"{self.base_url}/v2/account/portfolio/history", headers=self.headers
        ) as response:
            assert response.status == 200, await response.text()
            history = await response.json()
            assert all(field in history for field in ["timestamp", "equity"])
