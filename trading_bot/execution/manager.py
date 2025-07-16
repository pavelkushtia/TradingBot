"""Order execution manager with mock and real exchange support."""

import asyncio
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..core.config import Config
from ..core.exceptions import OrderExecutionError
from ..core.logging import TradingLogger
from ..core.models import Order, OrderSide, OrderStatus, OrderType, Trade
from ..core.shared_execution import SharedExecutionLogic


class ExecutionManager:
    """High-performance order execution manager."""

    def __init__(self, config: Config):
        """Initialize execution manager."""
        self.config = config
        self.logger = TradingLogger("execution_manager")

        # Initialize shared execution logic (same as backtesting)
        self.shared_execution = SharedExecutionLogic()

        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}

        # Performance tracking
        self.orders_submitted = 0
        self.orders_filled = 0
        self.orders_cancelled = 0
        self.total_volume = Decimal("0")

        # Mock execution for simulation
        self.mock_mode = config.exchange.environment == "sandbox"
        self.mock_fill_delay = 0.1  # seconds

    async def initialize(self) -> None:
        """Initialize execution manager."""
        self.logger.logger.info("Execution manager initialized")

        if self.mock_mode:
            self.logger.logger.info("Running in mock execution mode")

    async def shutdown(self) -> None:
        """Shutdown execution manager."""
        # Cancel all pending orders
        for order_id in list(self.pending_orders.keys()):
            try:
                await self.cancel_order(order_id)
            except Exception as e:
                self.logger.log_error(
                    e, {"context": "shutdown_cancel", "order_id": order_id}
                )

        self.logger.logger.info("Execution manager shutdown")

    async def submit_order(self, order: Order) -> Optional[Order]:
        """Submit an order for execution."""
        try:
            # Assign order ID if not provided
            if not order.id:
                order.id = str(uuid.uuid4())

            # Set timestamps
            order.created_at = datetime.now(timezone.utc)
            order.updated_at = order.created_at

            # Validate order
            if not await self._validate_order(order):
                raise OrderExecutionError(f"Order validation failed: {order.id}")

            # Add to pending orders
            self.pending_orders[order.id] = order
            self.orders_submitted += 1

            self.logger.log_order(order.dict(), "submitted")

            if self.mock_mode:
                # Schedule mock execution
                asyncio.create_task(self._mock_execute_order(order))
            else:
                # Submit to real exchange
                await self._submit_to_exchange(order)

            return order

        except Exception as e:
            self.logger.log_error(
                e, {"context": "order_submission", "order": order.dict()}
            )
            order.status = OrderStatus.REJECTED
            return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            if order_id not in self.pending_orders:
                self.logger.logger.warning(
                    f"Order {order_id} not found for cancellation"
                )
                return False

            order = self.pending_orders[order_id]

            if self.mock_mode:
                # Mock cancellation
                await self._mock_cancel_order(order)
            else:
                # Cancel on real exchange
                await self._cancel_on_exchange(order)

            # Update order status
            order.status = OrderStatus.CANCELED
            order.updated_at = datetime.now(timezone.utc)

            # Move to completed orders
            self.completed_orders[order_id] = order
            del self.pending_orders[order_id]

            self.orders_cancelled += 1
            self.logger.log_order(order.dict(), "cancelled")

            return True

        except Exception as e:
            self.logger.log_error(
                e, {"context": "order_cancellation", "order_id": order_id}
            )
            return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current status of an order."""
        # Check pending orders first
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]

        # Check completed orders
        if order_id in self.completed_orders:
            return self.completed_orders[order_id]

        return None

    async def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return list(self.pending_orders.values())

    async def get_recent_trades(self, hours: int = 24) -> List[Trade]:
        """Get recent trades (mock implementation)."""
        # In a real implementation, this would query the exchange or database
        return []

    async def _validate_order(self, order: Order) -> bool:
        """Validate order parameters."""
        # Basic validation
        if order.quantity <= 0:
            self.logger.logger.error(f"Invalid quantity: {order.quantity}")
            return False

        if order.type == OrderType.LIMIT and (not order.price or order.price <= 0):
            self.logger.logger.error(f"Invalid limit price: {order.price}")
            return False

        # Validate stop loss orders
        if order.type == OrderType.STOP and (
            not order.stop_price or order.stop_price <= 0
        ):
            self.logger.logger.error(f"Invalid stop price: {order.stop_price}")
            return False

        # Symbol validation (basic)
        if not order.symbol or len(order.symbol) < 1:
            self.logger.logger.error(f"Invalid symbol: {order.symbol}")
            return False

        return True

    async def _submit_to_exchange(self, order: Order) -> None:
        """Submit order to real exchange (Alpaca implementation)."""

        # If in mock mode, skip real exchange submission
        if self.mock_mode:
            self.logger.logger.info(
                f"Mock mode: Skipping real exchange submission for order {order.id}"
            )
            return

        self.logger.logger.info(f"Submitting order to exchange: {order.id}")

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                url = f"{self.config.exchange.base_url}/v2/orders"
                headers = {
                    "APCA-API-KEY-ID": self.config.exchange.api_key,
                    "APCA-API-SECRET-KEY": self.config.exchange.secret_key,
                    "Content-Type": "application/json",
                }
                order_data = {
                    "symbol": order.symbol,
                    "qty": str(order.quantity),
                    "side": order.side.value,
                    "type": order.type.value,
                    "time_in_force": "day",  # Default to day orders
                }
                if order.price:
                    order_data["limit_price"] = str(order.price)

                async with session.post(
                    url, json=order_data, headers=headers
                ) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        order.id = result["id"]
                        self.logger.logger.info(
                            f"Order submitted successfully: {order.id}"
                        )
                    else:
                        error_text = await response.text()
                        raise OrderExecutionError(
                            f"Failed to submit order: {response.status} - {error_text}"
                        )
        except Exception as e:
            self.logger.log_error(
                e, {"context": "exchange_submission", "order_id": order.id}
            )
            raise OrderExecutionError(f"Order submission failed: {e}")

    async def _cancel_on_exchange(self, order: Order) -> None:
        """Cancel order on real exchange."""

        # If in mock mode, skip real exchange cancellation
        if self.mock_mode:
            self.logger.logger.info(
                f"Mock mode: Skipping real exchange cancellation for order {order.id}"
            )
            return

        self.logger.logger.info(f"Cancelling order on exchange: {order.id}")

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                url = f"{self.config.exchange.base_url}/v2/orders/{order.id}"
                headers = {
                    "APCA-API-KEY-ID": self.config.exchange.api_key,
                    "APCA-API-SECRET-KEY": self.config.exchange.secret_key,
                }
                async with session.delete(url, headers=headers) as response:
                    if response.status == 204:
                        self.logger.logger.info(
                            f"Order cancelled successfully: {order.id}"
                        )
                    else:
                        error_text = await response.text()
                        raise OrderExecutionError(
                            f"Failed to cancel order: {response.status} - {error_text}"
                        )
        except Exception as e:
            self.logger.log_error(
                e, {"context": "exchange_cancellation", "order_id": order.id}
            )
            raise OrderExecutionError(f"Order cancellation failed: {e}")

    async def _mock_execute_order(self, order: Order) -> None:
        """Mock order execution for simulation."""
        try:
            # Simulate execution delay
            await asyncio.sleep(self.mock_fill_delay)

            # Check if order was cancelled
            if order.id not in self.pending_orders:
                return

            # Simulate different execution scenarios
            import random

            # 95% fill rate for market orders, 80% for limit orders
            fill_probability = 0.95 if order.type == OrderType.MARKET else 0.80

            if random.random() < fill_probability:
                await self._mock_fill_order(order)
            else:
                # Order remains pending or gets cancelled
                if order.type == OrderType.LIMIT:
                    # Limit orders can stay pending
                    self.logger.logger.info(f"Limit order {order.id} remains pending")
                else:
                    # Market orders that don't fill get rejected
                    await self._mock_reject_order(order, "Insufficient liquidity")

        except Exception as e:
            self.logger.log_error(
                e, {"context": "mock_execution", "order_id": order.id}
            )

    async def _mock_fill_order(self, order: Order) -> None:
        """Mock order fill."""
        try:
            # Simulate realistic fill price
            fill_price = (
                order.price
                if order.price
                else self._get_mock_market_price(order.symbol)
            )

            # Add some slippage for market orders
            if order.type == OrderType.MARKET:
                slippage_factor = Decimal("0.001")  # 0.1% slippage
                if order.side == OrderSide.BUY:
                    fill_price *= 1 + slippage_factor
                else:
                    fill_price *= 1 - slippage_factor

            # Create trade
            trade = self.shared_execution.create_trade_from_order(order, fill_price)

            # Update order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = fill_price
            order.updated_at = datetime.now(timezone.utc)

            # Move to completed orders
            if order.id:
                self.completed_orders[order.id] = order
                del self.pending_orders[order.id]

            # Update statistics
            self.orders_filled += 1
            self.total_volume += order.quantity * fill_price

            self.logger.log_order(order.dict(), "filled")
            self.logger.log_trade(trade.dict())

        except Exception as e:
            self.logger.log_error(e, {"context": "mock_fill", "order_id": order.id})

    async def _mock_reject_order(self, order: Order, reason: str) -> None:
        """Mock order rejection."""
        order.status = OrderStatus.REJECTED
        order.updated_at = datetime.now(timezone.utc)
        order.metadata["rejection_reason"] = reason

        # Move to completed orders
        if order.id:
            self.completed_orders[order.id] = order
            del self.pending_orders[order.id]

        self.logger.log_order(order.dict(), "rejected")

    async def _mock_cancel_order(self, order: Order) -> None:
        """Mock order cancellation."""
        # Just log the cancellation - actual status update happens in cancel_order
        self.logger.logger.info(f"Mock cancelling order: {order.id}")

    def _get_mock_market_price(self, symbol: str) -> Decimal:
        """Get mock market price for a symbol."""
        # Simple mock price generation
        import random

        base_prices = {
            "AAPL": 150.0,
            "GOOGL": 2500.0,
            "MSFT": 300.0,
            "TSLA": 800.0,
            "SPY": 400.0,
        }

        base_price = base_prices.get(symbol, 100.0)
        # Add some random variation
        variation = random.uniform(-0.02, 0.02)  # ±2%
        mock_price = base_price * (1 + variation)

        return Decimal(str(round(mock_price, 2)))

    def _calculate_commission(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Calculate commission for a trade (using shared execution logic)."""
        return self.shared_execution.calculate_commission(quantity, price)

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        fill_rate = (
            (self.orders_filled / self.orders_submitted)
            if self.orders_submitted > 0
            else 0
        )

        return {
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_cancelled": self.orders_cancelled,
            "fill_rate": fill_rate,
            "total_volume": str(self.total_volume),
            "pending_orders": len(self.pending_orders),
            "mock_mode": self.mock_mode,
        }
