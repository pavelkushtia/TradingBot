"""Order execution manager with mock and real exchange support."""

import asyncio
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..core.config import Config
from ..core.events import EventBus, OrderEvent, SignalEvent
from ..core.exceptions import OrderExecutionError
from ..core.logging import TradingLogger
from ..core.models import Order, OrderSide, OrderStatus, OrderType, Trade
from ..core.shared_execution import SharedExecutionLogic


class ExecutionManager:
    """High-performance order execution manager."""

    def __init__(self, config: Config, event_bus: EventBus):
        """Initialize execution manager."""
        self.config = config
        self.event_bus = event_bus
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

        # Paper trading mode
        self.paper_trading_mode = config.trading.paper_trading
        self.paper_trading_fill_delay = 0.1  # seconds

        # Order synchronization lock to prevent race conditions
        self._order_lock = asyncio.Lock()

    async def on_signal(self, event: SignalEvent) -> None:
        """Handle new signal."""
        order = await self._signal_to_order(event.signal)
        if order:
            await self.submit_order(order)

    async def initialize(self) -> None:
        """Initialize execution manager."""
        self.logger.logger.info("Execution manager initialized")

        if self.paper_trading_mode:
            self.logger.logger.info("Running in paper trading mode")

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

            if self.paper_trading_mode:
                # Schedule paper trading execution
                asyncio.create_task(self._paper_execute_order(order))
            else:
                # Submit to real exchange
                await self._submit_to_exchange(order)

            await self.event_bus.publish("order", OrderEvent(order))
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
            async with self._order_lock:
                if order_id not in self.pending_orders:
                    self.logger.logger.warning(
                        f"Order {order_id} not found for cancellation"
                    )
                    return False

                order = self.pending_orders[order_id]

                if self.paper_trading_mode:
                    # Paper trading cancellation
                    await self._paper_cancel_order(order)
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

                await self.event_bus.publish("order_updated", OrderEvent(order))
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

    async def _signal_to_order(self, signal: Any) -> Optional[Order]:
        """Convert a signal to an order."""
        # This is a placeholder. A more sophisticated implementation would
        # consider position sizing, risk, etc.
        if signal.signal_type == "buy":
            side = OrderSide.BUY
        elif signal.signal_type == "sell":
            side = OrderSide.SELL
        else:
            return None

        return Order(
            symbol=signal.symbol,
            quantity=Decimal("100"),  # Placeholder quantity
            side=side,
            type=OrderType.MARKET,
        )

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

        # If in paper trading mode, skip real exchange submission
        if self.paper_trading_mode:
            self.logger.logger.info(
                f"Paper trading mode: Skipping real exchange submission for order {order.id}"
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
                        order.status = OrderStatus.ACCEPTED
                        order.updated_at = datetime.now(timezone.utc)
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
        """Cancel order on real exchange (Alpaca implementation)."""

        if self.paper_trading_mode:
            self.logger.logger.info(
                f"Paper trading mode: Skipping real exchange cancellation for order {order.id}"
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
                            f"Order {order.id} cancelled successfully"
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

    async def _paper_execute_order(self, order: Order) -> None:
        """Simulate order execution for paper trading."""
        await asyncio.sleep(self.paper_trading_fill_delay)

        # Simulate fill or rejection
        if self._should_fill():
            await self._paper_fill_order(order)
        else:
            await self._paper_reject_order(order, "Market closed")

    async def _paper_fill_order(self, order: Order) -> None:
        """Simulate filling an order for paper trading."""
        async with self._order_lock:
            if order.id in self.pending_orders:
                market_price = self._get_mock_market_price(order.symbol)

                # Update order
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.filled_avg_price = market_price
                order.updated_at = datetime.now(timezone.utc)
                order.commission = self._calculate_commission(
                    order.quantity, market_price
                )

                # Move to completed
                self.completed_orders[order.id] = order
                del self.pending_orders[order.id]

                self.orders_filled += 1
                self.total_volume += order.quantity * market_price
                self.logger.log_order(order.dict(), "filled")

                # Create and publish trade
                trade = Trade(
                    id=str(uuid.uuid4()),
                    order_id=order.id,
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=market_price,
                    side=order.side,
                    commission=order.commission,
                    timestamp=order.updated_at,
                )
                await self.event_bus.publish("trade", OrderEvent(order=order))

    async def _paper_reject_order(self, order: Order, reason: str) -> None:
        """Simulate rejecting an order for paper trading."""
        async with self._order_lock:
            if order.id in self.pending_orders:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now(timezone.utc)
                self.completed_orders[order.id] = order
                del self.pending_orders[order.id]
                self.logger.log_order(order.dict(), "rejected", {"reason": reason})

    async def _paper_cancel_order(self, order: Order) -> None:
        """Simulate cancelling an order for paper trading."""
        self.logger.logger.info(
            f"Paper trading: Simulating cancellation for order {order.id}"
        )
        # No external state to change, just log and update status

    def _should_fill(self) -> bool:
        """Randomly decide if a paper order should be filled."""
        # Simple logic: 95% chance of fill
        import random

        return random.random() < 0.95

    def _get_mock_market_price(self, symbol: str) -> Decimal:
        """Get a mock market price for a symbol."""
        # In a real paper trading system, this would fetch the current
        # market price from a data provider.
        # For this simulation, we'll use a random price around a baseline.
        import random

        if symbol == "AAPL":
            return Decimal(str(random.uniform(150.0, 155.0)))
        elif symbol == "GOOGL":
            return Decimal(str(random.uniform(2800.0, 2850.0)))
        else:
            return Decimal(str(random.uniform(100.0, 105.0)))

    def _calculate_commission(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Calculate commission for a trade."""
        return self.shared_execution.calculate_commission(quantity, price)

    def get_stats(self) -> Dict[str, Any]:
        """Get execution manager statistics."""
        return {
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_cancelled": self.orders_cancelled,
            "total_volume": str(self.total_volume),
            "pending_orders": len(self.pending_orders),
            "paper_trading_mode": self.paper_trading_mode,
        }
