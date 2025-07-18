"""Advanced order types for sophisticated trading strategies."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.logging import TradingLogger
from ..core.models import Order, OrderSide, OrderStatus, OrderType


class AdvancedOrderType(Enum):
    """Advanced order types."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"
    BRACKET = "bracket"


@dataclass
class OrderTriggerResult:
    """Result from order trigger check."""

    triggered: bool
    trigger_price: Optional[Decimal] = None
    message: str = ""


@dataclass
class StopLossConfig:
    """Configuration for stop-loss orders."""

    symbol: str
    side: str
    quantity: Decimal
    base_price: Decimal
    stop_percentage: Decimal
    stop_price: Optional[Decimal] = None
    trailing: bool = False
    trail_amount: Optional[Decimal] = None

    def __post_init__(self):
        if self.stop_price is None:
            # Calculate stop price from base price and percentage
            if self.side.lower() == "sell":
                self.stop_price = self.base_price * (1 - self.stop_percentage)
            else:
                self.stop_price = self.base_price * (1 + self.stop_percentage)


@dataclass
class TakeProfitConfig:
    """Configuration for take-profit orders."""

    symbol: str
    side: str
    quantity: Decimal
    base_price: Decimal
    target_percentage: Decimal
    target_price: Optional[Decimal] = None

    def __post_init__(self):
        if self.target_price is None:
            # Calculate target price from base price and percentage
            if self.side.lower() == "sell":
                self.target_price = self.base_price * (1 + self.target_percentage)
            else:
                self.target_price = self.base_price * (1 - self.target_percentage)


class BaseAdvancedOrder:
    """Base class for advanced order types."""

    def __init__(self, symbol: str, side: OrderSide, quantity: Decimal):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.created_at = datetime.now(timezone.utc)
        self.status = OrderStatus.NEW
        self.order_type = AdvancedOrderType.STOP_LOSS  # Override in subclasses

    def should_trigger(self, current_price: Decimal) -> bool:
        """Check if order should trigger at current price."""
        raise NotImplementedError

    def get_trigger_result(self, current_price: Decimal) -> OrderTriggerResult:
        """Get detailed trigger result."""
        triggered = self.should_trigger(current_price)
        return OrderTriggerResult(
            triggered=triggered,
            trigger_price=current_price if triggered else None,
            message=f"Order {'triggered' if triggered else 'not triggered'} at ${current_price}",
        )


class StopLossOrder(BaseAdvancedOrder):
    """Stop-loss order that triggers when price falls below stop price."""

    def __init__(
        self,
        config_or_symbol,
        side=None,
        quantity=None,
        stop_price=None,
        current_price=None,
    ):
        # Handle both config object and individual parameters
        if isinstance(config_or_symbol, StopLossConfig):
            config = config_or_symbol
            symbol = config.symbol
            side_obj = (
                OrderSide.SELL if config.side.lower() == "sell" else OrderSide.BUY
            )
            quantity_val = config.quantity
            stop_price_val = config.stop_price
            current_price_val = config.base_price
        else:
            # Individual parameters
            symbol = config_or_symbol
            side_obj = side
            quantity_val = quantity
            stop_price_val = stop_price
            current_price_val = current_price

        # Validate required parameters
        if (
            not symbol
            or not side_obj
            or not quantity_val
            or not stop_price_val
            or not current_price_val
        ):
            raise ValueError("Missing required parameters for StopLossOrder")

        # Type assertions after validation
        assert side_obj is not None
        assert quantity_val is not None
        assert stop_price_val is not None
        assert current_price_val is not None

        super().__init__(symbol, side_obj, quantity_val)
        self.stop_price = stop_price_val
        self.current_price = current_price_val
        self.order_type = AdvancedOrderType.STOP_LOSS

        # Validate stop price logic
        if side_obj == OrderSide.SELL and stop_price_val >= current_price_val:
            raise ValueError("Sell stop-loss must be below current price")
        elif side_obj == OrderSide.BUY and stop_price_val <= current_price_val:
            raise ValueError("Buy stop-loss must be above current price")

    def should_trigger(self, current_price: Decimal) -> bool:
        """Check if stop-loss should trigger."""
        if self.side == OrderSide.SELL:
            # Sell stop-loss triggers when price falls below stop price
            return current_price <= self.stop_price
        else:
            # Buy stop-loss triggers when price rises above stop price
            return current_price >= self.stop_price

    def update_current_price(self, new_price: Decimal) -> None:
        """Update current price tracking."""
        self.current_price = new_price


class TakeProfitOrder(BaseAdvancedOrder):
    """Take-profit order that triggers when price reaches target."""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        target_price: Decimal,
        current_price: Decimal,
    ):
        super().__init__(symbol, side, quantity)
        self.target_price = target_price
        self.current_price = current_price
        self.order_type = AdvancedOrderType.TAKE_PROFIT

        # Validate target price logic
        if side == OrderSide.SELL and target_price <= current_price:
            raise ValueError("Sell take-profit must be above current price")
        elif side == OrderSide.BUY and target_price >= current_price:
            raise ValueError("Buy take-profit must be below current price")

    def should_trigger(self, current_price: Decimal) -> bool:
        """Check if take-profit should trigger."""
        if self.side == OrderSide.SELL:
            # Sell take-profit triggers when price rises to target
            return current_price >= self.target_price
        else:
            # Buy take-profit triggers when price falls to target
            return current_price <= self.target_price

    def update_current_price(self, new_price: Decimal) -> None:
        """Update current price tracking."""
        self.current_price = new_price


class TrailingStopOrder(BaseAdvancedOrder):
    """Trailing stop order that adjusts stop price as market moves favorably."""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        trail_amount: Decimal,
        current_price: Decimal,
        trail_percent: Optional[Decimal] = None,
    ):
        super().__init__(symbol, side, quantity)
        self.trail_amount = trail_amount
        self.trail_percent = trail_percent
        self.current_price = current_price
        self.order_type = AdvancedOrderType.TRAILING_STOP

        # Initialize stop price and tracking
        if side == OrderSide.SELL:
            self.stop_price = current_price - trail_amount
            self.highest_price = current_price
        else:
            self.stop_price = current_price + trail_amount
            self.lowest_price = current_price

    def should_trigger(self, current_price: Decimal) -> bool:
        """Check if trailing stop should trigger."""
        if self.side == OrderSide.SELL:
            return current_price <= self.stop_price
        else:
            return current_price >= self.stop_price

    def update_price(self, new_price: Decimal) -> bool:
        """Update trailing stop based on new price. Returns True if stop moved."""
        self.current_price = new_price
        stop_moved = False

        if self.side == OrderSide.SELL:
            # For sell orders, trail up as price increases
            if new_price > self.highest_price:
                self.highest_price = new_price
                new_stop = new_price - self.trail_amount

                # Only move stop up, never down
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                    stop_moved = True
        else:
            # For buy orders, trail down as price decreases
            if new_price < self.lowest_price:
                self.lowest_price = new_price
                new_stop = new_price + self.trail_amount

                # Only move stop down, never up
                if new_stop < self.stop_price:
                    self.stop_price = new_stop
                    stop_moved = True

        return stop_moved


class OCOOrder:
    """One-Cancels-Other order - when one leg executes, the other is cancelled."""

    def __init__(
        self, primary_order: BaseAdvancedOrder, secondary_order: BaseAdvancedOrder
    ):
        self.id = str(uuid.uuid4())
        self.primary_order = primary_order
        self.secondary_order = secondary_order
        self.created_at = datetime.now(timezone.utc)
        self.status = OrderStatus.NEW
        self.triggered_order: Optional[BaseAdvancedOrder] = None
        self.cancelled_order: Optional[BaseAdvancedOrder] = None

        # Validate orders are for same symbol
        if primary_order.symbol != secondary_order.symbol:
            raise ValueError("OCO orders must be for the same symbol")

    def check_trigger(
        self, current_price: Decimal
    ) -> Tuple[Optional[BaseAdvancedOrder], Optional[BaseAdvancedOrder]]:
        """Check if either order should trigger. Returns (triggered, cancelled)."""
        if self.status != OrderStatus.NEW:
            return self.triggered_order, self.cancelled_order

        # Check primary order first
        if self.primary_order.should_trigger(current_price):
            self.triggered_order = self.primary_order
            self.cancelled_order = self.secondary_order
            self.status = OrderStatus.FILLED
            self.primary_order.status = OrderStatus.FILLED
            self.secondary_order.status = OrderStatus.CANCELED
            return self.triggered_order, self.cancelled_order

        # Check secondary order
        if self.secondary_order.should_trigger(current_price):
            self.triggered_order = self.secondary_order
            self.cancelled_order = self.primary_order
            self.status = OrderStatus.FILLED
            self.secondary_order.status = OrderStatus.FILLED
            self.primary_order.status = OrderStatus.CANCELED
            return self.triggered_order, self.cancelled_order

        return None, None

    @property
    def symbol(self) -> str:
        """Get symbol for the OCO order."""
        return self.primary_order.symbol


class BracketOrder:
    """Bracket order combines entry order with OCO exit orders (stop-loss + take-profit)."""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal,
        take_profit_price: Decimal,
    ):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.created_at = datetime.now(timezone.utc)
        self.status = OrderStatus.NEW

        # Create component orders
        self.entry_order = self._create_entry_order()
        self.oco_order = self._create_oco_exit_orders()

        # State tracking
        self.entry_filled = False

    def _create_entry_order(self) -> Order:
        """Create the entry order."""
        return Order(
            id=f"{self.id}_entry",
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type=OrderType.LIMIT,
            price=self.entry_price,
            created_at=self.created_at,
        )

    def _create_oco_exit_orders(self) -> OCOOrder:
        """Create OCO exit orders (stop-loss + take-profit)."""
        # Exit side is opposite of entry side
        exit_side = OrderSide.SELL if self.side == OrderSide.BUY else OrderSide.BUY

        # Create stop-loss and take-profit orders
        stop_loss = StopLossOrder(
            symbol=self.symbol,
            side=exit_side,
            quantity=self.quantity,
            stop_price=self.stop_loss_price,
            current_price=self.entry_price,
        )

        take_profit = TakeProfitOrder(
            symbol=self.symbol,
            side=exit_side,
            quantity=self.quantity,
            target_price=self.take_profit_price,
            current_price=self.entry_price,
        )

        return OCOOrder(primary_order=take_profit, secondary_order=stop_loss)

    def get_execution_sequence(self) -> List[Union[Order, OCOOrder]]:
        """Get the sequence of orders to execute."""
        return [self.entry_order, self.oco_order]

    def mark_entry_filled(self) -> None:
        """Mark the entry order as filled and activate exit orders."""
        self.entry_filled = True
        self.entry_order.status = OrderStatus.FILLED


class AdvancedOrderManager:
    """Manager for advanced order types with real-time monitoring."""

    def __init__(self):
        self.logger = TradingLogger("advanced_order_manager")

        # Order storage
        self.orders: Dict[str, BaseAdvancedOrder] = {}
        self.oco_orders: Dict[str, OCOOrder] = {}
        self.bracket_orders: Dict[str, BracketOrder] = {}

        # Monitoring
        self.current_prices: Dict[str, Decimal] = {}

    @property
    def active_orders(self) -> Dict[str, BaseAdvancedOrder]:
        """Get all active (pending) orders as a dictionary."""
        return {
            order_id: order
            for order_id, order in self.orders.items()
            if order.status == OrderStatus.NEW
        }

    def update_orders(
        self, symbol: str, price: Decimal, metadata: Dict[str, Any] = None
    ) -> List[BaseAdvancedOrder]:
        """Update orders for a specific symbol and price."""
        prices = {symbol: price}
        return self.update_prices(prices)

    def add_order(self, order: BaseAdvancedOrder) -> str:
        """Add any type of advanced order."""
        self.orders[order.id] = order
        self.logger.logger.info(
            f"Added {type(order).__name__} order {order.id} for {order.symbol}"
        )
        return order.id

    def get_order_status(self) -> Dict[str, Any]:
        """Get status of all orders."""
        active_count = sum(
            1 for order in self.orders.values() if order.status == OrderStatus.NEW
        )
        total_count = len(self.orders) + len(self.oco_orders) + len(self.bracket_orders)

        # Count orders by type
        orders_by_type = {}
        for order in self.orders.values():
            order_type = type(order).__name__
            orders_by_type[order_type] = orders_by_type.get(order_type, 0) + 1

        return {
            "total_orders": total_count,
            "orders_by_type": orders_by_type,
            "active_orders": {
                order_id: order
                for order_id, order in self.orders.items()
                if order.status == OrderStatus.NEW
            },
            "pending_count": active_count,
        }

    def add_stop_loss(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        stop_price: Decimal,
        current_price: Decimal,
    ) -> str:
        """Add a stop-loss order."""
        order = StopLossOrder(symbol, side, quantity, stop_price, current_price)
        self.orders[order.id] = order
        self.logger.logger.info(f"Added stop-loss order {order.id} for {symbol}")
        return order.id

    def add_take_profit(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        target_price: Decimal,
        current_price: Decimal,
    ) -> str:
        """Add a take-profit order."""
        order = TakeProfitOrder(symbol, side, quantity, target_price, current_price)
        self.orders[order.id] = order
        self.logger.logger.info(f"Added take-profit order {order.id} for {symbol}")
        return order.id

    def add_trailing_stop(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        trail_amount: Decimal,
        current_price: Decimal,
    ) -> str:
        """Add a trailing stop order."""
        order = TrailingStopOrder(symbol, side, quantity, trail_amount, current_price)
        self.orders[order.id] = order
        self.logger.logger.info(f"Added trailing stop order {order.id} for {symbol}")
        return order.id

    def add_oco_order(
        self, primary_order: BaseAdvancedOrder, secondary_order: BaseAdvancedOrder
    ) -> str:
        """Add an OCO order."""
        oco = OCOOrder(primary_order, secondary_order)
        self.oco_orders[oco.id] = oco
        self.logger.logger.info(f"Added OCO order {oco.id} for {oco.symbol}")
        return oco.id

    def add_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal,
        take_profit_price: Decimal,
    ) -> str:
        """Add a bracket order."""
        bracket = BracketOrder(
            symbol, side, quantity, entry_price, stop_loss_price, take_profit_price
        )
        self.bracket_orders[bracket.id] = bracket
        self.logger.logger.info(f"Added bracket order {bracket.id} for {symbol}")
        return bracket.id

    def update_prices(self, prices: Dict[str, Decimal]) -> List[BaseAdvancedOrder]:
        """Update current prices and check for triggered orders."""
        self.current_prices.update(prices)
        triggered_orders = []

        # Check individual orders
        for order in list(self.orders.values()):
            if order.status != OrderStatus.NEW:
                continue

            symbol_price = prices.get(order.symbol)
            if symbol_price is None:
                continue

            # Update trailing stops
            if isinstance(order, TrailingStopOrder):
                order.update_price(symbol_price)

            # Check for triggers
            if order.should_trigger(symbol_price):
                order.status = OrderStatus.FILLED
                triggered_orders.append(order)
                self.logger.logger.info(
                    f"Order {order.id} triggered at ${symbol_price}"
                )

        # Check OCO orders
        for oco in list(self.oco_orders.values()):
            if oco.status != OrderStatus.NEW:
                continue

            symbol_price = prices.get(oco.symbol)
            if symbol_price is None:
                continue

            triggered, cancelled = oco.check_trigger(symbol_price)
            if triggered:
                triggered_orders.append(triggered)
                self.logger.logger.info(f"OCO order {oco.id} triggered: {triggered.id}")

        return triggered_orders

    def get_active_orders(self) -> List[BaseAdvancedOrder]:
        """Get all active (pending) orders."""
        active = []

        # Individual orders
        for order in self.orders.values():
            if order.status == OrderStatus.NEW:
                active.append(order)

        # OCO orders
        for oco in self.oco_orders.values():
            if oco.status == OrderStatus.NEW:
                active.extend([oco.primary_order, oco.secondary_order])

        return active

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        # Check individual orders
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELED
            self.logger.logger.info(f"Cancelled order {order_id}")
            return True

        # Check OCO orders
        if order_id in self.oco_orders:
            oco = self.oco_orders[order_id]
            oco.status = OrderStatus.CANCELED
            oco.primary_order.status = OrderStatus.CANCELED
            oco.secondary_order.status = OrderStatus.CANCELED
            self.logger.logger.info(f"Cancelled OCO order {order_id}")
            return True

        return False

    def get_order_details(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status information for a specific order."""
        # Check individual orders
        if order_id in self.orders:
            order = self.orders[order_id]
            return {
                "id": order.id,
                "symbol": order.symbol,
                "type": order.order_type.value,
                "status": order.status.value,
                "side": order.side.value,
                "quantity": float(order.quantity),
                "created_at": order.created_at.isoformat(),
            }

        # Check OCO orders
        if order_id in self.oco_orders:
            oco = self.oco_orders[order_id]
            return {
                "id": oco.id,
                "type": "oco",
                "status": oco.status.value,
                "symbol": oco.symbol,
                "primary_order": oco.primary_order.id,
                "secondary_order": oco.secondary_order.id,
                "created_at": oco.created_at.isoformat(),
            }

        return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the order manager."""
        total_orders = (
            len(self.orders) + len(self.oco_orders) + len(self.bracket_orders)
        )

        individual_pending = sum(
            1 for o in self.orders.values() if o.status == OrderStatus.NEW
        )
        oco_pending = sum(
            1 for o in self.oco_orders.values() if o.status == OrderStatus.NEW
        )
        bracket_pending = sum(
            1 for o in self.bracket_orders.values() if o.status == OrderStatus.NEW
        )

        return {
            "total_orders": total_orders,
            "individual_orders": len(self.orders),
            "oco_orders": len(self.oco_orders),
            "bracket_orders": len(self.bracket_orders),
            "pending_orders": individual_pending + oco_pending + bracket_pending,
            "monitored_symbols": len(self.current_prices),
            "order_types_breakdown": {
                "stop_loss": sum(
                    1
                    for o in self.orders.values()
                    if o.order_type == AdvancedOrderType.STOP_LOSS
                ),
                "take_profit": sum(
                    1
                    for o in self.orders.values()
                    if o.order_type == AdvancedOrderType.TAKE_PROFIT
                ),
                "trailing_stop": sum(
                    1
                    for o in self.orders.values()
                    if o.order_type == AdvancedOrderType.TRAILING_STOP
                ),
            },
        }
