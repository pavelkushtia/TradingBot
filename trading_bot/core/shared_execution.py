"""Shared execution logic for both real trading and backtesting."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from uuid import uuid4

from .models import Order, OrderSide, OrderType, StrategySignal, Trade


class SharedExecutionLogic:
    """Shared logic for order execution between real trading and backtesting."""

    def __init__(
        self,
        commission_per_share: Decimal = Decimal("0.005"),
        min_commission: Decimal = Decimal("1.00"),
    ):
        """Initialize shared execution logic."""
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

    def signal_to_order(
        self, signal: StrategySignal, position_size: Decimal
    ) -> Optional[Order]:
        """Convert a strategy signal to an order (shared logic)."""
        if signal.signal_type == "hold" or position_size <= 0:
            return None

        return Order(
            symbol=signal.symbol,
            side=OrderSide.BUY if signal.signal_type == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=position_size,
            strategy_id=signal.strategy_name,
            metadata=signal.metadata,
        )

    def calculate_commission(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Calculate commission for a trade (shared logic)."""
        commission = max(quantity * self.commission_per_share, self.min_commission)
        return commission

    def simulate_execution_price(
        self, market_price: Decimal, signal_type: str, quantity: Decimal
    ) -> Decimal:
        """Simulate realistic execution price with slippage (shared logic)."""
        # Base slippage factors
        base_slippage = Decimal("0.001")  # 0.1% base slippage

        # Volume-based slippage (larger orders have more slippage)
        volume_factor = min(
            quantity / Decimal("1000"), Decimal("0.005")
        )  # Max 0.5% additional

        # Market impact simulation
        total_slippage = base_slippage + volume_factor

        if signal_type == "buy":
            # Buy orders: pay slightly more (adverse slippage)
            return market_price * (1 + total_slippage)
        else:
            # Sell orders: receive slightly less (adverse slippage)
            return market_price * (1 - total_slippage)

    def create_trade_from_order(self, order: Order, fill_price: Decimal) -> Trade:
        """Create a trade from a filled order (shared logic)."""
        return Trade(
            id=str(uuid4()),
            order_id=order.id or str(uuid4()),
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=datetime.now(timezone.utc),
            commission=self.calculate_commission(order.quantity, fill_price),
            strategy_id=order.strategy_id,
        )
