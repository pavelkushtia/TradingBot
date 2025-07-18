"""Main trading bot implementation."""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..database.manager import DatabaseManager
from ..execution.manager import ExecutionManager
from ..market_data.manager import MarketDataManager
from ..risk.manager import RiskManager
from ..strategy.manager import StrategyManager
from .config import Config
from .events import EventBus
from .exceptions import RiskManagementError, TradingBotError
from .logging import TradingLogger, setup_logging
from .models import Order, Portfolio, Position
from .shared_execution import SharedExecutionLogic
from .signal import StrategySignal


class TradingBot:
    """High-performance trading bot with async architecture."""

    def __init__(self, config: Config):
        """Initialize the trading bot."""
        self.config = config
        self.running = False
        self.start_time: Optional[datetime] = None

        # Setup logging
        setup_logging(config.logging)
        self.logger = TradingLogger("trading_bot")

        # Initialize event bus
        self.event_bus = EventBus()

        # Initialize managers
        self.market_data_manager = MarketDataManager(self.config, self.event_bus)
        self.strategy_manager = StrategyManager(self.config, self.event_bus)
        self.risk_manager = RiskManager(self.config, self.event_bus)
        self.execution_manager = ExecutionManager(self.config, self.event_bus)
        self.database_manager = DatabaseManager(self.config)

        # Initialize shared execution logic (same as backtesting)
        self.shared_execution = SharedExecutionLogic()

        # State
        self.portfolio: Optional[Portfolio] = None
        self.active_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}

        # Performance tracking
        self.start_portfolio_value: Optional[Decimal] = None
        self.daily_pnl: Decimal = Decimal("0")

    async def start(self) -> None:
        """Start the trading bot."""
        try:
            self.logger.logger.info("Starting trading bot...")
            self.start_time = datetime.now(timezone.utc)
            self.running = True

            # Initialize all managers
            await self._initialize_managers()

            # Load initial portfolio state
            await self._load_portfolio()

            # Setup event listeners
            self._setup_event_listeners()

            # Start main trading loop
            asyncio.create_task(self._run_trading_loop())

        except Exception as e:
            self.logger.log_error(e, {"context": "bot_start"})
            raise TradingBotError(f"Failed to start trading bot: {e}")

    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        self.logger.logger.info("Stopping trading bot...")
        self.running = False

        # Cancel all active orders
        await self._cancel_all_orders()

        # Close all positions if configured (optional)
        # await self._close_all_positions()

        # Shutdown managers
        await self._shutdown_managers()

        self.logger.logger.info("Trading bot stopped")

    def _setup_event_listeners(self) -> None:
        """Setup event listeners."""
        self.event_bus.subscribe("market_data", self.strategy_manager.on_market_data)
        self.event_bus.subscribe("signal", self.risk_manager.on_signal)
        self.event_bus.subscribe("signal_approved", self.execution_manager.on_signal)
        self.event_bus.subscribe("order_filled", self.database_manager.on_order_update)
        self.event_bus.subscribe("order_updated", self.database_manager.on_order_update)

    async def _initialize_managers(self) -> None:
        """Initialize all manager components."""
        await self.database_manager.initialize()
        await self.market_data_manager.initialize()
        await self.execution_manager.initialize()
        await self.strategy_manager.initialize()
        await self.risk_manager.initialize()

        # Initialize strategies with configured symbols
        await self._initialize_strategy_symbols()

    async def _initialize_strategy_symbols(self) -> None:
        """Initialize strategies with configured symbols."""
        configured_symbols = self.config.strategy.symbols
        configured_symbols = [s.strip() for s in configured_symbols if s.strip()]

        # Add configured symbols to all strategies
        for strategy in self.strategy_manager.strategies.values():
            for symbol in configured_symbols:
                strategy.symbols.add(symbol)

        self.logger.logger.info(
            f"Initialized strategies with symbols: {configured_symbols}"
        )

    async def _shutdown_managers(self) -> None:
        """Shutdown all manager components."""
        await self.strategy_manager.shutdown()
        await self.risk_manager.shutdown()
        await self.execution_manager.shutdown()
        await self.market_data_manager.shutdown()
        await self.database_manager.shutdown()

    async def _load_portfolio(self) -> None:
        """Load portfolio state from database, or create a new one."""
        self.portfolio = await self.database_manager.get_portfolio()
        if not self.portfolio:
            initial_capital = Decimal(str(self.config.trading.portfolio_value))
            self.portfolio = Portfolio(
                initial_capital=initial_capital,
                start_date=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                total_value=initial_capital,
                cash=initial_capital,
                buying_power=initial_capital,
            )
            self.logger.logger.info(
                "No existing portfolio found. Created a new one.",
                initial_capital=str(initial_capital),
            )

        self.start_portfolio_value = self.portfolio.total_value
        self.positions = self.portfolio.positions

        # Load active orders
        active_orders = await self.database_manager.get_active_orders()
        self.active_orders = {order.id: order for order in active_orders if order.id}

        self.logger.logger.info(
            "Portfolio loaded",
            total_value=str(self.portfolio.total_value),
            positions=len(self.positions),
            active_orders=len(self.active_orders),
        )

    async def _run_trading_loop(self) -> None:
        """Main trading loop."""
        self.logger.logger.info("Starting trading loop")

        while self.running:
            try:
                # Process market data updates
                await self._process_market_data()

                # Event-driven logic will handle the rest

                # Sleep briefly to avoid excessive CPU usage
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.log_error(e, {"context": "trading_loop"})
                # Continue running unless it's a critical error
                if isinstance(e, RiskManagementError):
                    self.logger.log_risk_event("critical_error", {"error": str(e)})
                    await self.stop()
                await asyncio.sleep(1.0)

    async def _process_market_data(self) -> None:
        """Process incoming market data updates."""
        # Get symbols from configuration and current positions/orders
        configured_symbols = self.config.strategy.symbols
        configured_symbols = [s.strip() for s in configured_symbols if s.strip()]

        # Combine configured symbols with current positions and orders
        symbols = set(
            configured_symbols
            + list(self.positions.keys())
            + [order.symbol for order in self.active_orders.values()]
        )

        if symbols:
            await self.market_data_manager.subscribe_symbols(list(symbols))

    async def _generate_signals(self) -> List[StrategySignal]:
        """Generate trading signals from all active strategies."""
        # This will be handled by the event bus
        return []

    async def _filter_signals(
        self, signals: List[StrategySignal]
    ) -> List[StrategySignal]:
        """Filter signals through risk management."""
        # This will be handled by the event bus
        return []

    async def _execute_signals(self, signals: List[StrategySignal]) -> None:
        """Execute approved trading signals."""
        # This will be handled by the event bus
        pass

    async def _signal_to_order(self, signal: StrategySignal) -> Optional[Order]:
        """Convert a strategy signal to an order (shared logic with backtesting)."""
        # Calculate position size based on risk management
        position_size = await self.risk_manager.calculate_position_size(
            signal.symbol, signal.price or Decimal("0"), self.portfolio
        )

        # Use shared execution logic (same as backtesting)
        return self.shared_execution.signal_to_order(signal, position_size)

    async def _update_portfolio(self) -> None:
        """Update portfolio state."""
        if not self.portfolio:
            return

        # Update position values with latest market prices
        for symbol, position in self.positions.items():
            latest_price = await self.market_data_manager.get_latest_price(symbol)
            if latest_price:
                position.market_value = position.quantity * latest_price
                position.unrealized_pnl = position.market_value - (
                    position.quantity * position.average_price
                )

        # Update portfolio totals
        self.portfolio.positions = self.positions
        self.portfolio.total_value = (
            self.portfolio.cash + self.portfolio.total_market_value
        )
        self.portfolio.total_pnl = self.portfolio.total_unrealized_pnl
        self.portfolio.updated_at = datetime.now(timezone.utc)

    async def _check_risk_limits(self) -> None:
        """Check if any risk limits are breached."""
        if not self.portfolio or not self.start_portfolio_value:
            return

        # Check daily loss limit
        daily_pnl_pct = (
            self.portfolio.total_value - self.start_portfolio_value
        ) / self.start_portfolio_value

        if daily_pnl_pct <= -self.config.risk.max_daily_loss:
            self.logger.log_risk_event(
                "daily_loss_limit_exceeded",
                {
                    "daily_pnl_pct": str(daily_pnl_pct),
                    "limit": str(self.config.risk.max_daily_loss),
                },
            )
            # Emergency stop
            await self.stop()

    async def _save_state(self) -> None:
        """Save current state to database."""
        if self.portfolio:
            await self.database_manager.save_portfolio(self.portfolio)

        # Save active orders
        for order in self.active_orders.values():
            await self.database_manager.save_order(order)

    async def _cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        for order_id, order in self.active_orders.items():
            try:
                await self.execution_manager.cancel_order(order_id)
                self.logger.log_order(order.dict(), "cancelled")
            except Exception as e:
                self.logger.log_error(
                    e, {"context": "cancel_order", "order_id": order_id}
                )

        self.active_orders.clear()

    async def _close_all_positions(self) -> None:
        """Close all open positions."""
        for symbol, position in self.positions.items():
            try:
                # Create market order to close position
                from .models import OrderSide, OrderType

                close_order = Order(
                    symbol=symbol,
                    side=(
                        OrderSide.SELL
                        if position.side.value == "long"
                        else OrderSide.BUY
                    ),
                    type=OrderType.MARKET,
                    quantity=abs(position.quantity),
                    strategy_id="close_all",
                )

                await self.execution_manager.submit_order(close_order)
                self.logger.log_position(position.dict(), "closed")

            except Exception as e:
                self.logger.log_error(
                    e, {"context": "close_position", "symbol": symbol}
                )

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return {
            "running": self.running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "portfolio_value": (
                str(self.portfolio.total_value) if self.portfolio else "0"
            ),
            "active_orders": len(self.active_orders),
            "open_positions": len(self.positions),
            "daily_pnl": str(self.daily_pnl),
        }
