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
from .exceptions import RiskManagementError, TradingBotError
from .logging import TradingLogger, setup_logging
from .models import Order, Portfolio, Position, StrategySignal


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

        # Initialize managers
        self.market_data_manager = MarketDataManager(config)
        self.strategy_manager = StrategyManager(config)
        self.risk_manager = RiskManager(config)
        self.execution_manager = ExecutionManager(config)
        self.database_manager = DatabaseManager(config)

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

            # Start main trading loop
            await self._run_trading_loop()

        except Exception as e:
            self.logger.log_error(e, {"context": "bot_start"})
            raise TradingBotError(f"Failed to start trading bot: {e}")

    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        self.logger.logger.info("Stopping trading bot...")
        self.running = False

        # Cancel all active orders
        await self._cancel_all_orders()

        # Close all positions if configured
        if self.config.trading.close_positions_on_stop:
            await self._close_all_positions()

        # Shutdown managers
        await self._shutdown_managers()

        self.logger.logger.info("Trading bot stopped")

    async def _initialize_managers(self) -> None:
        """Initialize all manager components."""
        await self.database_manager.initialize()
        await self.market_data_manager.initialize()
        await self.execution_manager.initialize()
        await self.strategy_manager.initialize()
        await self.risk_manager.initialize()

    async def _shutdown_managers(self) -> None:
        """Shutdown all manager components."""
        await self.strategy_manager.shutdown()
        await self.risk_manager.shutdown()
        await self.execution_manager.shutdown()
        await self.market_data_manager.shutdown()
        await self.database_manager.shutdown()

    async def _load_portfolio(self) -> None:
        """Load portfolio state from database."""
        self.portfolio = await self.database_manager.get_portfolio()
        if self.portfolio:
            self.start_portfolio_value = self.portfolio.total_value
            self.positions = self.portfolio.positions

            # Load active orders
            active_orders = await self.database_manager.get_active_orders()
            self.active_orders = {
                order.id: order for order in active_orders if order.id
            }

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

                # Generate strategy signals
                signals = await self._generate_signals()

                # Process signals through risk management
                filtered_signals = await self._filter_signals(signals)

                # Execute approved signals
                await self._execute_signals(filtered_signals)

                # Update portfolio and positions
                await self._update_portfolio()

                # Check risk limits
                await self._check_risk_limits()

                # Save state to database
                await self._save_state()

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
        # Get latest market data for all tracked symbols
        symbols = list(self.positions.keys()) + [
            order.symbol for order in self.active_orders.values()
        ]

        if symbols:
            await self.market_data_manager.subscribe_symbols(symbols)

    async def _generate_signals(self) -> List[StrategySignal]:
        """Generate trading signals from all active strategies."""
        return await self.strategy_manager.generate_signals()

    async def _filter_signals(
        self, signals: List[StrategySignal]
    ) -> List[StrategySignal]:
        """Filter signals through risk management."""
        filtered_signals = []

        for signal in signals:
            try:
                # Check if signal passes risk management
                is_approved = await self.risk_manager.evaluate_signal(
                    signal, self.portfolio
                )

                if is_approved:
                    filtered_signals.append(signal)
                else:
                    self.logger.log_risk_event(
                        "signal_rejected",
                        {
                            "symbol": signal.symbol,
                            "signal_type": signal.signal_type,
                            "strategy": signal.strategy_name,
                        },
                    )
            except Exception as e:
                self.logger.log_error(
                    e, {"context": "signal_filtering", "signal": signal.dict()}
                )

        return filtered_signals

    async def _execute_signals(self, signals: List[StrategySignal]) -> None:
        """Execute approved trading signals."""
        for signal in signals:
            try:
                # Convert signal to order
                order = await self._signal_to_order(signal)

                if order:
                    # Submit order for execution
                    executed_order = await self.execution_manager.submit_order(order)

                    if executed_order and executed_order.id:
                        self.active_orders[executed_order.id] = executed_order
                        self.logger.log_order(executed_order.dict(), "submitted")

                        # Save order to database
                        await self.database_manager.save_order(executed_order)

            except Exception as e:
                self.logger.log_error(
                    e, {"context": "signal_execution", "signal": signal.dict()}
                )

    async def _signal_to_order(self, signal: StrategySignal) -> Optional[Order]:
        """Convert a strategy signal to an order."""
        # This is a simplified implementation
        # In practice, you'd have more sophisticated order sizing logic

        if signal.signal_type == "hold":
            return None

        # Calculate position size based on risk management
        position_size = await self.risk_manager.calculate_position_size(
            signal.symbol, signal.price or Decimal("0"), self.portfolio
        )

        if position_size <= 0:
            return None

        from .models import OrderSide, OrderType

        return Order(
            symbol=signal.symbol,
            side=OrderSide.BUY if signal.signal_type == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=position_size,
            strategy_id=signal.strategy_name,
            metadata=signal.metadata,
        )

    async def _update_portfolio(self) -> None:
        """Update portfolio with latest market data and positions."""
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
