"""Strategy management and execution coordination."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core.config import Config
from ..core.events import EventBus, MarketDataEvent, SignalEvent
from ..core.exceptions import StrategyError
from ..core.logging import TradingLogger
from ..core.models import MarketData, Quote
from ..core.signal import StrategySignal
from .base import BaseStrategy
from .breakout import BreakoutStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum_crossover import MomentumCrossoverStrategy


class StrategyManager:
    """Manager for multiple trading strategies."""

    def __init__(self, config: Config, event_bus: EventBus):
        """Initialize strategy manager."""
        self.config = config
        self.event_bus = event_bus
        self.logger = TradingLogger("strategy_manager")

        # Load strategies from config
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_classes: Dict[str, type[BaseStrategy]] = {
            "momentum_crossover": MomentumCrossoverStrategy,
            "mean_reversion": MeanReversionStrategy,
            "breakout": BreakoutStrategy,
        }
        self.signal_count = 0
        self.last_signal_time: Optional[datetime] = None

    async def initialize(self) -> None:
        """Initialize strategies."""
        await self.add_strategy(
            self.config.strategy.default_strategy, self.config.strategy.parameters
        )

    async def on_market_data(self, event: MarketDataEvent) -> None:
        """Handle new market data."""
        self.logger.logger.debug(f"Market data received for {event.market_data.symbol}")
        for strategy in self.strategies.values():
            try:
                await strategy.on_bar(event.market_data.symbol, event.market_data)
            except Exception as e:
                self.logger.log_error(
                    e,
                    {
                        "context": "market_data_update",
                        "symbol": event.market_data.symbol,
                    },
                )
        signals = await self.generate_signals()
        for signal in signals:
            await self.event_bus.publish("signal", SignalEvent(signal))

    async def shutdown(self) -> None:
        """Shutdown strategy manager."""
        self.logger.logger.info("Shutting down strategy manager...")

        for strategy in self.strategies.values():
            await strategy.cleanup()

        self.strategies.clear()
        self.logger.logger.info("Strategy manager shutdown")

    async def add_strategy(
        self, strategy_name: str, parameters: Dict[str, Any]
    ) -> None:
        """Add a new strategy."""
        if strategy_name not in self.strategy_classes:
            raise StrategyError(f"Unknown strategy: {strategy_name}")

        if strategy_name in self.strategies:
            self.logger.logger.warning(
                f"Strategy {strategy_name} already exists, replacing..."
            )

        try:
            strategy_class = self.strategy_classes[strategy_name]
            strategy = strategy_class(strategy_name, parameters)
            await strategy.initialize()

            self.strategies[strategy_name] = strategy

            self.logger.logger.info(f"Added strategy: {strategy_name}")

        except Exception as e:
            raise StrategyError(f"Failed to add strategy {strategy_name}: {e}")

    async def remove_strategy(self, strategy_name: str) -> None:
        """Remove a strategy."""
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            await strategy.cleanup()
            del self.strategies[strategy_name]

            self.logger.logger.info(f"Removed strategy: {strategy_name}")

    async def generate_signals(self) -> List[StrategySignal]:
        """Generate signals from all active strategies."""
        all_signals = []

        for strategy_name, strategy in self.strategies.items():
            try:
                signals = await strategy.generate_signals()

                for signal in signals:
                    signal.strategy_name = strategy_name
                    all_signals.append(signal)
                    self.logger.logger.debug(f"Signal generated: {signal.dict()}")

                    self.logger.log_strategy_signal(signal.dict())

                self.signal_count += len(signals)
                if signals:
                    self.last_signal_time = datetime.now(timezone.utc)

            except Exception as e:
                self.logger.log_error(
                    e, {"context": "signal_generation", "strategy": strategy_name}
                )

        return all_signals

    async def update_market_data(self, symbol: str, data: MarketData) -> None:
        """Update strategies with new market data."""
        for strategy in self.strategies.values():
            try:
                await strategy.on_bar(symbol, data)
            except Exception as e:
                self.logger.log_error(
                    e, {"context": "market_data_update", "symbol": symbol}
                )

    async def update_quote(self, quote: Quote) -> None:
        """Update strategies with new quote data."""
        for strategy in self.strategies.values():
            try:
                await strategy.on_quote(quote)
            except Exception as e:
                self.logger.log_error(
                    e, {"context": "quote_update", "symbol": quote.symbol}
                )

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get status of all strategies."""
        status = {}

        for name, strategy in self.strategies.items():
            status[name] = {
                "enabled": strategy.enabled,
                "parameters": strategy.parameters,
                "symbols": list(strategy.symbols),
                "performance": strategy.get_performance_metrics(),
            }

        return status

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy manager statistics."""
        return {
            "active_strategies": len(self.strategies),
            "total_signals": self.signal_count,
            "last_signal_time": (
                self.last_signal_time.isoformat() if self.last_signal_time else None
            ),
            "strategies": list(self.strategies.keys()),
        }
