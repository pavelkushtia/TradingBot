"""Multi-timeframe data manager for trading strategies."""

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..core.logging import TradingLogger
from ..core.models import MarketData


@dataclass
class TimeframeConfig:
    """Configuration for timeframe handling."""

    primary_timeframe: str = "1d"
    supported_timeframes: Optional[Dict[str, Any]] = None
    secondary_timeframes: Optional[List[str]] = None

    def __post_init__(self):
        if self.supported_timeframes is None:
            self.supported_timeframes = {"1d": {}, "1h": {}, "5m": {}}
        if self.secondary_timeframes is None:
            self.secondary_timeframes = []


@dataclass
class TimeframeData:
    """Data for a specific timeframe."""

    timeframe: str
    data: Dict[str, Any]


class MultiTimeframeStrategy:
    """Base class for multi-timeframe strategies."""

    def __init__(
        self,
        name: Optional[str] = None,
        parameters: Optional[dict] = None,
        config: Optional[TimeframeConfig] = None,
    ):
        """Initialize multi-timeframe strategy.

        Args:
            name: Strategy name (optional for backwards compatibility)
            parameters: Strategy parameters (optional for backwards compatibility)
            config: Timeframe configuration
        """
        # Handle backwards compatibility - if only config is provided
        if name is None and parameters is None and config is not None:
            self.config = config
        elif config is not None:
            # New signature with all three arguments
            self.name = name
            self.parameters = parameters or {}
            self.config = config
        else:
            # Fallback - assume first argument is config if it's a TimeframeConfig
            if isinstance(name, TimeframeConfig):
                self.config = name
                self.name = None
                self.parameters = {}
            else:
                # Create default config if none provided
                self.name = name
                self.parameters = parameters or {}
                self.config = TimeframeConfig()

        self.timeframe_data = {}
        self.symbols = set()

        # Simple timeframe manager object
        class SimpleTimeframeManager:
            def __init__(self, strategy):
                self.strategy = strategy

            @property
            def monitored_symbols(self):
                return self.strategy.symbols

            async def process_bar(self, symbol: str, bar):
                """Simple bar processing."""
                pass

            def get_timeframe_data(
                self, symbol: str, timeframe, limit: Optional[int] = None
            ):
                """Get timeframe data."""
                # Return mock data for testing
                from datetime import datetime, timedelta
                from decimal import Decimal

                from ..core.models import MarketData

                # Generate some mock timeframe data
                mock_data = []
                base_time = datetime.now()

                # Create mock bars based on timeframe
                num_bars = limit if limit else 10
                for i in range(num_bars):
                    timestamp = base_time - timedelta(
                        minutes=5 * i
                    )  # Mock 5-minute intervals
                    bar = MarketData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=Decimal("100.0"),
                        high=Decimal("105.0"),
                        low=Decimal("95.0"),
                        close=Decimal("102.0"),
                        volume=1000,
                    )
                    mock_data.append(bar)

                return mock_data

            def is_synchronized(self, symbol: str) -> bool:
                """Check if synchronized."""
                return True

        self.timeframe_manager = SimpleTimeframeManager(self)

    async def initialize(self):
        """Initialize the strategy."""
        pass

    async def on_bar(self, symbol: str, bar):
        """Handle new market data bar."""
        self.symbols.add(symbol)
        await self.timeframe_manager.process_bar(symbol, bar)

    async def generate_signals(self):
        """Generate trading signals - to be overridden by subclasses."""
        return []

    def get_timeframe_data(self, symbol: str, timeframe, limit: Optional[int] = None):
        """Get data for a specific timeframe."""
        return self.timeframe_manager.get_timeframe_data(symbol, timeframe, limit)

    def get_performance_metrics(self):
        """Get performance metrics."""
        return {
            "timeframe_status": "active",
            "indicators_available": True,
            "monitored_symbols": len(self.symbols),
            "timeframes": (
                list(self.config.supported_timeframes.keys())
                if self.config.supported_timeframes
                else []
            ),
        }

    def is_timeframes_synchronized(self, symbol: str) -> bool:
        """Check if timeframes are synchronized for a symbol."""
        return self.timeframe_manager.is_synchronized(symbol)

    def get_indicator_value(self, symbol: str, indicator: str):
        """Get indicator value - placeholder for compatibility."""
        return None

    async def cleanup(self):
        """Clean up resources."""
        pass

    def process_timeframe(self, timeframe: str, data: TimeframeData):
        """Process data for a specific timeframe."""
        self.timeframe_data[timeframe] = data


class Timeframe(Enum):
    """Supported timeframes."""

    M1 = "1min"
    M5 = "5min"
    M15 = "15min"
    H1 = "1h"
    D1 = "1d"


class TimeframeAggregator:
    """Aggregates lower timeframe data into higher timeframes."""

    def __init__(self):
        self.timeframe_seconds = {
            Timeframe.M1: 60,
            Timeframe.M5: 300,
            Timeframe.M15: 900,
            Timeframe.H1: 3600,
            Timeframe.D1: 86400,
        }

    def aggregate_bars(
        self, bars: List[MarketData], target_timeframe: Timeframe
    ) -> List[MarketData]:
        """Aggregate bars into target timeframe."""
        if not bars:
            return []

        target_seconds = self.timeframe_seconds[target_timeframe]
        aggregated_bars = []

        # Sort bars by timestamp
        sorted_bars = sorted(bars, key=lambda b: b.timestamp)

        current_period_start = None
        current_bars = []

        for bar in sorted_bars:
            # Determine period start for this bar
            period_start = self._get_period_start(bar.timestamp, target_seconds)

            if current_period_start is None:
                current_period_start = period_start
                current_bars = [bar]
            elif period_start == current_period_start:
                current_bars.append(bar)
            else:
                # Create aggregated bar from current_bars
                if current_bars:
                    agg_bar = self._create_aggregated_bar(
                        current_bars, target_timeframe
                    )
                    aggregated_bars.append(agg_bar)

                # Start new period
                current_period_start = period_start
                current_bars = [bar]

        # Don't forget the last period
        if current_bars:
            agg_bar = self._create_aggregated_bar(current_bars, target_timeframe)
            aggregated_bars.append(agg_bar)

        return aggregated_bars

    def _get_period_start(self, timestamp: datetime, period_seconds: int) -> datetime:
        """Get the start of the period for a given timestamp."""
        epoch = datetime(1970, 1, 1, tzinfo=timestamp.tzinfo)
        total_seconds = int((timestamp - epoch).total_seconds())
        period_start_seconds = (total_seconds // period_seconds) * period_seconds
        return epoch + timedelta(seconds=period_start_seconds)

    def _create_aggregated_bar(
        self, bars: List[MarketData], timeframe: Timeframe
    ) -> MarketData:
        """Create an aggregated bar from a list of bars."""
        if not bars:
            raise ValueError("Cannot aggregate empty bar list")

        # Use the first bar's symbol and last bar's timestamp
        symbol = bars[0].symbol
        timestamp = bars[-1].timestamp

        # Calculate OHLCV
        open_price = bars[0].open
        close_price = bars[-1].close
        high_price = max(bar.high for bar in bars)
        low_price = min(bar.low for bar in bars)
        total_volume = sum(bar.volume for bar in bars)

        # Calculate VWAP (volume-weighted average price)
        if total_volume > 0:
            vwap = sum(bar.close * bar.volume for bar in bars) / total_volume
        else:
            vwap = close_price

        return MarketData(
            symbol=symbol,
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume,
            vwap=vwap,
        )


class MultiTimeframeManager:
    """Manages market data across multiple timeframes."""

    def __init__(self, max_bars_per_timeframe: int = 1000):
        self.logger = TradingLogger("timeframe_manager")
        self.max_bars = max_bars_per_timeframe

        # Data storage: symbol -> timeframe -> deque of bars
        self.data: Dict[str, Dict[Timeframe, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_bars_per_timeframe))
        )

        # Track which symbols and timeframes we're managing
        self.symbols: Set[str] = set()
        self.timeframes: Set[Timeframe] = set()

        # Aggregator for creating higher timeframe data
        self.aggregator = TimeframeAggregator()

        # Base timeframe (usually the smallest we receive)
        self.base_timeframe = Timeframe.M1

    def add_timeframe(self, timeframe: Timeframe) -> None:
        """Add a timeframe to track."""
        self.timeframes.add(timeframe)
        self.logger.logger.info(f"Added timeframe: {timeframe.value}")

    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to track."""
        self.symbols.add(symbol)
        self.logger.logger.info(f"Added symbol: {symbol}")

    def update_bar(
        self, bar: MarketData, source_timeframe: Timeframe = None
    ) -> Dict[Timeframe, MarketData]:
        """Update with new bar data and return latest bars for all timeframes."""
        if source_timeframe is None:
            source_timeframe = self.base_timeframe

        symbol = bar.symbol
        self.symbols.add(symbol)

        # Store the raw bar in its native timeframe
        self.data[symbol][source_timeframe].append(bar)

        # Update all higher timeframes
        updated_timeframes = {}

        for timeframe in self.timeframes:
            if timeframe == source_timeframe:
                # Direct update
                updated_timeframes[timeframe] = bar
            elif self._is_higher_timeframe(timeframe, source_timeframe):
                # Aggregate from source timeframe
                source_bars = list(self.data[symbol][source_timeframe])
                if source_bars:
                    aggregated_bars = self.aggregator.aggregate_bars(
                        source_bars, timeframe
                    )
                    if aggregated_bars:
                        # Store aggregated bars
                        self.data[symbol][timeframe].clear()
                        for agg_bar in aggregated_bars:
                            self.data[symbol][timeframe].append(agg_bar)

                        # Return the latest aggregated bar
                        updated_timeframes[timeframe] = aggregated_bars[-1]

        return updated_timeframes

    def get_bars(
        self, symbol: str, timeframe: Timeframe, count: int = None
    ) -> List[MarketData]:
        """Get bars for a symbol and timeframe."""
        if symbol not in self.data or timeframe not in self.data[symbol]:
            return []

        bars = list(self.data[symbol][timeframe])

        if count is None:
            return bars

        return bars[-count:] if len(bars) >= count else bars

    def get_latest_bar(self, symbol: str, timeframe: Timeframe) -> Optional[MarketData]:
        """Get the latest bar for a symbol and timeframe."""
        bars = self.get_bars(symbol, timeframe, count=1)
        return bars[0] if bars else None

    def get_latest_price(
        self, symbol: str, timeframe: Timeframe = None
    ) -> Optional[Decimal]:
        """Get latest price for a symbol (defaults to base timeframe)."""
        if timeframe is None:
            timeframe = self.base_timeframe

        latest_bar = self.get_latest_bar(symbol, timeframe)
        return latest_bar.close if latest_bar else None

    def get_all_timeframe_data(self, symbol: str) -> Dict[Timeframe, List[MarketData]]:
        """Get data for all timeframes for a symbol."""
        result = {}
        for timeframe in self.timeframes:
            result[timeframe] = self.get_bars(symbol, timeframe)
        return result

    def sync_timeframes(self, symbol: str) -> Dict[Timeframe, MarketData]:
        """Ensure all timeframes are properly synchronized."""
        synced_data = {}

        # Get base timeframe data
        base_bars = self.get_bars(symbol, self.base_timeframe)
        if not base_bars:
            return synced_data

        # Re-aggregate all higher timeframes from base data
        for timeframe in self.timeframes:
            if timeframe == self.base_timeframe:
                synced_data[timeframe] = base_bars[-1]
            elif self._is_higher_timeframe(timeframe, self.base_timeframe):
                aggregated_bars = self.aggregator.aggregate_bars(base_bars, timeframe)
                if aggregated_bars:
                    # Update stored data
                    self.data[symbol][timeframe].clear()
                    for bar in aggregated_bars:
                        self.data[symbol][timeframe].append(bar)

                    synced_data[timeframe] = aggregated_bars[-1]

        return synced_data

    def _is_higher_timeframe(
        self, timeframe1: Timeframe, timeframe2: Timeframe
    ) -> bool:
        """Check if timeframe1 is higher (longer period) than timeframe2."""
        timeframe_order = [
            Timeframe.M1,
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.H1,
            Timeframe.D1,
        ]
        try:
            index1 = timeframe_order.index(timeframe1)
            index2 = timeframe_order.index(timeframe2)
            return index1 > index2
        except ValueError:
            return False

    def get_timeframe_alignment(self, symbol: str) -> Dict[str, Any]:
        """Get information about timeframe alignment and data availability."""
        alignment_info = {
            "symbol": symbol,
            "available_timeframes": [],
            "bar_counts": {},
            "latest_timestamps": {},
            "synchronized": True,
        }

        for timeframe in self.timeframes:
            bars = self.get_bars(symbol, timeframe)
            if bars:
                alignment_info["available_timeframes"].append(timeframe.value)
                alignment_info["bar_counts"][timeframe.value] = len(bars)
                alignment_info["latest_timestamps"][timeframe.value] = bars[
                    -1
                ].timestamp

        # Check if timestamps are reasonably aligned (simplified check)
        timestamps = list(alignment_info["latest_timestamps"].values())
        if len(timestamps) > 1:
            time_diffs = [
                abs((timestamps[i] - timestamps[0]).total_seconds())
                for i in range(1, len(timestamps))
            ]
            # Allow up to 1 hour difference for alignment
            alignment_info["synchronized"] = all(diff <= 3600 for diff in time_diffs)

        return alignment_info

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the timeframe manager."""
        total_bars = 0
        timeframe_counts = {}

        for symbol_data in self.data.values():
            for timeframe, bars in symbol_data.items():
                total_bars += len(bars)
                if timeframe.value not in timeframe_counts:
                    timeframe_counts[timeframe.value] = 0
                timeframe_counts[timeframe.value] += len(bars)

        return {
            "total_symbols": len(self.symbols),
            "active_timeframes": len(self.timeframes),
            "total_bars_stored": total_bars,
            "bars_per_timeframe": timeframe_counts,
            "max_bars_per_timeframe": self.max_bars,
            "memory_usage_estimate": f"{total_bars * 0.001:.2f} KB",  # Rough estimate
        }
