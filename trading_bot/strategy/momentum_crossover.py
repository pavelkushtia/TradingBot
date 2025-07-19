"""Momentum crossover strategy implementation."""

from typing import Any, Dict, List, Optional

from ..core.signal import StrategySignal
from .base import BaseStrategy


class MomentumCrossoverStrategy(BaseStrategy):
    """
    Simple momentum crossover strategy.

    Generates buy signals when short-term moving average crosses above
    long-term moving average. Generates sell signals when short-term moving
    average crosses below long-term moving average.
    """

    def __init__(self, name: str, parameters: Dict[str, Any]):
        """Initialize momentum crossover strategy."""
        super().__init__(name, parameters)

        # Strategy parameters
        self.short_window = parameters.get("short_window", 10)
        self.long_window = parameters.get("long_window", 30)
        self.min_strength_threshold = parameters.get("min_strength_threshold", 0.01)

        # State tracking
        self.previous_signals: Dict[str, str] = {}  # Track last signal per symbol

    async def generate_signals(self) -> List[StrategySignal]:
        """Generate momentum crossover signals."""
        signals = []

        for symbol in self.symbols:
            try:
                signal = await self._analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
            except Exception:
                # Log error but continue with other symbols
                pass

        return signals

    async def _analyze_symbol(self, symbol: str) -> Optional[StrategySignal]:
        """Analyze a single symbol for signals."""
        # Need at least long_window bars
        bars = self.get_bars(
            symbol, self.long_window + 5
        )  # Extra bars for confirmation
        if len(bars) < self.long_window + 1:
            return None

        # Calculate moving averages
        short_ma_current = self.calculate_sma(symbol, self.short_window)
        long_ma_current = self.calculate_sma(symbol, self.long_window)

        if short_ma_current is None or long_ma_current is None:
            return None

        # Calculate previous moving averages for crossover detection
        bars_minus_one = bars[:-1]
        if len(bars_minus_one) < self.long_window:
            return None

        # Temporarily store current bars and calculate previous MAs
        current_bars = self.market_data[symbol]
        self.market_data[symbol] = list(bars_minus_one)

        short_ma_previous = self.calculate_sma(symbol, self.short_window)
        long_ma_previous = self.calculate_sma(symbol, self.long_window)

        # Restore current bars
        self.market_data[symbol] = current_bars

        if short_ma_previous is None or long_ma_previous is None:
            return None

        # Detect crossover
        signal_type = None
        strength = 0.0

        # Bullish crossover: short MA crosses above long MA
        if short_ma_previous <= long_ma_previous and short_ma_current > long_ma_current:
            signal_type = "buy"

            # Calculate signal strength based on the magnitude of the crossover
            strength = float(abs(short_ma_current - long_ma_current) / long_ma_current)

        # Bearish crossover: short MA crosses below long MA
        elif (
            short_ma_previous >= long_ma_previous and short_ma_current < long_ma_current
        ):
            signal_type = "sell"

            # Calculate signal strength
            strength = float(abs(short_ma_current - long_ma_current) / long_ma_current)

        # Check if signal meets minimum strength threshold
        if signal_type and strength >= self.min_strength_threshold:
            # Avoid duplicate signals
            last_signal = self.previous_signals.get(symbol)
            if last_signal != signal_type:
                self.previous_signals[symbol] = signal_type

                # Add additional metadata
                metadata = {
                    "short_ma": float(short_ma_current),
                    "long_ma": float(long_ma_current),
                    "short_window": self.short_window,
                    "long_window": self.long_window,
                    "crossover_strength": strength,
                }

                return self.create_signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=min(strength * 2, 1.0),  # Scale strength but cap at 1.0
                    metadata=metadata,
                )

        return None
