"""Breakout strategy implementation."""

from decimal import Decimal
from typing import Any, Dict, List

from ..core.models import StrategySignal
from .base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy that identifies price breakouts from consolidation ranges.

    Generates buy signals when price breaks above resistance with volume confirmation.
    Generates sell signals when price breaks below support with volume confirmation.
    """

    def __init__(self, name: str, parameters: Dict[str, Any]):
        """Initialize breakout strategy."""
        super().__init__(name, parameters)

        # Strategy parameters
        self.lookback_window = parameters.get("lookback_window", 20)
        self.breakout_threshold = parameters.get(
            "breakout_threshold", 0.02
        )  # 2% breakout
        self.volume_multiplier = parameters.get(
            "volume_multiplier", 1.5
        )  # 1.5x average volume
        self.min_consolidation_periods = parameters.get("min_consolidation_periods", 10)

        # State tracking
        self.previous_signals: Dict[str, str] = {}
        self.support_levels: Dict[str, Decimal] = {}
        self.resistance_levels: Dict[str, Decimal] = {}

    async def generate_signals(self) -> List[StrategySignal]:
        """Generate breakout signals."""
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

    async def _analyze_symbol(self, symbol: str) -> StrategySignal:
        """Analyze a single symbol for breakout signals."""
        bars = self.get_bars(symbol, self.lookback_window + 10)
        if len(bars) < self.lookback_window + 5:
            return None

        current_price = self.get_latest_price(symbol)
        if current_price is None:
            return None

        # Calculate support and resistance levels
        recent_bars = bars[-self.lookback_window :]
        support, resistance = self._calculate_support_resistance(recent_bars)

        if support is None or resistance is None:
            return None

        # Store levels for reference
        self.support_levels[symbol] = support
        self.resistance_levels[symbol] = resistance

        # Check if price is in consolidation (between support and resistance)
        range_size = (resistance - support) / support
        if range_size < self.breakout_threshold:
            return None  # Range too small for meaningful breakout

        # Calculate average volume
        avg_volume = sum(bar.volume for bar in recent_bars) / len(recent_bars)
        current_volume = bars[-1].volume

        # Check volume confirmation
        volume_confirmed = current_volume >= (avg_volume * self.volume_multiplier)

        signal_type = None
        strength = 0.0
        metadata = {
            "current_price": float(current_price),
            "support_level": float(support),
            "resistance_level": float(resistance),
            "range_size": float(range_size),
            "current_volume": current_volume,
            "average_volume": int(avg_volume),
            "volume_confirmed": volume_confirmed,
        }

        # Check for upward breakout
        if current_price > resistance:
            breakout_strength = (current_price - resistance) / resistance

            if breakout_strength >= self.breakout_threshold:
                signal_type = "buy"
                strength = min(breakout_strength / self.breakout_threshold, 1.0)

                # Boost strength if volume is confirmed
                if volume_confirmed:
                    strength = min(strength * 1.3, 1.0)

        # Check for downward breakout
        elif current_price < support:
            breakout_strength = (support - current_price) / support

            if breakout_strength >= self.breakout_threshold:
                signal_type = "sell"
                strength = min(breakout_strength / self.breakout_threshold, 1.0)

                # Boost strength if volume is confirmed
                if volume_confirmed:
                    strength = min(strength * 1.3, 1.0)

        # Generate signal if conditions are met
        if signal_type and strength > 0:
            # Avoid duplicate signals
            last_signal = self.previous_signals.get(symbol)
            if last_signal != signal_type:
                self.previous_signals[symbol] = signal_type

                return self.create_signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=strength,
                    metadata=metadata,
                )

        return None

    def _calculate_support_resistance(self, bars: List) -> tuple:
        """Calculate support and resistance levels from recent bars."""
        if len(bars) < self.min_consolidation_periods:
            return None, None

        # Get high and low prices
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]

        # Simple approach: use highest high and lowest low
        # In practice, you might use more sophisticated pivot point detection
        resistance = max(highs)
        support = min(lows)

        # Alternative: use percentile-based levels for more robust S/R
        highs_sorted = sorted(highs, reverse=True)
        lows_sorted = sorted(lows)

        # Use 90th percentile for resistance, 10th percentile for support
        resistance_idx = int(len(highs_sorted) * 0.1)  # Top 10%
        support_idx = int(len(lows_sorted) * 0.1)  # Bottom 10%

        resistance = (
            highs_sorted[resistance_idx]
            if resistance_idx < len(highs_sorted)
            else resistance
        )
        support = (
            lows_sorted[support_idx] if support_idx < len(lows_sorted) else support
        )

        return support, resistance
