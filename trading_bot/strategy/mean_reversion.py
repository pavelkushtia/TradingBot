"""Mean reversion strategy implementation."""

from typing import Any, Dict, List

from ..core.models import StrategySignal
from .base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands and RSI.

    Generates buy signals when price is oversold (low RSI) and near lower
    Bollinger Band. Generates sell signals when price is overbought (high RSI)
    and near upper Bollinger Band.
    """

    def __init__(self, name: str, parameters: Dict[str, Any]):
        """Initialize mean reversion strategy."""
        super().__init__(name, parameters)

        # Strategy parameters
        self.bollinger_window = parameters.get("bollinger_window", 20)
        self.bollinger_std = parameters.get("bollinger_std", 2)
        self.rsi_window = parameters.get("rsi_window", 14)
        self.rsi_oversold = parameters.get("rsi_oversold", 30)
        self.rsi_overbought = parameters.get("rsi_overbought", 70)
        self.min_price_distance = parameters.get(
            "min_price_distance", 0.02
        )  # 2% from band

        # State tracking
        self.previous_signals: Dict[str, str] = {}

    async def generate_signals(self) -> List[StrategySignal]:
        """Generate mean reversion signals."""
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
        """Analyze a single symbol for mean reversion signals."""
        # Need sufficient data for both Bollinger Bands and RSI
        min_bars = max(self.bollinger_window, self.rsi_window) + 5
        bars = self.get_bars(symbol, min_bars)

        if len(bars) < min_bars:
            return None

        current_price = self.get_latest_price(symbol)
        if current_price is None:
            return None

        # Calculate Bollinger Bands
        bollinger = self.calculate_bollinger_bands(
            symbol, self.bollinger_window, self.bollinger_std
        )
        if bollinger is None:
            return None

        # Calculate RSI
        rsi = self.calculate_rsi(symbol, self.rsi_window)
        if rsi is None:
            return None

        # Determine signal
        signal_type = None
        strength = 0.0
        metadata = {
            "current_price": float(current_price),
            "bollinger_upper": float(bollinger["upper"]),
            "bollinger_middle": float(bollinger["middle"]),
            "bollinger_lower": float(bollinger["lower"]),
            "rsi": float(rsi),
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
        }

        # Check for oversold conditions (potential buy signal)
        if rsi <= self.rsi_oversold:
            # Calculate distance from lower Bollinger Band
            distance_from_lower = (current_price - bollinger["lower"]) / bollinger[
                "lower"
            ]

            if distance_from_lower <= self.min_price_distance:
                signal_type = "buy"

                # Strength based on how oversold and how close to lower band
                rsi_strength = (self.rsi_oversold - rsi) / self.rsi_oversold
                band_strength = (
                    max(0, self.min_price_distance - distance_from_lower)
                    / self.min_price_distance
                )
                strength = (rsi_strength + band_strength) / 2

        # Check for overbought conditions (potential sell signal)
        elif rsi >= self.rsi_overbought:
            # Calculate distance from upper Bollinger Band
            distance_from_upper = (bollinger["upper"] - current_price) / bollinger[
                "upper"
            ]

            if distance_from_upper <= self.min_price_distance:
                signal_type = "sell"

                # Strength based on how overbought and how close to upper band
                rsi_strength = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
                band_strength = (
                    max(0, self.min_price_distance - distance_from_upper)
                    / self.min_price_distance
                )
                strength = (rsi_strength + band_strength) / 2

        # Generate signal if conditions are met
        if signal_type and strength > 0:
            # Avoid duplicate signals
            last_signal = self.previous_signals.get(symbol)
            if last_signal != signal_type:
                self.previous_signals[symbol] = signal_type

                return self.create_signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=min(strength, 1.0),
                    metadata=metadata,
                )

        return None
