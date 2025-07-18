"""Momentum strategy template."""

from decimal import Decimal
from typing import Any, Dict, List, Optional

from ...core.models import StrategySignal
from ..base import BaseStrategy


class MomentumTemplate(BaseStrategy):
    """
    Momentum strategy template.

    This strategy identifies and follows strong price trends, entering positions
    when momentum is confirmed by multiple indicators.

    Key Components:
    - Moving average crossovers for trend direction
    - MACD for momentum confirmation
    - RSI for avoiding overbought/oversold extremes
    - Volume confirmation
    """

    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        Initialize momentum strategy.

        Parameters:
        - fast_ma: Fast moving average period (default: 10)
        - slow_ma: Slow moving average period (default: 20)
        - macd_fast: MACD fast period (default: 12)
        - macd_slow: MACD slow period (default: 26)
        - macd_signal: MACD signal period (default: 9)
        - rsi_period: RSI period (default: 14)
        - rsi_min: Minimum RSI for long positions (default: 40)
        - rsi_max: Maximum RSI for long positions (default: 80)
        - volume_multiplier: Volume must be X times average (default: 1.2)
        - min_price_change: Minimum price change to consider (default: 0.02)
        """
        default_params = {
            "fast_ma": 10,
            "slow_ma": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "rsi_period": 14,
            "rsi_min": 40,
            "rsi_max": 80,
            "volume_multiplier": 1.2,
            "min_price_change": 0.02,
            "position_size": 0.06,
            "profit_target": 0.05,  # 5% profit target
            "stop_loss": 0.03,  # 3% stop loss
            "trailing_stop": True,  # Use trailing stop
        }
        default_params.update(parameters)
        super().__init__(name, default_params)

    async def generate_signals(self) -> List[StrategySignal]:
        """Generate momentum signals."""
        signals = []

        for symbol in self.symbols:
            signal = await self._analyze_symbol(symbol)
            if signal:
                signals.append(signal)

        return signals

    async def _analyze_symbol(self, symbol: str) -> Optional[StrategySignal]:
        """Analyze a symbol for momentum opportunities."""

        # Need sufficient data
        min_bars = max(self.parameters["slow_ma"], self.parameters["macd_slow"]) + 10
        bars = self.get_bars(symbol, min_bars)
        if len(bars) < min_bars:
            return None

        latest_bar = bars[-1]
        current_price = latest_bar.close

        # Get technical indicators
        fast_ma = self.calculate_ema(symbol, self.parameters["fast_ma"])
        slow_ma = self.calculate_ema(symbol, self.parameters["slow_ma"])
        macd = self.get_macd(symbol)
        rsi = self.calculate_rsi(symbol, self.parameters["rsi_period"])

        if not all([fast_ma, slow_ma, macd, rsi]):
            return None

        # Volume confirmation
        recent_volume = sum(bar.volume for bar in bars[-5:]) / 5
        avg_volume = sum(bar.volume for bar in bars[-20:]) / 20
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0

        # Price momentum check
        price_5_bars_ago = bars[-6].close if len(bars) >= 6 else current_price
        price_change_pct = (current_price - price_5_bars_ago) / price_5_bars_ago

        metadata = {
            "fast_ma": float(fast_ma),
            "slow_ma": float(slow_ma),
            "macd": {
                "macd": float(macd["macd"]),
                "signal": float(macd["signal"]),
                "histogram": float(macd["histogram"]),
            },
            "rsi": float(rsi),
            "volume_ratio": float(volume_ratio),
            "price_change_pct": float(price_change_pct),
            "strategy_type": "momentum",
        }

        # Long momentum signal
        if self._check_long_momentum(
            fast_ma, slow_ma, macd, rsi, volume_ratio, price_change_pct
        ):

            # Calculate signal strength
            strength = self._calculate_momentum_strength(
                fast_ma, slow_ma, macd, rsi, volume_ratio, price_change_pct, "long"
            )

            quantity = self._calculate_position_size(current_price)

            metadata["signal_reason"] = "bullish_momentum"
            metadata["profit_target"] = float(
                current_price * (1 + Decimal(str(self.parameters["profit_target"])))
            )
            metadata["stop_loss"] = float(
                current_price * (1 - Decimal(str(self.parameters["stop_loss"])))
            )

            return self.create_signal(
                symbol=symbol,
                signal_type="BUY",
                strength=strength,
                price=current_price,
                quantity=quantity,
                metadata=metadata,
            )

        # Short momentum signal (for instruments that allow shorting)
        elif self._check_short_momentum(
            fast_ma, slow_ma, macd, rsi, volume_ratio, price_change_pct
        ):

            strength = self._calculate_momentum_strength(
                fast_ma, slow_ma, macd, rsi, volume_ratio, price_change_pct, "short"
            )

            quantity = self._calculate_position_size(current_price)

            metadata["signal_reason"] = "bearish_momentum"
            metadata["profit_target"] = float(
                current_price * (1 - Decimal(str(self.parameters["profit_target"])))
            )
            metadata["stop_loss"] = float(
                current_price * (1 + Decimal(str(self.parameters["stop_loss"])))
            )

            return self.create_signal(
                symbol=symbol,
                signal_type="SELL",
                strength=strength,
                price=current_price,
                quantity=quantity,
                metadata=metadata,
            )

        return None

    def _check_long_momentum(
        self,
        fast_ma: Decimal,
        slow_ma: Decimal,
        macd: Dict[str, Decimal],
        rsi: Decimal,
        volume_ratio: float,
        price_change_pct: float,
    ) -> bool:
        """Check for bullish momentum conditions."""

        # Moving average trend
        ma_bullish = fast_ma > slow_ma

        # MACD confirmation
        macd_bullish = macd["macd"] > macd["signal"] and macd["histogram"] > 0

        # RSI in acceptable range (not overbought)
        rsi_ok = self.parameters["rsi_min"] <= rsi <= self.parameters["rsi_max"]

        # Volume confirmation
        volume_ok = volume_ratio >= self.parameters["volume_multiplier"]

        # Price momentum
        momentum_ok = price_change_pct >= self.parameters["min_price_change"]

        return ma_bullish and macd_bullish and rsi_ok and volume_ok and momentum_ok

    def _check_short_momentum(
        self,
        fast_ma: Decimal,
        slow_ma: Decimal,
        macd: Dict[str, Decimal],
        rsi: Decimal,
        volume_ratio: float,
        price_change_pct: float,
    ) -> bool:
        """Check for bearish momentum conditions."""

        # Moving average trend
        ma_bearish = fast_ma < slow_ma

        # MACD confirmation
        macd_bearish = macd["macd"] < macd["signal"] and macd["histogram"] < 0

        # RSI in acceptable range (not oversold)
        rsi_ok = (
            (100 - self.parameters["rsi_max"])
            <= rsi
            <= (100 - self.parameters["rsi_min"])
        )

        # Volume confirmation
        volume_ok = volume_ratio >= self.parameters["volume_multiplier"]

        # Price momentum (negative)
        momentum_ok = price_change_pct <= -self.parameters["min_price_change"]

        return ma_bearish and macd_bearish and rsi_ok and volume_ok and momentum_ok

    def _calculate_momentum_strength(
        self,
        fast_ma: Decimal,
        slow_ma: Decimal,
        macd: Dict[str, Decimal],
        rsi: Decimal,
        volume_ratio: float,
        price_change_pct: float,
        direction: str,
    ) -> float:
        """Calculate momentum signal strength."""

        strength_components = []

        # MA spread strength
        ma_spread = abs(fast_ma - slow_ma) / slow_ma
        ma_strength = min(1.0, ma_spread * 20)  # Scale to 0-1
        strength_components.append(ma_strength)

        # MACD strength
        macd_strength = min(1.0, abs(macd["histogram"]) * 10)
        strength_components.append(macd_strength)

        # RSI strength (distance from extreme)
        if direction == "long":
            rsi_strength = min(1.0, (rsi - self.parameters["rsi_min"]) / 30)
        else:
            rsi_strength = min(1.0, ((100 - self.parameters["rsi_max"]) - rsi) / 30)
        strength_components.append(max(0.1, rsi_strength))

        # Volume strength
        volume_strength = min(1.0, (volume_ratio - 1) * 2)
        strength_components.append(volume_strength)

        # Price momentum strength
        momentum_strength = min(1.0, abs(price_change_pct) * 20)
        strength_components.append(momentum_strength)

        # Average all components
        overall_strength = sum(strength_components) / len(strength_components)
        return max(0.1, min(1.0, overall_strength))

    def _calculate_position_size(self, price: Decimal) -> Decimal:
        """Calculate position size based on available capital."""
        available_capital = Decimal("100000")  # Default for template
        position_value = available_capital * Decimal(
            str(self.parameters["position_size"])
        )
        quantity = position_value / price
        return quantity

    def get_strategy_description(self) -> Dict[str, Any]:
        """Get strategy description and parameters."""
        return {
            "name": self.name,
            "type": "momentum",
            "description": "Momentum strategy using moving averages, MACD, and volume",
            "indicators_used": ["EMA", "MACD", "RSI", "Volume"],
            "signal_conditions": {
                "long": "Fast MA > Slow MA AND MACD bullish AND RSI not overbought AND volume spike",
                "short": "Fast MA < Slow MA AND MACD bearish AND RSI not oversold AND volume spike",
            },
            "risk_management": {
                "profit_target": f"{self.parameters['profit_target']:.1%}",
                "stop_loss": f"{self.parameters['stop_loss']:.1%}",
                "position_size": f"{self.parameters['position_size']:.1%}",
                "trailing_stop": self.parameters["trailing_stop"],
            },
            "parameters": self.parameters,
            "best_markets": ["Trending markets", "High momentum periods"],
            "avoid_markets": ["Sideways markets", "Very low volatility"],
        }

    def backtest_ready(self) -> bool:
        """Check if strategy is ready for backtesting."""
        required_data_points = max(
            self.parameters["slow_ma"] + 10,
            self.parameters["macd_slow"] + self.parameters["macd_signal"] + 5,
            self.parameters["rsi_period"] + 5,
        )

        for symbol in self.symbols:
            bars = self.get_bars(symbol)
            if len(bars) < required_data_points:
                return False

        return len(self.symbols) > 0
