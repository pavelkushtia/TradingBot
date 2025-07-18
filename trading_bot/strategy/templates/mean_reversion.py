"""Mean reversion strategy template."""

from decimal import Decimal
from typing import Any, Dict, List, Optional

from ...core.models import MarketData, StrategySignal
from ..base import BaseStrategy


class MeanReversionTemplate(BaseStrategy):
    """
    Mean reversion strategy template.

    This strategy looks for stocks that have deviated significantly from their mean
    and bets on them reverting back to the mean.

    Key Components:
    - Bollinger Bands for mean reversion signals
    - RSI for overbought/oversold conditions
    - Moving averages for trend confirmation
    """

    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        Initialize mean reversion strategy.

        Parameters:
        - lookback_period: Period for calculating moving average (default: 20)
        - bollinger_std: Standard deviations for Bollinger Bands (default: 2.0)
        - rsi_period: Period for RSI calculation (default: 14)
        - rsi_oversold: RSI oversold threshold (default: 30)
        - rsi_overbought: RSI overbought threshold (default: 70)
        - min_volume: Minimum average volume for trade consideration (default: 100000)
        - position_size: Position size as percentage of capital (default: 0.05)
        """
        default_params = {
            "lookback_period": 20,
            "bollinger_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "min_volume": 100000,
            "position_size": 0.05,
            "profit_target": 0.03,  # 3% profit target
            "stop_loss": 0.02,  # 2% stop loss
        }
        default_params.update(parameters)
        super().__init__(name, default_params)

    async def generate_signals(self) -> List[StrategySignal]:
        """Generate mean reversion signals."""
        signals = []

        for symbol in self.symbols:
            signal = await self._analyze_symbol(symbol)
            if signal:
                signals.append(signal)

        return signals

    async def _analyze_symbol(self, symbol: str) -> Optional[StrategySignal]:
        """Analyze a symbol for mean reversion opportunities."""

        # Need sufficient data
        bars = self.get_bars(symbol, self.parameters["lookback_period"] + 10)
        if len(bars) < self.parameters["lookback_period"]:
            return None

        latest_bar = bars[-1]
        current_price = latest_bar.close

        # Check volume requirement
        avg_volume = sum(bar.volume for bar in bars[-10:]) / 10
        if avg_volume < self.parameters["min_volume"]:
            return None

        # Get technical indicators
        bollinger_bands = self.calculate_bollinger_bands(
            symbol, self.parameters["lookback_period"], self.parameters["bollinger_std"]
        )

        rsi = self.calculate_rsi(symbol, self.parameters["rsi_period"])
        sma = self.calculate_sma(symbol, self.parameters["lookback_period"])

        if not all([bollinger_bands, rsi, sma]):
            return None

        # Mean reversion signals
        signal_type = None
        strength = 0.0
        metadata = {
            "bollinger_bands": {
                "upper": float(bollinger_bands["upper"]),
                "middle": float(bollinger_bands["middle"]),
                "lower": float(bollinger_bands["lower"]),
            },
            "rsi": float(rsi),
            "sma": float(sma),
            "current_price": float(current_price),
            "strategy_type": "mean_reversion",
        }

        # Long signal: Price below lower Bollinger Band + RSI oversold
        if (
            current_price <= bollinger_bands["lower"]
            and rsi <= self.parameters["rsi_oversold"]
        ):

            signal_type = "BUY"

            # Calculate strength based on how far below the band and RSI level
            price_deviation = (
                bollinger_bands["lower"] - current_price
            ) / bollinger_bands["middle"]
            rsi_strength = (self.parameters["rsi_oversold"] - rsi) / self.parameters[
                "rsi_oversold"
            ]
            strength = min(1.0, max(0.1, (price_deviation + rsi_strength) / 2))

            metadata["signal_reason"] = "oversold_mean_reversion"
            metadata["price_deviation"] = float(price_deviation)
            metadata["rsi_strength"] = float(rsi_strength)

        # Short signal: Price above upper Bollinger Band + RSI overbought
        elif (
            current_price >= bollinger_bands["upper"]
            and rsi >= self.parameters["rsi_overbought"]
        ):

            signal_type = "SELL"

            # Calculate strength
            price_deviation = (
                current_price - bollinger_bands["upper"]
            ) / bollinger_bands["middle"]
            rsi_strength = (rsi - self.parameters["rsi_overbought"]) / (
                100 - self.parameters["rsi_overbought"]
            )
            strength = min(1.0, max(0.1, (price_deviation + rsi_strength) / 2))

            metadata["signal_reason"] = "overbought_mean_reversion"
            metadata["price_deviation"] = float(price_deviation)
            metadata["rsi_strength"] = float(rsi_strength)

        if signal_type:
            # Calculate position size
            quantity = self._calculate_position_size(current_price)

            # Add profit target and stop loss levels
            if signal_type == "BUY":
                metadata["profit_target"] = float(
                    current_price * (1 + Decimal(str(self.parameters["profit_target"])))
                )
                metadata["stop_loss"] = float(
                    current_price * (1 - Decimal(str(self.parameters["stop_loss"])))
                )
            else:
                metadata["profit_target"] = float(
                    current_price * (1 - Decimal(str(self.parameters["profit_target"])))
                )
                metadata["stop_loss"] = float(
                    current_price * (1 + Decimal(str(self.parameters["stop_loss"])))
                )

            return self.create_signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                price=current_price,
                quantity=quantity,
                metadata=metadata,
            )

        return None

    def _calculate_position_size(self, price: Decimal) -> Decimal:
        """Calculate position size based on available capital."""
        # This is simplified - in real implementation would integrate with portfolio manager
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
            "type": "mean_reversion",
            "description": "Mean reversion strategy using Bollinger Bands and RSI",
            "indicators_used": ["Bollinger Bands", "RSI", "SMA"],
            "signal_conditions": {
                "long": "Price below lower Bollinger Band AND RSI oversold",
                "short": "Price above upper Bollinger Band AND RSI overbought",
            },
            "risk_management": {
                "profit_target": f"{self.parameters['profit_target']:.1%}",
                "stop_loss": f"{self.parameters['stop_loss']:.1%}",
                "position_size": f"{self.parameters['position_size']:.1%}",
            },
            "parameters": self.parameters,
            "best_markets": ["Range-bound markets", "High volatility periods"],
            "avoid_markets": ["Strong trending markets", "Very low volatility"],
        }

    def backtest_ready(self) -> bool:
        """Check if strategy is ready for backtesting."""
        required_data_points = max(
            self.parameters["lookback_period"] + 10, self.parameters["rsi_period"] + 5
        )

        for symbol in self.symbols:
            bars = self.get_bars(symbol)
            if len(bars) < required_data_points:
                return False

        return len(self.symbols) > 0

    async def on_bar(self, symbol: str, data: MarketData) -> None:
        """Handle new market data."""
        await super().on_bar(symbol, data)

        # Optional: Add real-time signal generation here
        # This would be useful for live trading
