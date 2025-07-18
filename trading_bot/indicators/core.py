"""Simple and working technical indicators implementation."""

from collections import deque
from typing import Any, Dict, List, Optional

from ..core.logging import TradingLogger
from ..core.models import MarketData


class IndicatorResult:
    """Result object for technical indicators."""

    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value


class TechnicalIndicator:
    """Base class for technical indicators."""

    def __init__(self, period_or_config=14):
        # Handle both integer period and IndicatorConfig objects
        if hasattr(period_or_config, "period"):
            # It's an IndicatorConfig object
            self.period = period_or_config.period
            self.config = period_or_config
        else:
            # It's a simple integer period
            self.period = period_or_config
            self.config = None

        self.values = deque(maxlen=1000)  # Keep last 1000 values
        self.ready = False

    def update(self, value_or_bar) -> Optional[IndicatorResult]:
        """Update indicator with new value."""
        # Handle both float values and MarketData objects
        if hasattr(value_or_bar, "close"):
            # It's a MarketData object
            value = float(value_or_bar.close)
        else:
            # It's a simple float value
            value = float(value_or_bar)

        self.values.append(value)

        if len(self.values) >= self.period:
            self.ready = True
            result_value = self.calculate()
            if result_value is not None:
                return IndicatorResult(self.get_name(), result_value)
        return None

    def calculate(self) -> float:
        """Calculate indicator value. Override in subclasses."""
        raise NotImplementedError

    def get_name(self) -> str:
        """Get indicator name. Override in subclasses."""
        return "INDICATOR"


class SMA(TechnicalIndicator):
    """Simple Moving Average."""

    def calculate(self) -> float:
        """Calculate SMA."""
        recent_values = list(self.values)[-self.period :]
        return sum(recent_values) / len(recent_values)

    def get_name(self) -> str:
        """Get indicator name."""
        return "SMA"


class EMA(TechnicalIndicator):
    """Exponential Moving Average."""

    def __init__(self, period_or_config=14):
        super().__init__(period_or_config)
        self.alpha = 2.0 / (self.period + 1)
        self.ema_value = None

    def calculate(self) -> float:
        """Calculate EMA."""
        current_value = self.values[-1]

        if self.ema_value is None:
            # Initialize with SMA
            recent_values = list(self.values)[-self.period :]
            self.ema_value = sum(recent_values) / len(recent_values)
        else:
            # Calculate EMA
            self.ema_value = (current_value * self.alpha) + (
                self.ema_value * (1 - self.alpha)
            )

        return self.ema_value

    def get_name(self) -> str:
        """Get indicator name."""
        return "EMA"


class RSI(TechnicalIndicator):
    """Relative Strength Index."""

    def __init__(self, period_or_config=14):
        super().__init__(period_or_config)
        self.gains = deque(maxlen=1000)
        self.losses = deque(maxlen=1000)

    def update(self, value_or_bar) -> Optional[IndicatorResult]:
        """Update RSI with new value."""
        # Handle both float values and MarketData objects
        if hasattr(value_or_bar, "close"):
            # It's a MarketData object
            value = float(value_or_bar.close)
        else:
            # It's a simple float value
            value = float(value_or_bar)

        if len(self.values) > 0:
            change = value - self.values[-1]
            gain = max(0, change)
            loss = abs(min(0, change))

            self.gains.append(gain)
            self.losses.append(loss)

        self.values.append(value)

        if len(self.values) >= self.period + 1:
            self.ready = True
            result_value = self.calculate()
            if result_value is not None:
                return IndicatorResult(self.get_name(), result_value)
        return None

    def calculate(self) -> float:
        """Calculate RSI."""
        recent_gains = list(self.gains)[-self.period :]
        recent_losses = list(self.losses)[-self.period :]

        avg_gain = sum(recent_gains) / len(recent_gains)
        avg_loss = sum(recent_losses) / len(recent_losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_name(self) -> str:
        """Get indicator name."""
        return "RSI"


class MACD(TechnicalIndicator):
    """MACD (Moving Average Convergence Divergence)."""

    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        super().__init__(slow_period)
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
        self.signal_ema = EMA(signal_period)
        self.macd_line = deque(maxlen=1000)

    def update(self, value_or_bar) -> Optional[IndicatorResult]:
        """Update MACD with new value."""
        # Handle both float values and MarketData objects
        if hasattr(value_or_bar, "close"):
            # It's a MarketData object
            value = float(value_or_bar.close)
        else:
            # It's a simple float value
            value = float(value_or_bar)

        fast_ema_result = self.fast_ema.update(value)
        slow_ema_result = self.slow_ema.update(value)

        if fast_ema_result is not None and slow_ema_result is not None:
            # Extract values from IndicatorResult objects
            fast_ema_val = (
                fast_ema_result.value
                if hasattr(fast_ema_result, "value")
                else fast_ema_result
            )
            slow_ema_val = (
                slow_ema_result.value
                if hasattr(slow_ema_result, "value")
                else slow_ema_result
            )

            macd_val = fast_ema_val - slow_ema_val
            self.macd_line.append(macd_val)

            signal_result = self.signal_ema.update(macd_val)

            if signal_result is not None:
                signal_val = (
                    signal_result.value
                    if hasattr(signal_result, "value")
                    else signal_result
                )
                histogram = macd_val - signal_val
                self.ready = True

                result_dict = {
                    "macd": macd_val,
                    "signal": signal_val,
                    "histogram": histogram,
                }
                return IndicatorResult(self.get_name(), result_dict)

        return None

    def get_name(self) -> str:
        """Get indicator name."""
        return "MACD"


class BollingerBands(TechnicalIndicator):
    """Bollinger Bands."""

    def __init__(self, period_or_config=20, std_dev: float = 2.0):
        super().__init__(period_or_config)
        self.std_dev = std_dev

    def calculate(self) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        recent_values = list(self.values)[-self.period :]

        middle = sum(recent_values) / len(recent_values)  # SMA
        variance = sum((x - middle) ** 2 for x in recent_values) / len(recent_values)
        std = variance ** 0.5

        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)

        return {"upper": upper, "middle": middle, "lower": lower}

    def get_name(self) -> str:
        """Get indicator name."""
        return "BBANDS"


class ATR(TechnicalIndicator):
    """Average True Range."""

    def __init__(self, period_or_config=14):
        super().__init__(period_or_config)
        self.high_values = deque(maxlen=1000)
        self.low_values = deque(maxlen=1000)
        self.close_values = deque(maxlen=1000)
        self.tr_values = deque(maxlen=1000)

    def update(self, bar_or_high, low=None, close=None) -> Optional[IndicatorResult]:
        """Update ATR with new OHLC values."""
        # Handle both MarketData objects and separate values
        if hasattr(bar_or_high, "high"):
            # It's a MarketData object
            high = float(bar_or_high.high)
            low = float(bar_or_high.low)
            close = float(bar_or_high.close)
        else:
            # Separate values provided
            high = float(bar_or_high)
            low = float(low) if low is not None else high
            close = float(close) if close is not None else high

        self.high_values.append(high)
        self.low_values.append(low)
        self.close_values.append(close)

        if len(self.close_values) > 1:
            # Calculate True Range
            prev_close = self.close_values[-2]
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_range = max(tr1, tr2, tr3)
            self.tr_values.append(true_range)

            if len(self.tr_values) >= self.period:
                self.ready = True
                # Calculate ATR as SMA of True Range
                recent_tr = list(self.tr_values)[-self.period :]
                atr_value = sum(recent_tr) / len(recent_tr)
                return IndicatorResult(self.get_name(), atr_value)

        return None

    def get_name(self) -> str:
        """Get indicator name."""
        return "ATR"


class Stochastic(TechnicalIndicator):
    """Stochastic Oscillator."""

    def __init__(self, period_or_config=14, d_period: int = 3):
        if hasattr(period_or_config, "period"):
            # It's a config object
            k_period = period_or_config.period
        else:
            k_period = period_or_config

        super().__init__(k_period)
        self.k_period = k_period
        self.d_period = d_period
        self.high_values = deque(maxlen=1000)
        self.low_values = deque(maxlen=1000)
        self.close_values = deque(maxlen=1000)
        self.k_values = deque(maxlen=1000)

    def update(self, bar_or_high, low=None, close=None) -> Optional[IndicatorResult]:
        """Update Stochastic with new HLC values."""
        # Handle both MarketData objects and separate values
        if hasattr(bar_or_high, "high"):
            # It's a MarketData object
            high = float(bar_or_high.high)
            low = float(bar_or_high.low)
            close = float(bar_or_high.close)
        else:
            # Separate values provided
            high = float(bar_or_high)
            low = float(low) if low is not None else high
            close = float(close) if close is not None else high

        self.high_values.append(high)
        self.low_values.append(low)
        self.close_values.append(close)

        if len(self.close_values) >= self.k_period:
            # Calculate %K
            recent_highs = list(self.high_values)[-self.k_period :]
            recent_lows = list(self.low_values)[-self.k_period :]

            highest_high = max(recent_highs)
            lowest_low = min(recent_lows)

            if highest_high != lowest_low:
                k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                # Handle flat price range - use neutral value (50%)
                k_percent = 50.0

            self.k_values.append(k_percent)

            # Calculate %D (SMA of %K)
            if len(self.k_values) >= self.d_period:
                self.ready = True
                recent_k = list(self.k_values)[-self.d_period :]
                d_percent = sum(recent_k) / len(recent_k)

                result_dict = {"%K": k_percent, "%D": d_percent}
                return IndicatorResult(self.get_name(), result_dict)

        return None

    def get_name(self) -> str:
        """Get indicator name."""
        return "STOCH"


class SimpleIndicatorManager:
    """Simple indicator manager that works with existing strategies."""

    def __init__(self):
        self.indicators: Dict[str, Dict[str, TechnicalIndicator]] = {}
        self.logger = TradingLogger("indicator_manager")

    def add_indicator(
        self, symbol: str, indicator_name: str, config=None, **kwargs
    ) -> bool:
        """Add an indicator for a symbol."""
        try:
            if symbol not in self.indicators:
                self.indicators[symbol] = {}

            # Extract parameters from config or kwargs
            if config is not None:
                # Use config object (for IndicatorConfig)
                if hasattr(config, "period"):
                    kwargs["period"] = config.period
                if hasattr(config, "parameters") and config.parameters:
                    kwargs.update(config.parameters)

            # Create indicator instance
            if indicator_name == "SMA":
                period = kwargs.get("period", 20)
                self.indicators[symbol][indicator_name] = SMA(period)
            elif indicator_name == "EMA":
                period = kwargs.get("period", 12)
                self.indicators[symbol][indicator_name] = EMA(period)
            elif indicator_name == "RSI":
                period = kwargs.get("period", 14)
                self.indicators[symbol][indicator_name] = RSI(period)
            elif indicator_name == "MACD":
                fast = kwargs.get("fast_period", 12)
                slow = kwargs.get("slow_period", 26)
                signal = kwargs.get("signal_period", 9)
                self.indicators[symbol][indicator_name] = MACD(fast, slow, signal)
            elif indicator_name == "BBANDS":
                period = kwargs.get("period", 20)
                std_dev = kwargs.get("std_dev", 2.0)
                self.indicators[symbol][indicator_name] = BollingerBands(
                    period, std_dev
                )
            elif indicator_name == "ATR":
                period = kwargs.get("period", 14)
                self.indicators[symbol][indicator_name] = ATR(period)
            elif indicator_name == "STOCH":
                k_period = kwargs.get("k_period", 14)
                d_period = kwargs.get("d_period", 3)
                self.indicators[symbol][indicator_name] = Stochastic(k_period, d_period)
            else:
                return False

            return True

        except Exception as e:
            self.logger.logger.error(
                f"Failed to add indicator {indicator_name} for {symbol}: {e}"
            )
            return False

    def update_indicators_with_results(
        self, symbol: str, bar: MarketData
    ) -> Dict[str, IndicatorResult]:
        """Update all indicators for a symbol with new bar data and return IndicatorResult objects."""
        results = {}

        if symbol not in self.indicators:
            return results

        close_price = float(bar.close)
        high_price = float(bar.high)
        low_price = float(bar.low)

        for name, indicator in self.indicators[symbol].items():
            try:
                if name in ["SMA", "EMA", "RSI"]:
                    result = indicator.update(close_price)
                elif name == "MACD":
                    result = indicator.update(close_price)
                elif name == "BBANDS":
                    result = indicator.update(close_price)
                elif name == "ATR":
                    result = indicator.update(bar, high_price, low_price)
                elif name == "STOCH":
                    result = indicator.update(bar, high_price, low_price)
                else:
                    continue

                if result is not None:
                    results[name] = result

            except Exception as e:
                self.logger.logger.error(f"Error updating {name} for {symbol}: {e}")

        return results

    def update_indicators(self, symbol: str, bar: MarketData) -> Dict[str, Any]:
        """Update all indicators for a symbol with new bar data."""
        results = {}

        if symbol not in self.indicators:
            return results

        close_price = float(bar.close)
        high_price = float(bar.high)
        low_price = float(bar.low)

        for name, indicator in self.indicators[symbol].items():
            try:
                if name in ["SMA", "EMA", "RSI"]:
                    result = indicator.update(close_price)
                elif name == "MACD":
                    result = indicator.update(close_price)
                elif name == "BBANDS":
                    result = indicator.update(close_price)
                elif name == "ATR":
                    result = indicator.update(bar, high_price, low_price)
                elif name == "STOCH":
                    result = indicator.update(bar, high_price, low_price)
                else:
                    continue

                if result is not None:
                    # Extract raw value from IndicatorResult for backward compatibility
                    if hasattr(result, "value"):
                        results[name] = result.value
                    else:
                        results[name] = result

            except Exception as e:
                self.logger.logger.error(f"Error updating {name} for {symbol}: {e}")

        return results

    def get_indicator_value(self, symbol: str, indicator_name: str) -> Any:
        """Get latest value for a specific indicator."""
        if symbol not in self.indicators:
            return None

        if indicator_name not in self.indicators[symbol]:
            return None

        indicator = self.indicators[symbol][indicator_name]

        # For indicators that maintain values, return the latest one
        if hasattr(indicator, "values") and indicator.values:
            return indicator.values[-1]

        return None

    def calculate_composite_signals(self, symbol: str) -> Dict[str, Any]:
        """Calculate composite trading signals from multiple indicators."""
        if symbol not in self.indicators:
            return {
                "overall_signal": "neutral",
                "trend_strength": 0.0,
                "momentum_strength": 0.0,
                "volatility_regime": "normal",
            }

        signals = {}
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0

        # Analyze SMA for trend
        if "SMA" in self.indicators[symbol]:
            sma_indicator = self.indicators[symbol]["SMA"]
            if hasattr(sma_indicator, "values") and len(sma_indicator.values) >= 2:
                recent_sma = sma_indicator.values[-1]
                prev_sma = sma_indicator.values[-2]
                if recent_sma > prev_sma:
                    bullish_signals += 1
                elif recent_sma < prev_sma:
                    bearish_signals += 1
                total_signals += 1

        # Analyze RSI for momentum
        rsi_value = None
        if "RSI" in self.indicators[symbol]:
            rsi_indicator = self.indicators[symbol]["RSI"]
            if hasattr(rsi_indicator, "values") and rsi_indicator.values:
                # RSI calculates its own value, use the calculate method if available
                try:
                    if rsi_indicator.ready:
                        rsi_value = rsi_indicator.calculate()
                        if rsi_value > 70:
                            bearish_signals += 1  # Overbought
                        elif rsi_value < 30:
                            bullish_signals += 1  # Oversold
                        total_signals += 1
                except:
                    pass

        # Analyze MACD for momentum confirmation
        if "MACD" in self.indicators[symbol]:
            macd_indicator = self.indicators[symbol]["MACD"]
            if (
                hasattr(macd_indicator, "macd_line")
                and len(macd_indicator.macd_line) >= 2
            ):
                recent_macd = macd_indicator.macd_line[-1]
                prev_macd = macd_indicator.macd_line[-2]
                if recent_macd > prev_macd and recent_macd > 0:
                    bullish_signals += 1
                elif recent_macd < prev_macd and recent_macd < 0:
                    bearish_signals += 1
                total_signals += 1

        # Calculate signal strengths
        if total_signals > 0:
            bullish_strength = bullish_signals / total_signals
            bearish_strength = bearish_signals / total_signals

            if bullish_strength > 0.6:
                overall_signal = "bullish"
            elif bearish_strength > 0.6:
                overall_signal = "bearish"
            else:
                overall_signal = "neutral"
        else:
            bullish_strength = 0.0
            bearish_strength = 0.0
            overall_signal = "neutral"

        # Determine volatility regime (simplified)
        volatility_regime = "normal"
        if "ATR" in self.indicators[symbol]:
            atr_indicator = self.indicators[symbol]["ATR"]
            if (
                hasattr(atr_indicator, "tr_values")
                and len(atr_indicator.tr_values) >= 10
            ):
                recent_atr = sum(list(atr_indicator.tr_values)[-5:]) / 5
                longer_atr = sum(list(atr_indicator.tr_values)[-10:]) / 10
                if recent_atr > longer_atr * 1.5:
                    volatility_regime = "high"
                elif recent_atr < longer_atr * 0.7:
                    volatility_regime = "low"

        return {
            "overall_signal": overall_signal,
            "trend_strength": max(bullish_strength, bearish_strength),
            "momentum_strength": bullish_strength - bearish_strength,  # Range: -1 to 1
            "volatility_regime": volatility_regime,
        }

    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators."""
        return ["SMA", "EMA", "RSI", "MACD", "BBANDS", "ATR", "STOCH"]
