"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from ..core.logging import TradingLogger
from ..core.models import MarketData, Quote
from ..core.signal import StrategySignal


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, name: str, parameters: Dict[str, Any]):
        """Initialize the strategy."""
        self.name = name
        self.parameters = parameters
        self.enabled = True
        self.symbols: Set[str] = set()

        # Performance tracking
        self.signals_generated = 0
        self.last_signal_time: Optional[datetime] = None

        # Market data storage
        self.market_data: Dict[str, List[MarketData]] = {}
        self.latest_quotes: Dict[str, Quote] = {}

        # Indicators management - use SimpleIndicatorManager
        try:
            from ..indicators import IndicatorManager

            self.indicator_manager = IndicatorManager()
            self._indicators_available = True
            self._setup_indicators()
        except Exception:
            # Fallback if indicators don't work
            self.indicator_manager = None
            self._indicators_available = False

    def _setup_indicators(self) -> None:
        """Setup default indicators for the strategy. Override in subclasses."""
        if not self._indicators_available:
            return

        # Default indicators that most strategies can use
        self._default_indicators = [
            ("SMA", {"period": 20}),
            ("EMA", {"period": 12}),
            ("RSI", {"period": 14}),
            ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
            ("BBANDS", {"period": 20, "std_dev": 2.0}),
            ("ATR", {"period": 14}),
        ]

    async def initialize(self) -> None:
        """Initialize the strategy (override if needed)."""

    async def cleanup(self) -> None:
        """Cleanup strategy resources (override if needed)."""

    @abstractmethod
    async def generate_signals(self) -> List[StrategySignal]:
        """Generate trading signals."""

    async def on_bar(self, symbol: str, data: MarketData) -> None:
        """Handle new market data bar."""
        if symbol not in self.market_data:
            self.market_data[symbol] = []
            # Setup indicators for new symbol
            self._setup_symbol_indicators(symbol)

        self.market_data[symbol].append(data)

        # Keep only recent data (configurable)
        max_bars = self.parameters.get("max_bars", 1000)
        if len(self.market_data[symbol]) > max_bars:
            self.market_data[symbol] = self.market_data[symbol][-max_bars:]

        self.symbols.add(symbol)

        # Update indicators
        if self._indicators_available and self.indicator_manager:
            try:
                self.indicator_manager.update_indicators(symbol, data)
            except Exception:
                # Silently fail if indicators don't work
                pass

    def _setup_symbol_indicators(self, symbol: str) -> None:
        """Setup indicators for a new symbol."""
        if not self._indicators_available or not self.indicator_manager:
            return

        for indicator_name, params in self._default_indicators:
            try:
                self.indicator_manager.add_indicator(symbol, indicator_name, **params)
            except Exception:
                # Skip if indicator fails to add
                pass

    async def on_quote(self, quote: Quote) -> None:
        """Handle new quote data."""
        self.latest_quotes[quote.symbol] = quote
        self.symbols.add(quote.symbol)

    def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """Get latest price for a symbol."""
        # Try quote first
        if symbol in self.latest_quotes:
            return self.latest_quotes[symbol].mid_price

        # Fall back to latest bar close
        if symbol in self.market_data and self.market_data[symbol]:
            return self.market_data[symbol][-1].close

        return None

    def get_bars(self, symbol: str, count: int = None) -> List[MarketData]:
        """Get recent bars for a symbol."""
        if symbol not in self.market_data:
            return []

        bars = self.market_data[symbol]
        if count is None:
            return bars

        return bars[-count:] if len(bars) >= count else bars

    def get_indicator_value(self, symbol: str, indicator_name: str) -> Any:
        """Get latest indicator value for a symbol."""
        if not self._indicators_available or not self.indicator_manager:
            return None

        try:
            return self.indicator_manager.get_indicator_value(symbol, indicator_name)
        except Exception:
            return None

    def get_composite_signals(self, symbol: str) -> Dict[str, Any]:
        """Get composite technical analysis signals."""
        if not self._indicators_available or not self.indicator_manager:
            return {}

        # Simple composite signals based on available indicators
        signals = {
            "trend_signal": "neutral",
            "momentum_signal": "neutral",
            "overall_signal": "neutral",
        }

        try:
            # Get indicator values
            sma = self.get_indicator_value(symbol, "SMA")
            ema = self.get_indicator_value(symbol, "EMA")
            rsi = self.get_indicator_value(symbol, "RSI")

            current_price = self.get_latest_price(symbol)
            if current_price:
                current_price = float(current_price)

                # Trend analysis
                if sma and ema:
                    if current_price > sma and ema > sma:
                        signals["trend_signal"] = "bullish"
                    elif current_price < sma and ema < sma:
                        signals["trend_signal"] = "bearish"

                # Momentum analysis
                if rsi:
                    if rsi > 70:
                        signals["momentum_signal"] = "overbought"
                    elif rsi < 30:
                        signals["momentum_signal"] = "oversold"
                    elif rsi > 50:
                        signals["momentum_signal"] = "bullish"
                    elif rsi < 50:
                        signals["momentum_signal"] = "bearish"

                # Overall signal
                if signals["trend_signal"] == "bullish" and signals[
                    "momentum_signal"
                ] in ["bullish", "oversold"]:
                    signals["overall_signal"] = "bullish"
                elif signals["trend_signal"] == "bearish" and signals[
                    "momentum_signal"
                ] in ["bearish", "overbought"]:
                    signals["overall_signal"] = "bearish"

        except Exception:
            # Return neutral signals if anything fails
            pass

        return signals

    # Legacy methods for backward compatibility - now use indicators when available
    def calculate_returns(self, symbol: str, periods: int = 1) -> List[Decimal]:
        """Calculate price returns for a symbol."""
        bars = self.get_bars(symbol)
        if len(bars) < periods + 1:
            return []

        returns = []
        for i in range(periods, len(bars)):
            current_price = bars[i].close
            previous_price = bars[i - periods].close
            return_pct = (current_price - previous_price) / previous_price
            returns.append(return_pct)

        return returns

    def calculate_sma(self, symbol: str, window: int) -> Optional[Decimal]:
        """Calculate Simple Moving Average - uses indicator system if available."""
        # Try using indicator system first
        if self._indicators_available:
            sma_value = self.get_indicator_value(symbol, "SMA")
            if sma_value is not None:
                return Decimal(str(sma_value))

        # Fallback to manual calculation
        bars = self.get_bars(symbol, window)
        if len(bars) < window:
            return None

        total = sum(bar.close for bar in bars[-window:])
        return total / window

    def calculate_ema(
        self, symbol: str, window: int, alpha: Optional[Decimal] = None
    ) -> Optional[Decimal]:
        """Calculate Exponential Moving Average - uses indicator system if available."""
        # Try using indicator system first
        if self._indicators_available:
            ema_value = self.get_indicator_value(symbol, "EMA")
            if ema_value is not None:
                return Decimal(str(ema_value))

        # Fallback to manual calculation
        bars = self.get_bars(symbol)
        if len(bars) < window:
            return None

        if alpha is None:
            alpha = Decimal("2") / (window + 1)

        prices = [bar.close for bar in bars]
        ema = prices[0]

        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def calculate_rsi(self, symbol: str, window: int = 14) -> Optional[Decimal]:
        """Calculate Relative Strength Index - uses indicator system if available."""
        # Try using indicator system first
        if self._indicators_available:
            rsi_value = self.get_indicator_value(symbol, "RSI")
            if rsi_value is not None:
                return Decimal(str(rsi_value))

        # Fallback to manual calculation
        bars = self.get_bars(symbol, window + 1)
        if len(bars) < window + 1:
            return None

        gains = []
        losses = []

        for i in range(1, len(bars)):
            change = bars[i].close - bars[i - 1].close
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))

        if len(gains) < window:
            return None

        avg_gain = sum(gains[-window:]) / window
        avg_loss = sum(losses[-window:]) / window

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_bollinger_bands(
        self, symbol: str, window: int = 20, std_dev: int = 2
    ) -> Optional[Dict[str, Decimal]]:
        """Calculate Bollinger Bands - uses indicator system if available."""
        # Try using indicator system first
        if self._indicators_available:
            bbands_value = self.get_indicator_value(symbol, "BBANDS")
            if bbands_value and isinstance(bbands_value, dict):
                return {
                    "upper": Decimal(str(bbands_value["upper"])),
                    "middle": Decimal(str(bbands_value["middle"])),
                    "lower": Decimal(str(bbands_value["lower"])),
                }

        # Fallback to manual calculation
        bars = self.get_bars(symbol, window)
        if len(bars) < window:
            return None

        prices = [bar.close for bar in bars[-window:]]
        sma = sum(prices) / len(prices)

        variance = sum((price - sma) ** 2 for price in prices) / len(prices)
        std = variance ** (Decimal("0.5"))

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        return {"upper": upper, "middle": sma, "lower": lower}

    def get_macd(self, symbol: str) -> Optional[Dict[str, Decimal]]:
        """Get MACD values - uses indicator system if available."""
        if self._indicators_available:
            macd_value = self.get_indicator_value(symbol, "MACD")
            if macd_value and isinstance(macd_value, dict):
                return {
                    "macd": Decimal(str(macd_value["macd"])),
                    "signal": Decimal(str(macd_value["signal"])),
                    "histogram": Decimal(str(macd_value["histogram"])),
                }
        return None

    def get_atr(self, symbol: str) -> Optional[Decimal]:
        """Get Average True Range value - uses indicator system if available."""
        if self._indicators_available:
            atr_value = self.get_indicator_value(symbol, "ATR")
            if atr_value is not None:
                return Decimal(str(atr_value))
        return None

    def get_stochastic(self, symbol: str) -> Optional[Dict[str, Decimal]]:
        """Get Stochastic oscillator values - uses indicator system if available."""
        if self._indicators_available:
            stoch_value = self.get_indicator_value(symbol, "STOCH")
            if stoch_value and isinstance(stoch_value, dict):
                return {
                    "k": Decimal(str(stoch_value["%K"])),
                    "d": Decimal(str(stoch_value["%D"])),
                }
        return None

    def create_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float = 1.0,
        price: Optional[Decimal] = None,
        quantity: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StrategySignal:
        """Create a strategy signal with enhanced metadata."""
        if price is None:
            price = self.get_latest_price(symbol)

        # Add technical indicators to metadata
        enhanced_metadata = metadata or {}

        # Add composite signals if available
        composite_signals = self.get_composite_signals(symbol)
        if composite_signals:
            enhanced_metadata["technical_analysis"] = composite_signals

        # Add individual indicator values if available
        if self._indicators_available:
            try:
                indicator_values = {}
                for indicator_name in ["SMA", "EMA", "RSI", "MACD", "BBANDS"]:
                    value = self.get_indicator_value(symbol, indicator_name)
                    if value is not None:
                        indicator_values[indicator_name] = value

                if indicator_values:
                    enhanced_metadata["indicators"] = indicator_values
            except Exception:
                pass

        signal = StrategySignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price=price,
            quantity=quantity,
            timestamp=datetime.now(timezone.utc),
            strategy_name=self.name,
            metadata=enhanced_metadata,
        )

        self.signals_generated += 1
        self.last_signal_time = signal.timestamp

        return signal

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        metrics = {
            "signals_generated": self.signals_generated,
            "last_signal_time": (
                self.last_signal_time.isoformat() if self.last_signal_time else None
            ),
            "symbols_tracked": len(self.symbols),
            "enabled": self.enabled,
            "indicators_available": self._indicators_available,
        }

        if self._indicators_available and self.indicator_manager:
            try:
                metrics["available_indicators"] = (
                    self.indicator_manager.get_available_indicators()
                )
                metrics["indicators_per_symbol"] = {
                    symbol: list(indicators.keys())
                    for symbol, indicators in self.indicator_manager.indicators.items()
                }
            except Exception:
                pass

        return metrics
