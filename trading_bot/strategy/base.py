"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from ..core.models import MarketData, Quote, StrategySignal


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

        self.market_data[symbol].append(data)

        # Keep only recent data (configurable)
        max_bars = self.parameters.get("max_bars", 1000)
        if len(self.market_data[symbol]) > max_bars:
            self.market_data[symbol] = self.market_data[symbol][-max_bars:]

        self.symbols.add(symbol)

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
        """Calculate Simple Moving Average."""
        bars = self.get_bars(symbol, window)
        if len(bars) < window:
            return None

        total = sum(bar.close for bar in bars[-window:])
        return total / window

    def calculate_ema(
        self, symbol: str, window: int, alpha: Optional[Decimal] = None
    ) -> Optional[Decimal]:
        """Calculate Exponential Moving Average."""
        bars = self.get_bars(symbol)
        if len(bars) < window:
            return None

        if alpha is None:
            alpha = Decimal("2") / (window + 1)

        ema = bars[0].close
        for bar in bars[1:]:
            ema = alpha * bar.close + (1 - alpha) * ema

        return ema

    def calculate_rsi(self, symbol: str, window: int = 14) -> Optional[Decimal]:
        """Calculate Relative Strength Index."""
        returns = self.calculate_returns(symbol, 1)
        if len(returns) < window:
            return None

        gains = [max(ret, Decimal("0")) for ret in returns[-window:]]
        losses = [abs(min(ret, Decimal("0"))) for ret in returns[-window:]]

        avg_gain = sum(gains) / window
        avg_loss = sum(losses) / window

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_bollinger_bands(
        self, symbol: str, window: int = 20, std_dev: int = 2
    ) -> Optional[Dict[str, Decimal]]:
        """Calculate Bollinger Bands."""
        bars = self.get_bars(symbol, window)
        if len(bars) < window:
            return None

        prices = [bar.close for bar in bars[-window:]]
        sma = sum(prices) / window

        # Calculate standard deviation
        variance = sum((price - sma) ** 2 for price in prices) / window
        std = variance ** Decimal("0.5")

        return {
            "upper": sma + (std_dev * std),
            "middle": sma,
            "lower": sma - (std_dev * std),
        }

    def create_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float = 1.0,
        price: Optional[Decimal] = None,
        quantity: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StrategySignal:
        """Create a strategy signal."""
        if price is None:
            price = self.get_latest_price(symbol)

        signal = StrategySignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price=price,
            quantity=quantity,
            timestamp=datetime.utcnow(),
            strategy_name=self.name,
            metadata=metadata or {},
        )

        self.signals_generated += 1
        self.last_signal_time = signal.timestamp

        return signal

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        return {
            "signals_generated": self.signals_generated,
            "last_signal_time": (
                self.last_signal_time.isoformat() if self.last_signal_time else None
            ),
            "symbols_tracked": len(self.symbols),
            "enabled": self.enabled,
        }
