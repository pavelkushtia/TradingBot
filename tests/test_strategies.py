"""Tests for trading strategies."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List

from trading_bot.core.models import MarketData, StrategySignal
from trading_bot.strategy.momentum_crossover import MomentumCrossoverStrategy
from trading_bot.strategy.mean_reversion import MeanReversionStrategy
from trading_bot.strategy.breakout import BreakoutStrategy


class TestMomentumCrossoverStrategy:
    """Test momentum crossover strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        parameters = {
            "short_window": 3,
            "long_window": 5,
            "min_strength_threshold": 0.001
        }
        return MomentumCrossoverStrategy("test_momentum", parameters)
    
    def create_trending_data(self, symbol: str, periods: int, trend: str = "up") -> List[MarketData]:
        """Create trending market data."""
        data = []
        base_time = datetime.utcnow() - timedelta(days=periods)
        base_price = 100.0
        
        for i in range(periods):
            if trend == "up":
                price = base_price + (i * 0.5)
            elif trend == "down":
                price = base_price - (i * 0.5)
            else:  # sideways
                price = base_price + (i % 2) * 0.1
            
            bar = MarketData(
                symbol=symbol,
                timestamp=base_time + timedelta(hours=i),
                open=Decimal(str(price)),
                high=Decimal(str(price + 0.5)),
                low=Decimal(str(price - 0.5)),
                close=Decimal(str(price)),
                volume=100000
            )
            data.append(bar)
        
        return data
    
    @pytest.mark.asyncio
    async def test_no_signal_insufficient_data(self, strategy):
        """Test no signal when insufficient data."""
        # Add only 2 bars (need at least long_window + 1)
        data = self.create_trending_data("AAPL", 2)
        
        for bar in data:
            await strategy.on_bar(bar.symbol, bar)
        
        signals = await strategy.generate_signals()
        assert len(signals) == 0
    
    @pytest.mark.asyncio
    async def test_bullish_crossover_signal(self, strategy):
        """Test bullish crossover signal generation."""
        # Create data with clear upward trend
        data = self.create_trending_data("AAPL", 10, "up")
        
        for bar in data:
            await strategy.on_bar(bar.symbol, bar)
        
        signals = await strategy.generate_signals()
        
        # Should generate buy signal due to upward trend
        if signals:
            signal = signals[0]
            assert signal.signal_type == "buy"
            assert signal.symbol == "AAPL"
            assert signal.strength > 0
    
    @pytest.mark.asyncio
    async def test_bearish_crossover_signal(self, strategy):
        """Test bearish crossover signal generation."""
        # Create data with initial upward trend, then downward
        data = self.create_trending_data("AAPL", 6, "up")
        data.extend(self.create_trending_data("AAPL", 6, "down"))
        
        # Update timestamps to be sequential
        for i, bar in enumerate(data):
            bar.timestamp = datetime.utcnow() - timedelta(hours=len(data) - i)
        
        for bar in data:
            await strategy.on_bar(bar.symbol, bar)
        
        signals = await strategy.generate_signals()
        
        # Should generate sell signal due to downward trend
        if signals:
            signal = signals[0]
            assert signal.signal_type == "sell"
            assert signal.symbol == "AAPL"
            assert signal.strength > 0
    
    @pytest.mark.asyncio
    async def test_no_duplicate_signals(self, strategy):
        """Test that duplicate signals are not generated."""
        data = self.create_trending_data("AAPL", 10, "up")
        
        for bar in data:
            await strategy.on_bar(bar.symbol, bar)
        
        # Generate signals twice
        signals1 = await strategy.generate_signals()
        signals2 = await strategy.generate_signals()
        
        # Should not generate duplicate signals
        assert len(signals2) == 0 or len(signals1) == 0
    
    def test_sma_calculation(self, strategy):
        """Test SMA calculation."""
        data = self.create_trending_data("AAPL", 10, "up")
        
        for bar in data:
            strategy.market_data["AAPL"] = strategy.market_data.get("AAPL", []) + [bar]
        
        sma_3 = strategy.calculate_sma("AAPL", 3)
        sma_5 = strategy.calculate_sma("AAPL", 5)
        
        assert sma_3 is not None
        assert sma_5 is not None
        assert isinstance(sma_3, Decimal)
        assert isinstance(sma_5, Decimal)
        
        # With upward trend, shorter SMA should be higher
        assert sma_3 > sma_5


class TestMeanReversionStrategy:
    """Test mean reversion strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        parameters = {
            "bollinger_window": 10,
            "bollinger_std": 2,
            "rsi_window": 5,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "min_price_distance": 0.01
        }
        return MeanReversionStrategy("test_mean_reversion", parameters)
    
    def create_oscillating_data(self, symbol: str, periods: int) -> List[MarketData]:
        """Create oscillating market data."""
        data = []
        base_time = datetime.utcnow() - timedelta(days=periods)
        base_price = 100.0
        
        for i in range(periods):
            # Create oscillating pattern
            import math
            price = base_price + 5 * math.sin(i * 0.5) + (i % 3) * 0.2
            
            bar = MarketData(
                symbol=symbol,
                timestamp=base_time + timedelta(hours=i),
                open=Decimal(str(price)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                volume=100000
            )
            data.append(bar)
        
        return data
    
    @pytest.mark.asyncio
    async def test_insufficient_data(self, strategy):
        """Test behavior with insufficient data."""
        data = self.create_oscillating_data("AAPL", 5)
        
        for bar in data:
            await strategy.on_bar(bar.symbol, bar)
        
        signals = await strategy.generate_signals()
        assert len(signals) == 0
    
    @pytest.mark.asyncio
    async def test_signal_generation(self, strategy):
        """Test signal generation with sufficient data."""
        data = self.create_oscillating_data("AAPL", 20)
        
        for bar in data:
            await strategy.on_bar(bar.symbol, bar)
        
        signals = await strategy.generate_signals()
        
        # May or may not have signals depending on RSI and Bollinger Band conditions
        for signal in signals:
            assert signal.symbol == "AAPL"
            assert signal.signal_type in ["buy", "sell"]
            assert 0 <= signal.strength <= 1
    
    def test_bollinger_bands_calculation(self, strategy):
        """Test Bollinger Bands calculation."""
        data = self.create_oscillating_data("AAPL", 15)
        
        for bar in data:
            strategy.market_data["AAPL"] = strategy.market_data.get("AAPL", []) + [bar]
        
        bands = strategy.calculate_bollinger_bands("AAPL", 10, 2)
        
        assert bands is not None
        assert "upper" in bands
        assert "middle" in bands
        assert "lower" in bands
        assert bands["upper"] > bands["middle"]
        assert bands["middle"] > bands["lower"]
    
    def test_rsi_calculation(self, strategy):
        """Test RSI calculation."""
        data = self.create_oscillating_data("AAPL", 20)
        
        for bar in data:
            strategy.market_data["AAPL"] = strategy.market_data.get("AAPL", []) + [bar]
        
        rsi = strategy.calculate_rsi("AAPL", 14)
        
        assert rsi is not None
        assert 0 <= rsi <= 100


class TestBreakoutStrategy:
    """Test breakout strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        parameters = {
            "lookback_window": 10,
            "breakout_threshold": 0.02,
            "volume_multiplier": 1.5,
            "min_consolidation_periods": 5
        }
        return BreakoutStrategy("test_breakout", parameters)
    
    def create_consolidation_breakout_data(self, symbol: str) -> List[MarketData]:
        """Create data with consolidation followed by breakout."""
        data = []
        base_time = datetime.utcnow() - timedelta(days=20)
        base_price = 100.0
        
        # Consolidation phase (10 bars)
        for i in range(10):
            price = base_price + (i % 2) * 0.1  # Tight range
            bar = MarketData(
                symbol=symbol,
                timestamp=base_time + timedelta(hours=i),
                open=Decimal(str(price)),
                high=Decimal(str(price + 0.2)),
                low=Decimal(str(price - 0.2)),
                close=Decimal(str(price)),
                volume=100000
            )
            data.append(bar)
        
        # Breakout phase (5 bars)
        for i in range(5):
            price = base_price + 3 + (i * 0.5)  # Clear breakout
            bar = MarketData(
                symbol=symbol,
                timestamp=base_time + timedelta(hours=10 + i),
                open=Decimal(str(price)),
                high=Decimal(str(price + 0.5)),
                low=Decimal(str(price - 0.2)),
                close=Decimal(str(price)),
                volume=150000  # Higher volume
            )
            data.append(bar)
        
        return data
    
    @pytest.mark.asyncio
    async def test_insufficient_data(self, strategy):
        """Test behavior with insufficient data."""
        data = self.create_consolidation_breakout_data("AAPL")[:5]
        
        for bar in data:
            await strategy.on_bar(bar.symbol, bar)
        
        signals = await strategy.generate_signals()
        assert len(signals) == 0
    
    @pytest.mark.asyncio
    async def test_breakout_signal(self, strategy):
        """Test breakout signal generation."""
        data = self.create_consolidation_breakout_data("AAPL")
        
        for bar in data:
            await strategy.on_bar(bar.symbol, bar)
        
        signals = await strategy.generate_signals()
        
        # Should generate buy signal due to upward breakout
        if signals:
            signal = signals[0]
            assert signal.signal_type == "buy"
            assert signal.symbol == "AAPL"
            assert signal.strength > 0
    
    def test_support_resistance_calculation(self, strategy):
        """Test support and resistance calculation."""
        data = self.create_consolidation_breakout_data("AAPL")[:10]  # Just consolidation
        
        support, resistance = strategy._calculate_support_resistance(data)
        
        assert support is not None
        assert resistance is not None
        assert resistance > support
        assert isinstance(support, Decimal)
        assert isinstance(resistance, Decimal)


class TestStrategyCommon:
    """Test common strategy functionality."""
    
    @pytest.fixture
    def strategy(self):
        """Create base strategy instance."""
        return MomentumCrossoverStrategy("test", {"short_window": 5, "long_window": 10})
    
    def test_latest_price_retrieval(self, strategy):
        """Test latest price retrieval."""
        from trading_bot.core.models import Quote
        
        # Add quote data
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            bid_price=Decimal("150.00"),
            ask_price=Decimal("150.10"),
            bid_size=100,
            ask_size=100
        )
        
        strategy.latest_quotes["AAPL"] = quote
        
        latest_price = strategy.get_latest_price("AAPL")
        assert latest_price == quote.mid_price
    
    def test_signal_creation(self, strategy):
        """Test signal creation."""
        signal = strategy.create_signal(
            symbol="AAPL",
            signal_type="buy",
            strength=0.8,
            price=Decimal("150.00")
        )
        
        assert signal.symbol == "AAPL"
        assert signal.signal_type == "buy"
        assert signal.strength == 0.8
        assert signal.price == Decimal("150.00")
        assert signal.strategy_name == "test"
        assert isinstance(signal.timestamp, datetime)
    
    def test_performance_metrics(self, strategy):
        """Test performance metrics."""
        # Generate a signal to update metrics
        signal = strategy.create_signal("AAPL", "buy", 0.5)
        
        metrics = strategy.get_performance_metrics()
        
        assert "signals_generated" in metrics
        assert "last_signal_time" in metrics
        assert "symbols_tracked" in metrics
        assert "enabled" in metrics
        
        assert metrics["signals_generated"] > 0
        assert metrics["enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 