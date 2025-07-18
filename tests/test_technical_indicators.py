"""Tests for technical indicators system."""

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from trading_bot.core.models import MarketData
from trading_bot.indicators import (ATR, EMA, MACD, OBV, RSI, SMA, WMA,
                                    BollingerBands, IndicatorConfig,
                                    IndicatorManager, Stochastic, WilliamsR)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    data = []
    base_price = 100.0

    for i in range(50):
        # Create some realistic price movement
        price_change = np.random.normal(0, 0.02)  # 2% volatility
        base_price *= 1 + price_change

        high = base_price * (1 + abs(np.random.normal(0, 0.01)))
        low = base_price * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(np.random.uniform(100000, 1000000))

        bar = MarketData(
            symbol="TEST",
            timestamp=datetime.now(timezone.utc),
            open=Decimal(str(base_price)),
            high=Decimal(str(high)),
            low=Decimal(str(low)),
            close=Decimal(str(base_price)),
            volume=volume,
            vwap=Decimal(str(base_price)),
        )
        data.append(bar)

    return data


class TestIndicatorManager:
    """Test the indicator manager."""

    def test_indicator_manager_creation(self):
        """Test indicator manager can be created."""
        manager = IndicatorManager()
        assert manager is not None
        assert len(manager.get_available_indicators()) > 0
        assert "SMA" in manager.get_available_indicators()
        assert "RSI" in manager.get_available_indicators()

    def test_add_indicator(self):
        """Test adding indicators to manager."""
        manager = IndicatorManager()
        config = IndicatorConfig(period=20)

        manager.add_indicator("TEST", "SMA", config)
        assert "TEST" in manager.indicators
        assert "SMA" in manager.indicators["TEST"]

    def test_update_indicators(self, sample_market_data):
        """Test updating indicators with market data."""
        manager = IndicatorManager()
        config = IndicatorConfig(period=10)

        manager.add_indicator("TEST", "SMA", config)
        manager.add_indicator("TEST", "RSI", config)

        results = {}
        for bar in sample_market_data:
            results = manager.update_indicators("TEST", bar)

        # Should have results after enough data
        assert "SMA" in results
        assert "RSI" in results
        assert isinstance(results["SMA"], float)
        assert isinstance(results["RSI"], float)


class TestSMA:
    """Test Simple Moving Average indicator."""

    def test_sma_calculation(self, sample_market_data):
        """Test SMA calculation accuracy."""
        config = IndicatorConfig(period=10)
        sma = SMA(config)

        # Add data
        result = None
        for bar in sample_market_data:
            result = sma.update(bar)

        assert result is not None
        assert result.name == "SMA"
        assert isinstance(result.value, float)
        assert result.value > 0

        # Verify calculation manually
        last_10_closes = [float(bar.close) for bar in sample_market_data[-10:]]
        expected_sma = sum(last_10_closes) / 10

        assert abs(result.value - expected_sma) < 0.001


class TestEMA:
    """Test Exponential Moving Average indicator."""

    def test_ema_calculation(self, sample_market_data):
        """Test EMA calculation."""
        config = IndicatorConfig(period=10)
        ema = EMA(config)

        result = None
        for bar in sample_market_data:
            result = ema.update(bar)

        assert result is not None
        assert result.name == "EMA"
        assert isinstance(result.value, float)
        assert result.value > 0


class TestRSI:
    """Test Relative Strength Index indicator."""

    def test_rsi_calculation(self, sample_market_data):
        """Test RSI calculation."""
        config = IndicatorConfig(period=14)
        rsi = RSI(config)

        result = None
        for bar in sample_market_data:
            result = rsi.update(bar)

        assert result is not None
        assert result.name == "RSI"
        assert isinstance(result.value, float)
        assert 0 <= result.value <= 100


class TestMACD:
    """Test MACD indicator."""

    def test_macd_calculation(self, sample_market_data):
        """Test MACD calculation."""
        config = IndicatorConfig(
            period=26,  # Use slow period as base period
            parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9},
        )
        macd = MACD(config)

        result = None
        for bar in sample_market_data:
            result = macd.update(bar)

        assert result is not None
        assert result.name == "MACD"
        assert isinstance(result.value, dict)
        assert "macd" in result.value
        assert "signal" in result.value
        assert "histogram" in result.value


class TestBollingerBands:
    """Test Bollinger Bands indicator."""

    def test_bollinger_bands_calculation(self, sample_market_data):
        """Test Bollinger Bands calculation."""
        config = IndicatorConfig(period=20, parameters={"std_dev": 2.0})
        bbands = BollingerBands(config)

        result = None
        for bar in sample_market_data:
            result = bbands.update(bar)

        assert result is not None
        assert result.name == "BBANDS"
        assert isinstance(result.value, dict)
        assert "upper" in result.value
        assert "middle" in result.value
        assert "lower" in result.value

        # Upper should be higher than middle, which should be higher than lower
        assert result.value["upper"] > result.value["middle"]
        assert result.value["middle"] > result.value["lower"]


class TestATR:
    """Test Average True Range indicator."""

    def test_atr_calculation(self, sample_market_data):
        """Test ATR calculation."""
        config = IndicatorConfig(period=14)
        atr = ATR(config)

        result = None
        for bar in sample_market_data:
            result = atr.update(bar)

        assert result is not None
        assert result.name == "ATR"
        assert isinstance(result.value, float)
        assert result.value >= 0


class TestStochastic:
    """Test Stochastic Oscillator indicator."""

    def test_stochastic_calculation(self, sample_market_data):
        """Test Stochastic calculation."""
        config = IndicatorConfig(period=14, parameters={"k_period": 14, "d_period": 3})
        stoch = Stochastic(config)

        result = None
        for bar in sample_market_data:
            result = stoch.update(bar)

        assert result is not None
        assert result.name == "STOCH"
        assert isinstance(result.value, dict)
        assert "%K" in result.value
        assert "%D" in result.value

        # Values should be between 0 and 100
        assert 0 <= result.value["%K"] <= 100
        assert 0 <= result.value["%D"] <= 100


class TestWMA:
    """Test Weighted Moving Average indicator."""

    def test_wma_calculation(self, sample_market_data):
        """Test WMA calculation accuracy."""
        config = IndicatorConfig(period=10)
        wma = WMA(config)

        result = None
        for bar in sample_market_data:
            result = wma.update(bar)

        assert result is not None
        assert result.name == "WMA"
        assert isinstance(result.value, float)
        assert result.value > 0

        # Verify calculation manually
        last_10_closes = [float(bar.close) for bar in sample_market_data[-10:]]
        weights = list(range(1, 11))
        expected_wma = sum(x * w for x, w in zip(last_10_closes, weights)) / sum(
            weights
        )

        assert abs(result.value - expected_wma) < 0.001


class TestWilliamsR:
    """Test Williams %R indicator."""

    def test_williams_r_calculation(self, sample_market_data):
        """Test Williams %R calculation."""
        config = IndicatorConfig(period=14)
        willr = WilliamsR(config)

        result = None
        for bar in sample_market_data:
            result = willr.update(bar)

        assert result is not None
        assert result.name == "WILLR"
        assert isinstance(result.value, float)
        assert -100 <= result.value <= 0


class TestOBV:
    """Test On-Balance Volume indicator."""

    def test_obv_calculation(self, sample_market_data):
        """Test OBV calculation."""
        obv = OBV()

        result = None
        for bar in sample_market_data:
            result = obv.update(bar)

        assert result is not None
        assert result.name == "OBV"
        assert isinstance(result.value, float)

        # Manual calculation for verification
        manual_obv = 0
        for i in range(1, len(sample_market_data)):
            if sample_market_data[i].close > sample_market_data[i - 1].close:
                manual_obv += float(sample_market_data[i].volume)
            elif sample_market_data[i].close < sample_market_data[i - 1].close:
                manual_obv -= float(sample_market_data[i].volume)

        assert abs(result.value - manual_obv) < 1  # Allow for small float differences


class TestIndicatorComposite:
    """Test composite indicator functionality."""

    def test_composite_signals(self, sample_market_data):
        """Test composite signal generation."""
        manager = IndicatorManager()

        # Add multiple indicators
        config = IndicatorConfig(period=14)
        manager.add_indicator("TEST", "SMA", config)
        manager.add_indicator("TEST", "RSI", config)
        manager.add_indicator("TEST", "MACD", config)

        # Update with data
        for bar in sample_market_data:
            manager.update_indicators("TEST", bar)

        # Get composite signals
        signals = manager.calculate_composite_signals("TEST")

        assert "trend_strength" in signals
        assert "momentum_strength" in signals
        assert "volatility_regime" in signals
        assert "overall_signal" in signals

        assert signals["overall_signal"] in ["bullish", "bearish", "neutral"]
        assert signals["volatility_regime"] in ["high", "normal", "low"]


class TestIndicatorConfiguration:
    """Test indicator configuration options."""

    def test_different_sources(self, sample_market_data):
        """Test indicators with different data sources."""
        # Test with different price sources
        sources = ["open", "high", "low", "close", "hlc3", "hl2", "ohlc4"]

        for source in sources:
            config = IndicatorConfig(period=10, source=source)
            sma = SMA(config)

            result = None
            for bar in sample_market_data:
                result = sma.update(bar)

            assert result is not None
            assert result.value > 0

    def test_different_periods(self, sample_market_data):
        """Test indicators with different periods."""
        periods = [5, 10, 20, 50]

        for period in periods:
            if period <= len(sample_market_data):
                config = IndicatorConfig(period=period)
                sma = SMA(config)

                result = None
                for bar in sample_market_data:
                    result = sma.update(bar)

                if len(sample_market_data) >= period:
                    assert result is not None
                    assert result.value > 0


class TestIndicatorRobustness:
    """Test indicator robustness and error handling."""

    def test_insufficient_data(self):
        """Test indicators with insufficient data."""
        config = IndicatorConfig(period=20)
        sma = SMA(config)

        # Add only a few data points
        for i in range(5):
            bar = MarketData(
                symbol="TEST",
                timestamp=datetime.now(timezone.utc),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=10000,
                vwap=Decimal("100"),
            )
            result = sma.update(bar)
            # Should return None until we have enough data
            assert result is None

    def test_zero_values(self):
        """Test indicators with zero values."""
        config = IndicatorConfig(period=5)
        rsi = RSI(config)

        # Add data with no price movement (should not crash)
        for i in range(10):
            bar = MarketData(
                symbol="TEST",
                timestamp=datetime.now(timezone.utc),
                open=Decimal("100"),
                high=Decimal("100"),
                low=Decimal("100"),
                close=Decimal("100"),
                volume=10000,
                vwap=Decimal("100"),
            )
            result = rsi.update(bar)

        # Should handle zero returns gracefully
        assert result is not None
        assert isinstance(result.value, float)


if __name__ == "__main__":
    pytest.main([__file__])
