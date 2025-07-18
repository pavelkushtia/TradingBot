"""Test the working technical indicators implementation."""

import os
import sys
from datetime import datetime, timezone
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.core.models import MarketData
from trading_bot.indicators import IndicatorManager


def create_test_market_data(prices, volumes=None):
    """Create test market data from price list."""
    data = []
    if volumes is None:
        volumes = [1000] * len(prices)

    for i, (price, volume) in enumerate(zip(prices, volumes)):
        # Create realistic OHLC from close price
        high = price * 1.02  # 2% higher
        low = price * 0.98  # 2% lower
        open_price = price * 1.01 if i % 2 == 0 else price * 0.99

        bar = MarketData(
            symbol="TEST",
            timestamp=datetime.now(timezone.utc),
            open=Decimal(str(open_price)),
            high=Decimal(str(high)),
            low=Decimal(str(low)),
            close=Decimal(str(price)),
            volume=volume,
            vwap=Decimal(str(price)),
        )
        data.append(bar)

    return data


class TestIndicatorManager:
    """Test the IndicatorManager functionality."""

    def test_indicator_manager_creation(self):
        """Test that IndicatorManager can be created."""
        manager = IndicatorManager()
        assert manager is not None

        available = manager.get_available_indicators()
        assert "SMA" in available
        assert "EMA" in available
        assert "RSI" in available
        assert "MACD" in available
        assert "BBANDS" in available
        print("✅ IndicatorManager created successfully")

    def test_add_indicators(self):
        """Test adding indicators to the manager."""
        manager = IndicatorManager()

        # Test adding SMA
        result = manager.add_indicator("TEST", "SMA", period=20)
        assert result == True

        # Test adding RSI
        result = manager.add_indicator("TEST", "RSI", period=14)
        assert result == True

        # Test adding invalid indicator
        result = manager.add_indicator("TEST", "INVALID")
        assert result == False

        print("✅ Indicators added successfully")

    def test_sma_calculation(self):
        """Test SMA calculation."""
        manager = IndicatorManager()
        manager.add_indicator("TEST", "SMA", period=5)

        # Test data: [100, 101, 102, 103, 104]
        prices = [100, 101, 102, 103, 104]
        test_data = create_test_market_data(prices)

        results = {}
        for bar in test_data:
            results = manager.update_indicators("TEST", bar)

        # SMA of last 5 values should be 102
        assert "SMA" in results
        expected_sma = sum(prices) / len(prices)  # 102
        assert abs(results["SMA"] - expected_sma) < 0.01

        print("✅ SMA calculation working")

    def test_rsi_calculation(self):
        """Test RSI calculation."""
        manager = IndicatorManager()
        manager.add_indicator("TEST", "RSI", period=14)

        # Create uptrend data
        prices = [100 + i for i in range(20)]  # 100, 101, 102, ..., 119
        test_data = create_test_market_data(prices)

        results = {}
        for bar in test_data:
            results = manager.update_indicators("TEST", bar)

        # RSI should be high (>50) for uptrend
        assert "RSI" in results
        assert results["RSI"] > 50
        assert results["RSI"] <= 100

        print("✅ RSI calculation working")

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        manager = IndicatorManager()
        manager.add_indicator("TEST", "BBANDS", period=10, std_dev=2.0)

        # Use consistent prices to test bands
        prices = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100]
        test_data = create_test_market_data(prices)

        results = {}
        for bar in test_data:
            results = manager.update_indicators("TEST", bar)

        # Should have upper, middle, lower bands
        assert "BBANDS" in results
        bbands = results["BBANDS"]
        assert "upper" in bbands
        assert "middle" in bbands
        assert "lower" in bbands

        # Upper should be > middle > lower
        assert bbands["upper"] > bbands["middle"]
        assert bbands["middle"] > bbands["lower"]

        print("✅ Bollinger Bands calculation working")

    def test_multiple_indicators(self):
        """Test multiple indicators working together."""
        manager = IndicatorManager()

        # Add multiple indicators
        manager.add_indicator("TEST", "SMA", period=10)
        manager.add_indicator("TEST", "EMA", period=10)
        manager.add_indicator("TEST", "RSI", period=14)
        manager.add_indicator("TEST", "BBANDS", period=20)

        # Create enough data for all indicators
        prices = [100 + (i % 10) for i in range(25)]  # Oscillating prices
        test_data = create_test_market_data(prices)

        results = {}
        for bar in test_data:
            results = manager.update_indicators("TEST", bar)

        # All indicators should produce results
        assert "SMA" in results
        assert "EMA" in results
        assert "RSI" in results
        assert "BBANDS" in results

        print("✅ Multiple indicators working together")

    def test_get_indicator_value(self):
        """Test getting specific indicator values."""
        manager = IndicatorManager()
        manager.add_indicator("TEST", "SMA", period=5)
        manager.add_indicator("TEST", "RSI", period=14)

        # Add enough data
        prices = [100 + i for i in range(20)]
        test_data = create_test_market_data(prices)

        for bar in test_data:
            manager.update_indicators("TEST", bar)

        # Test getting individual values
        sma_value = manager.get_indicator_value("TEST", "SMA")
        rsi_value = manager.get_indicator_value("TEST", "RSI")

        assert sma_value is not None
        assert rsi_value is not None
        assert isinstance(sma_value, float)
        assert isinstance(rsi_value, float)

        print("✅ Get indicator value working")


def test_indicators_integration():
    """Integration test - verify indicators work like in real strategies."""
    manager = IndicatorManager()

    # Set up indicators like a real strategy would
    symbol = "AAPL"
    manager.add_indicator(symbol, "SMA", period=20)
    manager.add_indicator(symbol, "RSI", period=14)
    manager.add_indicator(symbol, "BBANDS", period=20, std_dev=2.0)

    # Simulate real market data
    base_price = 150.0
    prices = []
    for i in range(30):
        # Add some randomness to simulate real data
        price_change = (i % 3 - 1) * 0.5  # -0.5, 0, 0.5
        base_price += price_change
        prices.append(base_price)

    test_data = create_test_market_data(prices)

    # Process data like a strategy would
    indicator_results = {}
    for bar in test_data:
        results = manager.update_indicators(symbol, bar)
        if results:
            indicator_results = results

    # Verify we get results for all indicators
    assert "SMA" in indicator_results
    assert "RSI" in indicator_results
    assert "BBANDS" in indicator_results

    # Verify values are reasonable
    sma = indicator_results["SMA"]
    rsi = indicator_results["RSI"]
    bbands = indicator_results["BBANDS"]

    assert 140 < sma < 160  # Should be close to our base price
    assert 0 <= rsi <= 100  # RSI should be in valid range
    assert bbands["upper"] > bbands["middle"] > bbands["lower"]

    print("✅ Integration test passed - indicators work like in real strategies")


if __name__ == "__main__":
    # Run all tests
    test_manager = TestIndicatorManager()

    test_manager.test_indicator_manager_creation()
    test_manager.test_add_indicators()
    test_manager.test_sma_calculation()
    test_manager.test_rsi_calculation()
    test_manager.test_bollinger_bands()
    test_manager.test_multiple_indicators()
    test_manager.test_get_indicator_value()

    test_indicators_integration()

    print("\n🎉 All technical indicator tests passed!")
    print("✅ Advanced Technical Indicators Library is working properly")
    print("✅ Ready for integration with existing strategies")
