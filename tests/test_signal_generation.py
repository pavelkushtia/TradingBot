"""Test signal generation in momentum crossover strategy."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.core.models import MarketData
from trading_bot.strategy.momentum_crossover import MomentumCrossoverStrategy


def create_crossover_data(symbol: str) -> list[MarketData]:
    """Create data that should trigger crossover signals."""
    base_time = datetime.utcnow() - timedelta(days=50)
    data = []

    base_price = Decimal("100.0")

    # Create 50 days of data with clear crossover pattern
    for i in range(50 * 24):  # Hourly data
        timestamp = base_time + timedelta(hours=i)

        # Create pattern: decline -> rise -> decline for clear crossovers
        if i < 400:  # First ~17 days: decline
            price = base_price - (Decimal(str(i)) * Decimal("0.1"))
        elif i < 800:  # Next ~17 days: rise
            price = (
                base_price - Decimal("40.0") + (Decimal(str(i - 400)) * Decimal("0.2"))
            )
        else:  # Last ~16 days: decline
            price = (
                base_price + Decimal("40.0") - (Decimal(str(i - 800)) * Decimal("0.15"))
            )

        bar = MarketData(
            symbol=symbol,
            timestamp=timestamp,
            open=price,
            high=price + Decimal("0.5"),
            low=price - Decimal("0.5"),
            close=price,
            volume=Decimal("1000"),
            vwap=price,
        )
        data.append(bar)

    return data


async def test_signal_generation() -> list:
    """Test that signals are generated correctly."""
    print("=" * 60)
    print("TESTING SIGNAL GENERATION")
    print("=" * 60)

    # Create strategy
    strategy = MomentumCrossoverStrategy(
        "momentum_crossover",
        {"short_window": 10, "long_window": 30, "min_strength_threshold": 0.01},
    )

    # Initialize strategy
    await strategy.initialize()

    # Create test data
    data = create_crossover_data("AAPL")

    # Feed data to strategy
    signals_generated = []

    for bar in data:
        await strategy.on_bar(bar.symbol, bar)

        # Check for signals after we have enough data
        if len(strategy.market_data.get("AAPL", [])) >= 35:  # Need 30+ for long MA
            signals = await strategy.generate_signals()
            if signals:
                signals_generated.extend(signals)
                print(
                    f"Signal at {bar.timestamp}: {signals[0].signal_type} - Strength: {signals[0].strength:.3f}"
                )

    print(f"\nTotal signals generated: {len(signals_generated)}")

    # Analyze signals
    buy_signals = [s for s in signals_generated if s.signal_type == "buy"]
    sell_signals = [s for s in signals_generated if s.signal_type == "sell"]

    print(f"Buy signals: {len(buy_signals)}")
    print(f"Sell signals: {len(sell_signals)}")

    # Test moving averages calculation
    print("\nTesting Moving Averages:")
    print("-" * 30)

    if len(strategy.market_data.get("AAPL", [])) >= 30:
        short_ma = strategy.calculate_sma("AAPL", 10)
        long_ma = strategy.calculate_sma("AAPL", 30)

        print(f"Short MA (10): {short_ma}")
        print(f"Long MA (30): {long_ma}")
        print(f"Latest price: {strategy.get_latest_price('AAPL')}")

        if short_ma and long_ma:
            print(f"Short > Long: {short_ma > long_ma}")

    return signals_generated


if __name__ == "__main__":
    asyncio.run(test_signal_generation())
