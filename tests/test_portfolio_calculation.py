"""Test portfolio calculation bug in backtest engine."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.config import Config
from trading_bot.core.models import MarketData
from trading_bot.strategy.momentum_crossover import MomentumCrossoverStrategy


def create_test_data_no_crossover(symbol: str, days: int = 15) -> list[MarketData]:
    """Create test data that won't trigger crossover signals."""
    base_time = datetime.utcnow() - timedelta(days=days)
    data = []

    # Create data with steady upward trend - no crossovers
    base_price = Decimal("100.0")

    for i in range(days * 24):  # Hourly data
        timestamp = base_time + timedelta(hours=i)

        # Steady upward trend - no crossovers
        price = base_price + (Decimal(str(i)) * Decimal("0.1"))

        bar = MarketData(
            symbol=symbol,
            timestamp=timestamp,
            open=price,
            high=price + Decimal("0.5"),
            low=price - Decimal("0.5"),
            close=price,
            volume=1000,
            vwap=price,
        )
        data.append(bar)

    return data


def create_test_data_with_crossover(symbol: str, days: int = 15) -> list[MarketData]:
    """Create test data that will trigger crossover signals."""
    base_time = datetime.utcnow() - timedelta(days=days)
    data = []

    base_price = Decimal("100.0")

    for i in range(days * 24):  # Hourly data
        timestamp = base_time + timedelta(hours=i)

        # Create pattern: decline -> sharp rise -> decline
        if i < 120:  # First 5 days: decline
            price = base_price - (Decimal(str(i)) * Decimal("0.2"))
        elif i < 240:  # Next 5 days: sharp rise
            price = (
                base_price - Decimal("24.0") + (Decimal(str(i - 120)) * Decimal("0.5"))
            )
        else:  # Last 5 days: decline
            price = (
                base_price + Decimal("36.0") - (Decimal(str(i - 240)) * Decimal("0.3"))
            )

        bar = MarketData(
            symbol=symbol,
            timestamp=timestamp,
            open=price,
            high=price + Decimal("0.5"),
            low=price - Decimal("0.5"),
            close=price,
            volume=1000,
            vwap=price,
        )
        data.append(bar)

    return data


async def test_portfolio_calculation_bug() -> tuple:
    """Test that demonstrates the portfolio calculation bug."""
    print("=" * 60)
    print("TESTING PORTFOLIO CALCULATION BUG")
    print("=" * 60)

    # Create config
    config = Config()
    event_bus = MagicMock()

    # Create backtest engine
    engine = BacktestEngine(config, event_bus)

    # Create strategy
    strategy = MomentumCrossoverStrategy(
        "momentum_crossover",
        {"short_window": 10, "long_window": 30, "min_strength_threshold": 0.01},
    )

    # Test 1: No crossover data (should show 0 trades, 0 return)
    print("\n1. Testing with NO CROSSOVER data:")
    print("-" * 40)

    no_crossover_data = create_test_data_no_crossover("AAPL", 15)
    start_date = datetime.utcnow() - timedelta(days=15)
    end_date = datetime.utcnow()

    results1 = await engine.run_backtest(
        strategy, no_crossover_data, start_date, end_date
    )

    print(f"Initial Capital: ${results1['initial_capital']:,.2f}")
    print(f"Final Capital: ${results1['final_capital']:,.2f}")
    print(f"Total Return: {results1['total_return']:.2%}")
    print(f"Total Trades: {results1['total_trades']}")
    print(f"Winning Trades: {results1['winning_trades']}")
    print(f"Losing Trades: {results1['losing_trades']}")

    # Check for bug: return without trades
    if results1["total_return"] != 0 and results1["total_trades"] == 0:
        print("🚨 BUG DETECTED: Non-zero return with zero trades!")
        print(f"   Return: {results1['total_return']:.2%}")
        print(f"   Trades: {results1['total_trades']}")
    else:
        print("✅ No bug detected in test 1")

    # Test 2: With crossover data (should show trades and matching returns)
    print("\n2. Testing with CROSSOVER data:")
    print("-" * 40)

    # Reset engine for second test
    engine = BacktestEngine(config, event_bus)
    strategy = MomentumCrossoverStrategy(
        "momentum_crossover",
        {"short_window": 10, "long_window": 30, "min_strength_threshold": 0.01},
    )

    crossover_data = create_test_data_with_crossover("AAPL", 15)

    results2 = await engine.run_backtest(strategy, crossover_data, start_date, end_date)

    print(f"Initial Capital: ${results2['initial_capital']:,.2f}")
    print(f"Final Capital: ${results2['final_capital']:,.2f}")
    print(f"Total Return: {results2['total_return']:.2%}")
    print(f"Total Trades: {results2['total_trades']}")
    print(f"Winning Trades: {results2['winning_trades']}")
    print(f"Losing Trades: {results2['losing_trades']}")

    # Check for bug: return without trades
    if results2["total_return"] != 0 and results2["total_trades"] == 0:
        print("🚨 BUG DETECTED: Non-zero return with zero trades!")
        print(f"   Return: {results2['total_return']:.2%}")
        print(f"   Trades: {results2['total_trades']}")
    else:
        print("✅ No bug detected in test 2")

    # Test 3: Debug portfolio state
    print("\n3. Portfolio State Analysis:")
    print("-" * 40)

    if hasattr(engine, "portfolio") and engine.portfolio:
        print(f"Cash: ${engine.portfolio.cash:,.2f}")
        print(f"Total Market Value: ${engine.portfolio.total_market_value:,.2f}")
        print(f"Total Value: ${engine.portfolio.total_value:,.2f}")
        print(f"Positions: {len(engine.portfolio.positions)}")

        for symbol, position in engine.portfolio.positions.items():
            print(f"  {symbol}: {position.quantity} shares @ ${position.average_price}")
            print(f"    Market Value: ${position.market_value}")
            print(f"    Unrealized P&L: ${position.unrealized_pnl}")

    print(f"\nEquity Curve Points: {len(engine.equity_curve)}")
    if engine.equity_curve:
        print(f"First Point: {engine.equity_curve[0][1]}")
        print(f"Last Point: {engine.equity_curve[-1][1]}")

    return results1, results2


if __name__ == "__main__":
    asyncio.run(test_portfolio_calculation_bug())
