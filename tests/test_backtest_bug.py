"""Test to demonstrate the backtest bug where returns show without trades."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.config import Config
from trading_bot.core.models import MarketData
from trading_bot.strategy.momentum_crossover import MomentumCrossoverStrategy


async def test_backtest_bug() -> dict:
    """Test that demonstrates the backtest bug with returns but no trades."""
    print("=" * 60)
    print("TESTING BACKTEST BUG")
    print("=" * 60)

    # Create config
    config = Config()

    # Create backtest engine
    engine = BacktestEngine(config)

    # Create strategy
    strategy = MomentumCrossoverStrategy(
        "momentum_crossover",
        {"short_window": 10, "long_window": 30, "min_strength_threshold": 0.01},
    )

    # Create simple upward trending data (no crossovers)
    base_time = datetime.utcnow() - timedelta(days=15)
    data = []

    for i in range(15 * 24):  # 15 days of hourly data
        timestamp = base_time + timedelta(hours=i)
        price = Decimal("100.0") + (
            Decimal(str(i)) * Decimal("0.1")
        )  # Steady upward trend

        bar = MarketData(
            symbol="AAPL",
            timestamp=timestamp,
            open=price,
            high=price + Decimal("0.5"),
            low=price - Decimal("0.5"),
            close=price,
            volume=Decimal("1000"),
            vwap=price,
        )
        data.append(bar)

    # Run backtest
    start_date = datetime.utcnow() - timedelta(days=15)
    end_date = datetime.utcnow()

    results = await engine.run_backtest(strategy, data, start_date, end_date)

    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Losing Trades: {results['losing_trades']}")

    # Check for the bug
    if results["total_return"] != 0 and results["total_trades"] == 0:
        print("\n🚨 BUG DETECTED!")
        print("The backtest shows a return but no trades were executed.")
        print("This is impossible - you cannot have returns without trades.")
        print(f"Return: {results['total_return']:.2%}")
        print(f"Trades: {results['total_trades']}")

        # Debug portfolio state
        if hasattr(engine, "portfolio") and engine.portfolio:
            print("\nPortfolio Debug:")
            print(f"Cash: ${engine.portfolio.cash}")
            print(f"Total Value: ${engine.portfolio.total_value}")
            print(f"Positions: {len(engine.portfolio.positions)}")

            # Check if there are any positions (there shouldn't be)
            if engine.portfolio.positions:
                print("⚠️  Found positions even though no trades were executed!")
                for symbol, pos in engine.portfolio.positions.items():
                    print(f"  {symbol}: {pos.quantity} shares")
    else:
        print("\n✅ No bug detected - returns match trades")

    return results


if __name__ == "__main__":
    asyncio.run(test_backtest_bug())
