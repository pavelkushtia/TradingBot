"""Test using the actual data generation from main.py to reproduce the bug."""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.config import Config
from trading_bot.core.models import MarketData
from trading_bot.strategy.momentum_crossover import MomentumCrossoverStrategy


def generate_sample_data(symbol: str, days: int) -> list:
    """Generate sample market data that creates clear crossover signals.

    This is copied from main.py to reproduce the exact same data generation.
    """
    data = []
    base_time = datetime.utcnow() - timedelta(days=days)

    total_bars = days * 24
    prices = []

    # Create multiple phases to generate several crossovers
    # Phase 1: Declining trend (first 1/3)
    phase1_bars = total_bars // 3
    for i in range(phase1_bars):
        price = 200 - i * 1.2  # Decline
        price += random.uniform(-2, 2)  # Noise
        prices.append(max(price, 50))

    # Phase 2: Rising trend (middle 1/3) - creates bullish crossover
    phase2_bars = total_bars // 3
    start_price = prices[-1] if prices else 100
    for i in range(phase2_bars):
        price = start_price + i * 1.5  # Rise
        price += random.uniform(-2, 2)  # Noise
        prices.append(price)

    # Phase 3: Declining trend (final 1/3) - creates bearish crossover
    remaining_bars = total_bars - len(prices)
    start_price = prices[-1] if prices else 200
    for i in range(remaining_bars):
        price = start_price - i * 0.8  # Decline
        price += random.uniform(-2, 2)  # Noise
        prices.append(max(price, 50))

    # Create MarketData objects
    for i, price in enumerate(prices):
        bar = MarketData(
            symbol=symbol,
            timestamp=base_time + timedelta(hours=i),
            open=Decimal(str(round(price, 2))),
            high=Decimal(str(round(price + random.uniform(0, 1), 2))),
            low=Decimal(str(round(price - random.uniform(0, 1), 2))),
            close=Decimal(str(round(price, 2))),
            volume=random.randint(50000, 500000),
            # Note: vwap is not set in main.py, defaults to None
        )
        data.append(bar)

    return data


async def test_main_data_generation() -> dict:
    """Test using the exact data generation from main.py."""
    print("=" * 60)
    print("TESTING WITH MAIN.PY DATA GENERATION")
    print("=" * 60)

    # Create config
    config = Config()

    # Create backtest engine
    engine = BacktestEngine(config)

    # Create strategy with same parameters as main.py
    strategy = MomentumCrossoverStrategy(
        "momentum_crossover", config.strategy.parameters
    )

    # Generate data using main.py method
    symbol = "AAPL"
    days = 15
    market_data = generate_sample_data(symbol, days)

    print(f"Generated {len(market_data)} bars for {symbol}")
    print(f"First bar: {market_data[0].timestamp} - ${market_data[0].close}")
    print(f"Last bar: {market_data[-1].timestamp} - ${market_data[-1].close}")

    # Run backtest
    start_date = datetime.utcnow() - timedelta(days=days)
    end_date = datetime.utcnow()

    results = await engine.run_backtest(strategy, market_data, start_date, end_date)

    print("\nBacktest Results:")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Losing Trades: {results['losing_trades']}")

    # Check for the bug
    if results["total_return"] != 0 and results["total_trades"] == 0:
        print("🚨 BUG REPRODUCED!")
        print("The backtest shows a return but no trades were executed.")
        print("This matches the user's reported issue.")
        print(f"Return: {results['total_return']:.2%}")
        print(f"Trades: {results['total_trades']}")

        # Debug portfolio state
        if hasattr(engine, "portfolio") and engine.portfolio:
            print("\nPortfolio Debug:")
            print(f"Cash: ${engine.portfolio.cash}")
            print(f"Total Value: ${engine.portfolio.total_value}")
            print(f"Positions: {len(engine.portfolio.positions)}")

            if engine.portfolio.positions:
                print("⚠️  Found positions even though no trades were executed!")
                for symbol, pos in engine.portfolio.positions.items():
                    print(f"  {symbol}: {pos.quantity} shares @ ${pos.average_price}")
                    print(f"    Market Value: ${pos.market_value}")
                    print(f"    Unrealized P&L: ${pos.unrealized_pnl}")

        # Check equity curve
        if hasattr(engine, "equity_curve") and engine.equity_curve:
            print("\nEquity Curve:")
            print(f"  Points: {len(engine.equity_curve)}")
            print(f"  Start: ${engine.equity_curve[0][1]}")
            print(f"  End: ${engine.equity_curve[-1][1]}")
            print(
                f"  Change: ${engine.equity_curve[-1][1] - engine.equity_curve[0][1]}"
            )
    else:
        print("✅ No bug detected - returns match trades")

    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    asyncio.run(test_main_data_generation())
