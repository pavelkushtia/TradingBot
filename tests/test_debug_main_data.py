"""Debug test using main.py data generation to trace phantom positions."""

import asyncio
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.config import Config
from trading_bot.core.models import MarketData
from trading_bot.strategy.momentum_crossover import MomentumCrossoverStrategy


class DebugBacktestEngine(BacktestEngine):
    """Debug version of BacktestEngine to trace position creation."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.debug_log: list[str] = []

    def _update_portfolio_with_trade(self, trade):
        """Override to log trade execution."""
        self.debug_log.append(
            f"TRADE EXECUTED: {trade.side} {trade.quantity} {trade.symbol} @ ${trade.price}"
        )
        super()._update_portfolio_with_trade(trade)

    def _update_portfolio_value(self, bar):
        """Override to log portfolio updates."""
        positions_before = len(self.portfolio.positions) if self.portfolio else 0
        super()._update_portfolio_value(bar)
        positions_after = len(self.portfolio.positions) if self.portfolio else 0

        if positions_after > positions_before:
            self.debug_log.append(
                f"POSITION CREATED: {bar.symbol} during portfolio update (NO TRADE!)"
            )
            # Log portfolio state
            if self.portfolio and bar.symbol in self.portfolio.positions:
                pos = self.portfolio.positions[bar.symbol]
                self.debug_log.append(
                    f"  Position: {pos.quantity} shares @ ${pos.average_price}"
                )

    async def _execute_signal(self, signal, bar):
        """Override to log signal execution."""
        self.debug_log.append(
            f"SIGNAL: {signal.signal_type} for {signal.symbol} @ ${bar.close}"
        )
        await super()._execute_signal(signal, bar)

    def _calculate_position_size(self, signal, price):
        """Override to log position sizing."""
        size = super()._calculate_position_size(signal, price)
        self.debug_log.append(
            f"POSITION SIZE: {size} shares for {signal.symbol} @ ${price}"
        )
        return size


def generate_sample_data(symbol: str, days: int) -> list:
    """Generate sample data from main.py."""
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
        )
        data.append(bar)

    return data


async def test_debug_main_data() -> dict:
    """Debug test using main.py data generation."""
    print("=" * 60)
    print("DEBUGGING WITH MAIN.PY DATA")
    print("=" * 60)

    # Create config
    config = Config()

    # Create debug backtest engine
    engine = DebugBacktestEngine(config)

    # Create strategy
    strategy = MomentumCrossoverStrategy(
        "momentum_crossover", config.strategy.parameters
    )

    # Generate data using main.py method
    symbol = "AAPL"
    days = 15  # Same as user's test
    market_data = generate_sample_data(symbol, days)

    print(f"Generated {len(market_data)} bars for {symbol}")
    print("First few bars:")
    for i, bar in enumerate(market_data[:5]):
        print(f"  {i}: {bar.timestamp} - ${bar.close}")

    # Run backtest
    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    end_date = datetime.now(timezone.utc)

    results = await engine.run_backtest(strategy, market_data, start_date, end_date)

    print("\nBacktest Results:")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")

    # Print debug log (first 20 entries)
    print("\nDebug Log (first 20 entries):")
    for i, entry in enumerate(engine.debug_log[:20]):
        print(f"  {i + 1}: {entry}")

    if len(engine.debug_log) > 20:
        print(f"  ... and {len(engine.debug_log) - 20} more entries")

    # Check portfolio state
    if engine.portfolio:
        print("\nFinal Portfolio State:")
        print(f"Cash: ${engine.portfolio.cash}")
        print(f"Total Value: ${engine.portfolio.total_value}")
        print(f"Positions: {len(engine.portfolio.positions)}")

        for symbol, pos in engine.portfolio.positions.items():
            print(f"  {symbol}: {pos.quantity} shares @ ${pos.average_price}")
            print(f"    Market Value: ${pos.market_value}")
            print(f"    Unrealized P&L: ${pos.unrealized_pnl}")

    # Check trades list
    print(f"\nTrades List: {len(engine.trades)} trades")
    for trade in engine.trades:
        print(f"  {trade.side} {trade.quantity} {trade.symbol} @ ${trade.price}")

    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    asyncio.run(test_debug_main_data())
