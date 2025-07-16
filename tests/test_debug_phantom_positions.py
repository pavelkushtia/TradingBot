"""Debug test to trace where phantom positions are created."""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.config import Config
from trading_bot.core.models import MarketData
from trading_bot.strategy.momentum_crossover import MomentumCrossoverStrategy


class DebugBacktestEngine(BacktestEngine):
    """Debug version of BacktestEngine to trace position creation."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.debug_log = []

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


async def test_phantom_positions() -> dict:
    """Test to trace phantom position creation."""
    print("=" * 60)
    print("DEBUGGING PHANTOM POSITIONS")
    print("=" * 60)

    # Create config
    config = Config()

    # Create debug backtest engine
    engine = DebugBacktestEngine(config)

    # Create strategy
    strategy = MomentumCrossoverStrategy(
        "momentum_crossover", config.strategy.parameters
    )

    # Create simple data - just 3 bars with price changes
    base_time = datetime.utcnow() - timedelta(days=1)
    data = []

    prices = [100.0, 105.0, 110.0]  # Simple upward trend

    for i, price in enumerate(prices):
        bar = MarketData(
            symbol="AAPL",
            timestamp=base_time + timedelta(hours=i),
            open=Decimal(str(price)),
            high=Decimal(str(price + 1)),
            low=Decimal(str(price - 1)),
            close=Decimal(str(price)),
            volume=100000,
        )
        data.append(bar)

    print("Test data:")
    for bar in data:
        print(f"  {bar.timestamp}: ${bar.close}")

    # Run backtest
    start_date = base_time
    end_date = base_time + timedelta(hours=3)

    results = await engine.run_backtest(strategy, data, start_date, end_date)

    print("\nBacktest Results:")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")

    # Print debug log
    print("\nDebug Log:")
    for entry in engine.debug_log:
        print(f"  {entry}")

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
    random.seed(42)
    asyncio.run(test_phantom_positions())
