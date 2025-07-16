"""Test to demonstrate the fix for the backtest bug."""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.config import Config
from trading_bot.core.models import MarketData, PerformanceMetrics
from trading_bot.strategy.momentum_crossover import MomentumCrossoverStrategy


class FixedBacktestEngine(BacktestEngine):
    """Fixed version of BacktestEngine that properly handles open positions."""

    def _calculate_performance_metrics(
        self, start_date: datetime, end_date: datetime
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics with proper handling of open positions."""

        # Calculate trade-level metrics (closed trades only)
        winning_trades = 0
        losing_trades = 0
        total_profit = Decimal("0")
        total_loss = Decimal("0")
        wins = []
        losses = []

        # Group trades by symbol to calculate P&L
        symbol_trades = {}
        for trade in self.trades:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)

        # Calculate P&L for each symbol (closed trades only)
        for symbol, trades in symbol_trades.items():
            trades.sort(key=lambda x: x.timestamp)

            position_quantity = Decimal("0")
            position_cost = Decimal("0")

            for trade in trades:
                if trade.side.name == "BUY":
                    position_quantity += trade.quantity
                    position_cost += trade.quantity * trade.price + trade.commission
                else:  # SELL
                    if position_quantity > 0:
                        # Calculate P&L for this sale
                        sold_quantity = min(trade.quantity, position_quantity)
                        avg_cost = (
                            position_cost / position_quantity
                            if position_quantity > 0
                            else Decimal("0")
                        )

                        sale_proceeds = sold_quantity * trade.price - trade.commission
                        cost_basis = sold_quantity * avg_cost

                        pnl = sale_proceeds - cost_basis

                        if pnl > 0:
                            winning_trades += 1
                            total_profit += pnl
                            wins.append(pnl)
                        else:
                            losing_trades += 1
                            total_loss += abs(pnl)
                            losses.append(abs(pnl))

                        # Update position
                        position_quantity -= sold_quantity
                        if position_quantity > 0:
                            position_cost = position_cost * (
                                position_quantity / (position_quantity + sold_quantity)
                            )
                        else:
                            position_cost = Decimal("0")

        # Calculate metrics for closed trades
        total_closed_trades = winning_trades + losing_trades
        win_rate = (
            Decimal(str(winning_trades / total_closed_trades))
            if total_closed_trades > 0
            else Decimal("0")
        )

        average_win = sum(wins) / len(wins) if wins else Decimal("0")
        average_loss = sum(losses) / len(losses) if losses else Decimal("0")
        largest_win = max(wins) if wins else Decimal("0")
        largest_loss = max(losses) if losses else Decimal("0")

        profit_factor = total_profit / total_loss if total_loss > 0 else Decimal("0")

        # Calculate total return including unrealized P&L
        if self.portfolio:
            total_return = (
                self.portfolio.total_value - self.initial_capital
            ) / self.initial_capital
        else:
            total_return = Decimal("0")

        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()

        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown()

        # Total trades should include ALL executed trades, not just closed ones
        total_trades = len(self.trades)

        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,  # Fixed: count all trades
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            period_start=start_date,
            period_end=end_date,
        )


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


async def test_fixed_backtest() -> dict:
    """Test the fixed backtest engine."""
    print("=" * 60)
    print("TESTING FIXED BACKTEST ENGINE")
    print("=" * 60)

    # Test original engine
    config = Config()
    original_engine = BacktestEngine(config)
    strategy1 = MomentumCrossoverStrategy(
        "momentum_crossover", config.strategy.parameters
    )

    # Test fixed engine
    fixed_engine = FixedBacktestEngine(config)
    strategy2 = MomentumCrossoverStrategy(
        "momentum_crossover", config.strategy.parameters
    )

    # Generate data
    symbol = "AAPL"
    days = 15
    market_data = generate_sample_data(symbol, days)

    start_date = datetime.utcnow() - timedelta(days=days)
    end_date = datetime.utcnow()

    # Run original backtest
    results1 = await original_engine.run_backtest(
        strategy1, market_data, start_date, end_date
    )

    # Run fixed backtest
    results2 = await fixed_engine.run_backtest(
        strategy2, market_data, start_date, end_date
    )

    print("ORIGINAL ENGINE RESULTS:")
    print(f"  Total Trades: {results1['total_trades']}")
    print(f"  Total Return: {results1['total_return']:.2%}")
    print(f"  Final Capital: ${results1['final_capital']:,.2f}")
    print(f"  Actual Trades Executed: {len(original_engine.trades)}")

    print("\nFIXED ENGINE RESULTS:")
    print(f"  Total Trades: {results2['total_trades']}")
    print(f"  Total Return: {results2['total_return']:.2%}")
    print(f"  Final Capital: ${results2['final_capital']:,.2f}")
    print(f"  Actual Trades Executed: {len(fixed_engine.trades)}")

    # Check if bug is fixed
    if results1["total_trades"] == 0 and results1["total_return"] != 0:
        print("\n🚨 ORIGINAL BUG CONFIRMED:")
        print(f"  Shows {results1['total_return']:.2%} return with 0 trades")

    if results2["total_trades"] > 0 and results2["total_return"] != 0:
        print("\n✅ FIXED ENGINE WORKING:")
        print(
            f"  Shows {results2['total_return']:.2%} return with {results2['total_trades']} trades"
        )

    return results2


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(test_fixed_backtest())
