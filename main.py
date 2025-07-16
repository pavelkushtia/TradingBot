print("main.py: START")
#!/usr/bin/env python3
"""
High-Performance Trading Bot

A professional-grade trading bot with advanced features:
- Real-time market data processing
- Multiple trading strategies
- Risk management
- Order execution
- Backtesting capabilities
- Performance monitoring
"""

import asyncio
import os
import signal
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import click
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from trading_bot import Config, TradingBot
from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.exceptions import TradingBotError
from trading_bot.core.models import MarketData
from trading_bot.strategy.breakout import BreakoutStrategy
from trading_bot.strategy.mean_reversion import MeanReversionStrategy
from trading_bot.strategy.momentum_crossover import MomentumCrossoverStrategy

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


console = Console()


class TradingBotCLI:
    """Command-line interface for the trading bot."""

    def __init__(self):
        self.bot: Optional[TradingBot] = None
        self.config: Optional[Config] = None
        self.running = False

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            console.print(
                "\n[yellow]Received shutdown signal. "
                "Stopping bot gracefully...[/yellow]"
            )
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def check_existing_instances(self):
        """Check for existing trading bot instances."""
        current_pid = os.getpid()
        instances = []

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                # Skip our own process
                if proc.info["pid"] == current_pid:
                    continue

                cmdline = proc.info["cmdline"]
                if not cmdline:
                    continue

                # Check if this is a python process running main.py with 'run' command
                if (
                    "python" in cmdline[0].lower()
                    and any("main.py" in arg for arg in cmdline)
                    and any("run" in arg for arg in cmdline)
                ):
                    # Additional check: make sure it's actually a different process
                    # by checking if it started before us
                    try:
                        proc_create_time = proc.create_time()
                        current_create_time = psutil.Process(current_pid).create_time()

                        # Only consider it an existing instance if it started before us
                        if proc_create_time < current_create_time:
                            instances.append(proc.info["pid"])
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # If we can't get process times, skip this process
                        continue

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return instances

    async def run_bot(self, symbols: str):
        """Run the trading bot."""
        try:
            # Check for existing instances
            existing_instances = self.check_existing_instances()
            if existing_instances:
                console.print(
                    f"[yellow]Warning: Found {len(existing_instances)} existing trading bot instance(s) "
                    f"with PID(s): {', '.join(map(str, existing_instances))}[/yellow]"
                )
                console.print(
                    "[yellow]Multiple instances may cause WebSocket connection limit errors.[/yellow]"
                )
                console.print(
                    "[yellow]Consider stopping other instances or use different WebSocket endpoints.[/yellow]"
                )

                # Ask user if they want to continue
                response = input("Continue anyway? (y/N): ").lower()
                if response != "y":
                    console.print("[yellow]Aborted by user.[/yellow]")
                    return

            self.config = Config.from_env()
            # Override trading symbols with CLI argument
            if symbols:
                self.config.trading.trading_symbols = symbols

            self.bot = TradingBot(self.config)

            console.print(
                Panel.fit(
                    "[bold green]Trading Bot Starting[/bold green]\n"
                    f"Environment: {self.config.exchange.environment}\n"
                    f"Exchange: {self.config.exchange.name}\n"
                    f"Strategy: {self.config.strategy.default_strategy}\n"
                    f"Portfolio Value: "
                    f"${self.config.trading.portfolio_value:,.2f}\n"
                    f"Trading Symbols: {self.config.trading.trading_symbols}",
                    title="🚀 Trading Bot",
                )
            )

            # Setup signal handlers
            self.setup_signal_handlers()
            self.running = True

            # Start the bot
            await self.bot.start()

            # Monitor bot status
            while self.running:
                await asyncio.sleep(5)
                self.display_status()

            # Stop the bot
            await self.bot.stop()

        except TradingBotError as e:
            console.print(f"[red]Trading Bot Error: {e}[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected Error: {e}[/red]")
            sys.exit(1)

    def display_status(self):
        """Display bot status."""
        if not self.bot:
            return

        status = self.bot.get_status()

        table = Table(title="Trading Bot Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Running", "✅ Yes" if status["running"] else "❌ No")
        table.add_row("Portfolio Value", f"${status['portfolio_value']}")
        table.add_row("Active Orders", str(status["active_orders"]))
        table.add_row("Open Positions", str(status["open_positions"]))
        table.add_row("Daily P&L", f"${status['daily_pnl']}")

        if status["start_time"]:
            start_time = datetime.fromisoformat(status["start_time"])
            runtime = datetime.now(timezone.utc) - start_time
            table.add_row("Runtime", str(runtime).split(".")[0])

        console.print(table)

    async def run_backtest(self, strategy_name: str, symbol: str, days: int):
        """Run backtest for a strategy."""
        try:
            self.config = Config.from_env()

            # Create strategy
            strategy_classes = {
                "momentum_crossover": MomentumCrossoverStrategy,
                "mean_reversion": MeanReversionStrategy,
                "breakout": BreakoutStrategy,
            }

            if strategy_name not in strategy_classes:
                console.print(f"[red]Unknown strategy: {strategy_name}[/red]")
                return

            strategy_class = strategy_classes[strategy_name]

            # Use custom parameters for backtesting with lower thresholds
            backtest_parameters = {
                "short_window": 10,
                "long_window": 30,
                "min_strength_threshold": 0.0001,  # Lower threshold for backtesting
            }

            strategy = strategy_class(strategy_name, backtest_parameters)

            console.print(
                f"[green]Starting backtest for {strategy_name} strategy[/green]"
            )
            console.print(f"Symbol: {symbol}, Period: {days} days")

            # Generate sample market data
            market_data = self.generate_sample_data(symbol, days)

            # Run backtest
            backtest_engine = BacktestEngine(self.config)

            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            end_date = datetime.now(timezone.utc)

            with Progress() as progress:
                task = progress.add_task("Running backtest...", total=100)

                # Simulate progress
                for i in range(100):
                    await asyncio.sleep(0.01)
                    progress.update(task, advance=1)

                results = await backtest_engine.run_backtest(
                    strategy, market_data, start_date, end_date
                )

            # Display results
            self.display_backtest_results(results)

        except Exception as e:
            console.print(f"[red]Backtest Error: {e}[/red]")

    def generate_sample_data(self, symbol: str, days: int) -> list:
        """Generate sample market data that creates clear crossover signals."""
        import random

        # Set seed for consistent results
        random.seed(42)

        data = []
        base_time = datetime.now(timezone.utc) - timedelta(days=days)

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

    def display_backtest_results(self, results: dict):
        """Display backtest results."""
        console.print("\n[bold green]Backtest Results[/bold green]")

        # Summary table
        table = Table(title="Performance Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Strategy", results["strategy_name"])
        table.add_row(
            "Period",
            f"{results['start_date'][:10]} to {results['end_date'][:10]}",
        )
        table.add_row("Initial Capital", f"${results['initial_capital']:,.2f}")
        table.add_row("Final Capital", f"${results['final_capital']:,.2f}")
        table.add_row("Total Return", f"{results['total_return']:.2%}")
        table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        table.add_row("Max Drawdown", f"{results['max_drawdown']:.2%}")
        table.add_row("Win Rate", f"{results['win_rate']:.2%}")
        table.add_row("Total Trades", str(results["total_trades"]))
        table.add_row("Winning Trades", str(results["winning_trades"]))
        table.add_row("Losing Trades", str(results["losing_trades"]))
        table.add_row("Profit Factor", f"{results['profit_factor']:.2f}")

        console.print(table)

        # Trade summary
        if results["total_trades"] > 0:
            console.print("\n[bold]Recent Trades (last 5):[/bold]")
            trades_table = Table()
            trades_table.add_column("Time", style="cyan")
            trades_table.add_column("Symbol", style="green")
            trades_table.add_column("Side", style="yellow")
            trades_table.add_column("Quantity", style="magenta")
            trades_table.add_column("Price", style="red")

            for trade in results["trades"][-5:]:
                # Handle datetime object from trade timestamp
                timestamp_str = trade["timestamp"]
                if isinstance(timestamp_str, str):
                    display_time = timestamp_str[:19]
                else:
                    # Convert datetime to string format
                    display_time = timestamp_str.strftime("%Y-%m-%d %H:%M:%S")

                trades_table.add_row(
                    display_time,
                    trade["symbol"],
                    trade["side"],
                    str(trade["quantity"]),
                    f"${trade['price']:.2f}",
                )

            console.print(trades_table)


# CLI Commands
@click.group()
def cli():
    """High-Performance Trading Bot CLI"""


@cli.command()
@click.option(
    "--symbols",
    default=None,
    help="Comma-separated list of symbols to trade (e.g., AAPL,GOOGL,MSFT)",
)
def run(symbols):
    """Run the trading bot in live mode."""
    bot_cli = TradingBotCLI()
    asyncio.run(bot_cli.run_bot(symbols))


@cli.command()
@click.option(
    "--strategy",
    default="momentum_crossover",
    help="Strategy to backtest (momentum_crossover, mean_reversion, breakout)",
)
@click.option("--symbol", default="AAPL", help="Symbol to backtest")
@click.option("--days", default=30, help="Number of days to backtest")
def backtest(strategy, symbol, days):
    """Run backtest for a strategy."""
    bot_cli = TradingBotCLI()
    asyncio.run(bot_cli.run_backtest(strategy, symbol, days))


@cli.command()
def config():
    """Show current configuration."""
    try:
        config_obj = Config.from_env()

        table = Table(title="Trading Bot Configuration")
        table.add_column("Section", style="cyan")
        table.add_column("Parameter", style="green")
        table.add_column("Value", style="magenta")

        # Trading config
        table.add_row(
            "Trading",
            "Portfolio Value",
            f"${config_obj.trading.portfolio_value:,.2f}",
        )
        table.add_row(
            "",
            "Max Position Size",
            f"{config_obj.trading.max_position_size:.1%}",
        )
        table.add_row("", "Stop Loss", f"{config_obj.trading.stop_loss_percentage:.1%}")
        table.add_row(
            "",
            "Take Profit",
            f"{config_obj.trading.take_profit_percentage:.1%}",
        )

        # Risk config
        table.add_row("Risk", "Max Daily Loss", f"{config_obj.risk.max_daily_loss:.1%}")
        table.add_row("", "Max Open Positions", str(config_obj.risk.max_open_positions))
        table.add_row("", "Risk Free Rate", f"{config_obj.risk.risk_free_rate:.1%}")

        # Exchange config
        table.add_row("Exchange", "Name", config_obj.exchange.name)
        table.add_row("", "Environment", config_obj.exchange.environment)
        table.add_row("", "Base URL", config_obj.exchange.base_url)

        # Strategy config
        table.add_row(
            "Strategy",
            "Default Strategy",
            config_obj.strategy.default_strategy,
        )
        table.add_row("", "Parameters", str(config_obj.strategy.parameters))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Configuration Error: {e}[/red]")


@cli.command()
def test():
    """Run the test suite."""
    console.print("[green]Running test suite...[/green]")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True,
        )

        console.print(result.stdout)
        if result.stderr:
            console.print(f"[red]{result.stderr}[/red]")

        if result.returncode == 0:
            console.print("[green]✅ All tests passed![/green]")
        else:
            console.print("[red]❌ Some tests failed.[/red]")
            sys.exit(1)

    except FileNotFoundError:
        console.print("[red]pytest not found. Install with: pip install pytest[/red]")
        sys.exit(1)


@cli.command()
def version():
    """Show version information."""
    from trading_bot import __version__

    console.print(f"Trading Bot v{__version__}")


if __name__ == "__main__":
    cli()
