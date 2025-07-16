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
import sys
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trading_bot import TradingBot, Config
from trading_bot.core.exceptions import TradingBotError
from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.strategy.momentum_crossover import MomentumCrossoverStrategy
from trading_bot.strategy.mean_reversion import MeanReversionStrategy
from trading_bot.strategy.breakout import BreakoutStrategy
from trading_bot.core.models import MarketData

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
            console.print("\n[yellow]Received shutdown signal. Stopping bot gracefully...[/yellow]")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_bot(self):
        """Run the trading bot."""
        try:
            self.config = Config.from_env()
            self.bot = TradingBot(self.config)
            
            console.print(Panel.fit(
                "[bold green]Trading Bot Starting[/bold green]\n"
                f"Environment: {self.config.exchange.environment}\n"
                f"Exchange: {self.config.exchange.name}\n"
                f"Strategy: {self.config.strategy.default_strategy}\n"
                f"Portfolio Value: ${self.config.trading.portfolio_value:,.2f}",
                title="🚀 Trading Bot"
            ))
            
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
            runtime = datetime.utcnow() - start_time
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
                "breakout": BreakoutStrategy
            }
            
            if strategy_name not in strategy_classes:
                console.print(f"[red]Unknown strategy: {strategy_name}[/red]")
                return
            
            strategy_class = strategy_classes[strategy_name]
            strategy = strategy_class(strategy_name, self.config.strategy.parameters)
            
            console.print(f"[green]Starting backtest for {strategy_name} strategy[/green]")
            console.print(f"Symbol: {symbol}, Period: {days} days")
            
            # Generate sample market data (in real implementation, load from database/API)
            market_data = self.generate_sample_data(symbol, days)
            
            # Run backtest
            backtest_engine = BacktestEngine(self.config)
            
            start_date = datetime.utcnow() - timedelta(days=days)
            end_date = datetime.utcnow()
            
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
        """Generate sample market data for backtesting."""
        import random
        
        data = []
        base_time = datetime.utcnow() - timedelta(days=days)
        base_price = 100.0
        
        for i in range(days * 24):  # Hourly data
            # Random walk with slight upward bias
            price_change = random.uniform(-0.5, 0.6)
            base_price += price_change
            base_price = max(base_price, 50)  # Minimum price
            
            bar = MarketData(
                symbol=symbol,
                timestamp=base_time + timedelta(hours=i),
                open=round(base_price, 2),
                high=round(base_price + random.uniform(0, 2), 2),
                low=round(base_price - random.uniform(0, 2), 2),
                close=round(base_price, 2),
                volume=random.randint(50000, 500000)
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
        table.add_row("Period", f"{results['start_date'][:10]} to {results['end_date'][:10]}")
        table.add_row("Initial Capital", f"${results['initial_capital']:,.2f}")
        table.add_row("Final Capital", f"${results['final_capital']:,.2f}")
        table.add_row("Total Return", f"{results['total_return']:.2%}")
        table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        table.add_row("Max Drawdown", f"{results['max_drawdown']:.2%}")
        table.add_row("Win Rate", f"{results['win_rate']:.2%}")
        table.add_row("Total Trades", str(results['total_trades']))
        table.add_row("Winning Trades", str(results['winning_trades']))
        table.add_row("Losing Trades", str(results['losing_trades']))
        table.add_row("Profit Factor", f"{results['profit_factor']:.2f}")
        
        console.print(table)
        
        # Trade summary
        if results['total_trades'] > 0:
            console.print(f"\n[bold]Recent Trades (last 5):[/bold]")
            trades_table = Table()
            trades_table.add_column("Time", style="cyan")
            trades_table.add_column("Symbol", style="green")
            trades_table.add_column("Side", style="yellow")
            trades_table.add_column("Quantity", style="magenta")
            trades_table.add_column("Price", style="red")
            
            for trade in results['trades'][-5:]:
                trades_table.add_row(
                    trade['timestamp'][:19],
                    trade['symbol'],
                    trade['side'],
                    str(trade['quantity']),
                    f"${trade['price']:.2f}"
                )
            
            console.print(trades_table)


# CLI Commands
@click.group()
def cli():
    """High-Performance Trading Bot CLI"""
    pass


@cli.command()
def run():
    """Run the trading bot in live mode."""
    bot_cli = TradingBotCLI()
    asyncio.run(bot_cli.run_bot())


@cli.command()
@click.option('--strategy', default='momentum_crossover', 
              help='Strategy to backtest (momentum_crossover, mean_reversion, breakout)')
@click.option('--symbol', default='AAPL', help='Symbol to backtest')
@click.option('--days', default=30, help='Number of days to backtest')
def backtest(strategy, symbol, days):
    """Run backtest for a strategy."""
    bot_cli = TradingBotCLI()
    asyncio.run(bot_cli.run_backtest(strategy, symbol, days))


@cli.command()
def config():
    """Show current configuration."""
    try:
        config = Config.from_env()
        
        table = Table(title="Trading Bot Configuration")
        table.add_column("Section", style="cyan")
        table.add_column("Parameter", style="green")
        table.add_column("Value", style="magenta")
        
        # Trading config
        table.add_row("Trading", "Portfolio Value", f"${config.trading.portfolio_value:,.2f}")
        table.add_row("", "Max Position Size", f"{config.trading.max_position_size:.1%}")
        table.add_row("", "Stop Loss", f"{config.trading.stop_loss_percentage:.1%}")
        table.add_row("", "Take Profit", f"{config.trading.take_profit_percentage:.1%}")
        
        # Risk config
        table.add_row("Risk", "Max Daily Loss", f"{config.risk.max_daily_loss:.1%}")
        table.add_row("", "Max Open Positions", str(config.risk.max_open_positions))
        table.add_row("", "Risk Free Rate", f"{config.risk.risk_free_rate:.1%}")
        
        # Exchange config
        table.add_row("Exchange", "Name", config.exchange.name)
        table.add_row("", "Environment", config.exchange.environment)
        table.add_row("", "Base URL", config.exchange.base_url)
        
        # Strategy config
        table.add_row("Strategy", "Default Strategy", config.strategy.default_strategy)
        table.add_row("", "Parameters", str(config.strategy.parameters))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Configuration Error: {e}[/red]")


@cli.command()
def test():
    """Run the test suite."""
    import subprocess
    
    console.print("[green]Running test suite...[/green]")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", "-v", "--tb=short"
        ], capture_output=True, text=True)
        
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