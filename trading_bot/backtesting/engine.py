"""Backtesting engine for strategy validation and performance analysis."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import Config
from ..core.events import EventBus
from ..core.exceptions import BacktestError
from ..core.logging import TradingLogger
from ..core.models import (MarketData, Order, OrderSide, OrderStatus,
                           OrderType, PerformanceMetrics, Portfolio, Position,
                           PositionSide, Trade)
from ..core.shared_execution import SharedExecutionLogic
from ..risk.manager import RiskManager
from ..strategy.base import BaseStrategy
from .simple_metrics import SimplePerformanceMetrics


class BacktestEngine:
    """High-performance backtesting engine for strategy validation."""

    def __init__(self, config: Config, event_bus: EventBus):
        """Initialize backtesting engine."""
        self.config = config
        self.logger = TradingLogger("backtest_engine")
        self.event_bus = event_bus

        # Initialize risk manager (same as real trading)
        self.risk_manager = RiskManager(config, self.event_bus)

        # Initialize shared execution logic (same as real trading)
        self.shared_execution = SharedExecutionLogic(
            commission_per_share=Decimal("0.005"), min_commission=Decimal("1.00")
        )

        # Advanced performance calculator - use simplified version
        self.performance_calculator = SimplePerformanceMetrics(
            risk_free_rate=config.risk.risk_free_rate
        )

        # Backtest parameters
        self.initial_capital = Decimal(str(config.trading.portfolio_value))

        # State tracking
        self.portfolio: Optional[Portfolio] = None
        self.trades: List[Trade] = []
        self.orders: List[Order] = []
        self.daily_returns: List[Decimal] = []
        self.equity_curve: List[Tuple[datetime, Decimal]] = []

        # Performance metrics
        self.performance_metrics: Optional[PerformanceMetrics] = None
        self.advanced_metrics: Optional[SimplePerformanceMetrics] = None

    async def run_backtest(
        self,
        strategy: BaseStrategy,
        market_data: List[MarketData],
        start_date: datetime,
        end_date: datetime,
        benchmark_data: Optional[List[MarketData]] = None,
    ) -> Dict[str, Any]:
        """Run backtest for a strategy with historical data."""
        try:
            self.logger.logger.info(f"Starting backtest: {start_date} to {end_date}")

            # Initialize portfolio
            self._initialize_portfolio(start_date)

            # Initialize strategy
            await strategy.initialize()

            # Sort market data by timestamp
            market_data.sort(key=lambda x: x.timestamp)

            # Normalize timezone for comparison
            def normalize_datetime(dt):
                """Normalize datetime to UTC timezone for comparison."""
                if dt.tzinfo is None:
                    # If timezone-naive, assume it's UTC
                    return dt.replace(tzinfo=timezone.utc)
                return dt

            # Normalize all timestamps
            normalized_start = normalize_datetime(start_date)
            normalized_end = normalize_datetime(end_date)

            # Filter data by date range
            filtered_data = [
                bar
                for bar in market_data
                if normalized_start
                <= normalize_datetime(bar.timestamp)
                <= normalized_end
            ]

            if not filtered_data:
                raise BacktestError(
                    "No market data available for the specified date range"
                )

            # Process each bar
            for bar in filtered_data:
                await self._process_bar(strategy, bar)

            # Calculate comprehensive performance metrics
            self.advanced_metrics = self._calculate_advanced_performance_metrics(
                start_date, end_date, benchmark_data
            )

            # Calculate basic performance metrics for backward compatibility
            self.performance_metrics = self._calculate_basic_performance_metrics(
                start_date, end_date
            )

            # Create backtest results
            results = {
                "strategy_name": strategy.name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": float(self.initial_capital),
                "final_capital": float(self.portfolio.total_value),
                # Basic metrics for backward compatibility
                "total_return": float(self.performance_metrics.total_return),
                "sharpe_ratio": float(self.performance_metrics.sharpe_ratio),
                "max_drawdown": float(self.performance_metrics.max_drawdown),
                "win_rate": float(self.performance_metrics.win_rate),
                "total_trades": self.performance_metrics.total_trades,
                "winning_trades": self.performance_metrics.winning_trades,
                "losing_trades": self.performance_metrics.losing_trades,
                "profit_factor": float(self.performance_metrics.profit_factor),
                # Advanced metrics
                "advanced_metrics": {
                    "annualized_return": self.advanced_metrics.annualized_return,
                    "sortino_ratio": self.advanced_metrics.sortino_ratio,
                    "calmar_ratio": self.advanced_metrics.calmar_ratio,
                    "information_ratio": self.advanced_metrics.information_ratio,
                    "volatility": self.advanced_metrics.volatility,
                    "downside_volatility": self.advanced_metrics.downside_volatility,
                    "var_95": self.advanced_metrics.var_95,
                    "var_99": self.advanced_metrics.var_99,
                    "cvar_95": self.advanced_metrics.cvar_95,
                    "skewness": self.advanced_metrics.skewness,
                    "kurtosis": self.advanced_metrics.kurtosis,
                    "expectancy": self.advanced_metrics.expectancy,
                    "consecutive_wins": self.advanced_metrics.consecutive_wins,
                    "consecutive_losses": self.advanced_metrics.consecutive_losses,
                    "recovery_factor": self.advanced_metrics.recovery_factor,
                    "max_drawdown_duration": self.advanced_metrics.max_drawdown_duration,
                },
                # Benchmark comparison (if available)
                "benchmark_metrics": {},
                # Detailed data
                "trades": [trade.dict() for trade in self.trades],
                "equity_curve": [
                    (dt.isoformat(), float(value)) for dt, value in self.equity_curve
                ],
                "daily_returns": [float(ret) for ret in self.daily_returns],
                # Performance report
                "performance_report": self.performance_calculator.generate_performance_report(
                    self.advanced_metrics
                ),
            }

            # Add benchmark comparison if available
            if (
                self.advanced_metrics.alpha is not None
                and self.advanced_metrics.beta is not None
            ):
                results["benchmark_metrics"] = {
                    "alpha": self.advanced_metrics.alpha,
                    "beta": self.advanced_metrics.beta,
                    "correlation": self.advanced_metrics.correlation,
                    "tracking_error": self.advanced_metrics.tracking_error,
                }

            self.logger.logger.info(
                f"Backtest completed. Total return: "
                f"{self.advanced_metrics.total_return:.2%}, "
                f"Sharpe: {self.advanced_metrics.sharpe_ratio:.2f}, "
                f"Max DD: {self.advanced_metrics.max_drawdown:.2%}"
            )

            return results

        except Exception as e:
            self.logger.log_error(e, {"context": "run_backtest"})
            raise BacktestError(f"Backtest failed: {e}")

    def _calculate_advanced_performance_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        benchmark_data: Optional[List[MarketData]] = None,
    ) -> SimplePerformanceMetrics:
        """Calculate comprehensive advanced performance metrics."""

        # Convert equity curve to float tuples for the calculator
        equity_curve_float = [
            (timestamp, float(value)) for timestamp, value in self.equity_curve
        ]

        # Prepare benchmark returns if available
        benchmark_returns = None
        if benchmark_data:
            benchmark_returns = self._calculate_benchmark_returns(benchmark_data)

        return self.performance_calculator.calculate_comprehensive_metrics(
            equity_curve=equity_curve_float,
            trades=self.trades,
            benchmark_returns=benchmark_returns,
            initial_capital=float(self.initial_capital),
        )

    def _calculate_benchmark_returns(
        self, benchmark_data: List[MarketData]
    ) -> List[float]:
        """Calculate returns from benchmark data."""
        if len(benchmark_data) < 2:
            return []

        returns = []
        for i in range(1, len(benchmark_data)):
            prev_close = float(benchmark_data[i - 1].close)
            current_close = float(benchmark_data[i].close)
            if prev_close > 0:
                daily_return = (current_close - prev_close) / prev_close
                returns.append(daily_return)

        return returns

    # Keep existing methods for backward compatibility
    async def _process_bar(self, strategy: BaseStrategy, bar: MarketData) -> None:
        """Process a single market data bar."""
        try:
            # Update strategy with new bar
            await strategy.on_bar(bar.symbol, bar)

            # Generate signals
            signals = await strategy.generate_signals()

            # Execute signals
            for signal in signals:
                await self._execute_signal(signal, bar)

            # Update portfolio value
            self._update_portfolio_value(bar)

            # Record equity curve point
            self.equity_curve.append((bar.timestamp, self.portfolio.total_value))

        except Exception as e:
            self.logger.log_error(
                e, {"context": "process_bar", "timestamp": bar.timestamp}
            )

    async def _execute_signal(self, signal, bar: MarketData) -> None:
        """Execute a trading signal in the backtest."""
        try:
            # Apply risk management checks (same as real trading)
            if not await self.risk_manager.evaluate_signal(signal, self.portfolio):
                self.logger.logger.info(
                    f"Signal rejected by risk management: {signal.symbol}"
                )
                return

            # Calculate position size using risk manager (same as real trading)
            position_size = await self._calculate_position_size(signal, bar.close)

            if position_size <= 0:
                return

            # Create order using shared logic (same as real trading)
            order = self.shared_execution.signal_to_order(signal, position_size)
            if not order:
                return

            # Simulate realistic execution with slippage (same as real trading)
            fill_price = self.shared_execution.simulate_execution_price(
                bar.close, signal.signal_type, position_size
            )

            # Create order
            order = Order(
                id=f"backtest_{len(self.orders) + 1}",
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == "buy" else OrderSide.SELL,
                type=OrderType.MARKET,
                quantity=position_size,
                price=bar.close,
                status=OrderStatus.FILLED,
                filled_quantity=position_size,
                average_fill_price=fill_price,
                created_at=bar.timestamp,
                updated_at=bar.timestamp,
                strategy_id=signal.strategy_name,
            )

            self.orders.append(order)

            # Create trade
            trade = Trade(
                id=f"trade_{len(self.trades) + 1}",
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                timestamp=bar.timestamp,
                commission=self._calculate_commission(order.quantity, order.price),
                strategy_id=order.strategy_id,
            )

            self.trades.append(trade)

            # Update portfolio
            self._update_portfolio_with_trade(trade)

        except Exception as e:
            self.logger.log_error(
                e, {"context": "execute_signal", "signal": signal.dict()}
            )

    def _initialize_portfolio(self, start_date: datetime) -> None:
        """Initialize the portfolio for backtesting."""
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            start_date=start_date,
            total_value=self.initial_capital,
            buying_power=self.initial_capital,
            cash=self.initial_capital,
            updated_at=start_date,
        )
        self.trades = []
        self.orders = []
        self.daily_returns = []
        self.equity_curve = [(start_date, self.initial_capital)]
        self.logger.logger.info(
            f"Portfolio initialized with {self.initial_capital} capital."
        )

    async def _calculate_position_size(self, signal, price: Decimal) -> Decimal:
        """Calculate position size based on risk parameters."""
        if not self.portfolio:
            return Decimal("0")

        # Use the same risk manager as real trading for position sizing
        return await self.risk_manager.calculate_position_size(
            signal.symbol, price, self.portfolio
        )

    def _calculate_commission(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Calculate commission for a trade."""
        return self.shared_execution.calculate_commission(quantity, price)

    def _simulate_execution_price(
        self, market_price: Decimal, signal_type: str, quantity: Decimal
    ) -> Decimal:
        """Simulate realistic execution price with slippage (same as real trading)."""
        # Base slippage factors
        base_slippage = Decimal("0.001")  # 0.1% base slippage

        # Volume-based slippage (larger orders have more slippage)
        volume_factor = min(
            quantity / Decimal("1000"), Decimal("0.005")
        )  # Max 0.5% additional

        # Market impact simulation
        total_slippage = base_slippage + volume_factor

        if signal_type == "buy":
            # Buy orders: pay slightly more (adverse slippage)
            return market_price * (1 + total_slippage)
        else:
            # Sell orders: receive slightly less (adverse slippage)
            return market_price * (1 - total_slippage)

    def _update_portfolio_with_trade(self, trade: Trade) -> None:
        """Update portfolio state with a new trade."""
        if not self.portfolio:
            return

        # Calculate trade value including commission
        trade_value = trade.quantity * trade.price
        total_cost = trade_value + trade.commission

        if trade.side == OrderSide.BUY:
            # Buying: reduce cash, add/increase position
            self.portfolio.cash -= total_cost

            if trade.symbol in self.portfolio.positions:
                # Add to existing position
                existing_pos = self.portfolio.positions[trade.symbol]
                total_quantity = existing_pos.quantity + trade.quantity
                total_cost_basis = (
                    existing_pos.quantity * existing_pos.average_price
                ) + trade_value
                new_avg_price = total_cost_basis / total_quantity

                existing_pos.quantity = total_quantity
                existing_pos.average_price = new_avg_price
                existing_pos.updated_at = trade.timestamp
            else:
                # Create new position
                self.portfolio.positions[trade.symbol] = Position(
                    symbol=trade.symbol,
                    side=PositionSide.LONG,
                    quantity=trade.quantity,
                    average_price=trade.price,
                    market_value=trade_value,
                    unrealized_pnl=Decimal("0"),
                    created_at=trade.timestamp,
                    updated_at=trade.timestamp,
                    strategy_id=trade.strategy_id,
                )

        else:  # SELL
            # Selling: increase cash, reduce/remove position
            self.portfolio.cash += trade_value - trade.commission

            if trade.symbol in self.portfolio.positions:
                existing_pos = self.portfolio.positions[trade.symbol]

                if existing_pos.quantity >= trade.quantity:
                    # Partial or full sale
                    existing_pos.quantity -= trade.quantity
                    existing_pos.updated_at = trade.timestamp

                    # Remove position if fully sold
                    if existing_pos.quantity == 0:
                        del self.portfolio.positions[trade.symbol]
                else:
                    # This shouldn't happen in a well-designed system
                    self.logger.logger.warning(
                        f"Selling more shares than owned for {trade.symbol}"
                    )

    def _update_portfolio_value(self, bar: MarketData) -> None:
        """Update portfolio value based on current market prices."""
        if not self.portfolio:
            return

        # Update position values
        if bar.symbol in self.portfolio.positions:
            position = self.portfolio.positions[bar.symbol]
            position.market_value = position.quantity * bar.close
            position.unrealized_pnl = position.market_value - (
                position.quantity * position.average_price
            )
            position.updated_at = bar.timestamp

        # Update portfolio totals
        self.portfolio.total_value = (
            self.portfolio.cash + self.portfolio.total_market_value
        )
        self.portfolio.updated_at = bar.timestamp

    def _calculate_basic_performance_metrics(
        self, start_date: datetime, end_date: datetime
    ) -> PerformanceMetrics:
        """Calculate basic performance metrics for backward compatibility."""
        if not self.trades:
            return self._get_empty_performance_metrics(start_date, end_date)

        # Use the advanced metrics to populate basic metrics
        return PerformanceMetrics(
            total_return=Decimal(str(self.advanced_metrics.total_return)),
            sharpe_ratio=Decimal(str(self.advanced_metrics.sharpe_ratio)),
            max_drawdown=Decimal(str(self.advanced_metrics.max_drawdown)),
            win_rate=Decimal(str(self.advanced_metrics.win_rate)),
            profit_factor=Decimal(str(self.advanced_metrics.profit_factor)),
            total_trades=self.advanced_metrics.total_trades,
            winning_trades=self.advanced_metrics.winning_trades,
            losing_trades=self.advanced_metrics.losing_trades,
            average_win=Decimal(str(self.advanced_metrics.average_win)),
            average_loss=Decimal(str(self.advanced_metrics.average_loss)),
            largest_win=Decimal(str(self.advanced_metrics.largest_win)),
            largest_loss=Decimal(str(self.advanced_metrics.largest_loss)),
            period_start=start_date,
            period_end=end_date,
        )

    def _get_empty_performance_metrics(
        self, start_date: datetime, end_date: datetime
    ) -> PerformanceMetrics:
        """Get empty performance metrics when no trades were made."""
        return PerformanceMetrics(
            total_return=Decimal("0"),
            sharpe_ratio=Decimal("0"),
            max_drawdown=Decimal("0"),
            win_rate=Decimal("0"),
            profit_factor=Decimal("0"),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            average_win=Decimal("0"),
            average_loss=Decimal("0"),
            largest_win=Decimal("0"),
            largest_loss=Decimal("0"),
            period_start=start_date,
            period_end=end_date,
        )

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        if self.advanced_metrics:
            return self.performance_calculator.generate_performance_report(
                self.advanced_metrics
            )
        elif self.performance_metrics:
            # Fallback to basic performance metrics
            return self.performance_calculator.generate_performance_report(
                self.performance_metrics
            )
        else:
            return {}
