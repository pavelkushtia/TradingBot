"""Advanced performance metrics for backtesting and strategy evaluation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.models import OrderSide, Trade


@dataclass
class AdvancedPerformanceMetrics:
    """Advanced performance metrics for trading strategies."""

    # Basic Returns
    total_return: float
    annualized_return: float
    cumulative_return: float

    # Risk Metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # Drawdown Metrics
    max_drawdown: float
    max_drawdown_duration: int  # days
    average_drawdown: float
    recovery_factor: float

    # Volatility Metrics
    volatility: float
    downside_volatility: float
    upside_volatility: float

    # Trade Analysis
    win_rate: float
    profit_factor: float
    expectancy: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int

    # Risk-Adjusted Returns
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional VaR 95%

    # Distribution Metrics
    skewness: float
    kurtosis: float

    # Benchmark Comparison
    alpha: Optional[float] = None
    beta: Optional[float] = None
    correlation: Optional[float] = None
    tracking_error: Optional[float] = None

    # Period Analysis
    period_start: datetime = None
    period_end: datetime = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0


class PerformanceCalculator:
    """Calculator for advanced performance metrics."""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def calculate_comprehensive_metrics(
        self,
        equity_curve: List[Tuple[datetime, float]],
        trades: List[Trade],
        benchmark_returns: Optional[List[float]] = None,
        initial_capital: float = 100000.0,
    ) -> AdvancedPerformanceMetrics:
        """Calculate comprehensive performance metrics."""

        if not equity_curve or len(equity_curve) < 2:
            return self._get_empty_metrics()

        # Extract returns from equity curve
        returns = self._calculate_returns_from_equity_curve(equity_curve)

        if not returns:
            return self._get_empty_metrics()

        # Basic calculations
        total_return = (equity_curve[-1][1] - equity_curve[0][1]) / equity_curve[0][1]

        # Annualized return
        days = (equity_curve[-1][0] - equity_curve[0][0]).days
        years = max(days / 365.25, 1 / 365.25)  # Prevent division by zero
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # Risk metrics
        returns_array = np.array(returns)
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualized

        # Sharpe ratio
        excess_returns = returns_array - (self.risk_free_rate / 252)
        sharpe_ratio = (
            np.mean(excess_returns) / np.std(returns_array) * np.sqrt(252)
            if np.std(returns_array) > 0
            else 0
        )

        # Sortino ratio
        downside_returns = returns_array[returns_array < 0]
        downside_volatility = (
            np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        )
        sortino_ratio = (
            np.mean(excess_returns) / downside_volatility * np.sqrt(252)
            if downside_volatility > 0
            else 0
        )

        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(equity_curve)

        # Calmar ratio
        calmar_ratio = (
            annualized_return / abs(drawdown_metrics["max_drawdown"])
            if drawdown_metrics["max_drawdown"] != 0
            else 0
        )

        # Trade analysis
        trade_metrics = self._calculate_trade_metrics(trades)

        # VaR calculations
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        var_99 = np.percentile(returns_array, 1) if len(returns_array) > 0 else 0

        # Conditional VaR (Expected Shortfall)
        var_95_returns = returns_array[returns_array <= var_95]
        cvar_95 = np.mean(var_95_returns) if len(var_95_returns) > 0 else 0

        # Distribution metrics
        skewness = self._calculate_skewness(returns_array)
        kurtosis = self._calculate_kurtosis(returns_array)

        # Upside/Downside volatility
        upside_returns = returns_array[returns_array > 0]
        upside_volatility = (
            np.std(upside_returns) * np.sqrt(252) if len(upside_returns) > 0 else 0
        )

        # Benchmark comparison
        benchmark_metrics = None
        if benchmark_returns:
            benchmark_metrics = self._calculate_benchmark_metrics(
                returns, benchmark_returns
            )

        return AdvancedPerformanceMetrics(
            # Basic Returns
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=total_return,
            # Risk Metrics
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=(
                benchmark_metrics.get("information_ratio", 0)
                if benchmark_metrics
                else 0
            ),
            # Drawdown Metrics
            max_drawdown=drawdown_metrics["max_drawdown"],
            max_drawdown_duration=drawdown_metrics["max_drawdown_duration"],
            average_drawdown=drawdown_metrics["average_drawdown"],
            recovery_factor=(
                total_return / abs(drawdown_metrics["max_drawdown"])
                if drawdown_metrics["max_drawdown"] != 0
                else 0
            ),
            # Volatility Metrics
            volatility=volatility,
            downside_volatility=downside_volatility,
            upside_volatility=upside_volatility,
            # Trade Analysis
            win_rate=trade_metrics["win_rate"],
            profit_factor=trade_metrics["profit_factor"],
            expectancy=trade_metrics["expectancy"],
            average_win=trade_metrics["average_win"],
            average_loss=trade_metrics["average_loss"],
            largest_win=trade_metrics["largest_win"],
            largest_loss=trade_metrics["largest_loss"],
            consecutive_wins=trade_metrics["consecutive_wins"],
            consecutive_losses=trade_metrics["consecutive_losses"],
            # Risk-Adjusted Returns
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            # Distribution Metrics
            skewness=skewness,
            kurtosis=kurtosis,
            # Benchmark Comparison
            alpha=benchmark_metrics.get("alpha") if benchmark_metrics else None,
            beta=benchmark_metrics.get("beta") if benchmark_metrics else None,
            correlation=(
                benchmark_metrics.get("correlation") if benchmark_metrics else None
            ),
            tracking_error=(
                benchmark_metrics.get("tracking_error") if benchmark_metrics else None
            ),
            # Period Analysis
            period_start=equity_curve[0][0],
            period_end=equity_curve[-1][0],
            total_trades=trade_metrics["total_trades"],
            winning_trades=trade_metrics["winning_trades"],
            losing_trades=trade_metrics["losing_trades"],
        )

    def _calculate_returns_from_equity_curve(
        self, equity_curve: List[Tuple[datetime, float]]
    ) -> List[float]:
        """Calculate daily returns from equity curve."""
        returns = []
        for i in range(1, len(equity_curve)):
            prev_value = equity_curve[i - 1][1]
            current_value = equity_curve[i][1]
            if prev_value > 0:
                daily_return = (current_value - prev_value) / prev_value
                returns.append(daily_return)
        return returns

    def _calculate_drawdown_metrics(
        self, equity_curve: List[Tuple[datetime, float]]
    ) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics."""
        if len(equity_curve) < 2:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "average_drawdown": 0.0,
            }

        values = [point[1] for point in equity_curve]
        [point[0] for point in equity_curve]

        peak = values[0]
        max_drawdown = 0.0
        max_drawdown_duration = 0
        current_drawdown_duration = 0
        drawdowns = []

        for i, value in enumerate(values):
            if value > peak:
                peak = value
                current_drawdown_duration = 0
            else:
                current_drawdown_duration += 1
                drawdown = (peak - value) / peak
                drawdowns.append(drawdown)

                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                if current_drawdown_duration > max_drawdown_duration:
                    max_drawdown_duration = current_drawdown_duration

        average_drawdown = np.mean(drawdowns) if drawdowns else 0.0

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "average_drawdown": average_drawdown,
        }

    def _calculate_trade_metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate comprehensive trade analysis metrics."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
            }

        # Group trades by symbol to calculate P&L
        pnl_results = self._calculate_trade_pnl(trades)

        winning_trades = sum(1 for pnl in pnl_results if pnl > 0)
        losing_trades = sum(1 for pnl in pnl_results if pnl < 0)
        total_trades = len(pnl_results)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        wins = [pnl for pnl in pnl_results if pnl > 0]
        losses = [abs(pnl) for pnl in pnl_results if pnl < 0]

        total_wins = sum(wins)
        total_losses = sum(losses)

        profit_factor = (
            total_wins / total_losses
            if total_losses > 0
            else float("inf")
            if total_wins > 0
            else 0
        )

        average_win = np.mean(wins) if wins else 0
        average_loss = np.mean(losses) if losses else 0

        expectancy = (win_rate * average_win) - ((1 - win_rate) * average_loss)

        largest_win = max(wins) if wins else 0
        largest_loss = max(losses) if losses else 0

        # Calculate consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(
            pnl_results
        )

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "average_win": average_win,
            "average_loss": average_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "consecutive_wins": consecutive_wins,
            "consecutive_losses": consecutive_losses,
        }

    def _calculate_trade_pnl(self, trades: List[Trade]) -> List[float]:
        """Calculate P&L for closed positions."""
        symbol_positions = {}
        pnl_results = []

        for trade in sorted(trades, key=lambda x: x.timestamp):
            symbol = trade.symbol

            if symbol not in symbol_positions:
                symbol_positions[symbol] = {"quantity": 0, "avg_cost": 0}

            position = symbol_positions[symbol]

            if trade.side == OrderSide.BUY:
                # Add to position
                total_cost = position["quantity"] * position["avg_cost"]
                new_cost = float(trade.quantity * trade.price)
                new_quantity = position["quantity"] + float(trade.quantity)

                if new_quantity > 0:
                    position["avg_cost"] = (total_cost + new_cost) / new_quantity
                    position["quantity"] = new_quantity

            else:  # SELL
                if position["quantity"] > 0:
                    # Calculate P&L for the sold quantity
                    sold_quantity = min(float(trade.quantity), position["quantity"])
                    cost_basis = sold_quantity * position["avg_cost"]
                    sale_proceeds = sold_quantity * float(trade.price)

                    pnl = sale_proceeds - cost_basis - float(trade.commission)
                    pnl_results.append(pnl)

                    # Update position
                    position["quantity"] -= sold_quantity

        return pnl_results

    def _calculate_consecutive_trades(
        self, pnl_results: List[float]
    ) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        if not pnl_results:
            return 0, 0

        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_consecutive_wins = 0
        current_consecutive_losses = 0

        for pnl in pnl_results:
            if pnl > 0:
                current_consecutive_wins += 1
                current_consecutive_losses = 0
                max_consecutive_wins = max(
                    max_consecutive_wins, current_consecutive_wins
                )
            elif pnl < 0:
                current_consecutive_losses += 1
                current_consecutive_wins = 0
                max_consecutive_losses = max(
                    max_consecutive_losses, current_consecutive_losses
                )
            else:
                # Break even trade
                current_consecutive_wins = 0
                current_consecutive_losses = 0

        return max_consecutive_wins, max_consecutive_losses

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return float(skewness)

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        if len(returns) < 4:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
        return float(kurtosis)

    def _calculate_benchmark_metrics(
        self, strategy_returns: List[float], benchmark_returns: List[float]
    ) -> Dict[str, float]:
        """Calculate metrics relative to benchmark."""
        # Align returns length
        min_length = min(len(strategy_returns), len(benchmark_returns))
        strat_returns = np.array(strategy_returns[:min_length])
        bench_returns = np.array(benchmark_returns[:min_length])

        if len(strat_returns) < 2:
            return {}

        # Calculate beta
        covariance = np.cov(strat_returns, bench_returns)[0, 1]
        benchmark_variance = np.var(bench_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Calculate alpha
        risk_free_daily = self.risk_free_rate / 252
        alpha = np.mean(strat_returns) - (
            risk_free_daily + beta * (np.mean(bench_returns) - risk_free_daily)
        )
        alpha = alpha * 252  # Annualize

        # Calculate correlation
        correlation = (
            np.corrcoef(strat_returns, bench_returns)[0, 1]
            if len(strat_returns) > 1
            else 0
        )

        # Calculate tracking error
        excess_returns = strat_returns - bench_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)

        # Information ratio
        information_ratio = (
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            if np.std(excess_returns) > 0
            else 0
        )

        return {
            "alpha": alpha,
            "beta": beta,
            "correlation": correlation,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
        }

    def _get_empty_metrics(self) -> AdvancedPerformanceMetrics:
        """Get empty metrics for cases with insufficient data."""
        return AdvancedPerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            cumulative_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            average_drawdown=0.0,
            recovery_factor=0.0,
            volatility=0.0,
            downside_volatility=0.0,
            upside_volatility=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            skewness=0.0,
            kurtosis=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )

    def generate_performance_report(
        self, metrics: AdvancedPerformanceMetrics
    ) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "Summary": {
                "Total Return": f"{metrics.total_return:.2%}",
                "Annualized Return": f"{metrics.annualized_return:.2%}",
                "Sharpe Ratio": f"{metrics.sharpe_ratio:.2f}",
                "Max Drawdown": f"{metrics.max_drawdown:.2%}",
                "Win Rate": f"{metrics.win_rate:.2%}",
                "Profit Factor": f"{metrics.profit_factor:.2f}",
            },
            "Risk Metrics": {
                "Volatility": f"{metrics.volatility:.2%}",
                "Sortino Ratio": f"{metrics.sortino_ratio:.2f}",
                "Calmar Ratio": f"{metrics.calmar_ratio:.2f}",
                "VaR (95%)": f"{metrics.var_95:.2%}",
                "CVaR (95%)": f"{metrics.cvar_95:.2%}",
                "Skewness": f"{metrics.skewness:.2f}",
                "Kurtosis": f"{metrics.kurtosis:.2f}",
            },
            "Trade Analysis": {
                "Total Trades": metrics.total_trades,
                "Winning Trades": metrics.winning_trades,
                "Losing Trades": metrics.losing_trades,
                "Average Win": f"${metrics.average_win:.2f}",
                "Average Loss": f"${metrics.average_loss:.2f}",
                "Largest Win": f"${metrics.largest_win:.2f}",
                "Largest Loss": f"${metrics.largest_loss:.2f}",
                "Expectancy": f"${metrics.expectancy:.2f}",
                "Max Consecutive Wins": metrics.consecutive_wins,
                "Max Consecutive Losses": metrics.consecutive_losses,
            },
            "Drawdown Analysis": {
                "Max Drawdown": f"{metrics.max_drawdown:.2%}",
                "Max Drawdown Duration": f"{metrics.max_drawdown_duration} days",
                "Average Drawdown": f"{metrics.average_drawdown:.2%}",
                "Recovery Factor": f"{metrics.recovery_factor:.2f}",
            },
        }

        if metrics.alpha is not None:
            report["Benchmark Comparison"] = {
                "Alpha": f"{metrics.alpha:.2%}",
                "Beta": f"{metrics.beta:.2f}",
                "Correlation": f"{metrics.correlation:.2f}",
                "Tracking Error": f"{metrics.tracking_error:.2%}",
                "Information Ratio": f"{metrics.information_ratio:.2f}",
            }

        return report
