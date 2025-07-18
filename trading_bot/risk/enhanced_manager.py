"""Enhanced risk management with advanced position sizing and correlation analysis."""

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.logging import TradingLogger
from ..core.models import MarketData, Position


@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio."""

    var_95: float
    var_99: float
    expected_shortfall: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_risk: float


@dataclass
class PositionSizingResult:
    """Result from position sizing calculation."""

    symbol: str
    recommended_shares: Decimal
    risk_amount: Decimal
    max_loss_amount: Decimal
    position_value: Decimal
    risk_percentage: float
    sizing_method: str
    confidence: float


class VolatilityCalculator:
    """Calculate various volatility measures."""

    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days

    def calculate_historical_volatility(self, returns: List[float]) -> float:
        """Calculate annualized historical volatility."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        variance = np.mean([(r - mean_return) ** 2 for r in returns])
        daily_vol = math.sqrt(variance)

        # Annualize assuming 252 trading days
        annual_vol = daily_vol * math.sqrt(252)
        return annual_vol

    def calculate_ewma_volatility(
        self, returns: List[float], lambda_factor: float = 0.94
    ) -> float:
        """Calculate EWMA (Exponentially Weighted Moving Average) volatility."""
        if len(returns) < 2:
            return 0.0

        # Initialize with first return squared
        ewma_var = returns[0] ** 2

        for i in range(1, len(returns)):
            ewma_var = lambda_factor * ewma_var + (1 - lambda_factor) * (
                returns[i] ** 2
            )

        return math.sqrt(ewma_var * 252)  # Annualized

    def calculate_garch_volatility(self, returns: List[float]) -> float:
        """Simplified GARCH(1,1) volatility estimation."""
        if len(returns) < 10:
            return self.calculate_historical_volatility(returns)

        # GARCH(1,1) parameters (simplified estimation)
        omega = 0.000001  # Long-term variance
        alpha = 0.1  # ARCH parameter
        beta = 0.85  # GARCH parameter

        # Initialize variance
        variance = np.var(returns)

        # Calculate GARCH variance
        for return_val in returns[-20:]:  # Use last 20 observations
            variance = omega + alpha * (return_val ** 2) + beta * variance

        return math.sqrt(variance * 252)  # Annualized


class CorrelationAnalyzer:
    """Analyze correlations between assets."""

    def __init__(self, min_observations: int = 30):
        self.min_observations = min_observations

    def calculate_correlation_matrix(
        self, returns_data: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between assets."""
        symbols = list(returns_data.keys())
        correlation_matrix = {}

        for symbol1 in symbols:
            correlation_matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    correlation = self._calculate_correlation(
                        returns_data[symbol1], returns_data[symbol2]
                    )
                    correlation_matrix[symbol1][symbol2] = correlation

        return correlation_matrix

    def _calculate_correlation(
        self, returns1: List[float], returns2: List[float]
    ) -> float:
        """Calculate correlation between two return series."""
        if (
            len(returns1) < self.min_observations
            or len(returns2) < self.min_observations
        ):
            return 0.0

        # Use the shorter series length
        min_length = min(len(returns1), len(returns2))
        r1 = returns1[-min_length:]
        r2 = returns2[-min_length:]

        if len(r1) < 2:
            return 0.0

        try:
            correlation = np.corrcoef(r1, r2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def calculate_portfolio_correlation_risk(
        self, weights: Dict[str, float], correlation_matrix: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate portfolio correlation risk (diversification ratio)."""
        symbols = list(weights.keys())

        if len(symbols) < 2:
            return 0.0

        # Calculate weighted average correlation
        total_correlation = 0.0
        total_weight_pairs = 0.0

        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i != j:
                    weight1 = weights.get(symbol1, 0)
                    weight2 = weights.get(symbol2, 0)
                    correlation = correlation_matrix.get(symbol1, {}).get(symbol2, 0)

                    total_correlation += weight1 * weight2 * abs(correlation)
                    total_weight_pairs += weight1 * weight2

        if total_weight_pairs == 0:
            return 0.0

        avg_correlation = total_correlation / total_weight_pairs
        return avg_correlation


class PositionSizer:
    """Advanced position sizing algorithms."""

    def __init__(
        self, max_risk_per_trade: float = 0.02, max_portfolio_risk: float = 0.06
    ):
        self.max_risk_per_trade = max_risk_per_trade  # 2% max risk per trade
        self.max_portfolio_risk = max_portfolio_risk  # 6% max portfolio risk
        self.logger = TradingLogger("position_sizer")

    def fixed_fractional_sizing(
        self,
        account_value: Decimal,
        risk_amount: Decimal,
        entry_price: Decimal,
        stop_price: Decimal,
    ) -> PositionSizingResult:
        """Fixed fractional position sizing based on risk amount."""
        if stop_price == entry_price:
            return self._no_position_result("Invalid stop price")

        risk_per_share = abs(entry_price - stop_price)
        max_shares = (
            risk_amount / risk_per_share if risk_per_share > 0 else Decimal("0")
        )

        position_value = max_shares * entry_price
        risk_percentage = float(risk_amount / account_value) if account_value > 0 else 0

        return PositionSizingResult(
            symbol="",
            recommended_shares=max_shares,
            risk_amount=risk_amount,
            max_loss_amount=risk_amount,
            position_value=position_value,
            risk_percentage=risk_percentage,
            sizing_method="fixed_fractional",
            confidence=0.85,
        )

    def volatility_based_sizing(
        self,
        account_value: Decimal,
        volatility: float,
        entry_price: Decimal,
        target_volatility: float = 0.15,
    ) -> PositionSizingResult:
        """Position sizing based on volatility targeting."""
        if volatility <= 0:
            return self._no_position_result("Invalid volatility")

        # Scale position size inversely with volatility
        vol_scaling_factor = target_volatility / volatility
        base_allocation = float(account_value) * self.max_risk_per_trade

        adjusted_allocation = base_allocation * vol_scaling_factor
        adjusted_allocation = min(
            adjusted_allocation, float(account_value) * 0.1
        )  # Max 10% allocation

        max_shares = Decimal(str(adjusted_allocation)) / entry_price
        position_value = max_shares * entry_price

        return PositionSizingResult(
            symbol="",
            recommended_shares=max_shares,
            risk_amount=Decimal(str(adjusted_allocation)),
            max_loss_amount=Decimal(str(adjusted_allocation * 0.5)),  # Estimate
            position_value=position_value,
            risk_percentage=adjusted_allocation / float(account_value),
            sizing_method="volatility_based",
            confidence=0.75,
        )

    def kelly_criterion_sizing(
        self, win_rate: float, avg_win: float, avg_loss: float, account_value: Decimal
    ) -> PositionSizingResult:
        """Kelly Criterion position sizing."""
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return self._no_position_result("Invalid Kelly parameters")

        # Kelly fraction: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate

        kelly_fraction = (b * p - q) / b

        # Apply conservative scaling (quarter Kelly)
        conservative_kelly = max(0, min(0.25, kelly_fraction * 0.25))

        allocation = float(account_value) * conservative_kelly

        return PositionSizingResult(
            symbol="",
            recommended_shares=Decimal("0"),  # Will be calculated with price
            risk_amount=Decimal(str(allocation)),
            max_loss_amount=Decimal(str(allocation)),
            position_value=Decimal(str(allocation)),
            risk_percentage=conservative_kelly,
            sizing_method="kelly_criterion",
            confidence=0.6,
        )

    def atr_based_sizing(
        self,
        account_value: Decimal,
        atr: float,
        entry_price: Decimal,
        atr_multiplier: float = 2.0,
    ) -> PositionSizingResult:
        """ATR-based position sizing."""
        if atr <= 0:
            return self._no_position_result("Invalid ATR")

        # Use ATR multiple as stop distance
        stop_distance = atr * atr_multiplier
        risk_amount = float(account_value) * self.max_risk_per_trade

        max_shares = Decimal(str(risk_amount)) / Decimal(str(stop_distance))
        position_value = max_shares * entry_price

        return PositionSizingResult(
            symbol="",
            recommended_shares=max_shares,
            risk_amount=Decimal(str(risk_amount)),
            max_loss_amount=Decimal(str(risk_amount)),
            position_value=position_value,
            risk_percentage=self.max_risk_per_trade,
            sizing_method="atr_based",
            confidence=0.8,
        )

    def _no_position_result(self, reason: str) -> PositionSizingResult:
        """Return zero position sizing result."""
        return PositionSizingResult(
            symbol="",
            recommended_shares=Decimal("0"),
            risk_amount=Decimal("0"),
            max_loss_amount=Decimal("0"),
            position_value=Decimal("0"),
            risk_percentage=0.0,
            sizing_method="none",
            confidence=0.0,
        )


class EnhancedRiskManager:
    """Enhanced risk management with advanced features."""

    def __init__(
        self, max_portfolio_risk: float = 0.1, max_single_position: float = 0.05
    ):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_single_position = max_single_position
        self.logger = TradingLogger("enhanced_risk_manager")

        # Components
        self.volatility_calculator = VolatilityCalculator()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.position_sizer = PositionSizer()

        # Data storage
        self.returns_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=252)
        )  # 1 year
        self.price_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))

        # Risk tracking
        self.position_risks: Dict[str, float] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}

    def update_market_data(self, symbol: str, bar: MarketData) -> None:
        """Update market data for risk calculations."""
        current_price = float(bar.close)
        self.price_data[symbol].append(current_price)

        # Calculate returns
        if len(self.price_data[symbol]) > 1:
            prev_price = self.price_data[symbol][-2]
            daily_return = (current_price - prev_price) / prev_price
            self.returns_data[symbol].append(daily_return)

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: Decimal,
        stop_price: Optional[Decimal],
        account_value: Decimal,
        method: str = "fixed_fractional",
    ) -> PositionSizingResult:
        """Calculate optimal position size using specified method."""

        # Get risk amount
        risk_amount = account_value * Decimal(str(self.max_single_position))

        if method == "fixed_fractional" and stop_price:
            result = self.position_sizer.fixed_fractional_sizing(
                account_value, risk_amount, entry_price, stop_price
            )
        elif method == "volatility_based":
            volatility = self._get_volatility(symbol)
            result = self.position_sizer.volatility_based_sizing(
                account_value, volatility, entry_price
            )
        elif method == "atr_based":
            atr = self._get_atr(symbol)
            result = self.position_sizer.atr_based_sizing(
                account_value, atr, entry_price
            )
        elif method == "kelly":
            # Simplified Kelly - would need historical trade data
            result = self.position_sizer.kelly_criterion_sizing(
                win_rate=0.55, avg_win=0.02, avg_loss=0.01, account_value=account_value
            )
        else:
            result = self.position_sizer._no_position_result("Unknown method")

        result.symbol = symbol
        return result

    def calculate_volatility_stop(
        self,
        symbol: str,
        entry_price: Decimal,
        position_side: str,
        volatility_multiplier: float = 2.0,
    ) -> Optional[Decimal]:
        """Calculate volatility-based stop loss."""
        volatility = self._get_volatility(symbol)
        if volatility <= 0:
            return None

        # Convert annual volatility to daily
        daily_vol = volatility / math.sqrt(252)

        # Calculate stop distance
        stop_distance = entry_price * Decimal(str(daily_vol * volatility_multiplier))

        if position_side.lower() == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def assess_portfolio_risk(
        self, positions: Dict[str, Position], current_prices: Dict[str, Decimal]
    ) -> RiskMetrics:
        """Assess overall portfolio risk."""

        if not positions:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)

        # Calculate portfolio weights
        total_value = sum(
            abs(float(pos.quantity * current_prices.get(symbol, Decimal("0"))))
            for symbol, pos in positions.items()
        )

        if total_value == 0:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)

        weights = {}
        portfolio_returns = []

        for symbol, position in positions.items():
            if symbol in current_prices:
                position_value = abs(float(position.quantity * current_prices[symbol]))
                weights[symbol] = position_value / total_value

                # Get returns for portfolio calculation
                if symbol in self.returns_data and len(self.returns_data[symbol]) > 0:
                    symbol_returns = list(self.returns_data[symbol])

                    # Weight the returns
                    weighted_returns = [r * weights[symbol] for r in symbol_returns]
                    if not portfolio_returns:
                        portfolio_returns = weighted_returns
                    else:
                        # Add to existing portfolio returns
                        min_len = min(len(portfolio_returns), len(weighted_returns))
                        portfolio_returns = [
                            portfolio_returns[i] + weighted_returns[i]
                            for i in range(min_len)
                        ]

        # Calculate risk metrics
        if portfolio_returns:
            volatility = self.volatility_calculator.calculate_historical_volatility(
                portfolio_returns
            )
            var_95 = (
                np.percentile(portfolio_returns, 5)
                if len(portfolio_returns) > 20
                else 0
            )
            var_99 = (
                np.percentile(portfolio_returns, 1)
                if len(portfolio_returns) > 100
                else 0
            )

            # Expected Shortfall (Conditional VaR)
            es_95 = (
                np.mean([r for r in portfolio_returns if r <= var_95])
                if var_95 < 0
                else 0
            )

            # Sharpe ratio
            mean_return = np.mean(portfolio_returns)
            sharpe = mean_return / volatility if volatility > 0 else 0

            # Max drawdown (simplified)
            cumulative = np.cumprod([1 + r for r in portfolio_returns])
            rolling_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        else:
            volatility = var_95 = var_99 = es_95 = sharpe = max_drawdown = 0

        # Correlation risk
        correlation_risk = self._calculate_portfolio_correlation_risk(weights)

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=es_95,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            correlation_risk=correlation_risk,
        )

    def check_risk_limits(
        self,
        symbol: str,
        proposed_quantity: Decimal,
        current_price: Decimal,
        positions: Dict[str, Position],
    ) -> Dict[str, Any]:
        """Check if proposed position violates risk limits."""

        results = {
            "approved": True,
            "violations": [],
            "warnings": [],
            "max_allowed_quantity": proposed_quantity,
            "risk_metrics": {},
        }

        # Calculate position value
        position_value = abs(proposed_quantity * current_price)

        # Check single position limit
        # This would need account value - simplified for now
        account_value = Decimal("100000")  # Default for testing
        max_position_value = account_value * Decimal(str(self.max_single_position))

        if position_value > max_position_value:
            results["approved"] = False
            results["violations"].append(
                f"Position size ${position_value} exceeds limit ${max_position_value}"
            )
            results["max_allowed_quantity"] = max_position_value / current_price

        # Check correlation limits
        correlation_risk = self._get_correlation_risk(symbol, positions)
        if correlation_risk > 0.8:
            results["warnings"].append(f"High correlation risk: {correlation_risk:.2f}")

        # Check volatility
        volatility = self._get_volatility(symbol)
        if volatility > 0.4:  # 40% annual volatility threshold
            results["warnings"].append(f"High volatility: {volatility:.2%}")

        return results

    def _get_volatility(self, symbol: str) -> float:
        """Get current volatility estimate for symbol."""
        if symbol not in self.returns_data or len(self.returns_data[symbol]) < 10:
            return 0.2  # Default 20% volatility

        returns = list(self.returns_data[symbol])
        return self.volatility_calculator.calculate_historical_volatility(returns)

    def _get_atr(self, symbol: str) -> float:
        """Get Average True Range for symbol (simplified)."""
        if symbol not in self.price_data or len(self.price_data[symbol]) < 10:
            return 1.0  # Default ATR

        prices = list(self.price_data[symbol])

        # Simplified ATR calculation
        ranges = []
        for i in range(1, len(prices)):
            daily_range = abs(prices[i] - prices[i - 1])
            ranges.append(daily_range)

        return np.mean(ranges[-14:]) if ranges else 1.0  # 14-day ATR

    def _calculate_portfolio_correlation_risk(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio correlation risk."""
        if len(weights) < 2:
            return 0.0

        # Update correlation matrix
        returns_dict = {
            symbol: list(returns)
            for symbol, returns in self.returns_data.items()
            if len(returns) >= 30
        }

        if len(returns_dict) >= 2:
            self.correlation_matrix = (
                self.correlation_analyzer.calculate_correlation_matrix(returns_dict)
            )
            return self.correlation_analyzer.calculate_portfolio_correlation_risk(
                weights, self.correlation_matrix
            )

        return 0.0

    def _get_correlation_risk(
        self, symbol: str, positions: Dict[str, Position]
    ) -> float:
        """Get correlation risk for adding a symbol to existing positions."""
        if not positions or symbol not in self.correlation_matrix:
            return 0.0

        correlations = []
        for existing_symbol in positions.keys():
            if (
                existing_symbol in self.correlation_matrix
                and symbol in self.correlation_matrix[existing_symbol]
            ):
                correlation = self.correlation_matrix[existing_symbol][symbol]
                correlations.append(abs(correlation))

        return max(correlations) if correlations else 0.0

    def get_risk_dashboard(
        self, positions: Dict[str, Position], current_prices: Dict[str, Decimal]
    ) -> Dict[str, Any]:
        """Generate comprehensive risk dashboard."""

        portfolio_metrics = self.assess_portfolio_risk(positions, current_prices)

        # Position-level risks
        position_risks = {}
        for symbol, position in positions.items():
            if symbol in current_prices:
                volatility = self._get_volatility(symbol)
                correlation_risk = self._get_correlation_risk(symbol, positions)

                position_risks[symbol] = {
                    "volatility": volatility,
                    "correlation_risk": correlation_risk,
                    "position_value": float(
                        abs(position.quantity * current_prices[symbol])
                    ),
                    "atr": self._get_atr(symbol),
                }

        return {
            "portfolio_metrics": {
                "var_95": portfolio_metrics.var_95,
                "var_99": portfolio_metrics.var_99,
                "volatility": portfolio_metrics.volatility,
                "sharpe_ratio": portfolio_metrics.sharpe_ratio,
                "max_drawdown": portfolio_metrics.max_drawdown,
                "correlation_risk": portfolio_metrics.correlation_risk,
            },
            "position_risks": position_risks,
            "risk_limits": {
                "max_portfolio_risk": self.max_portfolio_risk,
                "max_single_position": self.max_single_position,
            },
            "monitored_symbols": len(self.returns_data),
            "data_quality": {
                symbol: len(returns) for symbol, returns in self.returns_data.items()
            },
        }
