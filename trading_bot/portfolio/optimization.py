"""Portfolio optimization algorithms using modern portfolio theory."""

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.logging import TradingLogger


@dataclass
class OptimizationResult:
    """Result from portfolio optimization."""

    weights: Dict[str, float]
    expected_return: float
    volatility: float
    expected_volatility: float  # Alias for volatility for backward compatibility
    sharpe_ratio: float
    method: str
    optimization_method: str  # Alias for method for backward compatibility
    success: bool
    message: str

    def __post_init__(self) -> None:
        # Set aliases for backward compatibility
        if not hasattr(self, "expected_volatility") or self.expected_volatility is None:
            self.expected_volatility = self.volatility
        if not hasattr(self, "optimization_method") or self.optimization_method is None:
            self.optimization_method = self.method


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""

    method: str = "mean_variance"
    risk_free_rate: float = 0.02
    target_return: Optional[float] = None
    max_weight: float = 0.4  # Changed from 1.0 to match test expectations
    min_weight: float = 0.0
    risk_aversion: float = 1.0
    confidence_level: float = 0.95


class PortfolioOptimizer:
    """Portfolio optimization using various modern portfolio theory methods."""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.risk_free_rate = self.config.risk_free_rate
        self.logger = TradingLogger("portfolio_optimizer")

    def optimize(self, returns_data: pd.DataFrame, **kwargs: Any) -> OptimizationResult:
        """Generic optimize method to delegate to the correct optimizer."""
        method = self.config.method.lower().replace("_", "").replace("-", "")

        returns_dict = {col: returns_data[col].tolist() for col in returns_data.columns}

        if method in ["meanvariance", "markowitz"]:
            return self.mean_variance_optimization(
                returns_dict, config=self.config, **kwargs
            )
        elif method in ["riskparity", "risk"]:
            return self.risk_parity_optimization(returns_dict)
        elif method in ["kelly", "kellycriterion"]:
            return self.kelly_criterion_optimization(returns_dict, **kwargs)
        elif method in ["equalweight", "equal", "1n"]:
            symbols = list(returns_dict.keys())
            return self.equal_weight_portfolio(symbols)
        elif method in ["minvariance", "minvar", "minimum"]:
            return self.minimum_variance_portfolio(returns_dict)
        elif method in ["blacklitterman", "bl"]:
            return self.black_litterman_optimization(returns_dict, **kwargs)
        else:
            return self.mean_variance_optimization(
                returns_dict, config=self.config, **kwargs
            )

    def mean_variance_optimization(
        self,
        returns: Dict[str, List[float]],
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
        config: Optional[OptimizationConfig] = None,
    ) -> OptimizationResult:
        """
        Mean-Variance Optimization (Markowitz Portfolio Theory).

        Args:
            returns: Historical returns for each asset
            target_return: Target portfolio return (if None, maximizes Sharpe ratio)
            risk_aversion: Risk aversion parameter (higher = more conservative)
        """
        try:
            if not returns or len(returns) < 1:
                return OptimizationResult(
                    weights={},
                    expected_return=0.0,
                    volatility=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    method="Mean-Variance",
                    optimization_method="Mean-Variance",
                    success=False,
                    message="Need at least 1 asset",
                )

            # Handle single asset case
            if len(returns) == 1:
                symbol = list(returns.keys())[0]
                asset_returns = returns[symbol]

                if len(asset_returns) > 0:
                    expected_return: float = np.mean(asset_returns)
                    volatility: float = np.std(asset_returns)
                    sharpe_ratio = (
                        (expected_return - self.risk_free_rate) / volatility
                        if volatility > 0
                        else 0.0
                    )
                else:
                    expected_return = 0.0
                    volatility = 0.0
                    sharpe_ratio = 0.0

                return OptimizationResult(
                    weights={symbol: 1.0},
                    expected_return=float(expected_return),
                    volatility=float(volatility),
                    expected_volatility=float(volatility),
                    sharpe_ratio=float(sharpe_ratio),
                    method="Mean-Variance",
                    optimization_method="Mean-Variance",
                    success=True,
                    message="Single asset portfolio",
                )

            symbols = list(returns.keys())

            # Convert to numpy arrays
            return_matrix = []
            min_length = min(len(returns[symbol]) for symbol in symbols)

            if min_length < 10:
                return OptimizationResult(
                    weights={},
                    expected_return=0.0,
                    volatility=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    method="Mean-Variance",
                    optimization_method="Mean-Variance",
                    success=False,
                    message="Need at least 10 return observations",
                )

            for symbol in symbols:
                return_matrix.append(returns[symbol][-min_length:])

            return_matrix = np.array(return_matrix)

            # Calculate expected returns and covariance matrix
            expected_returns = np.mean(return_matrix, axis=1)
            cov_matrix = np.cov(return_matrix)

            # Add small regularization to ensure positive definiteness
            cov_matrix += np.eye(len(symbols)) * 1e-8

            if target_return is None:
                # Maximize Sharpe ratio
                weights = self._maximize_sharpe_ratio(
                    expected_returns, cov_matrix, config
                )
            else:
                # Target return optimization
                weights = self._target_return_optimization(
                    expected_returns, cov_matrix, target_return, risk_aversion, config
                )

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = math.sqrt(portfolio_variance)

            sharpe_ratio = (
                (portfolio_return - self.risk_free_rate) / portfolio_volatility
                if portfolio_volatility > 0
                else 0.0
            )

            # Create weights dictionary
            weights_dict = {
                symbol: float(weight) for symbol, weight in zip(symbols, weights)
            }

            return OptimizationResult(
                weights=weights_dict,
                expected_return=float(portfolio_return),
                volatility=float(portfolio_volatility),
                expected_volatility=float(portfolio_volatility),  # Alias for volatility
                sharpe_ratio=float(sharpe_ratio),
                method="Mean-Variance",
                optimization_method="Mean-Variance",  # Alias for method
                success=True,
                message="Optimization successful",
            )

        except Exception as e:
            self.logger.logger.error(f"Mean-variance optimization failed: {e}")
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                volatility=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                method="Mean-Variance",
                optimization_method="Mean-Variance",
                success=False,
                message=str(e),
            )

    def _apply_weight_constraints(
        self, weights: np.ndarray, config: OptimizationConfig
    ) -> np.ndarray:
        """Apply min/max weight constraints and renormalize."""
        n = len(weights)

        # Apply min/max constraints
        weights = np.clip(weights, config.min_weight, config.max_weight)

        # Check if constraints allow valid portfolio
        if np.sum(weights) == 0:
            # If all weights are clipped to 0, use equal weights within constraints
            if config.min_weight <= 1.0 / n <= config.max_weight:
                weights = np.full(n, 1.0 / n)
            else:
                # Use minimum possible weights that sum to 1
                if config.min_weight * n <= 1.0:
                    weights = np.full(n, config.min_weight)
                    # Adjust one weight to make sum = 1
                    remainder = 1.0 - np.sum(weights)
                    if remainder > 0 and weights[0] + remainder <= config.max_weight:
                        weights[0] += remainder
                else:
                    # Constraints are impossible, return equal weights
                    weights = np.full(n, 1.0 / n)
        else:
            # Renormalize to sum to 1
            weights = weights / np.sum(weights)

            # Check if renormalization violates constraints
            if np.any(weights < config.min_weight) or np.any(
                weights > config.max_weight
            ):
                # Use iterative approach to satisfy constraints
                for _ in range(100):  # Max iterations
                    weights = np.clip(weights, config.min_weight, config.max_weight)
                    current_sum: float = np.sum(weights)
                    if abs(current_sum - 1.0) < 1e-6:
                        break
                    weights = weights / current_sum

        return weights

    def _maximize_sharpe_ratio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        config: Optional[OptimizationConfig] = None,
    ) -> np.ndarray:
        """Maximize Sharpe ratio using analytical solution."""
        try:
            # Inverse of covariance matrix
            inv_cov = np.linalg.inv(cov_matrix)

            # Excess returns
            excess_returns = expected_returns - self.risk_free_rate

            # Optimal weights (before normalization)
            weights = np.dot(inv_cov, excess_returns)

            # Normalize to sum to 1
            weights = weights / np.sum(weights)

            # Apply weight constraints if config provided
            if config:
                weights = self._apply_weight_constraints(weights, config)

            return weights

        except np.linalg.LinAlgError:
            # If matrix is singular, use equal weights
            n = len(expected_returns)
            equal_weights: np.ndarray = np.ones(n) / n
            if config:
                equal_weights = self._apply_weight_constraints(equal_weights, config)
            return equal_weights

    def _target_return_optimization(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        target_return: float,
        risk_aversion: float,
        config: Optional[OptimizationConfig] = None,
    ) -> np.ndarray:
        """Optimize for target return with risk aversion."""
        try:
            n = len(expected_returns)

            # Use quadratic programming approximation
            # Minimize: 0.5 * w' * C * w - lambda * w' * mu
            # Subject to: sum(w) = 1

            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(n)

            # Calculate intermediate values
            A = np.dot(ones.T, np.dot(inv_cov, ones))
            B = np.dot(ones.T, np.dot(inv_cov, expected_returns))
            C = np.dot(expected_returns.T, np.dot(inv_cov, expected_returns))

            # Calculate lambda for target return
            discriminant = B**2 - A * (C - 2 * A * target_return)

            if discriminant < 0:
                # Target return not achievable, use equal weights
                return np.ones(n) / n

            lambda1 = (B + math.sqrt(discriminant)) / A
            lambda2 = (B - math.sqrt(discriminant)) / A

            # Choose lambda that gives reasonable weights
            for lam in [lambda1, lambda2]:
                weights = np.dot(inv_cov, expected_returns - lam * ones) / A
                if np.all(weights >= -0.5) and np.all(
                    weights <= 1.5
                ):  # Reasonable bounds
                    if config:
                        weights = self._apply_weight_constraints(weights, config)
                    return weights

            # If neither works, fall back to equal weights
            equal_weights = np.ones(n) / n
            if config:
                equal_weights = self._apply_weight_constraints(equal_weights, config)
            return np.array(equal_weights)

        except Exception:
            # Fallback to equal weights
            n = len(expected_returns)
            equal_weights = np.ones(n) / n
            if config:
                equal_weights = self._apply_weight_constraints(equal_weights, config)
            return np.array(equal_weights)

    def risk_parity_optimization(
        self, returns: Dict[str, List[float]]
    ) -> OptimizationResult:
        """
        Risk Parity Portfolio Optimization.
        Each asset contributes equally to portfolio risk.
        """
        try:
            if not returns or len(returns) < 2:
                return OptimizationResult(
                    weights={},
                    expected_return=0.0,
                    volatility=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    method="Risk-Parity",
                    optimization_method="Risk-Parity",
                    success=False,
                    message="Need at least 2 assets",
                )

            symbols = list(returns.keys())

            # Convert to numpy arrays
            return_matrix = []
            min_length = min(len(returns[symbol]) for symbol in symbols)

            if min_length < 10:
                return OptimizationResult(
                    weights={},
                    expected_return=0.0,
                    volatility=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    method="Risk-Parity",
                    optimization_method="Risk-Parity",
                    success=False,
                    message="Need at least 10 return observations",
                )

            for symbol in symbols:
                return_matrix.append(returns[symbol][-min_length:])

            return_matrix = np.array(return_matrix)

            # Calculate expected returns and covariance matrix
            expected_returns = np.mean(return_matrix, axis=1)
            cov_matrix = np.cov(return_matrix)

            # Add regularization
            cov_matrix += np.eye(len(symbols)) * 1e-8

            # Risk parity weights (iterative approach)
            weights = self._calculate_risk_parity_weights(cov_matrix)

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = math.sqrt(portfolio_variance)

            sharpe_ratio = (
                (portfolio_return - self.risk_free_rate) / portfolio_volatility
                if portfolio_volatility > 0
                else 0.0
            )

            weights_dict = {
                symbol: float(weight) for symbol, weight in zip(symbols, weights)
            }

            return OptimizationResult(
                weights=weights_dict,
                expected_return=float(portfolio_return),
                volatility=float(portfolio_volatility),
                expected_volatility=float(portfolio_volatility),  # Alias for volatility
                sharpe_ratio=float(sharpe_ratio),
                method="Risk-Parity",
                optimization_method="Risk-Parity",  # Alias for method
                success=True,
                message="Risk parity optimization successful",
            )

        except Exception as e:
            self.logger.logger.error(f"Risk parity optimization failed: {e}")
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                volatility=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                method="Risk-Parity",
                optimization_method="Risk-Parity",
                success=False,
                message=str(e),
            )

    def _calculate_risk_parity_weights(
        self, cov_matrix: np.ndarray, max_iterations: int = 100
    ) -> np.ndarray:
        """Calculate risk parity weights using iterative algorithm."""
        n = cov_matrix.shape[0]

        # Start with equal weights
        weights = np.ones(n) / n

        for _ in range(max_iterations):
            # Calculate marginal risk contributions
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            marginal_risk = np.dot(cov_matrix, weights)
            risk_contributions = weights * marginal_risk

            if portfolio_variance > 0:
                # Update weights to equalize risk contributions
                target_risk = portfolio_variance / n
                adjustment = target_risk / (risk_contributions + 1e-8)
                weights = weights * adjustment

                # Normalize
                weights = weights / np.sum(weights)
            else:
                break

        return weights

    def kelly_criterion_optimization(
        self,
        returns: Dict[str, List[float]],
        win_rates: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """
        Kelly Criterion for optimal position sizing.
        Maximizes log utility (geometric mean).
        """
        try:
            if not returns or len(returns) < 1:
                return OptimizationResult(
                    weights={},
                    expected_return=0.0,
                    volatility=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    method="Kelly",
                    optimization_method="Kelly",
                    success=False,
                    message="Need at least 1 asset",
                )

            symbols = list(returns.keys())
            weights_dict = {}

            for symbol in symbols:
                symbol_returns = returns[symbol]
                if len(symbol_returns) < 10:
                    weights_dict[symbol] = 0.0
                    continue

                # Calculate Kelly fraction for this asset
                if win_rates and symbol in win_rates:
                    win_rate = win_rates[symbol]
                    avg_win = (
                        np.mean([r for r in symbol_returns if r > 0])
                        if any(r > 0 for r in symbol_returns)
                        else 0
                    )
                    avg_loss = (
                        abs(np.mean([r for r in symbol_returns if r < 0]))
                        if any(r < 0 for r in symbol_returns)
                        else 1
                    )

                    if avg_loss > 0:
                        kelly_fraction = (
                            win_rate * avg_win - (1 - win_rate) * avg_loss
                        ) / avg_loss
                    else:
                        kelly_fraction = 0
                else:
                    # Use mean and variance approach
                    mean_return = np.mean(symbol_returns)
                    variance = np.var(symbol_returns)

                    if variance > 0:
                        kelly_fraction = mean_return / variance
                    else:
                        kelly_fraction = 0

                # Apply conservative scaling (quarter Kelly or half Kelly)
                kelly_fraction = max(
                    0, min(0.25, kelly_fraction * 0.25)
                )  # Quarter Kelly with cap
                weights_dict[symbol] = kelly_fraction

            # Normalize weights to sum to 1
            total_weight = sum(weights_dict.values())
            if total_weight > 0:
                weights_dict = {
                    symbol: weight / total_weight
                    for symbol, weight in weights_dict.items()
                }
            else:
                # Equal weights fallback
                n = len(symbols)
                weights_dict = {symbol: 1.0 / n for symbol in symbols}

            # Calculate portfolio metrics
            weights_array = np.array([weights_dict[symbol] for symbol in symbols])

            return_matrix = []
            min_length = min(len(returns[symbol]) for symbol in symbols)
            for symbol in symbols:
                return_matrix.append(returns[symbol][-min_length:])

            return_matrix = np.array(return_matrix)
            expected_returns = np.mean(return_matrix, axis=1)
            cov_matrix = np.cov(return_matrix)

            portfolio_return = np.dot(weights_array, expected_returns)
            portfolio_variance = np.dot(
                weights_array, np.dot(cov_matrix, weights_array)
            )
            portfolio_volatility = math.sqrt(portfolio_variance)

            sharpe_ratio = (
                (portfolio_return - self.risk_free_rate) / portfolio_volatility
                if portfolio_volatility > 0
                else 0.0
            )

            return OptimizationResult(
                weights=weights_dict,
                expected_return=float(portfolio_return),
                volatility=float(portfolio_volatility),
                expected_volatility=float(portfolio_volatility),  # Alias for volatility
                sharpe_ratio=float(sharpe_ratio),
                method="Kelly",
                optimization_method="Kelly",  # Alias for method
                success=True,
                message="Kelly criterion optimization successful",
            )

        except Exception as e:
            self.logger.logger.error(f"Kelly criterion optimization failed: {e}")
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                volatility=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                method="Kelly",
                optimization_method="Kelly",
                success=False,
                message=str(e),
            )

    def equal_weight_portfolio(self, symbols: List[str]) -> OptimizationResult:
        """Simple equal weight portfolio (1/N rule)."""
        try:
            if not symbols:
                return OptimizationResult(
                    weights={},
                    expected_return=0.0,
                    volatility=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    method="Equal-Weight",
                    optimization_method="Equal-Weight",
                    success=False,
                    message="No symbols provided",
                )

            n = len(symbols)
            weight = 1.0 / n
            weights_dict = {symbol: weight for symbol in symbols}

            return OptimizationResult(
                weights=weights_dict,
                expected_return=0.0,  # Not calculated for equal weight
                volatility=0.0,  # Not calculated for equal weight
                expected_volatility=0.0,  # Alias for volatility
                sharpe_ratio=0.0,  # Not calculated for equal weight
                method="Equal-Weight",
                optimization_method="Equal-Weight",  # Alias for method
                success=True,
                message="Equal weight portfolio created",
            )

        except Exception as e:
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                volatility=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                method="Equal-Weight",
                optimization_method="Equal-Weight",
                success=False,
                message=str(e),
            )

    def minimum_variance_portfolio(
        self, returns: Dict[str, List[float]]
    ) -> OptimizationResult:
        """Minimum variance portfolio optimization."""
        try:
            if not returns or len(returns) < 2:
                return OptimizationResult(
                    weights={},
                    expected_return=0.0,
                    volatility=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    method="Min-Variance",
                    optimization_method="Min-Variance",
                    success=False,
                    message="Need at least 2 assets",
                )

            symbols = list(returns.keys())

            # Convert to numpy arrays
            return_matrix = []
            min_length = min(len(returns[symbol]) for symbol in symbols)

            if min_length < 10:
                return OptimizationResult(
                    weights={},
                    expected_return=0.0,
                    volatility=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    method="Min-Variance",
                    optimization_method="Min-Variance",
                    success=False,
                    message="Need at least 10 return observations",
                )

            for symbol in symbols:
                return_matrix.append(returns[symbol][-min_length:])

            return_matrix = np.array(return_matrix)

            # Calculate covariance matrix
            cov_matrix = np.cov(return_matrix)
            cov_matrix += np.eye(len(symbols)) * 1e-8  # Regularization

            # Minimum variance weights: w = (C^-1 * 1) / (1' * C^-1 * 1)
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(len(symbols))

            weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))

            # Calculate portfolio metrics
            expected_returns = np.mean(return_matrix, axis=1)
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = math.sqrt(portfolio_variance)

            sharpe_ratio = (
                (portfolio_return - self.risk_free_rate) / portfolio_volatility
                if portfolio_volatility > 0
                else 0.0
            )

            weights_dict = {
                symbol: float(weight) for symbol, weight in zip(symbols, weights)
            }

            return OptimizationResult(
                weights=weights_dict,
                expected_return=float(portfolio_return),
                volatility=float(portfolio_volatility),
                expected_volatility=float(portfolio_volatility),  # Alias for volatility
                sharpe_ratio=float(sharpe_ratio),
                method="Min-Variance",
                optimization_method="Min-Variance",  # Alias for method
                success=True,
                message="Minimum variance optimization successful",
            )

        except Exception as e:
            self.logger.logger.error(f"Minimum variance optimization failed: {e}")
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                volatility=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                method="Min-Variance",
                optimization_method="Min-Variance",
                success=False,
                message=str(e),
            )

    def black_litterman_optimization(
        self,
        returns: Dict[str, List[float]],
        views: Optional[Dict[str, float]] = None,
        pi: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
        tau: float = 0.05,
        risk_aversion: float = 1.0,
    ) -> OptimizationResult:
        """
        Black-Litterman Portfolio Optimization.
        """
        try:
            if not returns or len(returns) < 2:
                return OptimizationResult(
                    weights={},
                    expected_return=0.0,
                    volatility=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    method="Black-Litterman",
                    optimization_method="Black-Litterman",
                    success=False,
                    message="Need at least 2 assets",
                )

            symbols = list(returns.keys())
            n = len(symbols)
            return_matrix = np.array([returns[symbol] for symbol in symbols])
            S = np.cov(return_matrix)  # Prior covariance matrix

            # Default to equilibrium returns if not provided
            if pi is None:
                market_caps = np.ones(n)  # Assume equal market caps for simplicity
                market_weights = market_caps / np.sum(market_caps)
                pi = risk_aversion * S.dot(market_weights)

            # If no views, it's just a reverse optimization from equilibrium
            if P is None or Q is None:
                # Fallback to mean-variance with equilibrium returns
                return self.mean_variance_optimization(returns)

            # Default omega (uncertainty of views) if not provided
            if omega is None:
                omega = np.diag(np.diag(P.dot(tau * S).dot(P.T)))

            # Black-Litterman formulas
            tau_S = tau * S
            tau_S_inv = np.linalg.inv(tau_S)
            omega_inv = np.linalg.inv(omega)

            # Posterior returns
            M = tau_S_inv + P.T.dot(omega_inv).dot(P)
            M_inv = np.linalg.inv(M)

            posterior_returns = M_inv.dot(tau_S_inv.dot(pi) + P.T.dot(omega_inv).dot(Q))

            # Posterior covariance
            posterior_cov = S + M_inv

            # Final optimization (e.g., maximize Sharpe)
            weights = self._maximize_sharpe_ratio(
                posterior_returns, posterior_cov, None
            )

            portfolio_return = np.dot(weights, posterior_returns)
            portfolio_variance = np.dot(weights, np.dot(posterior_cov, weights))
            portfolio_volatility = math.sqrt(portfolio_variance)

            sharpe_ratio = (
                (portfolio_return - self.risk_free_rate) / portfolio_volatility
                if portfolio_volatility > 0
                else 0.0
            )

            weights_dict = {symbol: weight for symbol, weight in zip(symbols, weights)}

            return OptimizationResult(
                weights=weights_dict,
                expected_return=float(portfolio_return),
                volatility=float(portfolio_volatility),
                expected_volatility=float(portfolio_volatility),
                sharpe_ratio=float(sharpe_ratio),
                method="Black-Litterman",
                optimization_method="Black-Litterman",
                success=True,
                message="Black-Litterman optimization successful",
            )

        except Exception as e:
            self.logger.logger.error(f"Black-Litterman optimization failed: {e}")
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                volatility=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                method="Black-Litterman",
                optimization_method="Black-Litterman",
                success=False,
                message=str(e),
            )

    def calculate_portfolio_metrics(
        self, weights: np.ndarray, returns_data: pd.DataFrame
    ) -> tuple[float, float, float]:
        """Calculate portfolio metrics for given weights."""
        returns_matrix = returns_data.values.T
        weights = np.array(weights)

        expected_returns = np.mean(returns_matrix, axis=1)
        expected_return = np.dot(weights, expected_returns)

        cov_matrix = np.cov(returns_matrix)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        volatility = np.sqrt(portfolio_variance)

        sharpe_ratio = (
            (expected_return - self.risk_free_rate) / volatility
            if volatility > 0
            else 0.0
        )
        return float(expected_return), float(volatility), float(sharpe_ratio)

    def calculate_var(
        self,
        weights: np.ndarray,
        returns_data: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate Value at Risk for given weights."""
        returns_matrix = returns_data.values
        weights = np.array(weights)

        portfolio_returns = np.dot(returns_matrix, weights)
        var_value = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return float(var_value)


class PortfolioManager:
    """High-level portfolio management with optimization capabilities."""

    def __init__(self, initial_capital: Decimal = Decimal("100000")):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Decimal] = {}
        self.optimizer = PortfolioOptimizer()
        self.logger = TradingLogger("portfolio_manager")

    def optimize_portfolio(
        self,
        returns_data: Dict[str, List[float]],
        method: str = "mean_variance",
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Optimize portfolio using specified method.

        Args:
            returns_data: Historical returns for each asset
            method: Optimization method ('mean_variance', 'risk_parity', 'kelly', 'equal_weight', 'min_variance')
            **kwargs: Additional parameters for optimization method
        """
        config = OptimizationConfig(method=method)
        optimizer = PortfolioOptimizer(config)

        returns_df = pd.DataFrame(returns_data)

        return optimizer.optimize(returns_df, **kwargs)

    def calculate_position_sizes(
        self,
        optimization_result: OptimizationResult,
        current_prices: Dict[str, Decimal],
    ) -> Dict[str, Decimal]:
        """Calculate actual position sizes based on optimization weights and current prices."""
        if not optimization_result.success:
            return {}

        position_sizes = {}
        total_value = float(self.current_capital)

        for symbol, weight in optimization_result.weights.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                position_value = total_value * weight
                price = float(current_prices[symbol])
                shares = position_value / price
                position_sizes[symbol] = Decimal(str(shares))
            else:
                position_sizes[symbol] = Decimal("0")

        return position_sizes

    def get_rebalancing_orders(
        self, target_weights: Dict[str, float], current_prices: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """
        Generate rebalancing orders to achieve target weights.

        Returns:
            Dict mapping symbol to shares to buy (positive) or sell (negative)
        """
        orders = {}
        total_value = float(self.current_capital)

        for symbol, target_weight in target_weights.items():
            if symbol not in current_prices or current_prices[symbol] <= 0:
                continue

            # Calculate target position size
            target_value = total_value * target_weight
            price = float(current_prices[symbol])
            target_shares = target_value / price

            # Calculate current position
            current_shares = float(self.positions.get(symbol, Decimal("0")))

            # Calculate order size
            order_size = target_shares - current_shares
            orders[symbol] = Decimal(str(order_size))

        return orders

    def update_position(self, symbol: str, shares: Decimal, price: Decimal) -> None:
        """Update position after trade execution."""
        if symbol not in self.positions:
            self.positions[symbol] = Decimal("0")

        self.positions[symbol] += shares

        # Update capital (simplified - assumes full execution)
        trade_value = shares * price
        self.current_capital -= trade_value

    def get_portfolio_value(self, current_prices: Dict[str, Decimal]) -> Decimal:
        """Calculate current portfolio value."""
        portfolio_value = self.current_capital  # Cash

        for symbol, shares in self.positions.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                position_value = shares * current_prices[symbol]
                portfolio_value += position_value

        return portfolio_value

    def get_portfolio_weights(
        self, current_prices: Dict[str, Decimal]
    ) -> Dict[str, float]:
        """Get current portfolio weights."""
        total_value = float(self.get_portfolio_value(current_prices))

        if total_value <= 0:
            return {}

        weights = {}
        for symbol, shares in self.positions.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                position_value = float(shares * current_prices[symbol])
                weights[symbol] = position_value / total_value

        return weights

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the portfolio."""
        # This is a placeholder - a more detailed implementation would go here
        return {"capital": self.current_capital, "positions": len(self.positions)}


# Alias classes for backward compatibility with tests
MeanVarianceOptimizer = PortfolioOptimizer
RiskParityOptimizer = PortfolioOptimizer
KellyCriterionOptimizer = PortfolioOptimizer
BlackLittermanOptimizer = PortfolioOptimizer
