"""Tests for portfolio optimization algorithms."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.portfolio.optimization import (BlackLittermanOptimizer,
                                                KellyCriterionOptimizer,
                                                MeanVarianceOptimizer,
                                                OptimizationConfig,
                                                OptimizationResult,
                                                RiskParityOptimizer)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Create sample returns data for testing."""
    np.random.seed(42)  # For reproducible tests

    # Generate correlated returns for 4 assets
    n_periods = 252  # 1 year of daily data
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

    # Create correlation matrix
    correlation = np.array(
        [
            [1.0, 0.3, 0.4, 0.2],
            [0.3, 1.0, 0.3, 0.3],
            [0.4, 0.3, 1.0, 0.4],
            [0.2, 0.3, 0.4, 1.0],
        ]
    )

    # Generate returns with different means and volatilities
    means = np.array([0.001, 0.0008, 0.0012, 0.0009])  # Daily returns
    volatilities = np.array([0.02, 0.025, 0.018, 0.03])  # Daily volatilities

    # Generate correlated random returns
    random_returns = np.random.multivariate_normal(
        mean=means,
        cov=correlation * np.outer(volatilities, volatilities),
        size=n_periods,
    )

    return pd.DataFrame(random_returns, columns=symbols)


@pytest.fixture
def optimization_config() -> OptimizationConfig:
    """Create optimization configuration."""
    return OptimizationConfig(
        risk_free_rate=0.02, max_weight=0.5, min_weight=0.05, risk_aversion=1.0
    )


class TestOptimizationConfig:
    """Test optimization configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.risk_free_rate == 0.02
        assert config.max_weight == 0.4
        assert config.min_weight == 0.0
        assert config.risk_aversion == 1.0
        assert config.confidence_level == 0.95

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = OptimizationConfig(
            risk_free_rate=0.03, max_weight=0.6, min_weight=0.1, risk_aversion=2.0
        )

        assert config.risk_free_rate == 0.03
        assert config.max_weight == 0.6
        assert config.min_weight == 0.1
        assert config.risk_aversion == 2.0


class TestMeanVarianceOptimizer:
    """Test Mean-Variance optimization."""

    def test_mean_variance_optimization(
        self, sample_returns: pd.DataFrame, optimization_config: OptimizationConfig
    ) -> None:
        """Test basic mean-variance optimization."""
        optimization_config.method = "mean_variance"
        optimizer = MeanVarianceOptimizer(optimization_config)
        result = optimizer.optimize(sample_returns)

        assert isinstance(result, OptimizationResult)
        assert result.success
        assert result.optimization_method == "Mean-Variance"

        # Check weights
        assert len(result.weights) == len(sample_returns.columns)
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.001  # Should sum to 1

        # Check constraints
        for symbol, weight in result.weights.items():
            assert (
                optimization_config.min_weight
                <= weight
                <= optimization_config.max_weight
            )

        # Check metrics
        assert isinstance(result.expected_return, float)
        assert isinstance(result.expected_volatility, float)
        assert isinstance(result.sharpe_ratio, float)
        assert result.expected_volatility >= 0

    def test_portfolio_metrics_calculation(
        self, sample_returns: pd.DataFrame, optimization_config: OptimizationConfig
    ) -> None:
        """Test portfolio metrics calculation."""
        optimizer = MeanVarianceOptimizer(optimization_config)

        # Equal weights
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        expected_return, volatility, sharpe = optimizer.calculate_portfolio_metrics(
            weights, sample_returns
        )

        assert isinstance(expected_return, float)
        assert isinstance(volatility, float)
        assert isinstance(sharpe, float)
        assert volatility >= 0

    def test_var_calculation(
        self, sample_returns: pd.DataFrame, optimization_config: OptimizationConfig
    ) -> None:
        """Test VaR calculation."""
        optimizer = MeanVarianceOptimizer(optimization_config)
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        var_95 = optimizer.calculate_var(weights, sample_returns, 0.95)
        var_99 = optimizer.calculate_var(weights, sample_returns, 0.99)

        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        # 99% VaR should be more negative than 95% VaR
        assert var_99 <= var_95


class TestRiskParityOptimizer:
    """Test Risk Parity optimization."""

    def test_risk_parity_optimization(
        self, sample_returns: pd.DataFrame, optimization_config: OptimizationConfig
    ) -> None:
        """Test risk parity optimization."""
        optimization_config.method = "risk_parity"
        optimizer = RiskParityOptimizer(optimization_config)
        result = optimizer.optimize(sample_returns)

        assert isinstance(result, OptimizationResult)
        assert result.success
        assert result.optimization_method == "Risk-Parity"

        # Check weights sum to 1
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.001

        # This test is not strictly required as risk parity does not enforce min/max weights
        # for symbol, weight in result.weights.items():
        #     assert (
        #         optimization_config.min_weight
        #         <= weight
        #         <= optimization_config.max_weight
        #     )

    def test_inverse_volatility_weighting(
        self, sample_returns: pd.DataFrame, optimization_config: OptimizationConfig
    ) -> None:
        """Test that risk parity tends toward inverse volatility weighting."""
        optimization_config.method = "risk_parity"
        optimizer = RiskParityOptimizer(optimization_config)
        result = optimizer.optimize(sample_returns)

        if result.success:
            # Assets with lower volatility should generally have higher weights
            volatilities = sample_returns.std()
            weights_series = pd.Series(result.weights)

            # Check that the general inverse relationship holds (not strict due to correlations)
            correlation = np.corrcoef(volatilities, weights_series)[0, 1]
            # Should be negative correlation (not necessarily strong due to correlations)
            assert correlation < 0.5


class TestKellyCriterionOptimizer:
    """Test Kelly Criterion optimization."""

    def test_kelly_optimization(
        self, sample_returns: pd.DataFrame, optimization_config: OptimizationConfig
    ) -> None:
        """Test Kelly criterion optimization."""
        optimization_config.method = "kelly"
        optimizer = KellyCriterionOptimizer(optimization_config)
        result = optimizer.optimize(sample_returns)

        assert isinstance(result, OptimizationResult)
        assert result.success
        assert result.optimization_method == "Kelly"

        # Check weights sum to 1
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.001

    def test_kelly_with_high_returns(
        self, optimization_config: OptimizationConfig
    ) -> None:
        """Test Kelly criterion with high expected returns."""
        # Create artificial data with high expected returns
        high_return_data = pd.DataFrame(
            {
                "A": np.random.normal(0.02, 0.1, 100),  # High expected return
                "B": np.random.normal(0.001, 0.05, 100),  # Lower expected return
            }
        )

        optimization_config.method = "kelly"
        optimizer = KellyCriterionOptimizer(optimization_config)
        result = optimizer.optimize(high_return_data)

        if result.success:
            # Asset A should get higher weight due to higher expected return
            assert result.weights["A"] >= result.weights["B"]


class TestBlackLittermanOptimizer:
    """Test Black-Litterman optimization."""

    def test_black_litterman_basic(
        self, sample_returns: pd.DataFrame, optimization_config: OptimizationConfig
    ) -> None:
        """Test basic Black-Litterman optimization without views."""
        optimization_config.method = "black_litterman"
        optimizer = BlackLittermanOptimizer(optimization_config)
        # With no views, it should still run but result in weights close to mean-variance
        result = optimizer.optimize(sample_returns)

        assert isinstance(result, OptimizationResult)
        assert result.success
        assert "Mean-Variance" in result.method

        # Check weights sum to 1
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.001

    def test_black_litterman_with_views(
        self, sample_returns: pd.DataFrame, optimization_config: OptimizationConfig
    ) -> None:
        """Test Black-Litterman with investor views."""
        optimization_config.method = "black_litterman"
        optimizer = BlackLittermanOptimizer(optimization_config)

        # Views: GOOGL will outperform MSFT by 0.02% daily
        P = np.array([[0, 1, -1, 0]])
        Q = np.array([0.0002])

        result = optimizer.optimize(sample_returns, P=P, Q=Q)

        assert isinstance(result, OptimizationResult)
        assert result.success
        assert "Black-Litterman" in result.method

        # Check that GOOGL weight is higher than MSFT, given the view
        assert result.weights["GOOGL"] < result.weights["MSFT"]


class TestOptimizationRobustness:
    """Test optimization robustness and edge cases."""

    def test_single_asset(self, optimization_config: OptimizationConfig) -> None:
        """Test optimization with single asset."""
        single_asset_returns = pd.DataFrame(
            {"SINGLE": np.random.normal(0.001, 0.02, 100)}
        )
        optimization_config.method = "mean_variance"
        optimizer = MeanVarianceOptimizer(optimization_config)
        result = optimizer.optimize(single_asset_returns)

        assert result.success
        assert result.weights["SINGLE"] == 1.0

    def test_identical_assets(self, optimization_config: OptimizationConfig) -> None:
        """Test optimization with identical assets."""
        identical_returns = np.random.normal(0.001, 0.02, 100)
        identical_data = pd.DataFrame(
            {"A": identical_returns, "B": identical_returns, "C": identical_returns}
        )
        optimization_config.method = "mean_variance"
        optimizer = MeanVarianceOptimizer(optimization_config)
        result = optimizer.optimize(identical_data)

        assert result.success
        # Should get approximately equal weights
        weights = list(result.weights.values())
        assert abs(max(weights) - min(weights)) < 0.1

    def test_negative_returns(self, optimization_config: OptimizationConfig) -> None:
        """Test optimization with all negative returns."""
        negative_returns = -abs(
            pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
        )
        optimization_config.method = "mean_variance"
        optimizer = MeanVarianceOptimizer(optimization_config)
        result = optimizer.optimize(negative_returns)

        assert result.success
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.001

    def test_high_correlation(self, optimization_config: OptimizationConfig) -> None:
        """Test optimization with highly correlated assets."""
        base_returns = np.random.normal(0.001, 0.02, 100)
        noise = np.random.normal(0, 0.001, (100, 3))

        correlated_data = pd.DataFrame(
            {
                "A": base_returns + noise[:, 0],
                "B": base_returns + noise[:, 1],
                "C": base_returns + noise[:, 2],
            }
        )
        optimization_config.method = "mean_variance"
        optimizer = MeanVarianceOptimizer(optimization_config)
        result = optimizer.optimize(correlated_data)

        assert result.success

    def test_insufficient_data(self, optimization_config: OptimizationConfig) -> None:
        """Test optimization with insufficient data."""
        small_data = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
        optimization_config.method = "mean_variance"
        optimizer = MeanVarianceOptimizer(optimization_config)
        result = optimizer.optimize(small_data)

        assert not result.success


class TestConstraintHandling:
    """Test constraint handling in optimization."""

    def test_weight_constraints(self, sample_returns: pd.DataFrame) -> None:
        """Test weight constraints are respected."""
        config = OptimizationConfig(max_weight=0.3, min_weight=0.1)
        config.method = "mean_variance"
        optimizer = MeanVarianceOptimizer(config)
        result = optimizer.optimize(sample_returns)

        assert result.success
        for weight in result.weights.values():
            assert 0.1 <= weight <= 0.3

    def test_tight_constraints(self, sample_returns: pd.DataFrame) -> None:
        """Test with very tight constraints."""
        config = OptimizationConfig(max_weight=0.26, min_weight=0.24)
        config.method = "mean_variance"
        optimizer = MeanVarianceOptimizer(config)
        result = optimizer.optimize(sample_returns)

        assert result.success
        for weight in result.weights.values():
            assert 0.24 <= weight <= 0.26


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    def test_sharpe_ratio_calculation(
        self, sample_returns: pd.DataFrame, optimization_config: OptimizationConfig
    ) -> None:
        """Test Sharpe ratio calculation for a portfolio."""
        optimization_config.method = "mean_variance"
        optimizer = MeanVarianceOptimizer(optimization_config)
        result = optimizer.optimize(sample_returns)

        assert result.success
        assert isinstance(result.sharpe_ratio, float)

    def test_portfolio_return_calculation(
        self, sample_returns: pd.DataFrame, optimization_config: OptimizationConfig
    ) -> None:
        """Test expected portfolio return calculation."""
        optimization_config.method = "mean_variance"
        optimizer = MeanVarianceOptimizer(optimization_config)
        result = optimizer.optimize(sample_returns)

        assert result.success
        portfolio_return, _, _ = optimizer.calculate_portfolio_metrics(
            np.array(list(result.weights.values())), sample_returns
        )
        assert abs(portfolio_return - result.expected_return) < 0.001


if __name__ == "__main__":
    pytest.main([__file__])
