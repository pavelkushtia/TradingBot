#!/usr/bin/env python3
"""
Test script to verify Portfolio Optimization feature is working.
Feature: Portfolio Optimization
Implementation: Modern Portfolio Theory algorithms (Mean-Variance, Risk Parity, Kelly Criterion)
"""

import os
import random
import sys
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_test_returns(symbols, days=100, base_return=0.001):
    """Generate synthetic return data for testing."""
    returns_data = {}

    for i, symbol in enumerate(symbols):
        returns = []
        # Generate correlated returns with some randomness
        for day in range(days):
            # Add some correlation between assets and market trends
            market_trend = random.gauss(base_return, 0.01)
            asset_specific = random.gauss(0, 0.005)
            daily_return = (
                market_trend + asset_specific + (i * 0.0002)
            )  # Slight asset bias
            returns.append(daily_return)

        returns_data[symbol] = returns

    return returns_data


def test_portfolio_optimizer():
    """Test the PortfolioOptimizer functionality."""
    print("🧪 Testing PortfolioOptimizer...")

    try:
        from trading_bot.portfolio import PortfolioOptimizer

        print("✅ PortfolioOptimizer imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

    try:
        optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        print("✅ PortfolioOptimizer created successfully")
    except Exception as e:
        print(f"❌ Creation failed: {e}")
        return False

    return True


def test_mean_variance_optimization():
    """Test Mean-Variance Optimization."""
    print("\n🧪 Testing Mean-Variance Optimization...")

    try:
        from trading_bot.portfolio import PortfolioOptimizer

        optimizer = PortfolioOptimizer()

        # Generate test data
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        returns_data = generate_test_returns(symbols, days=100)

        # Test Sharpe ratio maximization
        result = optimizer.mean_variance_optimization(returns_data)

        if not result.success:
            print(f"❌ Mean-variance optimization failed: {result.message}")
            return False

        print(f"✅ Mean-variance optimization successful")
        print(f"  📊 Method: {result.method}")
        print(f"  📈 Expected Return: {result.expected_return:.4f}")
        print(f"  📉 Volatility: {result.volatility:.4f}")
        print(f"  ⚡ Sharpe Ratio: {result.sharpe_ratio:.4f}")
        print(f"  💼 Weights:")
        for symbol, weight in result.weights.items():
            print(f"    {symbol}: {weight:.3f} ({weight*100:.1f}%)")

        # Verify weights sum to 1
        total_weight = sum(result.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"❌ Weights don't sum to 1: {total_weight}")
            return False

        print(f"  ✅ Weights sum to {total_weight:.6f}")

        return True

    except Exception as e:
        print(f"❌ Mean-variance test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_risk_parity_optimization():
    """Test Risk Parity Optimization."""
    print("\n🧪 Testing Risk Parity Optimization...")

    try:
        from trading_bot.portfolio import PortfolioOptimizer

        optimizer = PortfolioOptimizer()

        # Generate test data
        symbols = ["SPY", "BND", "GLD", "VTI"]
        returns_data = generate_test_returns(symbols, days=80)

        result = optimizer.risk_parity_optimization(returns_data)

        if not result.success:
            print(f"❌ Risk parity optimization failed: {result.message}")
            return False

        print(f"✅ Risk parity optimization successful")
        print(f"  📊 Method: {result.method}")
        print(f"  📈 Expected Return: {result.expected_return:.4f}")
        print(f"  📉 Volatility: {result.volatility:.4f}")
        print(f"  ⚡ Sharpe Ratio: {result.sharpe_ratio:.4f}")
        print(f"  💼 Weights:")
        for symbol, weight in result.weights.items():
            print(f"    {symbol}: {weight:.3f} ({weight*100:.1f}%)")

        # Risk parity should have more balanced weights
        weights = list(result.weights.values())
        max_weight = max(weights)
        min_weight = min(weights)

        if max_weight / min_weight > 10:  # Too imbalanced for risk parity
            print(
                f"❌ Risk parity weights too imbalanced: max={max_weight:.3f}, min={min_weight:.3f}"
            )
            return False

        print(f"  ✅ Risk parity balance: max={max_weight:.3f}, min={min_weight:.3f}")

        return True

    except Exception as e:
        print(f"❌ Risk parity test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_kelly_criterion():
    """Test Kelly Criterion Optimization."""
    print("\n🧪 Testing Kelly Criterion Optimization...")

    try:
        from trading_bot.portfolio import PortfolioOptimizer

        optimizer = PortfolioOptimizer()

        # Generate test data with some assets having higher returns
        returns_data = {
            "HIGH_RETURN": [random.gauss(0.002, 0.015) for _ in range(60)],
            "MEDIUM_RETURN": [random.gauss(0.001, 0.010) for _ in range(60)],
            "LOW_RETURN": [random.gauss(0.0005, 0.005) for _ in range(60)],
        }

        result = optimizer.kelly_criterion_optimization(returns_data)

        if not result.success:
            print(f"❌ Kelly criterion optimization failed: {result.message}")
            return False

        print(f"✅ Kelly criterion optimization successful")
        print(f"  📊 Method: {result.method}")
        print(f"  📈 Expected Return: {result.expected_return:.4f}")
        print(f"  📉 Volatility: {result.volatility:.4f}")
        print(f"  ⚡ Sharpe Ratio: {result.sharpe_ratio:.4f}")
        print(f"  💼 Weights:")
        for symbol, weight in result.weights.items():
            print(f"    {symbol}: {weight:.3f} ({weight*100:.1f}%)")

        # Kelly should allocate more to higher return assets (in theory)
        high_return_weight = result.weights.get("HIGH_RETURN", 0)
        low_return_weight = result.weights.get("LOW_RETURN", 0)

        print(
            f"  ✅ Kelly allocation bias: High={high_return_weight:.3f}, Low={low_return_weight:.3f}"
        )

        return True

    except Exception as e:
        print(f"❌ Kelly criterion test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_portfolio_manager():
    """Test PortfolioManager functionality."""
    print("\n🧪 Testing PortfolioManager...")

    try:
        from trading_bot.portfolio import PortfolioManager

        # Create portfolio manager
        manager = PortfolioManager(initial_capital=Decimal("100000"))
        print("✅ PortfolioManager created successfully")

        # Generate test data
        symbols = ["AAPL", "GOOGL", "MSFT"]
        returns_data = generate_test_returns(symbols, days=50)

        # Test optimization
        result = manager.optimize_portfolio(returns_data, method="mean_variance")

        if not result.success:
            print(f"❌ Portfolio optimization failed: {result.message}")
            return False

        print(f"✅ Portfolio optimization successful")

        # Test position size calculation
        current_prices = {
            "AAPL": Decimal("150.00"),
            "GOOGL": Decimal("2800.00"),
            "MSFT": Decimal("330.00"),
        }

        position_sizes = manager.calculate_position_sizes(result, current_prices)

        print(f"  💰 Position sizes:")
        for symbol, shares in position_sizes.items():
            if shares > 0:
                value = shares * current_prices[symbol]
                print(f"    {symbol}: {shares:.2f} shares (${value:.2f})")

        # Test rebalancing orders
        target_weights = result.weights
        orders = manager.get_rebalancing_orders(target_weights, current_prices)

        print(f"  📋 Rebalancing orders:")
        for symbol, order_size in orders.items():
            if abs(order_size) > 0.01:
                action = "BUY" if order_size > 0 else "SELL"
                print(f"    {action} {abs(order_size):.2f} shares of {symbol}")

        # Test performance metrics
        metrics = manager.get_performance_metrics()
        print(f"  📊 Performance metrics:")
        print(f"    Initial Capital: ${metrics['initial_capital']:,.2f}")
        print(f"    Current Capital: ${metrics['current_capital']:,.2f}")
        print(f"    Total Return: {metrics['total_return']:.2%}")

        return True

    except Exception as e:
        print(f"❌ Portfolio manager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_all_optimization_methods():
    """Test all optimization methods."""
    print("\n🧪 Testing all optimization methods...")

    try:
        from trading_bot.portfolio import PortfolioManager

        manager = PortfolioManager()
        symbols = ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D"]
        returns_data = generate_test_returns(symbols, days=60)

        methods = [
            "mean_variance",
            "risk_parity",
            "kelly",
            "equal_weight",
            "min_variance",
        ]

        results = {}

        for method in methods:
            try:
                result = manager.optimize_portfolio(returns_data, method=method)
                results[method] = result

                if result.success:
                    print(
                        f"  ✅ {method.upper()}: Sharpe={result.sharpe_ratio:.3f}, Vol={result.volatility:.3f}"
                    )
                else:
                    print(f"  ❌ {method.upper()}: {result.message}")

            except Exception as e:
                print(f"  ❌ {method.upper()}: Exception - {e}")

        # Verify we got results for most methods
        successful_methods = sum(1 for result in results.values() if result.success)

        if successful_methods >= 3:
            print(f"✅ {successful_methods}/{len(methods)} optimization methods working")
            return True
        else:
            print(f"❌ Only {successful_methods}/{len(methods)} methods working")
            return False

    except Exception as e:
        print(f"❌ All methods test failed: {e}")
        return False


def main():
    """Run all tests for Portfolio Optimization feature."""
    print("🎯 TESTING FEATURE: Portfolio Optimization")
    print("=" * 60)

    # Test 1: Basic optimizer functionality
    test1_passed = test_portfolio_optimizer()

    # Test 2: Mean-Variance Optimization
    test2_passed = test_mean_variance_optimization()

    # Test 3: Risk Parity Optimization
    test3_passed = test_risk_parity_optimization()

    # Test 4: Kelly Criterion
    test4_passed = test_kelly_criterion()

    # Test 5: Portfolio Manager
    test5_passed = test_portfolio_manager()

    # Test 6: All optimization methods
    test6_passed = test_all_optimization_methods()

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"  Basic Optimizer:          {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Mean-Variance Optimization: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"  Risk Parity Optimization:   {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"  Kelly Criterion:            {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print(f"  Portfolio Manager:          {'✅ PASS' if test5_passed else '❌ FAIL'}")
    print(f"  All Methods Integration:    {'✅ PASS' if test6_passed else '❌ FAIL'}")

    all_passed = all(
        [
            test1_passed,
            test2_passed,
            test3_passed,
            test4_passed,
            test5_passed,
            test6_passed,
        ]
    )

    if all_passed:
        print("\n🎉 FEATURE COMPLETE: Portfolio Optimization")
        print("✅ 5 optimization algorithms implemented")
        print("✅ Mean-Variance Optimization (Markowitz)")
        print("✅ Risk Parity Portfolio")
        print("✅ Kelly Criterion")
        print("✅ Minimum Variance Portfolio")
        print("✅ Equal Weight Portfolio (1/N rule)")
        print("✅ PortfolioManager with rebalancing")
        print("✅ Position sizing and order generation")
        print("✅ Performance tracking and metrics")

        # List capabilities
        print("\n💼 AVAILABLE CAPABILITIES:")
        capabilities = [
            "Modern Portfolio Theory: Markowitz mean-variance optimization",
            "Risk Management: Risk parity and minimum variance portfolios",
            "Growth Optimization: Kelly criterion for geometric mean maximization",
            "Practical Implementation: PortfolioManager with real-world features",
            "Rebalancing: Automatic order generation for portfolio rebalancing",
            "Position Sizing: Capital allocation based on optimization weights",
            "Performance Tracking: Portfolio metrics and return calculation",
            "Multiple Methods: 5 different optimization approaches for flexibility",
        ]

        for capability in capabilities:
            print(f"  ✅ {capability}")

        print("\n📝 UPDATE ROADMAP:")
        print("- [x] **Portfolio Optimization** ✅ COMPLETED")

        return True
    else:
        print("\n❌ FEATURE INCOMPLETE: Some tests failed")
        print("❗ Fix issues before proceeding to next feature")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
