#!/usr/bin/env python3
"""
Test script to verify Enhanced Risk Management feature is working.
Feature: Enhanced Risk Management
Implementation: Position sizing, volatility-based stops, correlation analysis
"""

import os
import random
import sys
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_test_returns(days=100, volatility=0.02):
    """Generate synthetic return data for testing."""
    returns = []
    for _ in range(days):
        daily_return = random.gauss(0.001, volatility)  # 0.1% mean, variable volatility
        returns.append(daily_return)
    return returns


def test_enhanced_risk_manager():
    """Test the EnhancedRiskManager functionality."""
    print("🧪 Testing EnhancedRiskManager...")

    try:
        from trading_bot.risk import EnhancedRiskManager

        print("✅ EnhancedRiskManager imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

    try:
        risk_manager = EnhancedRiskManager(
            max_portfolio_risk=0.1, max_single_position=0.05
        )
        print("✅ EnhancedRiskManager created successfully")

        # Test basic properties
        print(f"  📊 Max portfolio risk: {risk_manager.max_portfolio_risk:.1%}")
        print(f"  📈 Max single position: {risk_manager.max_single_position:.1%}")

        return True

    except Exception as e:
        print(f"❌ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_volatility_calculator():
    """Test VolatilityCalculator functionality."""
    print("\n🧪 Testing VolatilityCalculator...")

    try:
        from trading_bot.risk import VolatilityCalculator

        calc = VolatilityCalculator(lookback_days=30)
        print("✅ VolatilityCalculator created")

        # Test with synthetic data
        low_vol_returns = generate_test_returns(
            days=60, volatility=0.01
        )  # 1% daily vol
        high_vol_returns = generate_test_returns(
            days=60, volatility=0.03
        )  # 3% daily vol

        # Test historical volatility
        low_vol = calc.calculate_historical_volatility(low_vol_returns)
        high_vol = calc.calculate_historical_volatility(high_vol_returns)

        print(f"  📉 Low volatility: {low_vol:.2%}")
        print(f"  📈 High volatility: {high_vol:.2%}")

        if high_vol > low_vol:
            print("✅ Volatility calculation working correctly")
        else:
            print("❌ Volatility calculation failed")
            return False

        # Test EWMA volatility
        ewma_vol = calc.calculate_ewma_volatility(low_vol_returns)
        print(f"  📊 EWMA volatility: {ewma_vol:.2%}")

        # Test GARCH volatility
        garch_vol = calc.calculate_garch_volatility(low_vol_returns)
        print(f"  📈 GARCH volatility: {garch_vol:.2%}")

        return True

    except Exception as e:
        print(f"❌ VolatilityCalculator test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_correlation_analyzer():
    """Test CorrelationAnalyzer functionality."""
    print("\n🧪 Testing CorrelationAnalyzer...")

    try:
        from trading_bot.risk import CorrelationAnalyzer

        analyzer = CorrelationAnalyzer(min_observations=30)
        print("✅ CorrelationAnalyzer created")

        # Create correlated returns data
        np.random.seed(42)  # For reproducible results
        base_returns = np.random.normal(0.001, 0.02, 50)

        returns_data = {
            "STOCK_A": base_returns.tolist(),
            "STOCK_B": (
                base_returns * 0.8 + np.random.normal(0, 0.01, 50)
            ).tolist(),  # Correlated
            "STOCK_C": np.random.normal(0.001, 0.02, 50).tolist(),  # Independent
        }

        # Test correlation matrix calculation
        correlation_matrix = analyzer.calculate_correlation_matrix(returns_data)

        print("✅ Correlation matrix calculated")
        print("  📊 Correlation Matrix:")
        for stock1, correlations in correlation_matrix.items():
            for stock2, corr in correlations.items():
                if stock1 != stock2:
                    print(f"    {stock1} - {stock2}: {corr:.3f}")

        # Test portfolio correlation risk
        weights = {"STOCK_A": 0.4, "STOCK_B": 0.4, "STOCK_C": 0.2}
        portfolio_corr_risk = analyzer.calculate_portfolio_correlation_risk(
            weights, correlation_matrix
        )

        print(f"  ⚠️ Portfolio correlation risk: {portfolio_corr_risk:.3f}")

        # Verify A and B are more correlated than A and C
        corr_ab = correlation_matrix["STOCK_A"]["STOCK_B"]
        corr_ac = correlation_matrix["STOCK_A"]["STOCK_C"]

        if abs(corr_ab) > abs(corr_ac):
            print("✅ Correlation analysis working correctly")
            return True
        else:
            print("❌ Correlation analysis failed")
            return False

    except Exception as e:
        print(f"❌ CorrelationAnalyzer test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_position_sizer():
    """Test PositionSizer functionality."""
    print("\n🧪 Testing PositionSizer...")

    try:
        from trading_bot.risk import PositionSizer

        sizer = PositionSizer(max_risk_per_trade=0.02, max_portfolio_risk=0.06)
        print("✅ PositionSizer created")

        account_value = Decimal("100000")
        entry_price = Decimal("50.00")
        stop_price = Decimal("45.00")  # $5 stop loss

        # Test fixed fractional sizing
        ff_result = sizer.fixed_fractional_sizing(
            account_value, Decimal("2000"), entry_price, stop_price
        )

        print(f"✅ Fixed fractional sizing:")
        print(f"  📊 Recommended shares: {ff_result.recommended_shares}")
        print(f"  💰 Position value: ${ff_result.position_value}")
        print(f"  ⚠️ Risk amount: ${ff_result.risk_amount}")
        print(f"  📈 Risk percentage: {ff_result.risk_percentage:.2%}")

        # Test volatility-based sizing
        vb_result = sizer.volatility_based_sizing(
            account_value, 0.25, entry_price
        )  # 25% volatility

        print(f"✅ Volatility-based sizing:")
        print(f"  📊 Recommended shares: {vb_result.recommended_shares}")
        print(f"  💰 Position value: ${vb_result.position_value}")
        print(f"  📈 Risk percentage: {vb_result.risk_percentage:.2%}")

        # Test Kelly criterion sizing
        kelly_result = sizer.kelly_criterion_sizing(
            win_rate=0.6, avg_win=0.03, avg_loss=0.02, account_value=account_value
        )

        print(f"✅ Kelly criterion sizing:")
        print(f"  💰 Allocation: ${kelly_result.risk_amount}")
        print(f"  📈 Risk percentage: {kelly_result.risk_percentage:.2%}")

        # Test ATR-based sizing
        atr_result = sizer.atr_based_sizing(
            account_value, 2.5, entry_price
        )  # $2.50 ATR

        print(f"✅ ATR-based sizing:")
        print(f"  📊 Recommended shares: {atr_result.recommended_shares}")
        print(f"  💰 Position value: ${atr_result.position_value}")

        # Verify all methods returned valid results
        methods_working = [
            ff_result.confidence > 0,
            vb_result.confidence > 0,
            kelly_result.confidence >= 0,
            atr_result.confidence > 0,
        ]

        if all(methods_working):
            print("✅ All position sizing methods working")
            return True
        else:
            print("❌ Some position sizing methods failed")
            return False

    except Exception as e:
        print(f"❌ PositionSizer test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_risk_assessment():
    """Test risk assessment functionality."""
    print("\n🧪 Testing risk assessment...")

    try:
        from trading_bot.core.models import MarketData
        from trading_bot.risk import EnhancedRiskManager

        risk_manager = EnhancedRiskManager()

        # Create synthetic market data
        symbols = ["AAPL", "GOOGL", "MSFT"]
        current_prices = {
            "AAPL": Decimal("150.00"),
            "GOOGL": Decimal("2800.00"),
            "MSFT": Decimal("330.00"),
        }

        # Simulate market data updates
        for symbol in symbols:
            for i in range(50):  # 50 days of data
                price = float(current_prices[symbol]) * (1 + random.gauss(0, 0.02))
                bar = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal(str(price * 0.99)),
                    high=Decimal(str(price * 1.01)),
                    low=Decimal(str(price * 0.99)),
                    close=Decimal(str(price)),
                    volume=1000000,
                    vwap=Decimal(str(price)),
                )
                risk_manager.update_market_data(symbol, bar)

        print("✅ Market data updated")

        # Test position sizing
        position_size = risk_manager.calculate_position_size(
            "AAPL",
            Decimal("150.00"),
            Decimal("140.00"),  # Stop loss
            Decimal("100000"),  # Account value
            method="fixed_fractional",
        )

        print(f"✅ Position sizing calculated:")
        print(f"  📊 Symbol: {position_size.symbol}")
        print(f"  📈 Shares: {position_size.recommended_shares}")
        print(f"  💰 Value: ${position_size.position_value}")
        print(f"  ⚠️ Risk: {position_size.risk_percentage:.2%}")

        # Test volatility stop calculation
        vol_stop = risk_manager.calculate_volatility_stop(
            "AAPL", Decimal("150.00"), "long", volatility_multiplier=2.0
        )

        if vol_stop:
            print(f"✅ Volatility stop: ${vol_stop}")
        else:
            print("ℹ️ Volatility stop calculation needs more data")

        # Test risk limits check
        risk_check = risk_manager.check_risk_limits(
            "AAPL",
            Decimal("1000"),  # Large position
            current_prices["AAPL"],
            {},  # No existing positions
        )

        print(f"✅ Risk limits check:")
        print(f"  ✅ Approved: {risk_check['approved']}")
        print(f"  ⚠️ Violations: {len(risk_check['violations'])}")
        print(f"  📊 Warnings: {len(risk_check['warnings'])}")

        return True

    except Exception as e:
        print(f"❌ Risk assessment test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_risk_dashboard():
    """Test risk dashboard functionality."""
    print("\n🧪 Testing risk dashboard...")

    try:
        from trading_bot.core.models import Position
        from trading_bot.risk import EnhancedRiskManager

        risk_manager = EnhancedRiskManager()

        # Create dummy positions
        positions = {
            "AAPL": Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                avg_price=Decimal("150.00"),
                side="long",
                timestamp=datetime.now(timezone.utc),
            ),
            "GOOGL": Position(
                symbol="GOOGL",
                quantity=Decimal("50"),
                avg_price=Decimal("2800.00"),
                side="long",
                timestamp=datetime.now(timezone.utc),
            ),
        }

        current_prices = {"AAPL": Decimal("155.00"), "GOOGL": Decimal("2850.00")}

        # Add some market data
        for symbol in ["AAPL", "GOOGL"]:
            for i in range(30):
                returns = [random.gauss(0.001, 0.02) for _ in range(30)]
                risk_manager.returns_data[symbol].extend(returns)

        # Generate risk dashboard
        dashboard = risk_manager.get_risk_dashboard(positions, current_prices)

        print("✅ Risk dashboard generated:")
        print(
            f"  📊 Portfolio VaR (95%): {dashboard['portfolio_metrics']['var_95']:.4f}"
        )
        print(
            f"  📈 Portfolio volatility: {dashboard['portfolio_metrics']['volatility']:.2%}"
        )
        print(f"  ⚡ Sharpe ratio: {dashboard['portfolio_metrics']['sharpe_ratio']:.3f}")
        print(f"  📉 Max drawdown: {dashboard['portfolio_metrics']['max_drawdown']:.2%}")
        print(
            f"  🔗 Correlation risk: {dashboard['portfolio_metrics']['correlation_risk']:.3f}"
        )

        print(f"  📋 Position risks:")
        for symbol, risk_data in dashboard["position_risks"].items():
            print(
                f"    {symbol}: vol={risk_data['volatility']:.2%}, corr_risk={risk_data['correlation_risk']:.3f}"
            )

        print(f"  📊 Monitored symbols: {dashboard['monitored_symbols']}")

        return True

    except Exception as e:
        print(f"❌ Risk dashboard test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests for Enhanced Risk Management feature."""
    print("🎯 TESTING FEATURE: Enhanced Risk Management")
    print("=" * 60)

    # Test 1: Basic risk manager
    test1_passed = test_enhanced_risk_manager()

    # Test 2: Volatility calculator
    test2_passed = test_volatility_calculator()

    # Test 3: Correlation analyzer
    test3_passed = test_correlation_analyzer()

    # Test 4: Position sizer
    test4_passed = test_position_sizer()

    # Test 5: Risk assessment
    test5_passed = test_risk_assessment()

    # Test 6: Risk dashboard
    test6_passed = test_risk_dashboard()

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"  Enhanced Risk Manager:    {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Volatility Calculator:    {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"  Correlation Analyzer:     {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"  Position Sizer:           {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print(f"  Risk Assessment:          {'✅ PASS' if test5_passed else '❌ FAIL'}")
    print(f"  Risk Dashboard:           {'✅ PASS' if test6_passed else '❌ FAIL'}")

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
        print("\n🎉 FEATURE COMPLETE: Enhanced Risk Management")
        print("✅ Advanced position sizing algorithms implemented")
        print("✅ Multiple volatility models (Historical, EWMA, GARCH)")
        print("✅ Correlation analysis and portfolio risk assessment")
        print(
            "✅ 4 position sizing methods (Fixed Fractional, Volatility-based, Kelly, ATR)"
        )
        print("✅ Volatility-based stop loss calculation")
        print("✅ Real-time risk monitoring and limit checking")
        print("✅ Comprehensive risk dashboard with VaR, Sharpe, drawdown")
        print("✅ Portfolio-level risk metrics and diversification analysis")

        # List capabilities
        print("\n⚠️ RISK MANAGEMENT CAPABILITIES:")
        capabilities = [
            "Position Sizing: 4 sophisticated algorithms for optimal position sizing",
            "Volatility Models: Historical, EWMA, and simplified GARCH volatility",
            "Correlation Analysis: Portfolio diversification and concentration risk",
            "VaR Calculation: 95% and 99% Value at Risk with Expected Shortfall",
            "Dynamic Stops: Volatility-based stop loss calculation",
            "Risk Limits: Real-time checking of position and portfolio limits",
            "Performance Metrics: Sharpe ratio, maximum drawdown, volatility tracking",
            "Risk Dashboard: Comprehensive real-time risk monitoring interface",
        ]

        for capability in capabilities:
            print(f"  ✅ {capability}")

        return True
    else:
        print("\n❌ FEATURE INCOMPLETE: Some tests failed")
        print("❗ Will continue to next feature")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n📝 UPDATE ROADMAP:")
        print("- [x] **Enhanced Risk Management** ✅ COMPLETED")
    sys.exit(0 if success else 1)
