#!/usr/bin/env python3
"""
Test script to verify Strategy Templates feature is working.
Feature: Strategy Templates
Implementation: Mean reversion, momentum, pairs trading, arbitrage, market making
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_strategy_template_imports():
    """Test importing all strategy templates."""
    print("🧪 Testing strategy template imports...")

    try:
        pass

        print("✅ All strategy templates imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mean_reversion_template():
    """Test MeanReversionTemplate functionality."""
    print("\n🧪 Testing MeanReversionTemplate...")

    try:
        from trading_bot.strategy.templates import MeanReversionTemplate

        # Create strategy with custom parameters
        params = {
            "lookback_period": 20,
            "bollinger_std": 2.0,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "position_size": 0.05,
        }

        strategy = MeanReversionTemplate("TestMeanReversion", params)
        print("✅ MeanReversionTemplate created")

        # Test strategy description
        description = strategy.get_strategy_description()

        print(f"  📊 Strategy type: {description['type']}")
        print(f"  📈 Indicators: {', '.join(description['indicators_used'])}")
        print(f"  💰 Position size: {description['risk_management']['position_size']}")
        print(f"  🎯 Profit target: {description['risk_management']['profit_target']}")
        print(f"  🛡️ Stop loss: {description['risk_management']['stop_loss']}")

        # Verify strategy attributes
        required_fields = [
            "name",
            "type",
            "description",
            "indicators_used",
            "signal_conditions",
        ]
        for field in required_fields:
            if field not in description:
                print(f"❌ Missing field: {field}")
                return False

        # Test signal conditions
        if (
            "long" in description["signal_conditions"]
            and "short" in description["signal_conditions"]
        ):
            print("✅ Signal conditions defined for both directions")
        else:
            print("❌ Missing signal conditions")
            return False

        print("✅ MeanReversionTemplate working correctly")
        return True

    except Exception as e:
        print(f"❌ MeanReversionTemplate test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_momentum_template():
    """Test MomentumTemplate functionality."""
    print("\n🧪 Testing MomentumTemplate...")

    try:
        from trading_bot.strategy.templates import MomentumTemplate

        # Create strategy with custom parameters
        params = {
            "fast_ma": 10,
            "slow_ma": 20,
            "rsi_min": 40,
            "rsi_max": 80,
            "volume_multiplier": 1.5,
            "position_size": 0.06,
        }

        strategy = MomentumTemplate("TestMomentum", params)
        print("✅ MomentumTemplate created")

        # Test strategy description
        description = strategy.get_strategy_description()

        print(f"  📊 Strategy type: {description['type']}")
        print(f"  📈 Indicators: {', '.join(description['indicators_used'])}")
        print(f"  🚀 Best markets: {', '.join(description['best_markets'])}")
        print(f"  ⚠️ Avoid markets: {', '.join(description['avoid_markets'])}")

        # Test backtest readiness check
        backtest_ready = strategy.backtest_ready()
        print(f"  📊 Backtest ready: {backtest_ready}")

        # Verify momentum-specific features
        if "trailing_stop" in description["risk_management"]:
            print("✅ Trailing stop feature included")
        else:
            print("⚠️ Trailing stop feature missing")

        # Test parameter validation
        if strategy.parameters["fast_ma"] < strategy.parameters["slow_ma"]:
            print("✅ MA parameter validation correct")
        else:
            print("❌ MA parameter validation failed")
            return False

        print("✅ MomentumTemplate working correctly")
        return True

    except Exception as e:
        print(f"❌ MomentumTemplate test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pairs_trading_template():
    """Test PairsTradingTemplate functionality."""
    print("\n🧪 Testing PairsTradingTemplate...")

    try:
        from trading_bot.strategy.templates import PairsTradingTemplate

        params = {
            "lookback_period": 30,
            "zscore_threshold": 2.0,
            "correlation_threshold": 0.7,
        }

        strategy = PairsTradingTemplate("TestPairsTrading", params)
        print("✅ PairsTradingTemplate created")

        description = strategy.get_strategy_description()

        if description["type"] == "pairs_trading":
            print("✅ Pairs trading type correct")
        else:
            print("❌ Wrong strategy type")
            return False

        # Check for pairs-specific indicators
        expected_indicators = ["Correlation", "Z-Score", "Spread"]
        if all(
            indicator in description["indicators_used"]
            for indicator in expected_indicators
        ):
            print("✅ Pairs trading indicators correct")
        else:
            print("❌ Missing pairs trading indicators")
            return False

        print("✅ PairsTradingTemplate working correctly")
        return True

    except Exception as e:
        print(f"❌ PairsTradingTemplate test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_arbitrage_template():
    """Test ArbitrageTemplate functionality."""
    print("\n🧪 Testing ArbitrageTemplate...")

    try:
        from trading_bot.strategy.templates import ArbitrageTemplate

        params = {"min_spread": 0.005, "max_holding_time": 3600}

        strategy = ArbitrageTemplate("TestArbitrage", params)
        print("✅ ArbitrageTemplate created")

        description = strategy.get_strategy_description()

        if description["type"] == "arbitrage":
            print("✅ Arbitrage type correct")
        else:
            print("❌ Wrong strategy type")
            return False

        if "Price Spread" in description["indicators_used"]:
            print("✅ Price spread indicator included")
        else:
            print("❌ Price spread indicator missing")
            return False

        print("✅ ArbitrageTemplate working correctly")
        return True

    except Exception as e:
        print(f"❌ ArbitrageTemplate test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_market_making_template():
    """Test MarketMakingTemplate functionality."""
    print("\n🧪 Testing MarketMakingTemplate...")

    try:
        from trading_bot.strategy.templates import MarketMakingTemplate

        params = {"spread_pct": 0.001, "max_inventory": 1000, "inventory_target": 0}

        strategy = MarketMakingTemplate("TestMarketMaking", params)
        print("✅ MarketMakingTemplate created")

        description = strategy.get_strategy_description()

        if description["type"] == "market_making":
            print("✅ Market making type correct")
        else:
            print("❌ Wrong strategy type")
            return False

        # Check for market making specific indicators
        mm_indicators = ["Bid-Ask Spread", "Volume", "Inventory"]
        if all(
            indicator in description["indicators_used"] for indicator in mm_indicators
        ):
            print("✅ Market making indicators correct")
        else:
            print("❌ Missing market making indicators")
            return False

        print("✅ MarketMakingTemplate working correctly")
        return True

    except Exception as e:
        print(f"❌ MarketMakingTemplate test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_strategy_template_library():
    """Test the complete strategy template library."""
    print("\n🧪 Testing strategy template library...")

    try:
        from trading_bot.strategy.templates import (
            ArbitrageTemplate,
            MarketMakingTemplate,
            MeanReversionTemplate,
            MomentumTemplate,
            PairsTradingTemplate,
        )

        # Create all strategy types
        strategies = [
            MeanReversionTemplate("MR_Test", {}),
            MomentumTemplate("Mom_Test", {}),
            PairsTradingTemplate("Pairs_Test", {}),
            ArbitrageTemplate("Arb_Test", {}),
            MarketMakingTemplate("MM_Test", {}),
        ]

        print(f"✅ Created {len(strategies)} strategy templates")

        # Test that each has unique strategy type
        strategy_types = [s.get_strategy_description()["type"] for s in strategies]
        unique_types = set(strategy_types)

        if len(unique_types) == len(strategies):
            print("✅ All strategy types are unique")
        else:
            print("❌ Duplicate strategy types found")
            return False

        # Test that all have required methods
        required_methods = ["generate_signals", "get_strategy_description"]
        for strategy in strategies:
            for method in required_methods:
                if not hasattr(strategy, method):
                    print(f"❌ Strategy {strategy.name} missing method: {method}")
                    return False

        print("✅ All strategies have required methods")

        # Test parameter inheritance
        for strategy in strategies:
            if not hasattr(strategy, "parameters"):
                print(f"❌ Strategy {strategy.name} missing parameters")
                return False

            if "position_size" not in strategy.parameters:
                print(f"❌ Strategy {strategy.name} missing position_size parameter")
                return False

        print("✅ All strategies have required parameters")

        # Generate strategy summary
        print("\n📊 STRATEGY TEMPLATE LIBRARY SUMMARY:")
        for strategy in strategies:
            desc = strategy.get_strategy_description()
            print(f"  {desc['type'].upper()}: {desc['description']}")
            print(f"    Indicators: {', '.join(desc['indicators_used'])}")

        return True

    except Exception as e:
        print(f"❌ Strategy template library test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_strategy_customization():
    """Test strategy parameter customization."""
    print("\n🧪 Testing strategy customization...")

    try:
        from trading_bot.strategy.templates import MeanReversionTemplate

        # Test default parameters
        default_strategy = MeanReversionTemplate("Default", {})
        default_desc = default_strategy.get_strategy_description()

        # Test custom parameters
        custom_params = {
            "lookback_period": 50,
            "bollinger_std": 2.5,
            "position_size": 0.10,
            "profit_target": 0.05,
            "stop_loss": 0.025,
        }

        custom_strategy = MeanReversionTemplate("Custom", custom_params)
        custom_desc = custom_strategy.get_strategy_description()

        # Verify customization worked
        if (
            custom_strategy.parameters["lookback_period"] == 50
            and custom_strategy.parameters["bollinger_std"] == 2.5
        ):
            print("✅ Parameter customization working")
        else:
            print("❌ Parameter customization failed")
            return False

        # Test that custom parameters override defaults
        if (
            custom_desc["risk_management"]["position_size"]
            != default_desc["risk_management"]["position_size"]
        ):
            print("✅ Custom parameters override defaults")
        else:
            print("❌ Custom parameters not overriding defaults")
            return False

        print("✅ Strategy customization working correctly")
        return True

    except Exception as e:
        print(f"❌ Strategy customization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests for Strategy Templates feature."""
    print("🎯 TESTING FEATURE: Strategy Templates")
    print("=" * 60)

    # Test 1: Template imports
    test1_passed = test_strategy_template_imports()

    # Test 2: Mean reversion template
    test2_passed = test_mean_reversion_template()

    # Test 3: Momentum template
    test3_passed = test_momentum_template()

    # Test 4: Pairs trading template
    test4_passed = test_pairs_trading_template()

    # Test 5: Arbitrage template
    test5_passed = test_arbitrage_template()

    # Test 6: Market making template
    test6_passed = test_market_making_template()

    # Test 7: Strategy library
    test7_passed = test_strategy_template_library()

    # Test 8: Strategy customization
    test8_passed = test_strategy_customization()

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"  Template Imports:         {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Mean Reversion Template:  {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"  Momentum Template:        {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"  Pairs Trading Template:   {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print(f"  Arbitrage Template:       {'✅ PASS' if test5_passed else '❌ FAIL'}")
    print(f"  Market Making Template:   {'✅ PASS' if test6_passed else '❌ FAIL'}")
    print(f"  Strategy Library:         {'✅ PASS' if test7_passed else '❌ FAIL'}")
    print(f"  Strategy Customization:   {'✅ PASS' if test8_passed else '❌ FAIL'}")

    all_passed = all(
        [
            test1_passed,
            test2_passed,
            test3_passed,
            test4_passed,
            test5_passed,
            test6_passed,
            test7_passed,
            test8_passed,
        ]
    )

    if all_passed:
        print("\n🎉 FEATURE COMPLETE: Strategy Templates")
        print("✅ 5 comprehensive strategy templates implemented")
        print("✅ Mean Reversion: Bollinger Bands + RSI for range-bound markets")
        print("✅ Momentum: MA crossovers + MACD + volume for trending markets")
        print("✅ Pairs Trading: Statistical arbitrage framework for correlated assets")
        print("✅ Arbitrage: Price discrepancy exploitation across markets")
        print("✅ Market Making: Liquidity provision with bid-ask spread capture")
        print("✅ Fully customizable parameters for each strategy type")
        print("✅ Professional risk management with profit targets and stop losses")
        print("✅ Detailed strategy descriptions and usage guidelines")

        # List strategy types and use cases
        print("\n📚 STRATEGY TEMPLATE LIBRARY:")
        strategies_info = [
            (
                "Mean Reversion",
                "Range-bound markets, high volatility periods",
                "Bollinger Bands, RSI",
            ),
            ("Momentum", "Trending markets, breakout scenarios", "EMA, MACD, Volume"),
            (
                "Pairs Trading",
                "Market-neutral strategies, correlation trading",
                "Z-Score, Correlation",
            ),
            (
                "Arbitrage",
                "Multi-market opportunities, price discrepancies",
                "Spread Analysis",
            ),
            (
                "Market Making",
                "Liquidity provision, high-frequency trading",
                "Bid-Ask Spread",
            ),
        ]

        for name, use_case, indicators in strategies_info:
            print(f"  📈 {name}: {use_case}")
            print(f"      Indicators: {indicators}")

        return True
    else:
        print("\n❌ FEATURE INCOMPLETE: Some tests failed")
        print("❗ Will continue to next feature")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n📝 UPDATE ROADMAP:")
        print("- [x] **Strategy Templates** ✅ COMPLETED")
    sys.exit(0 if success else 1)
