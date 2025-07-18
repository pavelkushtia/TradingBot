#!/usr/bin/env python3
"""Simple test for technical indicators."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_basic():
    print("Testing basic imports...")

    try:
        from trading_bot.indicators.core import SimpleIndicatorManager

        print("✅ Core imports work")
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False

    try:
        manager = SimpleIndicatorManager()
        print("✅ Manager created")
    except Exception as e:
        print(f"❌ Manager creation failed: {e}")
        return False

    try:
        indicators = manager.get_available_indicators()
        print(f"✅ Available indicators: {indicators}")
    except Exception as e:
        print(f"❌ Get indicators failed: {e}")
        return False

    return True


def test_sma():
    print("\nTesting SMA...")

    try:
        from trading_bot.indicators.core import SMA

        sma = SMA(period=3)

        # Add 5 values
        values = [10, 20, 30, 40, 50]
        results = []

        for value in values:
            result = sma.update(value)
            results.append(result)

        # Should get None, None, 20, 30, 40
        expected = [None, None, 20.0, 30.0, 40.0]

        for i, (result, expect) in enumerate(zip(results, expected)):
            if expect is None:
                if result is not None:
                    print(f"❌ Expected None at index {i}, got {result}")
                    return False
            else:
                if abs(result - expect) > 0.001:
                    print(f"❌ Expected {expect} at index {i}, got {result}")
                    return False

        print("✅ SMA calculations correct")
        return True

    except Exception as e:
        print(f"❌ SMA test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Simple Technical Indicators Test")
    print("=" * 40)

    test1 = test_basic()
    test2 = test_sma()

    if test1 and test2:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Advanced Technical Indicators Library is working")
    else:
        print("\n❌ Some tests failed")
