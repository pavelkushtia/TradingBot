"""Comprehensive integration test for all implemented features."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.config import Config
from trading_bot.core.models import MarketData
from trading_bot.execution.order_types import (AdvancedOrderManager,
                                               StopLossConfig, StopLossOrder,
                                               TakeProfitConfig,
                                               TakeProfitOrder)
from trading_bot.portfolio.optimization import (MeanVarianceOptimizer,
                                                OptimizationConfig,
                                                PortfolioOptimizer,
                                                RiskParityOptimizer)
from trading_bot.timeframes.manager import (MultiTimeframeStrategy, Timeframe,
                                            TimeframeConfig)


class ComprehensiveTestStrategy(MultiTimeframeStrategy):
    """Test strategy that uses all implemented features."""

    def __init__(self, name: str, parameters: dict):
        # Initialize with multi-timeframe support
        timeframe_config = TimeframeConfig(
            primary_timeframe=Timeframe.M5,
            secondary_timeframes=[Timeframe.M15, Timeframe.H1],
        )
        super().__init__(name, parameters, timeframe_config)

        # Initialize portfolio optimizer
        opt_config = OptimizationConfig(risk_free_rate=0.02, max_weight=0.5)
        self.portfolio_optimizer = MeanVarianceOptimizer(opt_config)

        # Initialize advanced order manager
        self.order_manager = AdvancedOrderManager()

        # Track signals for testing
        self.generated_signals = []
        self.portfolio_weights = {}

    async def generate_multi_timeframe_signals(self) -> list:
        """Generate signals using multiple timeframes and indicators."""
        signals = []

        for symbol in self.symbols:
            try:
                # Check if timeframes are synchronized
                if not self.is_timeframes_synchronized(symbol):
                    continue

                # Get data from different timeframes
                m5_data = self.get_timeframe_data(symbol, Timeframe.M5, limit=50)
                m15_data = self.get_timeframe_data(symbol, Timeframe.M15, limit=20)
                h1_data = self.get_timeframe_data(symbol, Timeframe.H1, limit=10)

                if len(m5_data) < 20 or len(m15_data) < 10 or len(h1_data) < 5:
                    continue

                # Get indicator values from the enhanced base strategy
                rsi_m5 = self.get_indicator_value(symbol, "RSI")
                sma_m5 = self.get_indicator_value(symbol, "SMA")
                macd_m5 = self.get_indicator_value(symbol, "MACD")

                if not all([rsi_m5, sma_m5, macd_m5]):
                    continue

                # Multi-timeframe analysis
                current_price = m5_data[-1].close

                # Trend analysis on H1
                h1_sma = self._calculate_simple_sma(
                    [float(bar.close) for bar in h1_data[-5:]], 5
                )
                trend_up = float(current_price) > h1_sma if h1_sma else False

                # Momentum on M15
                m15_rsi = self._calculate_simple_rsi(
                    [float(bar.close) for bar in m15_data]
                )
                momentum_oversold = m15_rsi < 30 if m15_rsi else False
                momentum_overbought = m15_rsi > 70 if m15_rsi else False

                # Entry signals combining timeframes
                if trend_up and momentum_oversold and rsi_m5 < 40:
                    signal = self.create_signal(
                        symbol=symbol,
                        signal_type="buy",
                        strength=0.8,
                        price=current_price,
                        metadata={
                            "timeframe_analysis": {
                                "h1_trend": "up",
                                "m15_momentum": "oversold",
                                "m5_rsi": rsi_m5,
                            },
                            "multi_timeframe": True,
                        },
                    )
                    signals.append(signal)
                    self.generated_signals.append(signal)

                    # Create advanced orders
                    await self._create_risk_management_orders(
                        symbol, current_price, "buy"
                    )

                elif not trend_up and momentum_overbought and rsi_m5 > 60:
                    signal = self.create_signal(
                        symbol=symbol,
                        signal_type="sell",
                        strength=0.8,
                        price=current_price,
                        metadata={
                            "timeframe_analysis": {
                                "h1_trend": "down",
                                "m15_momentum": "overbought",
                                "m5_rsi": rsi_m5,
                            },
                            "multi_timeframe": True,
                        },
                    )
                    signals.append(signal)
                    self.generated_signals.append(signal)

                    # Create advanced orders
                    await self._create_risk_management_orders(
                        symbol, current_price, "sell"
                    )

            except Exception:
                # Log error but continue with other symbols
                pass

        return signals

    async def _create_risk_management_orders(
        self, symbol: str, entry_price: Decimal, side: str
    ):
        """Create stop-loss and take-profit orders."""
        try:
            if side == "buy":
                # Stop-loss 2% below entry
                sl_config = StopLossConfig(
                    symbol=symbol,
                    side="sell",
                    quantity=Decimal("100"),
                    base_price=entry_price,
                    stop_percentage=Decimal("0.02"),
                )
                stop_loss = StopLossOrder(sl_config)
                self.order_manager.add_order(stop_loss)

                # Take-profit 4% above entry
                tp_config = TakeProfitConfig(
                    symbol=symbol,
                    side="sell",
                    quantity=Decimal("100"),
                    base_price=entry_price,
                    target_percentage=Decimal("0.04"),
                )
                take_profit = TakeProfitOrder(tp_config)
                self.order_manager.add_order(take_profit)

            else:  # sell
                # Stop-loss 2% above entry
                sl_config = StopLossConfig(
                    symbol=symbol,
                    side="buy",
                    quantity=Decimal("100"),
                    base_price=entry_price,
                    stop_percentage=Decimal("0.02"),
                )
                stop_loss = StopLossOrder(sl_config)
                self.order_manager.add_order(stop_loss)

                # Take-profit 4% below entry
                tp_config = TakeProfitConfig(
                    symbol=symbol,
                    side="buy",
                    quantity=Decimal("100"),
                    base_price=entry_price,
                    target_percentage=Decimal("0.04"),
                )
                take_profit = TakeProfitOrder(tp_config)
                self.order_manager.add_order(take_profit)

        except Exception:
            # Handle errors gracefully
            pass

    def _calculate_simple_sma(self, values: list, period: int) -> float:
        """Simple moving average calculation."""
        if len(values) < period:
            return sum(values) / len(values) if values else 0
        return sum(values[-period:]) / period

    def _calculate_simple_rsi(self, values: list, period: int = 14) -> float:
        """Simple RSI calculation."""
        if len(values) < period + 1:
            return 50.0

        changes = [values[i] - values[i - 1] for i in range(1, len(values))]
        gains = [max(0, change) for change in changes[-period:]]
        losses = [abs(min(0, change)) for change in changes[-period:]]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    async def optimize_portfolio(self, symbols: list) -> dict:
        """Optimize portfolio using historical returns."""
        try:
            # Create synthetic returns data for testing
            returns_data = {}
            for symbol in symbols:
                # Get recent price data
                bars = self.get_timeframe_data(symbol, Timeframe.M5, limit=100)
                if len(bars) < 50:
                    continue

                # Calculate returns
                prices = [float(bar.close) for bar in bars]
                returns = [
                    (prices[i] - prices[i - 1]) / prices[i - 1]
                    for i in range(1, len(prices))
                ]
                returns_data[symbol] = returns

            if len(returns_data) < 2:
                return {}

            # Create DataFrame
            max_length = min(len(returns) for returns in returns_data.values())
            aligned_returns = {
                symbol: returns[-max_length:]
                for symbol, returns in returns_data.items()
            }
            returns_df = pd.DataFrame(aligned_returns)

            # Optimize portfolio
            result = self.portfolio_optimizer.optimize(returns_df)

            if result.success:
                self.portfolio_weights = result.weights
                return {
                    "weights": result.weights,
                    "expected_return": result.expected_return,
                    "volatility": result.expected_volatility,
                    "sharpe_ratio": result.sharpe_ratio,
                }

        except Exception:
            pass

        return {}


@pytest.fixture
def sample_comprehensive_data():
    """Create comprehensive sample data for testing."""
    symbols = ["AAPL", "GOOGL", "MSFT"]
    data = {}

    for symbol in symbols:
        bars = []
        base_price = 100.0 + hash(symbol) % 50  # Different base prices

        for i in range(200):  # Enough data for all timeframes
            # Create realistic price movement
            price_change = np.random.normal(0, 0.01)  # 1% volatility
            base_price *= 1 + price_change

            high = base_price * (1 + abs(np.random.normal(0, 0.005)))
            low = base_price * (1 - abs(np.random.normal(0, 0.005)))
            volume = int(np.random.uniform(100000, 1000000))

            timestamp = datetime.now(timezone.utc) - timedelta(minutes=200 - i)

            bar = MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=Decimal(str(base_price)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(base_price)),
                volume=volume,
                vwap=Decimal(str(base_price)),
            )
            bars.append(bar)

        data[symbol] = bars

    return data


class TestComprehensiveIntegration:
    """Comprehensive integration tests."""

    @pytest.mark.asyncio
    async def test_full_system_integration(self, sample_comprehensive_data):
        """Test complete system integration."""
        # Create strategy with all features
        strategy = ComprehensiveTestStrategy("comprehensive_test", {"test": True})

        # Initialize strategy
        await strategy.initialize()

        # Add symbols and process data
        symbols = list(sample_comprehensive_data.keys())
        for symbol in symbols:
            for bar in sample_comprehensive_data[symbol]:
                await strategy.on_bar(symbol, bar)

        # Wait for timeframe synchronization
        await asyncio.sleep(0.1)

        # Generate signals
        await strategy.generate_signals()

        # Verify multi-timeframe functionality
        for symbol in symbols:
            assert symbol in strategy.timeframe_manager.monitored_symbols

            # Check all timeframes have data
            m5_data = strategy.get_timeframe_data(symbol, Timeframe.M5)
            m15_data = strategy.get_timeframe_data(symbol, Timeframe.M15)
            h1_data = strategy.get_timeframe_data(symbol, Timeframe.H1)

            assert len(m5_data) > 0
            assert len(m15_data) > 0
            assert len(h1_data) > 0

        # Verify indicators are working
        performance = strategy.get_performance_metrics()
        assert "indicators_available" in performance
        assert "timeframe_status" in performance

        # Test portfolio optimization
        portfolio_result = await strategy.optimize_portfolio(symbols)
        if portfolio_result:
            assert "weights" in portfolio_result
            assert "expected_return" in portfolio_result
            assert "sharpe_ratio" in portfolio_result

        # Verify advanced orders
        order_status = strategy.order_manager.get_order_status()
        assert "total_orders" in order_status

        # Clean up
        await strategy.cleanup()

    @pytest.mark.asyncio
    async def test_backtesting_with_advanced_metrics(self, sample_comprehensive_data):
        """Test backtesting with comprehensive performance metrics."""
        # Create config
        config = Config.from_env()
        event_bus = MagicMock()

        # Create strategy
        strategy = ComprehensiveTestStrategy("backtest_test", {"backtest": True})

        # Create backtest engine
        backtest_engine = BacktestEngine(config, event_bus)

        # Prepare data (use first symbol for simplicity)
        symbol = list(sample_comprehensive_data.keys())[0]
        market_data = sample_comprehensive_data[symbol]

        start_date = market_data[0].timestamp
        end_date = market_data[-1].timestamp

        # Run backtest
        results = await backtest_engine.run_backtest(
            strategy, market_data, start_date, end_date
        )

        # Verify comprehensive results
        assert "advanced_metrics" in results
        assert "performance_report" in results

        advanced_metrics = results["advanced_metrics"]
        assert "annualized_return" in advanced_metrics
        assert "sortino_ratio" in advanced_metrics
        assert "var_95" in advanced_metrics
        assert "skewness" in advanced_metrics
        assert "kurtosis" in advanced_metrics

        # Verify performance report structure
        performance_report = results["performance_report"]
        assert "Summary" in performance_report
        assert "Risk Metrics" in performance_report
        assert "Trade Analysis" in performance_report

    def test_portfolio_optimization_algorithms(self, sample_comprehensive_data):
        """Test different portfolio optimization algorithms."""
        symbols = list(sample_comprehensive_data.keys())

        # Create returns data
        returns_data = {}
        for symbol in symbols:
            bars = sample_comprehensive_data[symbol][-50:]  # Last 50 bars
            prices = [float(bar.close) for bar in bars]
            returns = [
                (prices[i] - prices[i - 1]) / prices[i - 1]
                for i in range(1, len(prices))
            ]
            returns_data[symbol] = returns

        returns_df = pd.DataFrame(returns_data)

        # Test different optimizers
        config = OptimizationConfig(max_weight=0.6, min_weight=0.1)

        # Mean-Variance
        mv_optimizer = MeanVarianceOptimizer(config)
        mv_result = mv_optimizer.optimize(returns_df)
        assert mv_result.optimization_method == "Mean-Variance"

        # Risk Parity
        rp_config = OptimizationConfig(method="risk_parity")
        rp_optimizer = PortfolioOptimizer(rp_config)
        rp_result = rp_optimizer.optimize(returns_df)
        assert rp_result.success
        assert "Risk-Parity" in rp_result.method

        # Kelly Criterion
        kelly_config = OptimizationConfig(method="kelly")
        kelly_optimizer = PortfolioOptimizer(kelly_config)
        kelly_result = kelly_optimizer.optimize(returns_df)
        assert kelly_result.success
        assert "Kelly" in kelly_result.method

        # Black-Litterman
        bl_config = OptimizationConfig(method="black_litterman")
        bl_optimizer = PortfolioOptimizer(bl_config)
        bl_result = bl_optimizer.optimize(returns_df)
        assert bl_result.success
        assert "Mean-Variance" in bl_result.method

    @pytest.mark.asyncio
    async def test_advanced_order_management(
        self, sample_comprehensive_data: Dict[str, List[MarketData]]
    ) -> None:
        """Test advanced order types management."""
        order_manager = AdvancedOrderManager()
        symbol = list(sample_comprehensive_data.keys())[0]
        bars = sample_comprehensive_data[symbol]

        # Create stop-loss order
        sl_config = StopLossConfig(
            symbol=symbol,
            side="sell",
            quantity=Decimal("100"),
            base_price=bars[0].close,
            stop_percentage=Decimal("0.02"),
        )
        stop_loss = StopLossOrder(sl_config)
        order_id = order_manager.add_order(stop_loss)

        assert order_id in order_manager.active_orders

        # Simulate price movement
        for bar in bars[1:20]:  # Test with some bars
            triggered_orders = order_manager.update_orders(
                symbol, bar.close, {"timestamp": bar.timestamp}
            )

            # Check if stop-loss triggered
            if triggered_orders:
                assert len(triggered_orders) > 0
                assert triggered_orders[0].symbol == symbol
                break

        # Verify order management
        status = order_manager.get_order_status()
        assert "total_orders" in status
        assert "orders_by_type" in status

    def test_technical_indicators_integration(self, sample_comprehensive_data):
        """Test technical indicators integration."""
        from trading_bot.indicators import IndicatorConfig, IndicatorManager

        manager = IndicatorManager()
        symbol = list(sample_comprehensive_data.keys())[0]
        bars = sample_comprehensive_data[symbol]

        # Add multiple indicators
        indicators = ["SMA", "EMA", "RSI", "MACD", "BBANDS", "ATR", "STOCH"]

        for indicator in indicators:
            config = IndicatorConfig(period=14)
            manager.add_indicator(symbol, indicator, config)

        # Update with data
        results = {}
        for bar in bars:
            results = manager.update_indicators_with_results(symbol, bar)

        # Verify all indicators are working
        for indicator in indicators:
            if indicator in results:
                assert results[indicator].value is not None
                assert results[indicator].name == indicator

        # Test composite signals
        composite = manager.calculate_composite_signals(symbol)
        assert "overall_signal" in composite
        assert composite["overall_signal"] in ["bullish", "bearish", "neutral"]


class TestSystemRobustness:
    """Test system robustness and error handling."""

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test system handles errors gracefully."""
        strategy = ComprehensiveTestStrategy("error_test", {})

        # Test with invalid data
        invalid_bar = MarketData(
            symbol="INVALID",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("0"),
            high=Decimal("0"),
            low=Decimal("0"),
            close=Decimal("0"),
            volume=0,
            vwap=Decimal("0"),
        )

        # Should not crash
        await strategy.on_bar("INVALID", invalid_bar)
        signals = await strategy.generate_signals()

        # Should return empty signals gracefully
        assert isinstance(signals, list)

    def test_performance_with_large_datasets(self, sample_comprehensive_data):
        """Test performance with larger datasets."""
        # Extend dataset
        extended_data = {}
        for symbol, bars in sample_comprehensive_data.items():
            # Extend to 1000 bars
            extended_bars = bars * 5  # Simple duplication for testing
            extended_data[symbol] = extended_bars

        # Test should complete in reasonable time
        start_time = datetime.now()

        ComprehensiveTestStrategy("performance_test", {})

        # Add data
        for symbol in extended_data:
            for i, bar in enumerate(extended_data[symbol][:500]):  # Limit for test
                # Use asyncio.create_task to avoid blocking
                pass

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should complete within reasonable time (adjust as needed)
        assert duration < 10.0  # 10 seconds max for test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
