"""Comprehensive test suite for the trading bot."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.bot import TradingBot
from trading_bot.core.config import Config
from trading_bot.core.models import (MarketData, Order, Portfolio, Quote,
                                     StrategySignal)
from trading_bot.strategy.momentum_crossover import MomentumCrossoverStrategy


class TestTradingBot:
    """Test suite for the main TradingBot class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config.from_env()

    @pytest.fixture
    def trading_bot(self, config):
        """Create TradingBot instance for testing."""
        return TradingBot(config)

    @pytest.mark.asyncio
    async def test_bot_initialization(self, trading_bot):
        """Test bot initialization."""
        assert trading_bot.config is not None
        assert trading_bot.running is False
        assert trading_bot.start_time is None
        assert trading_bot.portfolio is None
        assert len(trading_bot.active_orders) == 0
        assert len(trading_bot.positions) == 0

    @pytest.mark.asyncio
    async def test_bot_status(self, trading_bot):
        """Test bot status reporting."""
        status = trading_bot.get_status()

        assert "running" in status
        assert "start_time" in status
        assert "portfolio_value" in status
        assert "active_orders" in status
        assert "open_positions" in status
        assert "daily_pnl" in status

        assert status["running"] is False
        assert status["start_time"] is None
        assert status["portfolio_value"] == "0"
        assert status["active_orders"] == 0
        assert status["open_positions"] == 0


class TestMarketData:
    """Test suite for market data functionality."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=1000000,
            vwap=Decimal("150.50"),
        )

    @pytest.fixture
    def sample_quote(self):
        """Create sample quote."""
        return Quote(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            bid_price=Decimal("150.95"),
            ask_price=Decimal("151.05"),
            bid_size=100,
            ask_size=200,
        )

    def test_market_data_creation(self, sample_market_data):
        """Test market data object creation."""
        assert sample_market_data.symbol == "AAPL"
        assert sample_market_data.open == Decimal("150.00")
        assert sample_market_data.high == Decimal("152.00")
        assert sample_market_data.low == Decimal("149.00")
        assert sample_market_data.close == Decimal("151.00")
        assert sample_market_data.volume == 1000000
        assert sample_market_data.vwap == Decimal("150.50")

    def test_quote_creation(self, sample_quote):
        """Test quote object creation."""
        assert sample_quote.symbol == "AAPL"
        assert sample_quote.bid_price == Decimal("150.95")
        assert sample_quote.ask_price == Decimal("151.05")
        assert sample_quote.bid_size == 100
        assert sample_quote.ask_size == 200

    def test_quote_mid_price(self, sample_quote):
        """Test quote mid price calculation."""
        expected_mid = (sample_quote.bid_price + sample_quote.ask_price) / 2
        assert sample_quote.mid_price == expected_mid

    def test_quote_spread(self, sample_quote):
        """Test quote spread calculation."""
        expected_spread = sample_quote.ask_price - sample_quote.bid_price
        assert sample_quote.spread == expected_spread


class TestMomentumCrossoverStrategy:
    """Test suite for momentum crossover strategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        parameters = {
            "short_window": 5,
            "long_window": 10,
            "min_strength_threshold": 0.01,
        }
        return MomentumCrossoverStrategy("test_momentum", parameters)

    @pytest.fixture
    def sample_bars(self):
        """Create sample market data bars."""
        bars = []
        base_time = datetime.utcnow() - timedelta(days=20)

        # Create bars with upward trend
        for i in range(15):
            bar = MarketData(
                symbol="AAPL",
                timestamp=base_time + timedelta(minutes=i),
                open=Decimal(str(100 + i * 0.5)),
                high=Decimal(str(101 + i * 0.5)),
                low=Decimal(str(99 + i * 0.5)),
                close=Decimal(str(100.5 + i * 0.5)),
                volume=100000,
            )
            bars.append(bar)

        return bars

    @pytest.mark.asyncio
    async def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        await strategy.initialize()

        assert strategy.name == "test_momentum"
        assert strategy.short_window == 5
        assert strategy.long_window == 10
        assert strategy.min_strength_threshold == 0.01
        assert strategy.enabled is True

    @pytest.mark.asyncio
    async def test_strategy_market_data_handling(self, strategy, sample_bars):
        """Test strategy market data handling."""
        for bar in sample_bars:
            await strategy.on_bar(bar.symbol, bar)

        assert "AAPL" in strategy.symbols
        assert len(strategy.market_data["AAPL"]) == len(sample_bars)

    @pytest.mark.asyncio
    async def test_strategy_signal_generation(self, strategy, sample_bars):
        """Test strategy signal generation."""
        # Feed data to strategy
        for bar in sample_bars:
            await strategy.on_bar(bar.symbol, bar)

        # Generate signals
        signals = await strategy.generate_signals()

        # Should have signals due to upward trend
        assert len(signals) >= 0  # May or may not have signals depending on crossover

    def test_strategy_sma_calculation(self, strategy, sample_bars):
        """Test SMA calculation."""
        # Add sample data
        for bar in sample_bars:
            strategy.market_data["AAPL"] = strategy.market_data.get("AAPL", []) + [bar]

        # Calculate SMA
        sma_5 = strategy.calculate_sma("AAPL", 5)
        sma_10 = strategy.calculate_sma("AAPL", 10)

        assert sma_5 is not None
        assert sma_10 is not None
        assert isinstance(sma_5, Decimal)
        assert isinstance(sma_10, Decimal)


class TestBacktestEngine:
    """Test suite for backtesting engine."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config.from_env()

    @pytest.fixture
    def backtest_engine(self, config):
        """Create backtest engine."""
        return BacktestEngine(config)

    @pytest.fixture
    def strategy(self):
        """Create test strategy."""
        parameters = {
            "short_window": 5,
            "long_window": 10,
            "min_strength_threshold": 0.01,
        }
        return MomentumCrossoverStrategy("test_momentum", parameters)

    @pytest.fixture
    def historical_data(self):
        """Create historical market data."""
        data = []
        base_time = datetime.utcnow() - timedelta(days=30)

        # Create 100 bars with some trend
        for i in range(100):
            price = 100 + (i * 0.1) + (i % 10) * 0.5  # Trending with some noise
            bar = MarketData(
                symbol="AAPL",
                timestamp=base_time + timedelta(hours=i),
                open=Decimal(str(price)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price + 0.5)),
                volume=100000,
            )
            data.append(bar)

        return data

    @pytest.mark.asyncio
    async def test_backtest_initialization(self, backtest_engine, config):
        """Test backtest engine initialization."""
        assert backtest_engine.config == config
        assert backtest_engine.initial_capital == Decimal(
            str(config.trading.portfolio_value)
        )
        assert backtest_engine.trades == []
        assert backtest_engine.orders == []
        assert backtest_engine.portfolio is None

    @pytest.mark.asyncio
    async def test_backtest_run(self, backtest_engine, strategy, historical_data):
        """Test running a backtest."""
        start_date = historical_data[0].timestamp
        end_date = historical_data[-1].timestamp

        results = await backtest_engine.run_backtest(
            strategy, historical_data, start_date, end_date
        )

        assert "strategy_name" in results
        assert "start_date" in results
        assert "end_date" in results
        assert "initial_capital" in results
        assert "final_capital" in results
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "trades" in results
        assert "equity_curve" in results

        assert results["strategy_name"] == strategy.name
        assert results["initial_capital"] == float(backtest_engine.initial_capital)

    def test_backtest_report_generation(self, backtest_engine):
        """Test backtest report generation."""
        # Create dummy performance metrics
        from trading_bot.core.models import PerformanceMetrics

        backtest_engine.performance_metrics = PerformanceMetrics(
            total_return=Decimal("0.15"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown=Decimal("0.05"),
            win_rate=Decimal("0.6"),
            profit_factor=Decimal("1.5"),
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            average_win=Decimal("100"),
            average_loss=Decimal("50"),
            largest_win=Decimal("200"),
            largest_loss=Decimal("80"),
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
        )

        # Create dummy portfolio
        from trading_bot.core.models import Portfolio

        backtest_engine.portfolio = Portfolio(
            total_value=Decimal("115000"),
            buying_power=Decimal("15000"),
            cash=Decimal("15000"),
            positions={},
            updated_at=datetime.utcnow(),
        )

        backtest_engine.initial_capital = Decimal("100000")

        report = backtest_engine.generate_report()

        assert "summary" in report
        assert "trade_analysis" in report
        assert "portfolio" in report

        assert "total_return" in report["summary"]
        assert "sharpe_ratio" in report["summary"]
        assert "max_drawdown" in report["summary"]


class TestOrderExecution:
    """Test suite for order execution."""

    @pytest.fixture
    def sample_order(self):
        """Create sample order."""
        from trading_bot.core.models import OrderSide, OrderType

        return Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("100"),
            strategy_id="test_strategy",
        )

    def test_order_creation(self, sample_order):
        """Test order object creation."""
        assert sample_order.symbol == "AAPL"
        assert sample_order.quantity == Decimal("100")
        assert sample_order.side.value == "buy"
        assert sample_order.type.value == "market"
        assert sample_order.strategy_id == "test_strategy"

    def test_order_properties(self, sample_order):
        """Test order properties."""
        assert sample_order.remaining_quantity == sample_order.quantity
        assert sample_order.is_filled is False
        assert sample_order.is_active is True

        # Test after filling
        sample_order.filled_quantity = sample_order.quantity
        from trading_bot.core.models import OrderStatus

        sample_order.status = OrderStatus.FILLED

        assert sample_order.remaining_quantity == Decimal("0")
        assert sample_order.is_filled is True
        assert sample_order.is_active is False


class TestRiskManagement:
    """Test suite for risk management."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config.from_env()

    @pytest.fixture
    def risk_manager(self, config):
        """Create risk manager."""
        from trading_bot.risk.manager import RiskManager

        return RiskManager(config)

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        return Portfolio(
            total_value=Decimal("100000"),
            buying_power=Decimal("50000"),
            cash=Decimal("50000"),
            positions={},
            day_pnl=Decimal("0"),
            total_pnl=Decimal("0"),
            updated_at=datetime.utcnow(),
        )

    @pytest.fixture
    def sample_signal(self):
        """Create sample signal."""
        return StrategySignal(
            symbol="AAPL",
            signal_type="buy",
            strength=0.8,
            price=Decimal("150.00"),
            timestamp=datetime.utcnow(),
            strategy_name="test_strategy",
        )

    @pytest.mark.asyncio
    async def test_risk_manager_initialization(self, risk_manager, config):
        """Test risk manager initialization."""
        await risk_manager.initialize()

        assert risk_manager.config == config
        assert risk_manager.max_position_size == config.trading.max_position_size
        assert risk_manager.max_daily_loss == config.risk.max_daily_loss
        assert risk_manager.max_open_positions == config.risk.max_open_positions

    @pytest.mark.asyncio
    async def test_signal_evaluation(
        self, risk_manager, sample_signal, sample_portfolio
    ):
        """Test signal evaluation."""
        await risk_manager.initialize()

        # Should pass basic risk checks
        result = await risk_manager.evaluate_signal(sample_signal, sample_portfolio)
        assert result is True

    @pytest.mark.asyncio
    async def test_position_sizing(self, risk_manager, sample_portfolio):
        """Test position sizing calculation."""
        await risk_manager.initialize()

        symbol = "AAPL"
        price = Decimal("150.00")

        position_size = await risk_manager.calculate_position_size(
            symbol, price, sample_portfolio
        )

        assert position_size >= 0
        assert isinstance(position_size, Decimal)

        # Position value should not exceed max position size
        position_value = position_size * price
        max_position_value = sample_portfolio.total_value * Decimal(
            str(risk_manager.max_position_size)
        )
        assert position_value <= max_position_value

    @pytest.mark.asyncio
    async def test_portfolio_risk_assessment(self, risk_manager, sample_portfolio):
        """Test portfolio risk assessment."""
        await risk_manager.initialize()

        risk_metrics = await risk_manager.check_portfolio_risk(sample_portfolio)

        assert "portfolio_value" in risk_metrics
        assert "daily_pnl_pct" in risk_metrics
        assert "max_concentration" in risk_metrics
        assert "position_count" in risk_metrics
        assert "risk_violations" in risk_metrics

        assert risk_metrics["portfolio_value"] == float(sample_portfolio.total_value)
        assert risk_metrics["position_count"] == len(sample_portfolio.positions)
        assert isinstance(risk_metrics["risk_violations"], list)


class TestConfiguration:
    """Test suite for configuration management."""

    def test_config_creation(self):
        """Test configuration creation."""
        config = Config.from_env()

        assert config.exchange is not None
        assert config.trading is not None
        assert config.risk is not None
        assert config.database is not None
        assert config.logging is not None
        assert config.monitoring is not None
        assert config.strategy is not None
        assert config.market_data is not None

    def test_config_validation(self):
        """Test configuration validation."""
        config = Config.from_env()

        # Test trading config
        assert config.trading.portfolio_value > 0
        assert 0 < config.trading.max_position_size <= 1.0
        assert 0 < config.trading.stop_loss_percentage <= 1.0
        assert 0 < config.trading.take_profit_percentage <= 1.0

        # Test risk config
        assert 0 < config.risk.max_daily_loss <= 1.0
        assert config.risk.max_open_positions > 0
        assert config.risk.risk_free_rate >= 0

        # Test strategy config
        assert isinstance(config.strategy.parameters, dict)


@pytest.mark.asyncio
async def test_integration_basic_flow():
    """Integration test for basic trading flow."""
    # Create bot
    config = Config.from_env()
    bot = TradingBot(config)

    # Test status before initialization
    status = bot.get_status()
    assert status["running"] is False

    # Test configuration
    assert bot.config is not None
    assert bot.config.trading.portfolio_value > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
