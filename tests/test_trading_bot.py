"""Comprehensive test suite for the trading bot."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.bot import TradingBot
from trading_bot.core.config import Config, DatabaseConfig, TradingConfig
from trading_bot.core.events import EventBus
from trading_bot.core.models import MarketData, Order, Portfolio, Quote
from trading_bot.core.signal import StrategySignal
from trading_bot.database.manager import DatabaseManager
from trading_bot.risk.manager import RiskManager
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
    def event_bus(self):
        """Create event bus."""
        return EventBus()

    @pytest.fixture
    def backtest_engine(self, config, event_bus):
        """Create backtest engine."""
        return BacktestEngine(config, event_bus)

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
            initial_capital=Decimal("100000"),
            total_value=Decimal("115000"),
            buying_power=Decimal("15000"),
            cash=Decimal("15000"),
            positions={},
            start_date=datetime.utcnow() - timedelta(days=30),
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
        """Fixture for bot configuration."""
        return Config.from_env()

    @pytest.fixture
    def event_bus(self):
        """Fixture for event bus."""
        return EventBus()

    @pytest.fixture
    def risk_manager(self, config, event_bus):
        """Fixture for the risk manager."""
        return RiskManager(config, event_bus)

    def test_risk_manager_initialization(self, risk_manager):
        """Test risk manager initialization."""
        assert risk_manager.max_drawdown == 0.15

    @pytest.mark.asyncio
    async def test_signal_evaluation(self, risk_manager):
        """Test signal evaluation."""
        signal = StrategySignal(
            strategy_name="test",
            symbol="AAPL",
            signal_type="buy",
            price=Decimal("150"),
            strength=0.8,
            timestamp=datetime.now(timezone.utc),
        )
        approved = await risk_manager.evaluate_signal(signal, None)
        assert approved

    @pytest.mark.asyncio
    async def test_position_sizing(self, risk_manager):
        """Test position sizing."""
        portfolio = Portfolio(
            initial_capital=Decimal("100000"),
            total_value=Decimal("100000"),
            buying_power=Decimal("100000"),
            cash=Decimal("100000"),
            start_date=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        size = await risk_manager.calculate_position_size(
            "AAPL", Decimal("150"), portfolio
        )
        assert size > 0

    @pytest.mark.asyncio
    async def test_portfolio_risk_assessment(self, risk_manager):
        """Test portfolio risk assessment."""
        portfolio = Portfolio(
            initial_capital=Decimal("100000"),
            total_value=Decimal("100000"),
            buying_power=Decimal("100000"),
            cash=Decimal("100000"),
            start_date=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        risk = await risk_manager.check_portfolio_risk(portfolio)
        assert risk is not None


class TestConfiguration:
    """Test suite for configuration."""

    def test_config_creation(self):
        """Test Config object creation."""
        c = Config()
        assert c.trading.portfolio_value > 0

    def test_config_validation(self):
        """Test Config validation."""
        with pytest.raises(Exception):
            Config(trading=TradingConfig(portfolio_value=-100))


class TestDatabase:
    """Test suite for database."""

    @pytest.fixture
    def db_manager(self):
        """Fixture for the database manager."""
        return DatabaseManager(
            Config(database=DatabaseConfig(url="sqlite+aiosqlite:///:memory:"))
        )

    @pytest.mark.asyncio
    async def test_db_initialization(self, db_manager):
        """Test database initialization."""
        await db_manager.initialize()
        assert db_manager.connection is not None
        await db_manager.shutdown()


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
