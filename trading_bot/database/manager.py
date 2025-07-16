"""Database management for storing trading data and history."""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import aiosqlite

from ..core.config import Config
from ..core.exceptions import DatabaseError
from ..core.logging import TradingLogger
from ..core.models import Order, PerformanceMetrics, Portfolio, Position, Trade


class DatabaseManager:
    """Async database manager for trading data persistence."""

    def __init__(self, config: Config):
        """Initialize database manager."""
        self.config = config
        self.logger = TradingLogger("database_manager")
        self.db_path = self._extract_db_path(config.database.url)
        self.connection: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        try:
            self.connection = await aiosqlite.connect(self.db_path)
            await self._create_tables()
            self.logger.logger.info(f"Database initialized: {self.db_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database: {e}")

    async def shutdown(self) -> None:
        """Close database connection."""
        if self.connection:
            await self.connection.close()
        self.logger.logger.info("Database connection closed")

    async def save_portfolio(self, portfolio: Portfolio) -> None:
        """Save portfolio state to database."""
        try:
            if not self.connection:
                raise DatabaseError("Database not initialized")

            await self.connection.execute(
                """
                INSERT OR REPLACE INTO portfolios (
                    id, total_value, buying_power, cash, day_pnl, total_pnl,
                    positions_json, updated_at
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    float(portfolio.total_value),
                    float(portfolio.buying_power),
                    float(portfolio.cash),
                    float(portfolio.day_pnl),
                    float(portfolio.total_pnl),
                    json.dumps({k: v.dict() for k, v in portfolio.positions.items()}),
                    portfolio.updated_at.isoformat(),
                ),
            )

            await self.connection.commit()

        except Exception as e:
            self.logger.log_error(e, {"context": "save_portfolio"})
            raise DatabaseError(f"Failed to save portfolio: {e}")

    async def get_portfolio(self) -> Optional[Portfolio]:
        """Load portfolio from database."""
        try:
            if not self.connection:
                raise DatabaseError("Database not initialized")

            cursor = await self.connection.execute(
                """
                SELECT total_value, buying_power, cash, day_pnl, total_pnl,
                       positions_json, updated_at
                FROM portfolios WHERE id = 1
            """
            )

            row = await cursor.fetchone()
            if not row:
                return None

            positions_data = json.loads(row[5])
            positions = {}

            for symbol, pos_data in positions_data.items():
                positions[symbol] = Position(**pos_data)

            return Portfolio(
                total_value=Decimal(str(row[0])),
                buying_power=Decimal(str(row[1])),
                cash=Decimal(str(row[2])),
                positions=positions,
                day_pnl=Decimal(str(row[3])),
                total_pnl=Decimal(str(row[4])),
                updated_at=datetime.fromisoformat(row[5]),
            )

        except Exception as e:
            self.logger.log_error(e, {"context": "get_portfolio"})
            return None

    async def save_order(self, order: Order) -> None:
        """Save order to database."""
        try:
            if not self.connection:
                raise DatabaseError("Database not initialized")

            await self.connection.execute(
                """
                INSERT OR REPLACE INTO orders (
                    id, symbol, side, type, quantity, price, stop_price,
                    time_in_force, status, filled_quantity, average_fill_price,
                    created_at, updated_at, strategy_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    order.id,
                    order.symbol,
                    order.side.value,
                    order.type.value,
                    float(order.quantity),
                    float(order.price) if order.price else None,
                    float(order.stop_price) if order.stop_price else None,
                    order.time_in_force,
                    order.status.value,
                    float(order.filled_quantity),
                    (
                        float(order.average_fill_price)
                        if order.average_fill_price
                        else None
                    ),
                    order.created_at.isoformat() if order.created_at else None,
                    order.updated_at.isoformat() if order.updated_at else None,
                    order.strategy_id,
                    json.dumps(order.metadata),
                ),
            )

            await self.connection.commit()

        except Exception as e:
            self.logger.log_error(e, {"context": "save_order", "order_id": order.id})
            raise DatabaseError(f"Failed to save order: {e}")

    async def get_active_orders(self) -> List[Order]:
        """Get all active orders from database."""
        try:
            if not self.connection:
                raise DatabaseError("Database not initialized")

            cursor = await self.connection.execute(
                """
                SELECT id, symbol, side, type, quantity, price, stop_price,
                       time_in_force, status, filled_quantity, average_fill_price,
                       created_at, updated_at, strategy_id, metadata_json
                FROM orders
                WHERE status IN ('new', 'partially_filled')
                ORDER BY created_at DESC
            """
            )

            orders = []
            async for row in cursor:
                order = self._row_to_order(row)
                if order:
                    orders.append(order)

            return orders

        except Exception as e:
            self.logger.log_error(e, {"context": "get_active_orders"})
            return []

    async def save_trade(self, trade: Trade) -> None:
        """Save trade to database."""
        try:
            if not self.connection:
                raise DatabaseError("Database not initialized")

            await self.connection.execute(
                """
                INSERT OR REPLACE INTO trades (
                    id, order_id, symbol, side, quantity, price, timestamp,
                    commission, strategy_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade.id,
                    trade.order_id,
                    trade.symbol,
                    trade.side.value,
                    float(trade.quantity),
                    float(trade.price),
                    trade.timestamp.isoformat(),
                    float(trade.commission),
                    trade.strategy_id,
                ),
            )

            await self.connection.commit()

        except Exception as e:
            self.logger.log_error(e, {"context": "save_trade", "trade_id": trade.id})
            raise DatabaseError(f"Failed to save trade: {e}")

    async def get_trades(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> List[Trade]:
        """Get trades from database."""
        try:
            if not self.connection:
                raise DatabaseError("Database not initialized")

            query = """
                SELECT id, order_id, symbol, side, quantity, price, timestamp,
                       commission, strategy_id
                FROM trades
            """
            params = []

            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = await self.connection.execute(query, params)

            trades = []
            async for row in cursor:
                trade = self._row_to_trade(row)
                if trade:
                    trades.append(trade)

            return trades

        except Exception as e:
            self.logger.log_error(e, {"context": "get_trades"})
            return []

    async def save_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Save performance metrics to database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO performance_metrics
                    (total_return, sharpe_ratio, max_drawdown, win_rate,
                     total_trades, winning_trades, losing_trades, profit_factor,
                     created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        float(metrics.total_return),
                        float(metrics.sharpe_ratio),
                        float(metrics.max_drawdown),
                        float(metrics.win_rate),
                        metrics.total_trades,
                        metrics.winning_trades,
                        metrics.losing_trades,
                        float(metrics.profit_factor),
                        datetime.utcnow().isoformat(),
                    ),
                )
                await db.commit()

        except Exception as e:
            self.logger.log_error(e, {"context": "save_performance_metrics"})
            raise DatabaseError(f"Failed to save performance metrics: {e}")

    async def get_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """Get latest performance metrics from database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT total_return, sharpe_ratio, max_drawdown, win_rate,
                           total_trades, winning_trades, losing_trades, profit_factor
                    FROM performance_metrics
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                row = await cursor.fetchone()

                if not row:
                    return None

                return PerformanceMetrics(
                    total_return=Decimal(str(row[0])),
                    sharpe_ratio=Decimal(str(row[1])),
                    max_drawdown=Decimal(str(row[2])),
                    win_rate=Decimal(str(row[3])),
                    total_trades=row[4],
                    winning_trades=row[5],
                    losing_trades=row[6],
                    profit_factor=Decimal(str(row[7])),
                )

        except Exception as e:
            self.logger.log_error(e, {"context": "get_performance_metrics"})
            return None

    async def _create_tables(self) -> None:
        """Create database tables."""
        if not self.connection:
            raise DatabaseError("Database not initialized")

        # Portfolio table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY,
                total_value REAL NOT NULL,
                buying_power REAL NOT NULL,
                cash REAL NOT NULL,
                day_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                positions_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )

        # Orders table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                stop_price REAL,
                time_in_force TEXT NOT NULL,
                status TEXT NOT NULL,
                filled_quantity REAL NOT NULL,
                average_fill_price REAL,
                created_at TEXT,
                updated_at TEXT,
                strategy_id TEXT,
                metadata_json TEXT NOT NULL
            )
        """
        )

        # Trades table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                commission REAL NOT NULL,
                strategy_id TEXT,
                FOREIGN KEY (order_id) REFERENCES orders (id)
            )
        """
        )

        # Performance metrics table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY,
                total_return REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                win_rate REAL NOT NULL,
                profit_factor REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                average_win REAL NOT NULL,
                average_loss REAL NOT NULL,
                largest_win REAL NOT NULL,
                largest_loss REAL NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """
        )

        # Create indexes
        await self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_orders_status
            ON orders (status)
        """
        )

        await self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp
            ON trades (symbol, timestamp)
        """
        )

        await self.connection.commit()

    def _extract_db_path(self, db_url: str) -> str:
        """Extract database file path from URL."""
        if db_url.startswith("sqlite+aiosqlite:///"):
            return db_url.replace("sqlite+aiosqlite:///", "")
        elif db_url.startswith("sqlite:///"):
            return db_url.replace("sqlite:///", "")
        else:
            # Default fallback
            return "trading_bot.db"

    def _row_to_order(self, row: tuple) -> Optional[Order]:
        """Convert database row to Order object."""
        try:
            from ..core.models import OrderSide, OrderStatus, OrderType

            return Order(
                id=row[0],
                symbol=row[1],
                side=OrderSide(row[2]),
                type=OrderType(row[3]),
                quantity=Decimal(str(row[4])),
                price=Decimal(str(row[5])) if row[5] else None,
                stop_price=Decimal(str(row[6])) if row[6] else None,
                time_in_force=row[7],
                status=OrderStatus(row[8]),
                filled_quantity=Decimal(str(row[9])),
                average_fill_price=Decimal(str(row[10])) if row[10] else None,
                created_at=datetime.fromisoformat(row[11]) if row[11] else None,
                updated_at=datetime.fromisoformat(row[12]) if row[12] else None,
                strategy_id=row[13],
                metadata=json.loads(row[14]) if row[14] else {},
            )
        except Exception as e:
            self.logger.log_error(e, {"context": "row_to_order"})
            return None

    def _row_to_trade(self, row: tuple) -> Optional[Trade]:
        """Convert database row to Trade object."""
        try:
            from ..core.models import OrderSide

            return Trade(
                id=row[0],
                order_id=row[1],
                symbol=row[2],
                side=OrderSide(row[3]),
                quantity=Decimal(str(row[4])),
                price=Decimal(str(row[5])),
                timestamp=datetime.fromisoformat(row[6]),
                commission=Decimal(str(row[7])),
                strategy_id=row[8],
            )
        except Exception as e:
            self.logger.log_error(e, {"context": "row_to_trade"})
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {"db_path": self.db_path, "connected": self.connection is not None}
