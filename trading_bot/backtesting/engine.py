"""Backtesting engine for strategy validation and performance analysis."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.exceptions import BacktestError
from ..core.logging import TradingLogger
from ..core.models import MarketData, Portfolio, Order, Trade, PerformanceMetrics, OrderSide, OrderType, OrderStatus
from ..strategy.base import BaseStrategy


class BacktestEngine:
    """High-performance backtesting engine for strategy validation."""
    
    def __init__(self, config: Config):
        """Initialize backtesting engine."""
        self.config = config
        self.logger = TradingLogger("backtest_engine")
        
        # Backtest parameters
        self.initial_capital = Decimal(str(config.trading.portfolio_value))
        self.commission_per_share = Decimal("0.005")
        self.min_commission = Decimal("1.00")
        
        # State tracking
        self.portfolio: Optional[Portfolio] = None
        self.trades: List[Trade] = []
        self.orders: List[Order] = []
        self.daily_returns: List[Decimal] = []
        self.equity_curve: List[Tuple[datetime, Decimal]] = []
        
        # Performance metrics
        self.performance_metrics: Optional[PerformanceMetrics] = None
        
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        market_data: List[MarketData],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Run backtest for a strategy with historical data."""
        try:
            self.logger.logger.info(f"Starting backtest: {start_date} to {end_date}")
            
            # Initialize portfolio
            self._initialize_portfolio(start_date)
            
            # Initialize strategy
            await strategy.initialize()
            
            # Sort market data by timestamp
            market_data.sort(key=lambda x: x.timestamp)
            
            # Filter data by date range
            filtered_data = [
                bar for bar in market_data 
                if start_date <= bar.timestamp <= end_date
            ]
            
            if not filtered_data:
                raise BacktestError("No market data available for the specified date range")
            
            # Process each bar
            for bar in filtered_data:
                await self._process_bar(strategy, bar)
            
            # Calculate final performance metrics
            self.performance_metrics = self._calculate_performance_metrics(start_date, end_date)
            
            # Create backtest results
            results = {
                "strategy_name": strategy.name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": float(self.initial_capital),
                "final_capital": float(self.portfolio.total_value),
                "total_return": float(self.performance_metrics.total_return),
                "sharpe_ratio": float(self.performance_metrics.sharpe_ratio),
                "max_drawdown": float(self.performance_metrics.max_drawdown),
                "win_rate": float(self.performance_metrics.win_rate),
                "total_trades": self.performance_metrics.total_trades,
                "winning_trades": self.performance_metrics.winning_trades,
                "losing_trades": self.performance_metrics.losing_trades,
                "profit_factor": float(self.performance_metrics.profit_factor),
                "trades": [trade.dict() for trade in self.trades],
                "equity_curve": [(dt.isoformat(), float(value)) for dt, value in self.equity_curve],
                "daily_returns": [float(ret) for ret in self.daily_returns]
            }
            
            self.logger.logger.info(f"Backtest completed. Total return: {self.performance_metrics.total_return:.2%}")
            
            return results
            
        except Exception as e:
            self.logger.log_error(e, {"context": "run_backtest"})
            raise BacktestError(f"Backtest failed: {e}")
    
    async def _process_bar(self, strategy: BaseStrategy, bar: MarketData) -> None:
        """Process a single market data bar."""
        try:
            # Update strategy with new bar
            await strategy.on_bar(bar.symbol, bar)
            
            # Generate signals
            signals = await strategy.generate_signals()
            
            # Execute signals
            for signal in signals:
                await self._execute_signal(signal, bar)
            
            # Update portfolio value
            self._update_portfolio_value(bar)
            
            # Record equity curve point
            self.equity_curve.append((bar.timestamp, self.portfolio.total_value))
            
        except Exception as e:
            self.logger.log_error(e, {"context": "process_bar", "timestamp": bar.timestamp})
    
    async def _execute_signal(self, signal, bar: MarketData) -> None:
        """Execute a trading signal in the backtest."""
        try:
            # Calculate position size (simplified)
            position_size = self._calculate_position_size(signal, bar.close)
            
            if position_size <= 0:
                return
            
            # Create order
            order = Order(
                id=f"backtest_{len(self.orders) + 1}",
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == "buy" else OrderSide.SELL,
                type=OrderType.MARKET,
                quantity=position_size,
                price=bar.close,
                status=OrderStatus.FILLED,
                filled_quantity=position_size,
                average_fill_price=bar.close,
                created_at=bar.timestamp,
                updated_at=bar.timestamp,
                strategy_id=signal.strategy_name
            )
            
            self.orders.append(order)
            
            # Create trade
            trade = Trade(
                id=f"trade_{len(self.trades) + 1}",
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                timestamp=bar.timestamp,
                commission=self._calculate_commission(order.quantity, order.price),
                strategy_id=order.strategy_id
            )
            
            self.trades.append(trade)
            
            # Update portfolio
            self._update_portfolio_with_trade(trade)
            
        except Exception as e:
            self.logger.log_error(e, {"context": "execute_signal", "signal": signal.dict()})
    
    def _initialize_portfolio(self, start_date: datetime) -> None:
        """Initialize portfolio for backtesting."""
        self.portfolio = Portfolio(
            total_value=self.initial_capital,
            buying_power=self.initial_capital,
            cash=self.initial_capital,
            positions={},
            day_pnl=Decimal("0"),
            total_pnl=Decimal("0"),
            updated_at=start_date
        )
    
    def _calculate_position_size(self, signal, price: Decimal) -> Decimal:
        """Calculate position size for backtesting."""
        # Simple position sizing: 5% of portfolio value
        max_position_value = self.portfolio.total_value * Decimal("0.05")
        position_size = max_position_value / price
        
        # Ensure we have enough cash
        required_cash = position_size * price
        if required_cash > self.portfolio.cash:
            position_size = self.portfolio.cash / price
        
        # Round down to whole shares
        return Decimal(str(int(position_size)))
    
    def _calculate_commission(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Calculate commission for a trade."""
        commission = max(quantity * self.commission_per_share, self.min_commission)
        return commission
    
    def _update_portfolio_with_trade(self, trade: Trade) -> None:
        """Update portfolio state with a new trade."""
        if not self.portfolio:
            return
        
        # Calculate trade value including commission
        trade_value = trade.quantity * trade.price
        total_cost = trade_value + trade.commission
        
        if trade.side == OrderSide.BUY:
            # Buying: reduce cash, add/increase position
            self.portfolio.cash -= total_cost
            
            if trade.symbol in self.portfolio.positions:
                # Add to existing position
                existing_pos = self.portfolio.positions[trade.symbol]
                total_quantity = existing_pos.quantity + trade.quantity
                total_cost_basis = (existing_pos.quantity * existing_pos.average_price) + trade_value
                new_avg_price = total_cost_basis / total_quantity
                
                existing_pos.quantity = total_quantity
                existing_pos.average_price = new_avg_price
                existing_pos.updated_at = trade.timestamp
            else:
                # Create new position
                from ..core.models import Position, PositionSide
                
                self.portfolio.positions[trade.symbol] = Position(
                    symbol=trade.symbol,
                    side=PositionSide.LONG,
                    quantity=trade.quantity,
                    average_price=trade.price,
                    market_value=trade_value,
                    unrealized_pnl=Decimal("0"),
                    created_at=trade.timestamp,
                    updated_at=trade.timestamp,
                    strategy_id=trade.strategy_id
                )
        
        else:  # SELL
            # Selling: increase cash, reduce/remove position
            self.portfolio.cash += trade_value - trade.commission
            
            if trade.symbol in self.portfolio.positions:
                existing_pos = self.portfolio.positions[trade.symbol]
                
                if existing_pos.quantity >= trade.quantity:
                    # Partial or full sale
                    existing_pos.quantity -= trade.quantity
                    existing_pos.updated_at = trade.timestamp
                    
                    # Remove position if fully sold
                    if existing_pos.quantity == 0:
                        del self.portfolio.positions[trade.symbol]
                else:
                    # This shouldn't happen in a well-designed system
                    self.logger.logger.warning(f"Selling more shares than owned for {trade.symbol}")
    
    def _update_portfolio_value(self, bar: MarketData) -> None:
        """Update portfolio value based on current market prices."""
        if not self.portfolio:
            return
        
        # Update position values
        if bar.symbol in self.portfolio.positions:
            position = self.portfolio.positions[bar.symbol]
            position.market_value = position.quantity * bar.close
            position.unrealized_pnl = position.market_value - (position.quantity * position.average_price)
            position.updated_at = bar.timestamp
        
        # Update portfolio totals
        self.portfolio.total_value = self.portfolio.cash + self.portfolio.total_market_value
        self.portfolio.updated_at = bar.timestamp
    
    def _calculate_performance_metrics(self, start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return self._get_empty_performance_metrics(start_date, end_date)
        
        # Calculate trade-level metrics
        winning_trades = 0
        losing_trades = 0
        total_profit = Decimal("0")
        total_loss = Decimal("0")
        wins = []
        losses = []
        
        # Group trades by symbol to calculate P&L
        symbol_trades = {}
        for trade in self.trades:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)
        
        # Calculate P&L for each symbol
        for symbol, trades in symbol_trades.items():
            trades.sort(key=lambda x: x.timestamp)
            
            position_quantity = Decimal("0")
            position_cost = Decimal("0")
            
            for trade in trades:
                if trade.side == OrderSide.BUY:
                    position_quantity += trade.quantity
                    position_cost += trade.quantity * trade.price + trade.commission
                else:  # SELL
                    if position_quantity > 0:
                        # Calculate P&L for this sale
                        sold_quantity = min(trade.quantity, position_quantity)
                        avg_cost = position_cost / position_quantity if position_quantity > 0 else Decimal("0")
                        
                        sale_proceeds = sold_quantity * trade.price - trade.commission
                        cost_basis = sold_quantity * avg_cost
                        
                        pnl = sale_proceeds - cost_basis
                        
                        if pnl > 0:
                            winning_trades += 1
                            total_profit += pnl
                            wins.append(pnl)
                        else:
                            losing_trades += 1
                            total_loss += abs(pnl)
                            losses.append(abs(pnl))
                        
                        # Update position
                        position_quantity -= sold_quantity
                        if position_quantity > 0:
                            position_cost = position_cost * (position_quantity / (position_quantity + sold_quantity))
                        else:
                            position_cost = Decimal("0")
        
        # Calculate metrics
        total_trades = winning_trades + losing_trades
        win_rate = Decimal(str(winning_trades / total_trades)) if total_trades > 0 else Decimal("0")
        
        average_win = sum(wins) / len(wins) if wins else Decimal("0")
        average_loss = sum(losses) / len(losses) if losses else Decimal("0")
        largest_win = max(wins) if wins else Decimal("0")
        largest_loss = max(losses) if losses else Decimal("0")
        
        profit_factor = total_profit / total_loss if total_loss > 0 else Decimal("0")
        
        # Calculate total return
        total_return = (self.portfolio.total_value - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe ratio (simplified using daily returns)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            period_start=start_date,
            period_end=end_date
        )
    
    def _calculate_sharpe_ratio(self) -> Decimal:
        """Calculate Sharpe ratio from equity curve."""
        if len(self.equity_curve) < 2:
            return Decimal("0")
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_value = self.equity_curve[i-1][1]
            curr_value = self.equity_curve[i][1]
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(float(daily_return))
        
        if not returns:
            return Decimal("0")
        
        # Calculate Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return Decimal("0")
        
        # Annualize (assuming daily data)
        risk_free_rate = float(self.config.risk.risk_free_rate) / 252  # Daily risk-free rate
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
        
        return Decimal(str(round(sharpe, 4)))
    
    def _calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return Decimal("0")
        
        peak = self.equity_curve[0][1]
        max_drawdown = Decimal("0")
        
        for _, value in self.equity_curve:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _get_empty_performance_metrics(self, start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """Get empty performance metrics when no trades were made."""
        return PerformanceMetrics(
            total_return=Decimal("0"),
            sharpe_ratio=Decimal("0"),
            max_drawdown=Decimal("0"),
            win_rate=Decimal("0"),
            profit_factor=Decimal("0"),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            average_win=Decimal("0"),
            average_loss=Decimal("0"),
            largest_win=Decimal("0"),
            largest_loss=Decimal("0"),
            period_start=start_date,
            period_end=end_date
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        if not self.performance_metrics:
            return {}
        
        return {
            "summary": {
                "total_return": f"{self.performance_metrics.total_return:.2%}",
                "sharpe_ratio": f"{self.performance_metrics.sharpe_ratio:.2f}",
                "max_drawdown": f"{self.performance_metrics.max_drawdown:.2%}",
                "win_rate": f"{self.performance_metrics.win_rate:.2%}",
                "profit_factor": f"{self.performance_metrics.profit_factor:.2f}",
                "total_trades": self.performance_metrics.total_trades
            },
            "trade_analysis": {
                "winning_trades": self.performance_metrics.winning_trades,
                "losing_trades": self.performance_metrics.losing_trades,
                "average_win": f"${self.performance_metrics.average_win:.2f}",
                "average_loss": f"${self.performance_metrics.average_loss:.2f}",
                "largest_win": f"${self.performance_metrics.largest_win:.2f}",
                "largest_loss": f"${self.performance_metrics.largest_loss:.2f}"
            },
            "portfolio": {
                "initial_capital": f"${self.initial_capital:.2f}",
                "final_value": f"${self.portfolio.total_value:.2f}",
                "cash": f"${self.portfolio.cash:.2f}",
                "positions": len(self.portfolio.positions)
            }
        } 