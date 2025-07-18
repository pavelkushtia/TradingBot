"""Risk management manager for position sizing and risk controls."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from ..core.config import Config
from ..core.events import ApprovedSignalEvent, EventBus, SignalEvent
from ..core.logging import TradingLogger
from ..core.models import Portfolio, Position
from ..core.signal import StrategySignal


class RiskManager:
    """Oversees risk management across all trading activities."""

    def __init__(self, config: Config, event_bus: EventBus):
        """Initialize the RiskManager."""
        self.config = config
        self.logger = TradingLogger("risk_manager")
        self.event_bus = event_bus
        self.max_drawdown = config.risk.max_drawdown
        self.max_position_size = config.trading.max_position_size
        self.max_daily_loss = config.risk.max_daily_loss
        self.max_open_positions = config.risk.max_open_positions
        self.last_reset_date = datetime.now(timezone.utc).date()
        self.rejected_signals_count = 0
        self.daily_pnl = Decimal("0.0")

    async def on_signal(self, event: SignalEvent) -> None:
        """Handle new signal."""
        # This is a placeholder for where the portfolio would be retrieved
        portfolio = None
        is_approved = await self.evaluate_signal(event, portfolio)
        if is_approved:
            await self.event_bus.publish(
                "signal.approved", ApprovedSignalEvent(signal=event.signal)
            )

    async def initialize(self) -> None:
        """Initialize risk manager."""
        self.logger.logger.info("Risk manager initialized")

    async def shutdown(self) -> None:
        """Shutdown risk manager."""
        self.logger.logger.info("Risk manager shutdown")

    async def evaluate_signal(
        self, signal: StrategySignal, portfolio: Portfolio
    ) -> bool:
        """Evaluate if a signal passes risk management checks."""
        try:
            # Reset daily counters if new day
            await self._check_daily_reset()

            # Check daily loss limit
            if not await self._check_daily_loss_limit(portfolio):
                self.logger.log_risk_event(
                    "daily_loss_limit_exceeded",
                    {
                        "daily_pnl": str(self.daily_pnl),
                        "limit": str(self.max_daily_loss),
                    },
                )
                return False

            # Check maximum open positions
            if not await self._check_max_positions(portfolio):
                self.logger.log_risk_event(
                    "max_positions_exceeded",
                    {
                        "current_positions": (
                            len(portfolio.positions) if portfolio else 0
                        ),
                        "limit": self.max_open_positions,
                    },
                )
                return False

            # Check position size limits
            if not await self._check_position_size_limit(signal, portfolio):
                self.logger.log_risk_event(
                    "position_size_limit_exceeded",
                    {"symbol": signal.symbol, "signal_type": signal.signal_type},
                )
                return False

            # Check symbol concentration
            if not await self._check_symbol_concentration(signal, portfolio):
                self.logger.log_risk_event(
                    "symbol_concentration_exceeded", {"symbol": signal.symbol}
                )
                return False

            return True

        except Exception as e:
            self.logger.log_error(e, {"context": "signal_evaluation"})
            self.rejected_signals_count += 1
            return False

    async def calculate_position_size(
        self, symbol: str, price: Decimal, portfolio: Optional[Portfolio]
    ) -> Decimal:
        """Calculate appropriate position size based on risk parameters."""
        if not portfolio or price <= 0:
            return Decimal("0")

        try:
            # Base position size as percentage of portfolio value
            base_size = portfolio.total_value * Decimal(str(self.max_position_size))

            # Calculate number of shares
            max_shares = base_size / price

            # Apply additional constraints

            # 1. Volatility-based sizing (simplified)
            volatility_factor = await self._calculate_volatility_factor(symbol)
            adjusted_shares = max_shares * volatility_factor

            # 2. Existing position constraints
            existing_position = portfolio.positions.get(symbol)
            if existing_position:
                # Limit additional exposure
                current_exposure = abs(existing_position.quantity * price)
                max_additional = (
                    portfolio.total_value * Decimal("0.1")
                ) - current_exposure
                if max_additional <= 0:
                    return Decimal("0")
                adjusted_shares = min(adjusted_shares, max_additional / price)

            # 3. Available cash constraint
            required_cash = adjusted_shares * price
            if required_cash > portfolio.buying_power:
                adjusted_shares = portfolio.buying_power / price

            # 4. Minimum order size
            min_order = Decimal("1")  # Minimum 1 share
            if adjusted_shares < min_order:
                return Decimal("0")

            # Round down to whole shares
            final_shares = int(adjusted_shares)

            return Decimal(str(final_shares))

        except Exception as e:
            self.logger.log_error(e, {"context": "position_sizing", "symbol": symbol})
            return Decimal("0")

    async def calculate_stop_loss_price(
        self, position: Position, current_price: Decimal
    ) -> Decimal:
        """Calculate stop loss price for a position."""
        if position.side.value == "long":
            # Long position: stop loss below current price
            stop_price = current_price * (1 - Decimal(str(self.stop_loss_pct)))
        else:
            # Short position: stop loss above current price
            stop_price = current_price * (1 + Decimal(str(self.stop_loss_pct)))

        return stop_price

    async def calculate_take_profit_price(
        self, position: Position, current_price: Decimal
    ) -> Decimal:
        """Calculate take profit price for a position."""
        if position.side.value == "long":
            # Long position: take profit above current price
            take_profit = current_price * (1 + Decimal(str(self.take_profit_pct)))
        else:
            # Short position: take profit below current price
            take_profit = current_price * (1 - Decimal(str(self.take_profit_pct)))

        return take_profit

    async def check_portfolio_risk(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Comprehensive portfolio risk assessment."""
        risk_metrics = {}

        try:
            # Portfolio concentration
            total_value = portfolio.total_value
            max_position_value = Decimal("0")

            for position in portfolio.positions.values():
                position_value = abs(position.market_value)
                if position_value > max_position_value:
                    max_position_value = position_value

            concentration = (
                float(max_position_value / total_value) if total_value > 0 else 0
            )

            # Daily P&L as percentage
            daily_pnl_pct = (
                float(portfolio.day_pnl / total_value) if total_value > 0 else 0
            )

            # Number of positions
            position_count = len(portfolio.positions)

            risk_metrics = {
                "portfolio_value": float(total_value),
                "daily_pnl_pct": daily_pnl_pct,
                "max_concentration": concentration,
                "position_count": position_count,
                "max_positions_limit": self.max_open_positions,
                "daily_loss_limit": float(self.max_daily_loss),
                "concentration_limit": float(self.max_position_size),
                "risk_violations": [],
            }

            # Check for violations
            if daily_pnl_pct <= -float(self.max_daily_loss):
                risk_metrics["risk_violations"].append("daily_loss_limit")

            if concentration > float(self.max_position_size):
                risk_metrics["risk_violations"].append("concentration_limit")

            if position_count > self.max_open_positions:
                risk_metrics["risk_violations"].append("max_positions")

        except Exception as e:
            self.logger.log_error(e, {"context": "portfolio_risk_check"})

        return risk_metrics

    async def _check_daily_reset(self) -> None:
        """Reset daily counters if new trading day."""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self.last_reset_date:
            self.daily_pnl = Decimal("0")
            self.daily_trades = 0
            self.last_reset_date = current_date
            self.logger.logger.info("Daily risk counters reset")

    async def _check_daily_loss_limit(self, portfolio: Optional[Portfolio]) -> bool:
        """Check if daily loss limit is exceeded."""
        if not portfolio:
            return True

        daily_loss_pct = portfolio.day_pnl / portfolio.total_value
        return daily_loss_pct > -Decimal(str(self.max_daily_loss))

    async def _check_max_positions(self, portfolio: Optional[Portfolio]) -> bool:
        """Check if maximum positions limit is exceeded."""
        if not portfolio:
            return True

        return len(portfolio.positions) < self.max_open_positions

    async def _check_position_size_limit(
        self, signal: StrategySignal, portfolio: Optional[Portfolio]
    ) -> bool:
        """Check if position size would exceed limits."""
        if not portfolio or not signal.price:
            return True

        # Calculate what the position size would be
        position_size = await self.calculate_position_size(
            signal.symbol, signal.price, portfolio
        )
        position_value = position_size * signal.price

        # Check against portfolio percentage limit
        max_position_value = portfolio.total_value * Decimal(
            str(self.max_position_size)
        )

        return position_value <= max_position_value

    async def _check_symbol_concentration(
        self, signal: StrategySignal, portfolio: Optional[Portfolio]
    ) -> bool:
        """Check symbol concentration limits."""
        if not portfolio:
            return True

        # Allow signals for symbols we already have positions in
        if signal.symbol in portfolio.positions:
            return True

        # For new symbols, ensure we're not over-concentrated
        symbol_count = len(portfolio.positions)
        max_symbols = min(self.max_open_positions, 20)  # Reasonable limit

        return symbol_count < max_symbols

    async def _calculate_volatility_factor(self, symbol: str) -> Decimal:
        """Calculate volatility-based position sizing factor."""
        # Simplified volatility calculation
        # In practice, you'd use historical price data

        # Default moderate volatility factor
        base_factor = Decimal("1.0")

        # You could implement:
        # - Historical volatility calculation
        # - VIX-based adjustments
        # - Sector-specific volatility

        return base_factor

    def get_stats(self) -> Dict[str, Any]:
        """Get risk management statistics."""
        return {
            "daily_pnl": str(self.daily_pnl),
            "daily_trades": self.daily_trades,
            "rejected_signals": self.rejected_signals_count,
            "max_position_size": str(self.max_position_size),
            "max_daily_loss": str(self.max_daily_loss),
            "max_open_positions": self.max_open_positions,
            "last_reset_date": self.last_reset_date.isoformat(),
        }
