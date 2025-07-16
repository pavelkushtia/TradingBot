"""Data models for trading entities."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


class MarketData(BaseModel):
    """Market data model."""
    
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Optional[Decimal] = None
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Quote(BaseModel):
    """Real-time quote model."""
    
    symbol: str
    timestamp: datetime
    bid_price: Decimal
    ask_price: Decimal
    bid_size: int
    ask_size: int
    
    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Order(BaseModel):
    """Order model."""
    
    id: Optional[str] = None
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "day"
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Optional[Decimal] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (can still be filled)."""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat() if v else None
        }


class Position(BaseModel):
    """Position model."""
    
    symbol: str
    side: PositionSide
    quantity: Decimal
    average_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal("0")
    created_at: datetime
    updated_at: datetime
    strategy_id: Optional[str] = None
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of position."""
        return abs(self.quantity * self.average_price)
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Trade(BaseModel):
    """Trade execution model."""
    
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    commission: Decimal = Decimal("0")
    strategy_id: Optional[str] = None
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of trade."""
        return self.quantity * self.price
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Portfolio(BaseModel):
    """Portfolio model."""
    
    total_value: Decimal
    buying_power: Decimal
    cash: Decimal
    positions: Dict[str, Position] = Field(default_factory=dict)
    day_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    updated_at: datetime
    
    @property
    def total_market_value(self) -> Decimal:
        """Calculate total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values()) or Decimal("0")
    
    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values()) or Decimal("0")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class StrategySignal(BaseModel):
    """Strategy signal model."""
    
    symbol: str
    signal_type: str  # "buy", "sell", "hold"
    strength: float = Field(ge=-1.0, le=1.0)  # Signal strength from -1 to 1
    price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    timestamp: datetime
    strategy_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    
    total_return: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    period_start: datetime
    period_end: datetime
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        } 