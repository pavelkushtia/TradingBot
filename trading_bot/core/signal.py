from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


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
        json_encoders = {Decimal: str, datetime: lambda v: v.isoformat()}
