"""Simple technical indicators that work with existing strategies."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .core import (
    ATR,
    EMA,
    MACD,
    RSI,
    SMA,
    BollingerBands,
    SimpleIndicatorManager,
    Stochastic,
)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""

    name: str = "sma"
    period: int = 14
    parameters: Optional[Dict[str, Any]] = None
    source: str = "close"

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


# Alias for backward compatibility
IndicatorManager = SimpleIndicatorManager

__all__ = [
    "IndicatorManager",
    "SimpleIndicatorManager",
    "IndicatorConfig",
    "SMA",
    "EMA",
    "RSI",
    "MACD",
    "BollingerBands",
    "ATR",
    "Stochastic",
]
