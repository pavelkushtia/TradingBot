"""Pre-built strategy templates for common trading approaches."""

from .arbitrage import ArbitrageTemplate
from .market_making import MarketMakingTemplate
from .mean_reversion import MeanReversionTemplate
from .momentum import MomentumTemplate
from .pairs_trading import PairsTradingTemplate

__all__ = [
    "MeanReversionTemplate",
    "MomentumTemplate",
    "PairsTradingTemplate",
    "ArbitrageTemplate",
    "MarketMakingTemplate",
]
