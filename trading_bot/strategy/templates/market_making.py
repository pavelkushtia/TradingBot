"""Market making strategy template."""

from typing import Any, Dict, List

from ...core.models import StrategySignal
from ..base import BaseStrategy


class MarketMakingTemplate(BaseStrategy):
    """
    Market making strategy template.

    This strategy provides liquidity by placing buy and sell orders around
    the current market price, profiting from the bid-ask spread.
    """

    def __init__(self, name: str, parameters: Dict[str, Any]):
        default_params = {
            "spread_pct": 0.001,  # 0.1% spread from mid
            "max_inventory": 1000,  # Max shares to hold
            "inventory_target": 0,  # Target inventory (neutral)
            "position_size": 0.01,
        }
        default_params.update(parameters)
        super().__init__(name, default_params)

    async def generate_signals(self) -> List[StrategySignal]:
        """Generate market making signals."""
        # Simplified implementation
        return []

    def get_strategy_description(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "market_making",
            "description": "Liquidity provision with bid-ask spread capture",
            "indicators_used": ["Bid-Ask Spread", "Volume", "Inventory"],
            "parameters": self.parameters,
        }
