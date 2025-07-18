"""Risk management package."""

from .manager import RiskManager

try:
    from .enhanced_manager import (
        CorrelationAnalyzer,
        EnhancedRiskManager,
        PositionSizer,
        PositionSizingResult,
        RiskMetrics,
        VolatilityCalculator,
    )

    ENHANCED_RISK_AVAILABLE = True
    __all__ = [
        "RiskManager",
        "EnhancedRiskManager",
        "PositionSizer",
        "VolatilityCalculator",
        "CorrelationAnalyzer",
        "RiskMetrics",
        "PositionSizingResult",
    ]
except ImportError:
    ENHANCED_RISK_AVAILABLE = False
    __all__ = [
        "RiskManager",
    ]
