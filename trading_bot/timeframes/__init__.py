"""Multi-timeframe support for trading strategies."""

from .manager import (MultiTimeframeManager, MultiTimeframeStrategy, Timeframe,
                      TimeframeAggregator, TimeframeConfig, TimeframeData)

# Optional imports - these modules may not exist yet
try:
    pass

    AGGREGATOR_AVAILABLE = True
except ImportError:
    AGGREGATOR_AVAILABLE = False

try:
    pass

    SYNCHRONIZER_AVAILABLE = True
except ImportError:
    SYNCHRONIZER_AVAILABLE = False

__all__ = [
    "MultiTimeframeManager",
    "TimeframeConfig",
    "TimeframeData",
    "MultiTimeframeStrategy",
    "Timeframe",
    "TimeframeAggregator",
]

if SYNCHRONIZER_AVAILABLE:
    __all__.extend(
        [
            "TimeframeSynchronizer",
            "DataAlignment",
            "TimeframeConverter",
        ]
    )

if AGGREGATOR_AVAILABLE:
    __all__.extend(
        [
            "DataAggregator",
            "BarAggregator",
            "VolumeProfileAggregator",
        ]
    )
