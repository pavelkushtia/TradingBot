import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List

from .models import MarketData, Order
from .signal import StrategySignal


class BaseEvent(ABC):
    """Abstract base class for all events."""

    pass


class EventBus:
    """A simple asynchronous event bus."""

    def __init__(self):
        self._listeners = defaultdict(list)

    def subscribe(self, event_type: str, listener: Callable[..., Coroutine]):
        """Subscribe a listener to an event type."""
        self._listeners[event_type].append(listener)

    async def publish(self, event_type: str, *args, **kwargs):
        """Publish an event to all subscribed listeners."""
        if event_type in self._listeners:
            tasks = [
                listener(*args, **kwargs) for listener in self._listeners[event_type]
            ]
            await asyncio.gather(*tasks)


# Core Events
class MarketDataEvent:
    """Event for new market data."""

    def __init__(self, market_data: MarketData):
        self.market_data = market_data


class SignalEvent(BaseEvent):
    """Event triggered when a new signal is generated."""

    signal: StrategySignal


class ApprovedSignalEvent(BaseEvent):
    """Event triggered when a signal is approved by the risk manager."""

    signal: StrategySignal


class OrderEvent(BaseEvent):
    """Event triggered when a new order is created."""

    def __init__(self, order: Order):
        self.order = order
