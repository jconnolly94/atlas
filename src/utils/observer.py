from abc import ABC, abstractmethod
from typing import Dict, Any, List


class Observer(ABC):
    """Base observer interface for simulation events."""

    @abstractmethod
    def on_step_complete(self, data: Dict[str, Any]) -> None:
        """Called after each simulation step."""
        pass

    @abstractmethod
    def on_episode_complete(self, data: Dict[str, Any]) -> None:
        """Called after each episode is complete."""
        pass


class Observable:
    """Base class for objects that can be observed."""

    def __init__(self):
        self._observers: List[Observer] = []

    def add_observer(self, observer: Observer) -> None:
        """Add an observer to this object."""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: Observer) -> None:
        """Remove an observer from this object."""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_step_complete(self, data: Dict[str, Any]) -> None:
        """Notify all observers that a step is complete."""
        for observer in self._observers:
            observer.on_step_complete(data)

    def notify_episode_complete(self, data: Dict[str, Any]) -> None:
        """Notify all observers that an episode is complete."""
        for observer in self._observers:
            observer.on_episode_complete(data)