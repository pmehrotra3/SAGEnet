# BaseCallback.py
# Abstract base class for training callbacks.
# Subclasses must implement all three methods — on_training_start, on_episode_end,
# and on_training_end — to be used with any CMA-ES training script.
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com

from abc import ABC, abstractmethod


# =============================================================================
# BaseCallback
# =============================================================================

class BaseCallback(ABC):
    """
    Abstract base class for callbacks passed to model.learn().
    Subclasses must implement all three methods.
    """

    @abstractmethod
    def on_training_start(self) -> None:
        """Called once before training begins."""
        ...

    @abstractmethod
    def on_episode_end(self, ep_return: float, ep_length: int) -> None:
        """Called after every rollout with the episode return and length."""
        ...

    @abstractmethod
    def on_training_end(self) -> None:
        """Called once after training ends."""
        ...