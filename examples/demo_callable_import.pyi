"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict, override

from nonfig import MakeableModel

def helper(data: list[int], func: Callable[[int], float]) -> list[float]: ...

class process:  # noqa: N801
  """Configurable function.

  Call Arguments:
      data (list[int])

  Hyperparameters:
      multiplier (float)
  """

  class _BoundFunction:
    """Bound function with hyperparameters as attributes."""

    multiplier: float
    def __call__(self, data: list[int]) -> list[float]: ...

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for process.

    Configuration:
        multiplier (float)
    """

    multiplier: float

  class Config(MakeableModel[_BoundFunction]):
    """Configuration class for process.

    Configurable function.

    Configuration:
        multiplier (float)
    """

    multiplier: float
    def __init__(self, *, multiplier: float = ...) -> None: ...
    """Initialize configuration for process.

        Configuration:
            multiplier (float)
        """

    @override
    def make(self) -> process._BoundFunction: ...

  Type = _BoundFunction
  def __call__(self, data: list[int], multiplier: float = ...) -> list[float]: ...
