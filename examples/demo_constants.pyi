"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import TypedDict, override

from nonfig import MakeableModel

WINDOW_SIZE: ...
THRESHOLD: ...
MODE: ...
UNUSED_CONSTANT: ...

class process:  # noqa: N801
  """Process data using module constants as defaults.

  Call Arguments:
      data (list[float])

  Hyperparameters:
      window (int)
  """

  class _BoundFunction:
    """Bound function with hyperparameters as attributes."""

    window: int
    def __call__(self, data: list[float]) -> list[float]: ...

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for process.

    Configuration:
        window (int)
    """

    window: int

  class Config(MakeableModel[_BoundFunction]):
    """Configuration class for process.

    Process data using module constants as defaults.

    Configuration:
        window (int)
    """

    window: int
    def __init__(self, *, window: int = ...) -> None: ...
    """Initialize configuration for process.

        Configuration:
            window (int)
        """

    @override
    def make(self) -> process._BoundFunction: ...

  Type = _BoundFunction
  def __call__(self, data: list[float], window: int = ...) -> list[float]: ...
