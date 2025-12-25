"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import TypedDict, override

from nonfig import MakeableModel

ResultDict = dict[str, list[DataPoint]]
MAX_WINDOW: int = ...

class DataPoint:
  x: float
  y: float

class analyze:  # noqa: N801
  """Analyze data points.

  Uses:
  - DataPoint (local class) in parameter and return type
  - ResultDict (local type alias) in return type
  - MAX_WINDOW (module constant) as default

  Call Arguments:
      data (list[DataPoint])

  Hyperparameters:
      window (int)
  """

  class _BoundFunction:
    """Bound function with hyperparameters as attributes."""

    window: int
    def __call__(self, data: list[DataPoint]) -> ResultDict: ...

  class ConfigDict(TypedDict, total=False):
    """Configuration dictionary for analyze.

    Configuration:
        window (int)
    """

    window: int

  class Config(MakeableModel[_BoundFunction]):
    """Configuration class for analyze.

    Analyze data points.

    Uses:
    - DataPoint (local class) in parameter and return type
    - ResultDict (local type alias) in return type
    - MAX_WINDOW (module constant) as default

    Configuration:
        window (int)
    """

    window: int
    def __init__(self, *, window: int = ...) -> None: ...
    """Initialize configuration for analyze.

        Configuration:
            window (int)
        """

    @override
    def make(self) -> analyze._BoundFunction: ...

  Type = _BoundFunction
  def __call__(self, data: list[DataPoint], window: int = ...) -> ResultDict: ...
