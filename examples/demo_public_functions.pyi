"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import TypedDict, override

from nonfig import MakeableModel

def helper_function(data: list[int], multiplier: float = 2.0) -> list[float]: ...

class process:  # noqa: N801
  """Configurable function that uses the helper.

  This will have Config class generated, but helper_function
  should also be in the stub.

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

    Configurable function that uses the helper.

    This will have Config class generated, but helper_function
    should also be in the stub.

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
