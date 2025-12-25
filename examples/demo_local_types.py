"""Example demonstrating local type definitions in stubs.

When @configurable functions reference locally-defined classes or types,
those definitions are now included in the generated stub.
"""

from __future__ import annotations

from dataclasses import dataclass

from nonfig import Hyper, configurable


# Local class that will be referenced in signature
@dataclass
class DataPoint:
  x: float
  y: float


# Type alias that will be referenced
ResultDict = dict[str, list[DataPoint]]


# Module constant
MAX_WINDOW: int = 100


@configurable
def analyze(
  data: list[DataPoint],
  window: Hyper[int] = MAX_WINDOW,
) -> ResultDict:
  """Analyze data points.

  Uses:
  - DataPoint (local class) in parameter and return type
  - ResultDict (local type alias) in return type
  - MAX_WINDOW (module constant) as default
  """
  return {"results": data[:window]}
