"""Example demonstrating public function inclusion in stubs.

Both @configurable and regular public functions are included in the stub.
"""

from __future__ import annotations

from nonfig import Hyper, configurable


def helper_function(data: list[int], multiplier: float = 2.0) -> list[float]:
  """Public helper function that should be in the stub.

  This is NOT @configurable but is public (in __all__ implicitly).
  Type checkers need this signature.
  """
  return [x * multiplier for x in data]


@configurable
def process(
  data: list[int],
  multiplier: Hyper[float] = 2.0,
) -> list[float]:
  """Configurable function that uses the helper.

  This will have Config class generated, but helper_function
  should also be in the stub.
  """
  return helper_function(data, multiplier)


def _private_helper(x: int) -> int:
  """Private function - should NOT be in stub."""
  return x * 2


_ = _private_helper  # Reference to suppress unused warning
