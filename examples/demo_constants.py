"""Example demonstrating module constant extraction in stubs.

Module-level constants used as default values are now included in generated stubs.
"""

from __future__ import annotations

from nonfig import Hyper, configurable

# Module constants that will be used in default values
WINDOW_SIZE = 10
THRESHOLD = 0.75
MODE = "fast"
UNUSED_CONSTANT = 99  # This won't be in the stub


@configurable
def process(
  data: list[float],
  window: Hyper[int] = WINDOW_SIZE,
) -> list[float]:
  """Process data using module constants as defaults."""
  return data[:window]
