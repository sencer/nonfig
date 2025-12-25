"""Example demonstrating import of types used in public functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nonfig import Hyper, configurable

if TYPE_CHECKING:
  from collections.abc import Callable


def helper(data: list[int], func: Callable[[int], float]) -> list[float]:
  """Helper function using Callable - should import it in stub."""
  return [func(x) for x in data]


@configurable
def process(
  data: list[int],
  multiplier: Hyper[float] = 2.0,
) -> list[float]:
  """Configurable function."""
  return helper(data, lambda x: x * multiplier)
