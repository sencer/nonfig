from collections.abc import Callable
from dataclasses import dataclass
import os
import statistics
import time
from typing import Any

# Disable debug logging
os.environ["LOGURU_LEVEL"] = "WARNING"

from nonfig import DEFAULT, Hyper, configurable


def time_it(func: Callable[[], Any], iterations: int = 100_000) -> float:
  """Run iterations and return average time in microseconds."""
  start = time.perf_counter()
  for _ in range(iterations):
    func()
  end = time.perf_counter()
  return (end - start) / iterations * 1_000_000


def run_benchmark(
  name: str, func: Callable[[], Any], trials: int = 7, iterations: int = 100_000
):
  results = []
  # Warmup
  time_it(func, iterations=10_000)

  for _ in range(trials):
    results.append(time_it(func, iterations=iterations))

  median = statistics.median(results)
  stdev = statistics.stdev(results) if len(results) > 1 else 0
  print(f"{name:<40} | Median: {median:6.3f}μs | Stdev: {stdev:6.3f}μs")
  return median, stdev


def run_all_benchmarks():
  print(f"{'Metric':<40} | {'Performance':<20}")
  print("-" * 75)

  # --- FUNCTION PATTERN ---
  def raw_func(x: int, y: float, z: int = 1) -> float:
    return (x + z) * y

  @configurable
  def configured_func(x: Hyper[int], y: Hyper[float], z: int = 1) -> float:
    return (x + z) * y

  run_benchmark("Function: Raw Call", lambda: raw_func(10, 0.5, 2))
  run_benchmark(
    "Function: Direct Call (Decorated)", lambda: configured_func(10, 0.5, 2)
  )
  run_benchmark(
    "Function: Config Creation", lambda: configured_func.Config(x=10, y=0.5)
  )

  cfg_f = configured_func.Config(x=10, y=0.5)
  run_benchmark("Function: make()", cfg_f.make)

  made_f = cfg_f.make()
  run_benchmark("Function: Made Call", lambda: made_f(z=2))

  print("-" * 75)

  # --- CLASS PATTERN ---
  class RawClass:
    def __init__(self, x: int, y: float):
      self.x = x
      self.y = y

  @configurable
  class ConfiguredClass:
    def __init__(self, x: int, y: float):
      self.x = x
      self.y = y

  run_benchmark("Class: Raw Init", lambda: RawClass(10, 0.5))
  run_benchmark("Class: Direct Init (Decorated)", lambda: ConfiguredClass(10, 0.5))
  run_benchmark("Class: Config Creation", lambda: ConfiguredClass.Config(x=10, y=0.5))

  cfg_c = ConfiguredClass.Config(x=10, y=0.5)
  run_benchmark("Class: make()", cfg_c.make)

  print("-" * 75)

  # --- DATACLASS PATTERN ---
  @dataclass
  class RawData:
    x: int
    y: float

  @configurable
  @dataclass
  class ConfiguredData:
    x: int
    y: float

  run_benchmark("Dataclass: Raw Init", lambda: RawData(10, 0.5))
  run_benchmark("Dataclass: Direct Init (Decorated)", lambda: ConfiguredData(10, 0.5))
  run_benchmark(
    "Dataclass: Config Creation", lambda: ConfiguredData.Config(x=10, y=0.5)
  )

  cfg_d = ConfiguredData.Config(x=10, y=0.5)
  run_benchmark("Dataclass: make()", cfg_d.make)

  print("-" * 75)

  # --- VALIDATION PATTERN ---
  from nonfig import Ge, Le

  @configurable
  def validated_func(
    x: Hyper[int, Ge[0], Le[100]] = 10, y: Hyper[float, Ge[0.0], Le[1.0]] = 0.5
  ) -> float:
    return x * y

  run_benchmark(
    "Validation: Config Creation", lambda: validated_func.Config(x=50, y=0.75)
  )

  def run_full_validation():
    return validated_func.Config(x=50, y=0.75).make()()

  run_benchmark("Validation: Full Pattern", run_full_validation)

  print("-" * 75)

  # --- NESTING PATTERN ---
  @configurable
  @dataclass
  class Inner:
    x: int = 1

  @configurable
  @dataclass
  class Outer:
    inner: Hyper[Inner] = DEFAULT

  inner_cfg = Inner.Config(x=10)
  outer_cfg = Outer.Config(inner=inner_cfg)

  run_benchmark("Nesting: make() (Reused)", outer_cfg.make)

  def run_full_nesting():
    return Outer.Config(inner=Inner.Config(x=10)).make()

  run_benchmark("Nesting: Full make()", run_full_nesting)


if __name__ == "__main__":
  run_all_benchmarks()
