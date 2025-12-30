"""Robust micro-benchmark suite for nonfig.

Design principles for micro-optimization validation:
1. Use MIN time (not median) - represents true cost without interference
2. High iteration count to amortize measurement overhead
3. Multiple trials with warmup to reach steady state
4. Report both min and stdev to gauge stability
5. Side-by-side comparison in same process to reduce variance
"""

from collections.abc import Callable
from dataclasses import dataclass
import os
import statistics
import time
from typing import Any

# Disable debug logging
os.environ["LOGURU_LEVEL"] = "WARNING"

from nonfig import DEFAULT, Hyper, configurable

# Configuration
TRIALS = 15
ITERATIONS = 100_000


def time_ns(func: Callable[[], Any], iterations: int) -> float:
  """Run iterations and return total time in nanoseconds."""
  start = time.perf_counter_ns()
  for _ in range(iterations):
    func()
  return time.perf_counter_ns() - start


def benchmark(
  name: str,
  func: Callable[[], Any],
  trials: int = TRIALS,
  iterations: int = ITERATIONS,
) -> tuple[float, float]:
  """Run benchmark with warmup, return (min_ns_per_call, stdev_ns_per_call)."""
  # Warmup
  time_ns(func, iterations // 5)

  results_ns = []
  for _ in range(trials):
    total_ns = time_ns(func, iterations)
    results_ns.append(total_ns / iterations)
    # Progress dot
    print(".", end="", flush=True)

  min_ns = min(results_ns)
  stdev_ns = statistics.stdev(results_ns) if len(results_ns) > 1 else 0.0

  # Clear progress dots and print result
  print(f"\r{name:<40} | Min: {min_ns:7.1f}ns | Stdev: {stdev_ns:5.1f}ns")

  return min_ns, stdev_ns


def compare(
  name: str,
  baseline: Callable[[], Any],
  optimized: Callable[[], Any],
  trials: int = TRIALS,
  iterations: int = ITERATIONS,
) -> None:
  """Compare two implementations side-by-side, report speedup."""
  # Interleaved warmup
  time_ns(baseline, iterations // 5)
  time_ns(optimized, iterations // 5)

  baseline_results = []
  optimized_results = []

  # Interleave trials to reduce systematic bias
  for _ in range(trials):
    baseline_results.append(time_ns(baseline, iterations) / iterations)
    optimized_results.append(time_ns(optimized, iterations) / iterations)
    print(".", end="", flush=True)

  base_min = min(baseline_results)
  opt_min = min(optimized_results)
  speedup = (base_min - opt_min) / base_min * 100 if base_min > 0 else 0

  indicator = "+" if speedup > 5 else ("-" if speedup < -5 else "=")

  print(
    f"\r{name:<40} | Base: {base_min:6.1f}ns | Opt: {opt_min:6.1f}ns | {speedup:+5.1f}% {indicator}"
  )


def run_all_benchmarks():
  print("=" * 80)
  print("NONFIG MICRO-BENCHMARK SUITE")
  print(f"Method: MIN of {TRIALS} trials x {ITERATIONS:,} iterations")
  print("=" * 80)

  # --- FUNCTION PATTERN ---
  print("\n[1/6] FUNCTION PATTERN")

  def raw_func(x: int, y: float, z: int = 1) -> float:
    return (x + z) * y

  @configurable
  def configured_func(x: Hyper[int], y: Hyper[float], z: int = 1) -> float:
    return (x + z) * y

  benchmark("Function: Raw Call", lambda: raw_func(10, 0.5, 2))
  benchmark("Function: Direct Call (Decorated)", lambda: configured_func(10, 0.5, 2))
  benchmark("Function: Config Creation", lambda: configured_func.Config(x=10, y=0.5))

  cfg_f = configured_func.Config(x=10, y=0.5)
  benchmark("Function: make()", cfg_f.make)

  made_f = cfg_f.make()
  benchmark("Function: Made Call", lambda: made_f(z=2))

  # --- CLASS PATTERN ---
  print("\n[2/6] CLASS PATTERN")

  class RawClass:
    def __init__(self, x: int, y: float):
      self.x = x
      self.y = y

    def method(self, z: int) -> float:
      return (self.x + z) * self.y

  @configurable
  class ConfiguredClass:
    def __init__(self, x: int, y: float):
      self.x = x
      self.y = y

    def method(self, z: int) -> float:
      return (self.x + z) * self.y

  benchmark("Class: Raw Init", lambda: RawClass(10, 0.5))
  benchmark("Class: Direct Init (Decorated)", lambda: ConfiguredClass(10, 0.5))
  benchmark("Class: Config Creation", lambda: ConfiguredClass.Config(x=10, y=0.5))

  cfg_c = ConfiguredClass.Config(x=10, y=0.5)
  benchmark("Class: make()", cfg_c.make)

  raw_inst = RawClass(10, 0.5)
  direct_inst = ConfiguredClass(10, 0.5)
  made_inst = cfg_c.make()

  benchmark("Class: Method Call (Raw)", lambda: raw_inst.method(2))
  benchmark("Class: Method Call (Direct)", lambda: direct_inst.method(2))
  benchmark("Class: Method Call (Made)", lambda: made_inst.method(2))

  # --- DATACLASS PATTERN ---
  print("\n[3/6] DATACLASS PATTERN")

  @dataclass
  class RawData:
    x: int
    y: float

  @configurable
  @dataclass
  class ConfiguredData:
    x: int
    y: float

  benchmark("Dataclass: Raw Init", lambda: RawData(10, 0.5))
  benchmark("Dataclass: Direct Init (Decorated)", lambda: ConfiguredData(10, 0.5))
  benchmark("Dataclass: Config Creation", lambda: ConfiguredData.Config(x=10, y=0.5))

  cfg_d = ConfiguredData.Config(x=10, y=0.5)
  benchmark("Dataclass: make()", cfg_d.make)

  # --- VALIDATION PATTERN ---
  print("\n[4/6] VALIDATION PATTERN")

  from nonfig import Ge, Le

  @configurable
  def validated_func(
    x: Hyper[int, Ge[0], Le[100]] = 10, y: Hyper[float, Ge[0.0], Le[1.0]] = 0.5
  ) -> float:
    return x * y

  benchmark("Validation: Config Creation", lambda: validated_func.Config(x=50, y=0.75))

  def run_validation_full():
    return validated_func.Config(x=50, y=0.75).make()()

  benchmark("Validation: Full Pattern", run_validation_full)

  # --- NESTING PATTERN ---
  print("\n[5/6] NESTING PATTERN")

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

  benchmark("Nesting: make() (Reused Config)", outer_cfg.make)
  benchmark(
    "Nesting: Full make() (Fresh Config)",
    lambda: Outer.Config(inner=Inner.Config(x=10)).make(),
  )

  # --- OVERHEAD COMPARISON ---
  print("\n[6/6] OVERHEAD vs RAW (side-by-side)")

  compare(
    "Function Call Overhead",
    lambda: raw_func(10, 0.5, 2),
    lambda: configured_func(10, 0.5, 2),
  )

  compare(
    "Class Init Overhead",
    lambda: RawClass(10, 0.5),
    lambda: ConfiguredClass(10, 0.5),
  )

  compare(
    "Dataclass Init Overhead",
    lambda: RawData(10, 0.5),
    lambda: ConfiguredData(10, 0.5),
  )

  print("\n" + "=" * 80)
  print("Legend: + >5% faster | = within ±5% | - >5% slower")
  print("=" * 80)


if __name__ == "__main__":
  run_all_benchmarks()
