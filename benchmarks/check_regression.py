"""Automated performance regression check for CI with statistical significance."""

import sys
import timeit

import numpy as np

from nonfig import Hyper, configurable

# Threshold for BoundFunction call overhead in microseconds
# Typically 0.1 - 0.3us on most systems. 1.0us is a safe upper bound
# to catch significant regressions (e.g. accidentally re-extracting params).
MAX_OVERHEAD_US = 1.0


def plain(x: int, y: int) -> int:
  return x + y


@configurable
def decorated(x: int, y: Hyper[int] = 10) -> int:
  return x + y


def get_stats(times, iterations):
  # Convert total times to microseconds per call
  us_per_call = (np.array(times) / iterations) * 1_000_000
  mean = np.mean(us_per_call)
  std_dev = np.std(us_per_call, ddof=1)
  sem = std_dev / np.sqrt(len(times))
  # 95% Confidence Interval (approx 1.96 for n=30)
  margin_of_error = 1.96 * sem
  return mean, margin_of_error


def check_regression():
  iterations = 100_000
  repeats = 30

  # Prepare bound function once
  bound_fn = decorated.Config(y=10).make()

  print(f"Running benchmarks ({repeats} repeats, {iterations} iterations each)...")

  # Warmup
  timeit.timeit(lambda: plain(1, 10), number=1000)
  timeit.timeit(lambda: bound_fn(1), number=1000)

  # Benchmark
  t_plain_list = timeit.repeat(lambda: plain(1, 10), number=iterations, repeat=repeats)
  t_bound_list = timeit.repeat(lambda: bound_fn(1), number=iterations, repeat=repeats)

  mean_plain, ci_plain = get_stats(t_plain_list, iterations)
  mean_bound, ci_bound = get_stats(t_bound_list, iterations)

  overhead = mean_bound - mean_plain
  # Combined margin of error for the difference of means
  combined_ci = np.sqrt(ci_plain**2 + ci_bound**2)

  print("-" * 60)
  print(f"Baseline (raw call): {mean_plain:6.4f} \u00b1 {ci_plain:6.4f} \u00b5s/call")
  print(f"BoundFunction call:  {mean_bound:6.4f} \u00b1 {ci_bound:6.4f} \u00b5s/call")
  print(f"Measured Overhead:   {overhead:6.4f} \u00b1 {combined_ci:6.4f} \u00b5s/call")
  print("-" * 60)

  # We fail if the lower bound of the overhead estimate still exceeds our threshold.
  stat_overhead_min = overhead - combined_ci

  if stat_overhead_min > MAX_OVERHEAD_US:
    print("ERROR: Statistically significant performance regression detected!")
    print(
      f"Lower bound of overhead ({stat_overhead_min:.2f}\u00b5s) exceeds threshold ({MAX_OVERHEAD_US}\u00b5s)."
    )
    sys.exit(1)
  elif overhead > MAX_OVERHEAD_US:
    print(
      f"WARNING: Mean overhead ({overhead:.2f}\u00b5s) exceeds threshold, but is not statistically significant."
    )
    print(
      f"This might be due to extreme CI noise (Margin: \u00b1{combined_ci:.2f}\u00b5s)."
    )
  else:
    print("Performance check passed!")


if __name__ == "__main__":
  check_regression()
