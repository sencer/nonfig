from collections.abc import Callable
import json
import pathlib
import statistics
import sys
import time
from typing import Any

# Configuration - slightly reduced for faster collection across 10 commits
TRIALS = 10
ITERATIONS = 50_000

# We need to monkeypatch the benchmark function in bench_robust to collect results
# instead of just printing them.

results_data = {}


def time_ns(func: Callable[[], Any], iterations: int) -> float:
  start = time.perf_counter_ns()
  for _ in range(iterations):
    func()
  return time.perf_counter_ns() - start


def benchmark_collect(
  name: str, func: Callable[[], Any], trials: int = TRIALS, iterations: int = ITERATIONS
) -> tuple[float, float]:
  time_ns(func, iterations // 5)
  results_ns = []
  for _ in range(trials):
    results_ns.append(time_ns(func, iterations) / iterations)
  min_ns = min(results_ns)
  stdev_ns = statistics.stdev(results_ns) if len(results_ns) > 1 else 0.0
  results_data[name] = {"min_ns": min_ns, "stdev_ns": stdev_ns}
  return min_ns, stdev_ns


# Mock compare to also collect if needed, but benchmark() covers most
def compare_collect(
  name: str,
  baseline: Callable[[], Any],
  optimized: Callable[[], Any],
  trials: int = TRIALS,
  iterations: int = ITERATIONS,
) -> None:
  time_ns(baseline, iterations // 5)
  time_ns(optimized, iterations // 5)
  baseline_results = []
  optimized_results = []
  for _ in range(trials):
    baseline_results.append(time_ns(baseline, iterations) / iterations)
    optimized_results.append(time_ns(optimized, iterations) / iterations)

  base_min = min(baseline_results)
  opt_min = min(optimized_results)
  results_data[f"Compare: {name} (Base)"] = {"min_ns": base_min}
  results_data[f"Compare: {name} (Opt)"] = {"min_ns": opt_min}


if __name__ == "__main__":
  # Add project root to sys.path
  sys.path.insert(0, pathlib.Path.cwd())

  import benchmarks.bench_robust as br

  # Monkeypatch
  br.TRIALS = TRIALS
  br.ITERATIONS = ITERATIONS
  br.benchmark = benchmark_collect
  br.compare = compare_collect

  # Run
  try:
    br.run_all_benchmarks()
    print(json.dumps(results_data, indent=2))
  except (RuntimeError, ValueError, OSError) as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
