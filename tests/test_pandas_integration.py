"""Test edge cases specific to pandas integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nonfig import Ge, Hyper, configurable


def test_nan_handling_in_series() -> None:
  """Test handling of NaN values in pandas Series."""

  @configurable
  def process(
    data: pd.Series,
    skip_na: Hyper[bool] = True,
  ) -> float:
    if skip_na:
      return float(data.dropna().sum())
    return float(data.sum())

  config = process.Config(skip_na=True)
  fn = config.make()

  # Series with NaN
  series_with_nan = pd.Series([1.0, 2.0, np.nan, 4.0])
  result = fn(data=series_with_nan)
  assert result == 7.0  # 1 + 2 + 4


def test_empty_dataframe() -> None:
  """Test handling of empty DataFrame."""

  @configurable
  def process(
    data: pd.DataFrame,
    default_value: Hyper[float] = 0.0,
  ) -> float:
    if data.empty:
      return default_value
    return float(data.sum().sum())

  config = process.Config(default_value=42.0)
  fn = config.make()

  empty_df = pd.DataFrame()
  result = fn(data=empty_df)
  assert result == 42.0


def test_series_with_different_dtypes() -> None:
  """Test Series with various dtypes."""

  @configurable
  def process(
    data: pd.Series,
    multiplier: Hyper[float] = 2.0,
  ) -> pd.Series:
    return data * multiplier

  config = process.Config(multiplier=3.0)
  fn = config.make()

  # Test with int64
  int_series = pd.Series([1, 2, 3], dtype="int64")
  result = fn(data=int_series)
  # Note: multiplying int by float may change dtype
  assert list(result) == [3, 6, 9]

  # Test with float64
  float_series = pd.Series([1.5, 2.5], dtype="float64")
  result2 = fn(data=float_series)
  expected2 = pd.Series([4.5, 7.5], dtype="float64")
  assert result2.equals(expected2)


def test_series_index_preservation() -> None:
  """Test that custom Series index is preserved."""

  @configurable
  def process(
    data: pd.Series,
    offset: Hyper[float] = 10.0,
  ) -> pd.Series:
    return data + offset

  config = process.Config(offset=5.0)
  fn = config.make()

  # Series with custom index
  custom_series = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
  result = fn(data=custom_series)

  assert list(result.index) == ["a", "b", "c"]
  assert result["a"] == 6.0


def test_dataframe_column_operations() -> None:
  """Test operations on DataFrame columns."""

  @configurable
  def process(
    data: pd.DataFrame,
    column: str,
    threshold: Hyper[float, Ge[0.0]] = 0.5,
  ) -> pd.DataFrame:
    return data[data[column] > threshold]

  config = process.Config(threshold=2.0)
  fn = config.make()

  df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
  result = fn(data=df, column="a")

  assert len(result) == 2
  assert list(result["a"]) == [3, 4]
