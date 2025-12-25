"""Test performance and scale scenarios."""

from __future__ import annotations

import sys
from typing import ClassVar

from nonfig import Ge, Hyper, Le, MakeableModel, configurable


def test_many_hyperparameters() -> None:
  """Test function with many Hyper parameters (stress test)."""

  @configurable
  def process(
    p1: Hyper[int] = 1,
    p2: Hyper[int] = 2,
    p3: Hyper[int] = 3,
    p4: Hyper[int] = 4,
    p5: Hyper[int] = 5,
    p6: Hyper[int] = 6,
    p7: Hyper[int] = 7,
    p8: Hyper[int] = 8,
    p9: Hyper[int] = 9,
    p10: Hyper[int] = 10,
    p11: Hyper[int] = 11,
    p12: Hyper[int] = 12,
    p13: Hyper[int] = 13,
    p14: Hyper[int] = 14,
    p15: Hyper[int] = 15,
  ) -> int:
    return (
      p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p11 + p12 + p13 + p14 + p15
    )

  # Should handle many parameters
  config = process.Config(
    p1=10,
    p2=20,
    p3=30,
    p4=40,
    p5=50,
    p6=60,
    p7=70,
    p8=80,
    p9=90,
    p10=100,
    p11=110,
    p12=120,
    p13=130,
    p14=140,
    p15=150,
  )

  fn = config.make()
  result = fn()
  expected = sum(range(10, 160, 10))
  assert result == expected


def test_deeply_nested_config_chain() -> None:
  """Test deeply nested configuration chain (4+ levels)."""

  @configurable
  def level1(value: Hyper[int] = 1) -> int:
    return value

  @configurable
  def level2(multiplier: Hyper[int] = 2) -> int:
    config1 = level1.Config(value=10)
    fn1 = config1.make()
    return fn1() * multiplier

  @configurable
  def level3(addition: Hyper[int] = 3) -> int:
    config2 = level2.Config(multiplier=5)
    fn2 = config2.make()
    return fn2() + addition

  @configurable
  def level4(power: Hyper[int] = 2) -> int:
    config3 = level3.Config(addition=15)
    fn3 = config3.make()
    return fn3() ** power

  # Test 4-level deep nesting
  config = level4.Config(power=2)
  fn = config.make()
  result = fn()
  # level1: 10, level2: 10*5=50, level3: 50+15=65, level4: 65^2=4225
  assert result == 4225


def test_large_constraint_values() -> None:
  """Test with sys.maxsize and extreme values."""

  @configurable
  def process(
    value: Hyper[int, Ge[0], Le[sys.maxsize]] = 1000,
  ) -> int:
    return value

  # Test with large values
  large_value = sys.maxsize // 2
  config = process.Config(value=large_value)
  assert config.value == large_value

  fn = config.make()
  result = fn()
  assert result == large_value

  # Test with max value
  config2 = process.Config(value=sys.maxsize)
  assert config2.value == sys.maxsize


def test_config_serialization_performance() -> None:
  """Test serialization with moderately large data structures."""

  @configurable
  def process(
    data: Hyper[list[int]] = None,
    mapping: Hyper[dict[str, int]] = None,
  ) -> int:
    if mapping is None:
      mapping = {f"k{i}": i for i in range(50)}
    if data is None:
      data = list(range(100))
    return len(data) + len(mapping)

  # Create config with larger data
  large_data = list(range(1000))
  large_mapping = {f"key{i}": i for i in range(500)}

  config = process.Config(data=large_data, mapping=large_mapping)

  # Serialization should work
  data_dict = config.model_dump()
  assert len(data_dict["data"]) == 1000
  assert len(data_dict["mapping"]) == 500

  # Deserialization should work
  config2 = process.Config(**data_dict)
  assert config2.data == large_data
  assert config2.mapping == large_mapping


def test_decorator_overhead() -> None:
  """Test that decorator has minimal performance impact."""

  # Regular function
  def regular_function(value: int = 10) -> int:
    return value * 2

  # Decorated function
  @configurable
  def decorated_function(value: Hyper[int] = 10) -> int:
    return value * 2

  # Both should produce same results
  assert regular_function(5) == 10
  config = decorated_function.Config(value=5)
  fn = config.make()
  assert fn() == 10

  # Test that decorated function can still be called normally
  assert decorated_function(5) == 10

  # Test with class
  class RegularClass:
    def __init__(self, value: int = 10) -> None:
      self.value = value

  @configurable
  class DecoratedClass:
    Config: ClassVar[type[MakeableModel[object]]]

    def __init__(self, value: Hyper[int] = 10) -> None:
      self.value = value

  # Both should work similarly
  obj1 = RegularClass(5)
  assert obj1.value == 5

  obj2 = DecoratedClass(5)
  assert obj2.value == 5

  # Decorated class has Config
  config2 = DecoratedClass.Config(value=7)
  obj3 = config2.make()  # Now returns instance directly
  assert isinstance(obj3, DecoratedClass)
  assert obj3.value == 7
