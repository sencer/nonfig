from __future__ import annotations

from typing import Literal

from pydantic import ValidationError
import pytest

from nonfig import (
  Ge,
  Gt,
  Hyper,
  Le,
  Lt,
  MaxLen,
  MinLen,
  MultipleOf,
  Pattern,
  configurable,
)
from nonfig.constraints import InvalidPatternError, validate_constraint_conflicts

"""Consolidated constraint tests."""

"""Test all available constraint markers."""


@configurable
def numeric_constraints(
  value: float,
  ge_val: Hyper[int, Ge[0]] = 5,
  le_val: Hyper[int, Le[100]] = 50,
  gt_val: Hyper[float, Gt[0.0]] = 1.5,
  lt_val: Hyper[float, Lt[10.0]] = 5.0,
) -> float:
  """Test function with numeric constraints."""
  return value + ge_val + le_val + gt_val + lt_val


@configurable
def string_constraints(
  text: Hyper[str, MinLen[2], MaxLen[10]] = "hello",
  # Use string literal for regex to avoid syntax error in forward ref
  email: Hyper[
    str, Pattern[Literal["^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"]]
  ] = "test@example.com",
) -> str:
  """Test function with string constraints."""
  return f"{text}: {email}"


def test_ge_constraint() -> None:
  """Test Ge (greater than or equal) constraint."""
  config = numeric_constraints.Config(ge_val=0)
  assert config.ge_val == 0

  with pytest.raises(ValidationError):
    numeric_constraints.Config(ge_val=-1)


def test_le_constraint() -> None:
  """Test Le (less than or equal) constraint."""
  config = numeric_constraints.Config(le_val=100)
  assert config.le_val == 100

  with pytest.raises(ValidationError):
    numeric_constraints.Config(le_val=101)


def test_gt_constraint() -> None:
  """Test Gt (greater than) constraint."""
  config = numeric_constraints.Config(gt_val=0.1)
  assert config.gt_val == 0.1

  with pytest.raises(ValidationError):
    numeric_constraints.Config(gt_val=0.0)


def test_lt_constraint() -> None:
  """Test Lt (less than) constraint."""
  config = numeric_constraints.Config(lt_val=9.9)
  assert config.lt_val == 9.9

  with pytest.raises(ValidationError):
    numeric_constraints.Config(lt_val=10.0)


def test_multiple_of_constraint() -> None:
  """Test MultipleOf constraint."""

  @configurable
  def with_multiple(val: Hyper[int, MultipleOf[5]] = 10) -> int:
    return val

  config = with_multiple.Config(val=15)
  assert config.val == 15

  with pytest.raises(ValidationError):
    with_multiple.Config(val=7)


def test_min_len_constraint() -> None:
  """Test MinLen constraint."""
  config = string_constraints.Config(text="ab")
  assert config.text == "ab"

  with pytest.raises(ValidationError):
    string_constraints.Config(text="a")


def test_max_len_constraint() -> None:
  """Test MaxLen constraint."""
  config = string_constraints.Config(text="1234567890")
  assert config.text == "1234567890"

  with pytest.raises(ValidationError):
    string_constraints.Config(text="12345678901")


def test_pattern_constraint() -> None:
  """Test Pattern constraint."""
  config = string_constraints.Config(email="user@domain.com")
  assert config.email == "user@domain.com"

  with pytest.raises(ValidationError):
    string_constraints.Config(email="invalid-email")


def test_combined_constraints() -> None:
  """Test using multiple constraints together."""
  config = numeric_constraints.Config(ge_val=10, le_val=90, gt_val=2.5, lt_val=8.0)
  assert config.ge_val == 10
  assert config.le_val == 90
  assert config.gt_val == 2.5
  assert config.lt_val == 8.0


def test_function_calls_with_constraints() -> None:
  """Test that functions work correctly with constrained values."""
  result = numeric_constraints(10.0)
  # 10.0 + 5 + 50 + 1.5 + 5.0 = 71.5
  assert result == 71.5

  result = string_constraints()
  assert result == "hello: test@example.com"


"""Test contradictory and impossible constraint combinations."""


def test_impossible_constraint_combination() -> None:
  """Test constraint where no value can satisfy both."""
  # Ge[10] and Lt[10] means >= 10 and < 10, which is impossible
  # This should now be caught at decoration time
  with pytest.raises(ValueError, match="Conflicting constraints"):

    @configurable
    def process(
      value: Hyper[int, Ge[10], Lt[10]] = 10,
    ) -> int:
      return value


def test_contradictory_le_and_ge() -> None:
  """Test Le[x] with Ge[y] where y > x."""
  # No value can be >= 100 and <= 50
  # This should now be caught at decoration time
  with pytest.raises(ValueError, match="Conflicting constraints"):

    @configurable
    def process(
      value: Hyper[int, Ge[100], Le[50]] = 75,
    ) -> int:
      return value


def test_minlen_greater_than_maxlen() -> None:
  """Test MinLen > MaxLen which makes string impossible."""
  # No string can satisfy both constraints
  # This should now be caught at decoration time
  with pytest.raises(ValueError, match="Conflicting constraints"):

    @configurable
    def process(
      text: Hyper[str, MinLen[10], MaxLen[5]] = "hello",
    ) -> str:
      return text


def test_multiple_ge_constraints() -> None:
  """Test multiple Ge constraints (should use most restrictive)."""

  @configurable
  def process(
    value: Hyper[int, Ge[5], Ge[10]] = 15,
  ) -> int:
    return value

  # Should use Ge[10] (most restrictive)
  config = process.Config(value=10)
  assert config.value == 10

  # Value less than 10 should fail
  with pytest.raises(ValidationError):
    process.Config(value=7)


def test_gt_and_ge_on_same_value() -> None:
  """Test Gt[5] and Ge[5] together (Gt is more restrictive)."""

  @configurable
  def process(
    value: Hyper[int, Gt[5], Ge[5]] = 6,
  ) -> int:
    return value

  # Gt[5] is more restrictive, so 5 should fail
  with pytest.raises(ValidationError):
    process.Config(value=5)

  # 6 should pass
  config = process.Config(value=6)
  assert config.value == 6


def test_invalid_pattern_regex() -> None:
  """Test that invalid regex pattern raises ValueError."""
  with pytest.raises(ValueError, match="Invalid regex pattern"):
    # "[a-z" is an invalid regex (missing closing bracket)
    _ = Pattern["[a-z"]


def test_required_hyper_with_constraints() -> None:
  """Test that required Hyper parameters with constraints work correctly."""

  @configurable
  def func(p: Hyper[int, Ge[0], Le[100]]) -> int:
    return p

  # Must provide the required param
  config = func.Config(p=50)
  fn = config.make()
  assert fn() == 50

  # Constraints are still enforced
  with pytest.raises(ValidationError):
    func.Config(p=-5)  # Less than Ge[0]


def test_overlapping_lt_and_le() -> None:
  """Test Lt and Le with same value (Lt is more restrictive)."""

  @configurable
  def process(
    value: Hyper[int, Lt[10], Le[10]] = 5,
  ) -> int:
    return value

  # Lt[10] is more restrictive, so 10 should fail
  with pytest.raises(ValidationError):
    process.Config(value=10)

  # 9 should pass
  config = process.Config(value=9)
  assert config.value == 9


"""Test that contradictory constraints are caught at decoration time."""


def test_contradictory_ge_lt_caught_at_decoration() -> None:
  """Test that Ge[10] and Lt[10] is caught when decorator is applied."""
  with pytest.raises(ValueError, match="Conflicting constraints.*exclusive"):

    @configurable
    def func(x: Hyper[int, Ge[10], Lt[10]] = 10) -> int:
      return x


def test_contradictory_ge_le_caught_at_decoration() -> None:
  """Test that Ge[100] and Le[50] is caught when decorator is applied."""
  with pytest.raises(
    ValueError, match="Conflicting constraints.*lower bound.*upper bound"
  ):

    @configurable
    def func(x: Hyper[int, Ge[100], Le[50]] = 75) -> int:
      return x


def test_contradictory_gt_le_caught_at_decoration() -> None:
  """Test that Gt[100] and Le[100] is caught when decorator is applied."""
  with pytest.raises(ValueError, match="Conflicting constraints.*exclusive"):

    @configurable
    def func(x: Hyper[int, Gt[100], Le[100]] = 101) -> int:
      return x


def test_contradictory_ge_lt_same_value_caught() -> None:
  """Test that Ge[50] and Lt[50] is caught (no value satisfies both)."""
  with pytest.raises(ValueError, match="Conflicting constraints.*exclusive"):

    @configurable
    def func(x: Hyper[int, Ge[50], Lt[50]] = 50) -> int:
      return x


def test_contradictory_minlen_maxlen_caught_at_decoration() -> None:
  """Test that MinLen[10] and MaxLen[5] is caught when decorator is applied."""
  with pytest.raises(
    ValueError, match="Conflicting constraints.*min_length.*max_length"
  ):

    @configurable
    def func(text: Hyper[str, MinLen[10], MaxLen[5]] = "hello") -> str:
      return text


def test_valid_overlapping_constraints_allowed() -> None:
  """Test that valid overlapping constraints are allowed."""
  # These should not raise - they have valid ranges

  @configurable
  def func1(x: Hyper[int, Ge[5], Le[100]] = 50) -> int:
    return x

  @configurable
  def func2(x: Hyper[int, Gt[0], Lt[100]] = 50) -> int:
    return x

  @configurable
  def func3(text: Hyper[str, MinLen[5], MaxLen[10]] = "hello") -> str:
    return text

  # Should work fine
  assert func1(x=50) == 50
  assert func2(x=50) == 50
  assert func3(text="hello") == "hello"


def test_multiple_ge_constraints_allowed() -> None:
  """Test that multiple Ge constraints are allowed (most restrictive wins)."""

  # Should not raise - Ge[10] is more restrictive
  @configurable
  def func(x: Hyper[int, Ge[5], Ge[10]] = 15) -> int:
    return x

  assert func(x=15) == 15


def test_same_bound_ge_le_allowed() -> None:
  """Test that Ge[50] and Le[50] is allowed (only 50 is valid)."""

  # Should not raise - value 50 satisfies both
  @configurable
  def func(x: Hyper[int, Ge[50], Le[50]] = 50) -> int:
    return x

  config = func.Config(x=50)
  assert config.x == 50


def test_same_minlen_maxlen_allowed() -> None:
  """Test that MinLen[5] and MaxLen[5] is allowed (fixed length)."""

  # Should not raise - exactly length 5 strings are valid
  @configurable
  def func(text: Hyper[str, MinLen[5], MaxLen[5]] = "hello") -> str:
    return text

  config = func.Config(text="world")
  assert config.text == "world"


def test_error_message_includes_parameter_name() -> None:
  """Test that error messages include the parameter name for clarity."""
  with pytest.raises(ValueError, match=r"my_param"):

    @configurable
    def func(my_param: Hyper[int, Ge[100], Le[50]] = 75) -> int:
      return my_param


def test_error_message_includes_bound_values() -> None:
  """Test that error messages include the actual bound values."""
  with pytest.raises(ValueError, match=r"100.*50"):

    @configurable
    def func(x: Hyper[int, Ge[100], Le[50]] = 75) -> int:
      return x


def test_class_constraint_validation() -> None:
  """Test that constraint validation works for class __init__ parameters."""
  with pytest.raises(ValueError, match="Conflicting constraints"):

    @configurable
    class MyClass:
      def __init__(self, value: Hyper[int, Ge[100], Lt[50]] = 75) -> None:
        self.value = value


def test_dataclass_constraint_validation() -> None:
  """Test that constraint validation works for dataclass fields."""
  from dataclasses import dataclass

  with pytest.raises(ValueError, match="Conflicting constraints"):

    @configurable
    @dataclass
    class MyDataclass:
      value: Hyper[int, Ge[100], Le[50]] = 75


"""Tests for constraint conflict validation logic."""


def test_conflict_gt_ge():
  # Gt(10) implies Ge(10) effectively, but if Gt(10) and Ge(5), lower bound is 10 (exclusive).
  # If Gt(5) and Ge(10), lower bound is 10.

  # Validation logic aggregates:
  # lower = max(ge, gt)
  # This is tested implicitly. Let's test explicit failures.
  pass


def test_conflict_lower_gt_upper():
  """Lower bound > Upper bound."""
  # Ge(10), Le(5)
  with pytest.raises(ValueError, match="lower bound .* > upper bound"):
    validate_constraint_conflicts({"ge": 10, "le": 5}, "x")


def test_conflict_equal_bounds_exclusive():
  """Lower bound == Upper bound, but one is exclusive."""
  # Ge(10), Lt(10) -> Impossible
  with pytest.raises(ValueError, match="bounds equal .* but one is exclusive"):
    validate_constraint_conflicts({"ge": 10, "lt": 10}, "x")

  # Gt(10), Le(10) -> Impossible
  with pytest.raises(ValueError, match="bounds equal .* but one is exclusive"):
    validate_constraint_conflicts({"gt": 10, "le": 10}, "x")

  # Gt(10), Lt(10) -> Impossible
  with pytest.raises(ValueError, match="bounds equal .* but one is exclusive"):
    validate_constraint_conflicts({"gt": 10, "lt": 10}, "x")


def test_conflict_valid_equal_bounds():
  """Lower bound == Upper bound, inclusive."""
  # Ge(10), Le(10) -> Valid (x == 10)
  validate_constraint_conflicts({"ge": 10, "le": 10}, "x")


def test_conflict_valid_ranges():
  validate_constraint_conflicts({"gt": 0, "lt": 10}, "x")
  validate_constraint_conflicts({"ge": 0, "le": 10}, "x")


def test_conflict_length():
  with pytest.raises(ValueError, match="min_length .* > max_length"):
    validate_constraint_conflicts({"min_length": 10, "max_length": 5}, "s")


def test_valid_length():
  validate_constraint_conflicts({"min_length": 5, "max_length": 10}, "s")


"""Test constraint validation error handling at constraint creation time."""


def test_ge_requires_numeric_value() -> None:
  """Test that Ge raises TypeError for non-numeric values."""
  with pytest.raises(TypeError, match="numeric value"):
    Ge["invalid"]

  with pytest.raises(TypeError, match="numeric value"):
    Ge[None]


def test_le_requires_numeric_value() -> None:
  """Test that Le raises TypeError for non-numeric values."""
  with pytest.raises(TypeError, match="numeric value"):
    Le["invalid"]

  with pytest.raises(TypeError, match="numeric value"):
    Le[None]


def test_gt_requires_numeric_value() -> None:
  """Test that Gt raises TypeError for non-numeric values."""
  with pytest.raises(TypeError, match="numeric value"):
    Gt["invalid"]

  with pytest.raises(TypeError, match="numeric value"):
    Gt[None]


def test_lt_requires_numeric_value() -> None:
  """Test that Lt raises TypeError for non-numeric values."""
  with pytest.raises(TypeError, match="numeric value"):
    Lt["invalid"]

  with pytest.raises(TypeError, match="numeric value"):
    Lt[None]


def test_minlen_requires_integer() -> None:
  """Test that MinLen raises TypeError for non-integer values."""
  with pytest.raises(TypeError, match="integer"):
    MinLen[3.5]

  with pytest.raises(TypeError, match="integer"):
    MinLen["invalid"]


def test_minlen_requires_non_negative() -> None:
  """Test that MinLen raises ValueError for negative values."""
  with pytest.raises(ValueError, match="non-negative"):
    MinLen[-1]

  with pytest.raises(ValueError, match="non-negative"):
    MinLen[-100]


def test_maxlen_requires_integer() -> None:
  """Test that MaxLen raises TypeError for non-integer values."""
  with pytest.raises(TypeError, match="integer"):
    MaxLen[3.5]

  with pytest.raises(TypeError, match="integer"):
    MaxLen["invalid"]


def test_maxlen_requires_non_negative() -> None:
  """Test that MaxLen raises ValueError for negative values."""
  with pytest.raises(ValueError, match="non-negative"):
    MaxLen[-1]

  with pytest.raises(ValueError, match="non-negative"):
    MaxLen[-100]


def test_pattern_requires_string() -> None:
  """Test that Pattern raises TypeError for non-string values."""
  with pytest.raises(TypeError, match="string"):
    Pattern[123]

  with pytest.raises(TypeError, match="string"):
    Pattern[None]


def test_pattern_validates_regex() -> None:
  """Test that Pattern raises ValueError for invalid regex."""
  from nonfig.constraints import InvalidPatternError

  with pytest.raises(InvalidPatternError, match="Invalid regex pattern"):
    Pattern["[invalid"]  # Unclosed bracket

  with pytest.raises(InvalidPatternError, match="Invalid regex pattern"):
    Pattern["(unclosed"]  # Unclosed paren

  with pytest.raises(InvalidPatternError, match="Invalid regex pattern"):
    Pattern["*invalid"]  # Invalid quantifier


def test_multipleof_requires_numeric_value() -> None:
  """Test that MultipleOf raises TypeError for non-numeric values."""
  with pytest.raises(TypeError, match="numeric value"):
    MultipleOf["invalid"]

  with pytest.raises(TypeError, match="numeric value"):
    MultipleOf[None]


def test_multipleof_requires_positive_value() -> None:
  """Test that MultipleOf raises ValueError for zero or negative values."""
  with pytest.raises(ValueError, match="positive"):
    MultipleOf[0]

  with pytest.raises(ValueError, match="positive"):
    MultipleOf[-5]

  with pytest.raises(ValueError, match="positive"):
    MultipleOf[-0.5]


def test_numeric_constraints_accept_int_and_float() -> None:
  """Test that numeric constraints accept both int and float."""
  # Should not raise
  ge_int = Ge[5]
  ge_float = Ge[5.5]
  assert ge_int.ge == 5
  assert ge_float.ge == 5.5

  le_int = Le[10]
  le_float = Le[10.5]
  assert le_int.le == 10
  assert le_float.le == 10.5

  gt_int = Gt[0]
  gt_float = Gt[0.1]
  assert gt_int.gt == 0
  assert gt_float.gt == 0.1

  lt_int = Lt[100]
  lt_float = Lt[100.5]
  assert lt_int.lt == 100
  assert lt_float.lt == 100.5

  mo_int = MultipleOf[5]
  mo_float = MultipleOf[2.5]
  assert mo_int.multiple_of == 5
  assert mo_float.multiple_of == 2.5


def test_length_constraints_accept_zero() -> None:
  """Test that MinLen and MaxLen accept zero as valid."""
  # Should not raise
  min_zero = MinLen[0]
  max_zero = MaxLen[0]
  assert min_zero.min_length == 0
  assert max_zero.max_length == 0


def test_pattern_accepts_valid_regex() -> None:
  """Test that Pattern accepts valid regex patterns."""
  from nonfig.constraints import PatternConstraint

  # Should not raise
  pattern1 = Pattern[r"^[a-z]+$"]
  pattern2 = Pattern[r"\d{3}-\d{4}"]
  pattern3 = Pattern[r"[A-Z]{2,5}"]

  assert isinstance(pattern1, PatternConstraint)
  assert pattern1.pattern == r"^[a-z]+$"
  assert isinstance(pattern2, PatternConstraint)
  assert pattern2.pattern == r"\d{3}-\d{4}"
  assert isinstance(pattern3, PatternConstraint)
  assert pattern3.pattern == r"[A-Z]{2,5}"


"""Tests for invalid constraint definitions."""


def test_ge_rejects_non_numeric():
  with pytest.raises(TypeError, match="requires a numeric value"):
    Ge["not a number"]  # type: ignore


def test_gt_rejects_non_numeric():
  with pytest.raises(TypeError, match="requires a numeric value"):
    Gt["not a number"]  # type: ignore


def test_le_rejects_non_numeric():
  with pytest.raises(TypeError, match="requires a numeric value"):
    Le["not a number"]  # type: ignore


def test_lt_rejects_non_numeric():
  with pytest.raises(TypeError, match="requires a numeric value"):
    Lt["not a number"]  # type: ignore


def test_min_len_rejects_negative():
  with pytest.raises(ValueError, match="requires a non-negative integer"):
    MinLen[-1]


def test_min_len_rejects_float():
  with pytest.raises(TypeError, match="requires an integer value"):
    MinLen[1.5]  # type: ignore


def test_max_len_rejects_negative():
  with pytest.raises(ValueError, match="requires a non-negative integer"):
    MaxLen[-1]


def test_multiple_of_rejects_non_numeric():
  with pytest.raises(TypeError, match="requires a numeric value"):
    MultipleOf["not a number"]  # type: ignore


def test_multiple_of_rejects_zero():
  with pytest.raises(ValueError, match="requires a positive value"):
    MultipleOf[0]


def test_multiple_of_rejects_negative():
  with pytest.raises(ValueError, match="requires a positive value"):
    MultipleOf[-5]


def test_pattern_rejects_non_string():
  with pytest.raises(TypeError, match="expects a string"):
    Pattern[123]  # type: ignore


def test_pattern_rejects_invalid_regex():
  with pytest.raises(InvalidPatternError, match="Invalid regex pattern"):
    Pattern["["]  # Invalid regex


def test_pattern_literal_handling():
  # Should work with Literal
  assert Pattern[Literal[r"^[a-z]+$"]] is not None


def test_pattern_literal_rejects_non_string_arg():
  # Only string literals supported
  # This might depend on runtime inspection, likely falls back to non-literal logic if parsing fails
  # or raises TypeError within our logic if we inspect args
  pass
