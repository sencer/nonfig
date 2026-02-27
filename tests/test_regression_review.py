from typing import Any, cast

from annotated_types import BaseMetadata
from pydantic import ValidationError
import pytest

from nonfig import (
  Ge,
  Hyper,
  Le,
  configurable,
)
from nonfig.extraction import extract_constraints


# Global definitions to ensure they are available for get_type_hints
class CustomMetadata(BaseMetadata):
  """Mock metadata for testing preservation."""

  def __init__(self, value: str):
    self.value = value


@configurable
def func_ge_1(x: Hyper[int, Ge[10], Ge[20]] = 25):
  return x


@configurable
def func_ge_2(x: Hyper[int, Ge[20], Ge[10]] = 25):
  return x


@configurable
def func_le_1(x: Hyper[int, Le[10], Le[20]] = 5):
  return x


@configurable
def func_le_2(x: Hyper[int, Le[20], Le[10]] = 5):
  return x


def test_most_restrictive_resolution_ordering():
  """Regression test for most-restrictive-wins constraint resolution."""
  # Ge resolution: should always use Ge[20]
  c1 = func_ge_1.Config(x=20)
  fields = type(c1).model_fields

  assert "x" in fields
  assert c1.x == 20

  with pytest.raises(ValidationError, match="greater than or equal to 20"):
    func_ge_1.Config(x=15)

  c2 = func_ge_2.Config(x=20)
  assert c2.x == 20
  with pytest.raises(ValidationError, match="greater than or equal to 20"):
    func_ge_2.Config(x=15)

  # Le resolution: should always use Le[10]
  c3 = func_le_1.Config(x=10)
  assert c3.x == 10
  with pytest.raises(ValidationError, match="less than or equal to 10"):
    func_le_1.Config(x=15)

  c4 = func_le_2.Config(x=10)
  assert c4.x == 10
  with pytest.raises(ValidationError, match="less than or equal to 10"):
    func_le_2.Config(x=15)


def test_basemetadata_preservation():
  """Regression test for preservation of custom BaseMetadata."""

  class CustomMeta(BaseMetadata):
    def __init__(self, custom: str):
      self.custom = custom

  meta = CustomMeta(custom="keep-me")
  metadata = (meta,)

  _, leftovers = extract_constraints(metadata)

  # CustomMeta should be preserved in leftovers as it's NOT a handled constraint
  assert meta in leftovers
  assert cast("Any", leftovers[0]).custom == "keep-me"


def test_needs_transform_completeness():
  """Regression test for _needs_transform consistency with is_nested_type."""
  from nonfig.models import BoundFunction, DefaultSentinel, _needs_transform

  def dummy():
    pass

  bound = BoundFunction(dummy)
  sentinel = DefaultSentinel()

  assert _needs_transform(bound) is True
  assert _needs_transform(sentinel) is True
