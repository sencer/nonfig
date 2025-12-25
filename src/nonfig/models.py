"""Base model classes for nonfig."""

from __future__ import annotations

from collections.abc import Callable
from types import UnionType
from typing import Any, get_args, get_origin, override

from pydantic import BaseModel, ConfigDict, PrivateAttr

__all__ = [
  "BoundFunction",
  "MakeableModel",
  "is_makeable_model",
]


class MakeableModel[R](BaseModel):
  """
  Base class for generated Config classes.

  Provides:
  - Frozen (immutable) configuration
  - make() method to instantiate the target
  - Instance caching (in base class)
  - Arbitrary type support
  """

  model_config = ConfigDict(
    frozen=True,
    arbitrary_types_allowed=True,
  )

  _instance: R | None = PrivateAttr(default=None)

  def make(self) -> R:
    """Create an instance of the target from this config.

    Handles caching automatically. Subclasses should override _make_impl().
    """
    if self._instance is not None:
      return self._instance
    instance = self._make_impl()
    self._instance = instance
    return instance

  @override
  def __str__(self) -> str:
    return self.__repr__()

  def _make_impl(self) -> R:
    """Override this method to implement instance creation."""
    raise NotImplementedError("Subclasses must implement _make_impl()")


class BoundFunction[R]:
  """
  A callable wrapper that exposes hyperparameters as attributes.

  When a @configurable function's Config.make() is called, it returns
  a BoundFunction instead of a plain function. This allows accessing
  the bound hyperparameters:

      @configurable
      def process(data: pd.Series, window: Hyper[int] = 10) -> pd.Series:
          return data.rolling(window).mean()

      config = process.Config(window=20)
      fn = config.make()
      result = fn(data)      # Call it like a function
      print(fn.window)       # Access the bound hyperparameter (20)
  """

  __slots__ = ("_doc", "_func", "_hyper_kwargs", "_name")

  def __init__(
    self,
    func: Callable[..., R],
    hyper_kwargs: dict[str, Any],
  ) -> None:
    self._func = func
    self._hyper_kwargs = hyper_kwargs
    self._name = func.__name__
    self._doc = func.__doc__

  @property
  def __name__(self) -> str:
    return self._name

  @property
  @override
  def __doc__(self) -> str | None:  # pyright: ignore[reportIncompatibleVariableOverride]
    return self._doc

  @property
  def __wrapped__(self) -> Callable[..., R]:
    return self._func

  def __call__(self, *args: Any, **kwargs: Any) -> R:
    """Call the function with bound hyperparameters."""
    return self._func(*args, **self._hyper_kwargs, **kwargs)

  def __getattr__(self, name: str) -> Any:
    """Access bound hyperparameters as attributes."""
    if name.startswith("_"):
      raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    try:
      return self._hyper_kwargs[name]
    except KeyError:
      raise AttributeError(
        f"'{type(self).__name__}' has no attribute '{name}'"
      ) from None

  @override
  def __setattr__(self, name: str, value: Any) -> None:
    """Prevent modification of bound hyperparameters."""
    # Allow setting slots (internal use only)
    if name in self.__slots__:
      object.__setattr__(self, name, value)
      return

    # Check if it's a known hyperparameter
    if hasattr(self, "_hyper_kwargs") and name in self._hyper_kwargs:
      raise AttributeError(f"Hyperparameter '{name}' is read-only")

    # Otherwise it's an unknown attribute
    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

  @override
  def __repr__(self) -> str:
    params = ", ".join(f"{k}={v!r}" for k, v in self._hyper_kwargs.items())
    return f"{self._name}({params})"


def is_makeable_model(type_ann: Any) -> bool:
  """Check if a type annotation represents a MakeableModel."""
  if type_ann is None:
    return False

  # Direct class check
  if isinstance(type_ann, type) and issubclass(type_ann, MakeableModel):
    return True

  # Check Union types (T | T.Config)
  origin = get_origin(type_ann)
  if origin is UnionType:
    args = get_args(type_ann)
    return any(is_makeable_model(arg) for arg in args)

  return False
