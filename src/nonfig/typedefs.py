"""Core type definitions for nonfig."""

from typing import TYPE_CHECKING, Annotated, Any, Never, Protocol, override

__all__ = [
  "DEFAULT",
  "Configurable",
  "ConfigurableFunc",
  "DefaultSentinel",
  "Hyper",
  "HyperMarker",
]


class HyperMarker:
  """Sentinel class to mark parameters as hyperparameters."""

  __slots__ = ()


class DefaultSentinel:
  """Sentinel to indicate a field should use its type's default Config."""

  __slots__ = ()

  @override
  def __repr__(self) -> str:
    return "DEFAULT"


if TYPE_CHECKING:
  # At type-checking time, DEFAULT should be assignable to any type
  # Using Never makes it a bottom type that's assignable to anything
  DEFAULT: Never
else:
  DEFAULT = DefaultSentinel()


if TYPE_CHECKING:
  # At type-checking time, Hyper[T] is just T with metadata
  # This makes it fully transparent to type checkers
  type Hyper[T, *Args] = Annotated[T, HyperMarker, *Args]
else:

  class Hyper:
    """
    Type annotation for hyperparameters.

    Usage:
        x: Hyper[int]  # Simple hyperparameter
        x: Hyper[int, Ge[0], Le[100]]  # With constraints

    For classes/dataclasses: marks a field as a hyperparameter
    For functions: separates config args from call args
    """

    __slots__ = ()

    def __class_getitem__(cls, args: Any) -> Any:
      if not isinstance(args, tuple):
        # Single type: Hyper[int]
        return Annotated[args, HyperMarker]
      # Type with constraints: Hyper[int, Ge[0], Le[100]]
      inner_type = args[0]
      constraints = args[1:]
      return Annotated[inner_type, HyperMarker, *constraints]


if TYPE_CHECKING:
  from collections.abc import Callable

  from nonfig.models import BoundFunction, MakeableModel

  class ConfigurableFunc[**P, R]:
    """
    Phantom class to enable `fn.Type` syntax in type annotations.

    This class is never instantiated at runtime. It exists solely to trick
    type checkers into treating a @configurable function as a Class, which
    allows accessing attributes like .Type and .Config in type hints.
    """

    # The .Type attribute for typing arguments: def app(m: mul.Type): ...

    type Type = Callable[..., R]

    # The .Config attribute for configuration
    # Typed as a classmethod to simulate the constructor Config(...)
    @classmethod
    def Config(
      cls, *args: P.args, **kwargs: P.kwargs
    ) -> MakeableModel[BoundFunction[R]]: ...

    # __new__ allows the "class" to be called like a function returning R
    # This matches the runtime behavior where fn(args) -> R
    def __new__(cls, *args: P.args, **kwargs: P.kwargs) -> R: ...

  # TODO(python-3.13): Restore default type param [..., R_co = T_co] once 3.12 is dropped.
  # PEP 696 (Type defaults) is only available in Python 3.13+.
  class Configurable[T_co, **P, R_co](Protocol):
    """
    Protocol for classes and functions decorated with @configurable.

    Uses ParamSpec to capture the original signature, so Config accepts
    the same parameters. This provides good typing without stub generation.

    IMPORTANT - Dataclass Limitation:
        When stacking decorators like `@configurable @dataclass`, type checkers
        evaluate `@configurable` BEFORE `@dataclass` synthesizes `__init__`.
        This means ParamSpec captures an empty signature for dataclasses.

        For full IDE autocomplete with dataclasses, use one of:
        1. Runtime application: `Model = configurable(dataclass(_Model))`
        2. Generated stubs: `nonfig-stubgen src/`
        3. Regular classes with explicit `__init__`

    For regular classes, this works perfectly - all __init__ params are typed.
    For functions, Config accepts ALL params (not just Hyper) as a trade-off.

    Example (regular class - works):
        @configurable
        class Model:
            def __init__(self, x: int, y: str = "default") -> None: ...

        config = Model.Config(x=1, y="hi")  # Correctly typed params!
        instance = config.make()             # Returns Model

    Example (dataclass workaround):
        @dataclass
        class _Model:
            x: int
            y: str = "default"

        Model = configurable(_Model)
        config = Model.Config(x=1)  # Correctly typed params!
    """

    Config: Callable[P, MakeableModel[R_co]]
    Type: type[Any]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co: ...

else:
  # Runtime placeholder - only used for type checking
  Configurable = None
  ConfigurableFunc = None
