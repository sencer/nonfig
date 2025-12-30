"""nonfig - Automatic Pydantic config generation from function signatures."""

__version__ = "0.1.1"

from nonfig.cli.runner import run_cli
from nonfig.constraints import Ge, Gt, Le, Lt, MaxLen, MinLen, MultipleOf, Pattern
from nonfig.generation import configurable, wrap_external
from nonfig.loaders import load_json, load_toml, load_yaml
from nonfig.models import BoundFunction, ConfigValidationError, MakeableModel
from nonfig.typedefs import DEFAULT, Hyper, Leaf

# Note: Configurable is a type-only Protocol for static analysis.
# Import it from nonfig.typedefs in TYPE_CHECKING blocks:
#   if TYPE_CHECKING:
#       from nonfig.typedefs import Configurable

__all__ = [
  "DEFAULT",
  "BoundFunction",
  "ConfigValidationError",
  "Ge",
  "Gt",
  "Hyper",
  "Le",
  "Leaf",
  "Lt",
  "MakeableModel",
  "MaxLen",
  "MinLen",
  "MultipleOf",
  "Pattern",
  "__version__",
  "configurable",
  "load_json",
  "load_toml",
  "load_yaml",
  "run_cli",
  "wrap_external",
]
