"""nonfig - Automatic Pydantic config generation from function signatures."""

__version__ = "0.1.0"

from nonfig.constraints import Ge, Gt, Le, Lt, MaxLen, MinLen, MultipleOf, Pattern
from nonfig.generation import configurable
from nonfig.models import BoundFunction, MakeableModel
from nonfig.typedefs import DEFAULT, Hyper

# Note: Configurable is a type-only Protocol for static analysis.
# Import it from nonfig.typedefs in TYPE_CHECKING blocks:
#   if TYPE_CHECKING:
#       from nonfig.typedefs import Configurable

__all__ = [
  "DEFAULT",
  "BoundFunction",
  "Ge",
  "Gt",
  "Hyper",
  "Le",
  "Lt",
  "MakeableModel",
  "MaxLen",
  "MinLen",
  "MultipleOf",
  "Pattern",
  "__version__",
  "configurable",
]
