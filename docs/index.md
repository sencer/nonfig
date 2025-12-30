# nonfig

![CI](https://github.com/sencer/nonfig/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/sencer/nonfig/branch/master/graph/badge.svg)](https://app.codecov.io/github/sencer/nonfig)

**Automatic Pydantic config generation from class and function signatures.**

Turn any class into a configurable, serializable, validated component—just add
`@configurable` and mark tunable parameters with `Hyper[T]`. This simplifies the
creation of reproducible machine learning experiments and configurable applications by
reducing boilerplate and enforcing type safety and validation at definition time.

## Quick Example

```python
from nonfig import configurable, DEFAULT, Leaf

@configurable
class Optimizer:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum

@configurable
class Model:
    def __init__(
        self,
        hidden_size: int = 128,
        optimizer: Optimizer = DEFAULT,
        # Use Leaf[T] to accept an instance directly (no .Config transformation)
        data_loader: Leaf[DataLoader] = DEFAULT
    ):
        self.hidden_size = hidden_size
        self.optimizer = optimizer
```

## Installation

```bash
pip install nonfig
# Optional YAML support:
pip install nonfig[yaml]
```

## Core Concepts

### Classes & Dataclasses

The `@configurable` decorator generates a `.Config` class that captures parameters:

```python
from nonfig import configurable

@configurable
class Optimizer:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum

# Direct instantiation still works
opt = Optimizer(lr=0.001)

# Or use Config for validation + serialization
config = Optimizer.Config(lr=0.001)
opt = config.make()  # Returns Optimizer instance
```

For dataclasses, apply `@configurable` after `@dataclass` for full IDE support:

```python
from dataclasses import dataclass
from nonfig import configurable

@dataclass
class Model:
    hidden_size: int = 128
    dropout: float = 0.1

Model = configurable(Model)  # Full autocomplete!
```

### Functions with Hyper

For functions, use `Hyper[T]` to mark configurable parameters:

```python
from nonfig import configurable, Hyper, Ge, Gt

@configurable
def train(
    data: list[float],              # Runtime argument (not a hyperparameter)
    *,
    epochs: Hyper[int, Ge[1]] = 100,
    lr: Hyper[float, Gt[0.0]] = 0.001,
) -> dict:
    return {"trained": True}

# Create a configured function
trainer = train.Config(epochs=50, lr=0.01).make()
result = trainer(data=[1.0, 2.0, 3.0])
```

### Constraints & Validation

The `Hyper[T]` annotation attaches validation constraints:

```python
from nonfig import configurable, Hyper, Ge, Le, Gt, MinLen, Pattern

@configurable
class Network:
    def __init__(
        self,
        learning_rate: Hyper[float, Gt[0.0]],                 # Required, > 0
        dropout: Hyper[float, Ge[0.0], Le[1.0]] = 0.5,        # 0 <= x <= 1
        name: Hyper[str, MinLen[3], Pattern[r"^[a-z]+$"]] = "net",
    ):
        ...
```

**Available constraints:** `Ge` (>=), `Gt` (>), `Le` (≤), `Lt` (<), `MinLen`, `MaxLen`, `MultipleOf`, `Pattern`.

### Inheritance & Propagation

`nonfig` supports automatic inheritance. If you inherit from a `@configurable` class, the subclass is automatically made configurable.

Furthermore, `nonfig` uses **Smart Parameter Propagation**:
1.  If your subclass accepts `**kwargs` in `__init__`, `nonfig` assumes you are passing arguments to the base class.
2.  It automatically adds all configurable parameters from the Base class to the Subclass's Config.
3.  If you explicitly define a parameter in the Subclass, it overrides the Base parameter.

```python
@configurable
class Base:
    def __init__(self, x: int = 1):
        self.x = x

# ✅ Sub accepts **kwargs, so it inherits 'x' from Base
class Sub(Base):
    def __init__(self, y: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.y = y

# Config has both x and y!
config = Sub.Config(x=10, y=20)
obj = config.make()
assert obj.x == 10
assert obj.y == 20
```

### Cross-Field Validation

Define a `__config_validate__` hook for validation across multiple fields:

```python
@configurable
class Optimizer:
    def __init__(self, name: str = "Adam", momentum: float | None = None):
        self.name = name
        self.momentum = momentum

    @staticmethod
    def __config_validate__(config: "Optimizer.Config") -> "Optimizer.Config":
        if config.name == "SGD" and config.momentum is None:
            raise ValueError("SGD requires momentum to be set")
        return config

# This raises ValueError
Optimizer.Config(name="SGD", momentum=None)
```

### Leaf Markers

By default, any `@configurable` class used as a type hint is transformed into `T | T.Config`. If you want to force a parameter to accept only the raw instance (disabling nested configuration for that field), use `Leaf[T]`:

```python
from nonfig import configurable, Leaf

@configurable
class Processor:
    def __init__(self, model: Leaf[MyModel]):
        # 'model' must be a MyModel instance, not MyModel.Config
        self.model = model
```

### Generic Classes

Generic classes preserve type parameters in their Config:

```python
@configurable
class Container[T]:
    def __init__(self, count: int = 0):
        self.count = count

# PEP 695 style generics work
assert hasattr(Container.Config, "__type_params__")

# Generic[T] style also supported
from typing import Generic, TypeVar
T = TypeVar("T")

@configurable
class OldStyle(Generic[T]):
    ...
```

## Integration with JAX, Flax & jaxtyping

`nonfig` is designed to work seamlessly with the JAX ecosystem. It preserves `jaxtyping` annotations and supports Flax `nn.Module` configurations.

### `jaxtyping` & `beartype`

You can combine `@configurable` with runtime type checkers. `nonfig` correctly unwraps these decorators to extract hyperparameters and preserves dimension metadata in the generated Config.

```python
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype
from nonfig import configurable, Hyper

@configurable
@jaxtyped(typechecker=beartype)
def scale_tensor(
    tensor: Float[Array, "batch dim"],
    scale: Hyper[float] = 1.0
) -> Float[Array, "batch dim"]:
    return tensor * scale

# Dimension metadata "batch dim" is preserved in scale_tensor.Config
# Runtime checks correctly catch invalid shapes passed to the configured function
```

**Important: Decorator Order**
Always place `@configurable` as the **outermost** decorator for functions. This ensures that the configuration logic wraps the fully-decorated function (including type checks).

```python
# ✅ Correct: configurable is outer
@configurable
@jaxtyped(typechecker=beartype)
def func(...): ...

# ❌ Incorrect: jaxtyped is outer (bypasses type checking in .make())
@jaxtyped(typechecker=beartype)
@configurable
def func(...): ...
```

### Flax Modules

`nonfig` works with Flax modules, allowing you to easily configure layers and models:

```python
import flax.linen as nn

@configurable
class MyLayer(nn.Module):
    features: Hyper[int]
    dtype: Hyper[Any] = jnp.float32

    @nn.compact
    def __call__(self, x):
        # features is available as self.features
        kernel = self.param('kernel', nn.initializers.lecun_normal(),
                          (x.shape[-1], self.features))
        return jnp.dot(x, kernel)

# Configure the module
config = MyLayer.Config(features=128)
layer = config.make()  # Returns a valid MyLayer instance
```

## CLI & Tooling

### CLI Runner

Run configurable targets with command-line overrides:

```python
from nonfig import configurable, Hyper, run_cli

@configurable
def train(*, epochs: Hyper[int] = 10, lr: Hyper[float] = 0.01) -> dict:
    return {"epochs": epochs, "lr": lr}

if __name__ == "__main__":
    result = run_cli(train)  # Parse sys.argv
    print(result)
```

```bash
python train.py epochs=100 lr=0.001 optimizer.momentum=0.9
```

### Config Loaders

Load configs from JSON, TOML, or YAML files:

```python
from nonfig import load_json, load_toml, load_yaml

# Load and instantiate
data = load_toml("config.toml")
config = Model.Config(**data)
model = config.make()
```

### Stub Generation

For perfect IDE support (autocomplete and type checking), you can generate `.pyi` stub files for your configurable components:

```bash
nonfig-stubgen src/
```

This tool scans your source code for `@configurable` decorators and generates matching type stubs.

## Serialization

Configs are Pydantic models with full serialization support:

```python
config = Model.Config(hidden_size=256)

# To dict/JSON
config.model_dump()
config.model_dump_json()

# From dict/JSON
Model.Config.model_validate_json(json_string)
```

## Performance & Best Practices

### Performance

| Pattern | Typical Latency* | Notes |
| :--- | :--- | :--- |
| **Raw Instantiation** | ~0.34µs | Baseline Python class |
| **Direct Call** | ~0.34µs | Zero overhead on decorated class |
| **`Config.make()`** | ~1.18µs | Cached factory call |
| **`Config.fast_make()`** | ~0.54µs | Bypasses Pydantic validation |
| **Reused `make()`** | ~0.47µs | Hot path: repeatedly calling make() |
| **Full lifecycle** | ~3.80µs | `Config(...).make()` |

*Measured on Python 3.13, Linux x86_64, Intel(R) Core(TM) i5-7500T CPU @ 2.70GHz, 16GB RAM.*

### High-Performance Usage

Since `Config.make()` adds a small overhead (~1µs) per call, it is best practice to:

1.  **Configure High-Level Components:** Apply `@configurable` to top-level classes (e.g., `Optimizer`, `Model`, `Pipeline`) rather than low-level utility functions called in tight loops (e.g., `activation_fn`).
2.  **Make Once, Run Many:** Instantiate your configuration *outside* your main loop.

```python
# ✅ BEST PRACTICE: Make once at the top level
fn = train.Config(epochs=100).make()

# Then call the optimized bound function repeatedly
for batch in data:
    fn(batch)
```

For hot loops where you *must* create new objects dynamically, use `fast_make()`:

```python
# Bypasses Pydantic validation
for _ in range(1_000_000):
    model = Model.Config.fast_make(hidden_size=128)
```

### Use Keyword-Only Parameters

Place hyperparameters after `*` to make them keyword-only to avoid argument conflicts:

```python
# ✅ Recommended
@configurable
def train(data, *, epochs: Hyper[int] = 10, lr: Hyper[float] = 0.01):
    ...
```

## Advanced Topics

- **Circular dependencies** in nested configs are detected at decoration time.
- **Cycle detection** in `recursive_make` prevents infinite loops at runtime.
- **Reserved names**: Descriptive errors when clashing with Pydantic or nonfig internals.
- **Thread safety**: Concurrent config creation is fully supported.

## Known Limitations

### Function Config Overrides
When using `Config.make()` on a function, the returned callable (a `BoundFunction`) has its hyperparameters "baked in". You cannot override them by passing keyword arguments during the call, as this will raise a `TypeError` (multiple values for keyword argument).

```python
# Create function with baked-in params
fn = train.Config(epochs=50).make()

# ❌ This fails:
fn(data, epochs=100)

# ✅ Instead, create a new config:
fn = train.Config(epochs=100).make()
fn(data)
```

## API Reference

### Decorators & Types

| Export | Description |
|:-------|:------------|
| `configurable` | Decorator to make classes/functions configurable |
| `Hyper[T, ...]` | Mark a parameter as a hyperparameter with constraints |
| `Leaf[T]` | Mark a parameter as a leaf (no nested config transformation) |
| `DEFAULT` | Sentinel for nested configs with default values |
| `MakeableModel` | Base class for generated Config classes |
| `BoundFunction` | Wrapper for functions with bound hyperparameters |
| `ConfigValidationError` | Exception with readable error paths |

### Constraints

| Export | Description |
|:-------|:------------|
| `Ge[n]` | Greater than or equal |
| `Gt[n]` | Greater than |
| `Le[n]` | Less than or equal |
| `Lt[n]` | Less than |
| `MinLen[n]` | Minimum length |
| `MaxLen[n]` | Maximum length |
| `MultipleOf[n]` | Must be multiple of |
| `Pattern[r"..."]` | Regex pattern match |

### Utilities

| Export | Description |
|:-------|:------------|
| `run_cli(target, args)` | Run target with CLI overrides |
| `load_json(path)` | Load dict from JSON file |
| `load_toml(path)` | Load dict from TOML file |
| `load_yaml(path)` | Load dict from YAML file (requires `pyyaml`) |
| `nonfig-stubgen <path>` | CLI tool to generate .pyi stubs for IDE support |

## Comparison

| Feature | nonfig | gin-config | hydra | tyro |
| :--- | :--- | :--- | :--- | :--- |
| **Philosophy** | Config from code | Dependency injection | YAML-first | CLI from types |
| **Error Detection** | Decoration + Runtime | Runtime only | Runtime | CLI parse time |
| **Type Checking** | Full (.pyi stubs) | None | Partial | Full |
| **Boilerplate** | Minimal | Minimal | Moderate | Minimal |
| **Serialization** | Pydantic native | Custom | YAML | YAML/JSON |

## Contributing

Contributions welcome! Please open an issue or pull request.

## License

MIT License

---

**nonfig** — Configuration should be effortless.