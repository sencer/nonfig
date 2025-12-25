# nonfig

![CI](https://github.com/sencer/nonfig/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/sencer/nonfig/branch/master/graph/badge.svg)](https://app.codecov.io/github/sencer/nonfig)

**Automatic Pydantic config generation from class and function signatures.**

Turn any class into a configurable, serializable, validated component—just add
`@configurable` and mark tunable parameters with `Hyper[T]`. This simplifies the
creation of reproducible machine learning experiments and configurable applications by
reducing boilerplate and enforcing type safety and validation at definition time.

## Before & After

**Without nonfig** — manual config classes, validation, and factory methods:

```python
from dataclasses import dataclass

# 1. Define the Config class (Boilerplate)
@dataclass
class OptimizerConfig:
    learning_rate: float = 0.01
    momentum: float = 0.9

    # 2. Write a factory method to create the object
    def make(self) -> "Optimizer":
        return Optimizer(
            learning_rate=self.learning_rate,
            momentum=self.momentum
        )

# 3. Define the actual class
class Optimizer:
    def __init__(self, learning_rate: float, momentum: float):
        self.learning_rate = learning_rate
        self.momentum = momentum

# 4. Repeat for every component...
@dataclass
class ModelConfig:
    optimizer: OptimizerConfig
    hidden_size: int = 128

    def make(self) -> "Model":
        return Model(
            optimizer=self.optimizer.make(),
            hidden_size=self.hidden_size
        )

class Model:
    def __init__(self, optimizer: Optimizer, hidden_size: int):
        self.optimizer = optimizer
        self.hidden_size = hidden_size
```

**With nonfig** — automatic config generation with validation:

```python
from nonfig import configurable, DEFAULT

@configurable
class Optimizer:
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

@configurable
class Model:
    def __init__(
        self,
        hidden_size: int = 128,
        dropout: float = 0.1,
        optimizer: Optimizer = DEFAULT,  # Nested config with default
    ):
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.optimizer = optimizer

# Create config
config = Model.Config(hidden_size=256, dropout=0.2)

# Serialize/deserialize
json_str = config.model_dump_json()
loaded = Model.Config.model_validate_json(json_str)

# Instantiate
model = config.make()
# model.optimizer is now an instance of Optimizer
print(model.optimizer.learning_rate)  # 0.01
```

Fully type-safe out of the box—your IDE will autocomplete `.Config` parameters, validate
them, and catch errors before runtime.

## Installation

```bash
pip install nonfig
# or: uv add nonfig
```

## Core Concepts

### Classes

The `@configurable` decorator generates a `.Config` class that captures parameters:

```python
from nonfig import configurable

@configurable
class Optimizer:
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

# Direct instantiation still works
opt = Optimizer(learning_rate=0.001)

# Or use Config for validation + serialization
config = Optimizer.Config(learning_rate=0.001)
opt = config.make()  # Returns Optimizer instance
```

### Dataclasses

Dataclasses work seamlessly. Due to how Python's type system handles decorator stacking,
apply `@configurable` after defining the dataclass for full IDE support:

```python
from dataclasses import dataclass
from nonfig import configurable

@dataclass
class Model:
    hidden_size: int = 128
    dropout: float = 0.1

Model = configurable(Model)

config = Model.Config(hidden_size=256)  # Full autocomplete!
model = config.make()
```

### Functions

For functions, `nonfig` identifies configurable parameters via the `Hyper[T]`
annotation, or if the default value is `DEFAULT` or a nested configuration object.
Parameters without these markers are treated as runtime-only arguments.

```python
from nonfig import configurable, Hyper, Ge, Gt

@configurable
def train(
    data: list[float],              # Runtime data (not a hyperparameter)
    epochs: Hyper[int, Ge[1]] = 100,
    lr: Hyper[float, Gt[0.0]] = 0.001,
) -> dict:
    return {"trained": True}
```

Internally, `nonfig` treats this as if it were a dataclass with a `__call__` method,
where the `Hyper[T]` parameters become fields:

```python
@configurable
@dataclass
class train:
    epochs: Hyper[int, Ge[1]] = 100
    lr: Hyper[float, Gt[0.0]] = 0.001

    def __call__(self, data: list[float]) -> dict:
        return {"trained": True}
```

### Constraints & Validation

The `Hyper[T]` annotation attaches validation constraints to parameters:

```python
from dataclasses import dataclass
from nonfig import configurable, Hyper, Ge, Le, Gt, MinLen, Pattern

@dataclass
class Network:
    learning_rate: Hyper[float, Gt[0.0]]                # Required, > 0
    dropout: Hyper[float, Ge[0.0], Le[1.0]] = 0.5       # 0 <= x <= 1
    layers: Hyper[int, Ge[1], Le[100]] = 10             # 1 <= x <= 100
    name: Hyper[str, MinLen[3], Pattern[r"^[a-z]+$"]] = "net"

Network = configurable(Network)  # For full typing support
```

**Available constraints:** `Ge` (>=), `Gt` (>), `Le` (≤), `Lt` (<), `MinLen`,
`MaxLen`, `MultipleOf`, `Pattern`.

### Nested Configurations

Use `DEFAULT` to compose configs hierarchically—nested components use their own defaults
unless overridden:

```python
# Or: Pipeline = configurable(Pipeline) after class body for full typing support
@configurable
@dataclass
class Pipeline:
    model: Model = DEFAULT      # Uses Model's defaults
    optimizer: Optimizer = DEFAULT

# Override nested values
config = Pipeline.Config(
    model=Model.Config(hidden_size=512),
    optimizer=Optimizer.Config(learning_rate=0.001),
)
pipeline = config.make()  # All nested configs are instantiated
```

After `make()`, nested fields are instances:

```python
print(pipeline.model.hidden_size)      # 512
print(pipeline.optimizer.learning_rate)  # 0.001
```

### Collections & Lists

Use standard typed collections like `list[T]` and `dict[str, T]` for configurable
objects:

```python
from dataclasses import dataclass
from nonfig import configurable, DEFAULT

# Or: Layer = configurable(Layer) after class body for full typing support
@configurable
@dataclass
class Layer:
    size: int = 10

@configurable
@dataclass
class Network:
    layers: list[Layer] = DEFAULT

config = Network.Config(
    layers=[
        Layer.Config(size=32),
        Layer.Config(size=64),
    ]
)
net = config.make()
# net.layers is now [Layer(size=32), Layer(size=64)]
```

### Nested Functions

Nest configurable functions using the `.Type` attribute and the function itself as default:

```python
@configurable
def activation(x: float, limit: Hyper[float] = 1.0) -> float:
    return min(x, limit)

# Or: Layer = configurable(Layer) after class body for full typing support
@configurable
@dataclass
class Layer:
    act_fn: activation.Type = activation  # Use the function itself as default

# Direct instantiation uses the function with its defaults
layer = Layer()
print(layer.act_fn(2.0))  # Output: 1.0

# Or via Config for validation + serialization
config = Layer.Config()
layer = config.make()
print(layer.act_fn(2.0))  # Output: 1.0

# Override nested config
config = Layer.Config(act_fn=activation.Config(limit=3.0))
layer = config.make()
print(layer.act_fn(2.0))  # Output: 2.0
```

The `.Type` attribute provides correct type inference: it's typed as `Callable[..., R]`
where `R` is the return type. This works with type checkers out of the box.

### Literal Types & Enums

Use `Literal` or `Enum` for fixed choices:

```python
from typing import Literal
from enum import Enum

class Mode(str, Enum):
    FAST = "fast"
    SLOW = "slow"

@configurable
class Processor:
    def __init__(
        self,
        mode: Literal["train", "eval"] = "train",
        priority: Mode = Mode.FAST,
    ):
        self.mode = mode
        self.priority = priority
```

## Type Checking

### Built-in Type Support

`nonfig` provides full type inference out of the box. Your IDE understands `.Config`
and `.make()` without any extra steps:

```python
@configurable
class Model:
    def __init__(self, x: int, y: str = "default") -> None: ...

config = Model.Config(x=5, y="hello")  # Parameters are typed!
instance = config.make()                # Returns Model
```

For functions, IDE autocomplete shows all parameters (not just `Hyper` ones) in
`.Config()`—a minor trade-off for typed params without stubs.

### Stub Generation for Libraries

For library authors distributing configurable components, generate `.pyi` stubs for
complete and accurate type information:

```bash
nonfig-stubgen src/
```

This provides exact `Hyper`-only signatures for functions and full support for any
decorator stacking pattern.

## Serialization

Configs are Pydantic models with full serialization support:

```python
config = Model.Config(hidden_size=256)

# To dict/JSON
config.model_dump()
config.model_dump_json()

# From dict/JSON
Model.Config(**some_dict)
Model.Config.model_validate_json(json_string)
```

## Advanced Features

### Validation & Safety

- **Constraint conflicts** detected at decoration time: `Hyper[int, Ge[10], Le[5]]` →
  error

- **Invalid regex** patterns caught immediately

- **Circular dependencies** in nested configs → error

- **Reserved names**: `model_config` is reserved by Pydantic

### Thread Safety

`@configurable` is thread-safe for concurrent config creation and usage.

## Comparison

| Feature | nonfig | gin-config | hydra | tyro |
| :--- | :--- | :--- | :--- | :--- |
| **Philosophy** | Config from code | Dependency injection | YAML-first | CLI from types |
| **Error Detection** | Decoration + Runtime | Runtime only | Runtime | CLI parse time |
| **Type Checking** | Full (.pyi stubs) | None | Partial | Full |
| **Boilerplate** | Minimal | Minimal | Moderate | Minimal |
| **Serialization** | Pydantic native | Custom | YAML | YAML/JSON |
| **Best For** | Type-safe APIs, ML experiments | Google-style DI | Complex multi-run | CLI tools |

## Examples

See the [`examples/`](examples/) directory for complete working examples including:
- **ML Pipeline**: Nested configurations for model, optimizer, and data preprocessing
- **Nested Functions**: Using `fn.Type = fn` pattern for composable function configs
- **Configurable Functions**: Using `Hyper[T]` with constraints
- **Stub Generation**: IDE autocomplete with generated `.pyi` stubs

## Performance

Typical overhead on modern hardware*:

- Config creation: ~2µs
- `make()` overhead: ~2µs
- Direct function call overhead: ~0.1µs vs raw function
- Full pattern `Config().make()()`: ~6–7µs

Run `benchmarks/` to verify on your machine. *Measured on Python 3.12, Apple M2.

## Contributing

Contributions welcome! Please open an issue or pull request.

## License

MIT License

* * *

**nonfig** — Configuration should be effortless.
