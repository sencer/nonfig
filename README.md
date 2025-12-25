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

Use `DEFAULT` to compose configs hierarchically:

```python
@configurable
@dataclass
class Pipeline:
    model: Model = DEFAULT      # Uses Model's defaults
    optimizer: Optimizer = DEFAULT

config = Pipeline.Config(
    model=Model.Config(hidden_size=512),
)
pipeline = config.make()
```

## Performance & Optimization

`nonfig` is designed for high-performance applications like machine learning training
loops.

### Core Performance

| Pattern | Typical Latency* | Notes |
| :--- | :--- | :--- |
| **Raw Instantiation** | ~0.3µs | Baseline Python class creation |
| **Direct Call** | ~0.3µs | Calling decorated class/function (Zero Overhead) |
| **Reused `Config.make()`** | ~0.6µs | Metadata-cached factory call |
| **`Config.fast_make()`** | ~0.5µs | Bypasses Pydantic for maximum speed |
| **Full lifecycle** | ~3.3µs | `Config(...).make()` (includes validation) |

*\* Measured on Python 3.13, Linux x86_64.*

### High-Performance Usage (`fast_make`)

When you need to create instances in a hot loop and you already trust your parameters,
use the static `.fast_make()` method. It bypasses Pydantic's internal validation and
coercion logic for maximum speed:

```python
# Execution (Fast path, validation-free)
for _ in range(1_000_000):
    # OR use FastModel.Config.fast_make() to bypass Pydantic (~0.5µs)
    model = FastModel.Config.fast_make(learning_rate=0.01)
```

### Factory Pattern (Non-Caching)

Every call to `.make()` returns a **fresh instance**. This ensures that configurations
remain pure "recipes" and do not accidentally share mutable state between components.

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

## Advanced Features

- **Circular dependencies** in nested configs are detected and prevented.
- **Stub generation**: Use `nonfig-stubgen src/` for perfect IDE support in libraries.
- **Thread safe**: Concurrent config creation and usage is fully supported.

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

* * *

**nonfig** — Configuration should be effortless.