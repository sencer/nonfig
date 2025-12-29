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
from nonfig import configurable, DEFAULT

@configurable
class Optimizer:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum

@configurable
class Model:
    def __init__(self, hidden_size: int = 128, optimizer: Optimizer = DEFAULT):
        self.hidden_size = hidden_size
        self.optimizer = optimizer

# Create config, serialize, and instantiate
config = Model.Config(hidden_size=256)
json_str = config.model_dump_json()

model = config.make()
print(model.optimizer.lr)  # 0.01
```

## Installation

```bash
pip install nonfig
# Optional YAML support:
pip install nonfig[yaml]
```

## Features

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

Features:
- Dot notation for nested configs (`optimizer.lr=0.01`)
- Automatic type coercion based on Config field types

### Config Loaders

Load configs from JSON, TOML, or YAML files:

```python
from nonfig import load_json, load_toml, load_yaml

# Load and instantiate
data = load_toml("config.toml")
config = Model.Config(**data)
model = config.make()
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

## Performance

| Pattern | Typical Latency* | Notes |
| :--- | :--- | :--- |
| **Raw Instantiation** | ~0.3µs | Baseline Python class |
| **Direct Call** | ~0.3µs | Zero overhead on decorated class |
| **`Config.make()`** | ~1.3µs | Cached factory call |
| **`Config.fast_make()`** | ~0.5µs | Bypasses Pydantic validation |
| **Full lifecycle** | ~4.3µs | `Config(...).make()` |

*Measured on Python 3.13, Linux x86_64.*

### High-Performance Usage

For hot loops where parameters are already trusted:

```python
# Bypasses Pydantic validation (~0.5µs)
for _ in range(1_000_000):
    model = Model.Config.fast_make(hidden_size=128)
```

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

- **Circular dependencies** in nested configs are detected at decoration time
- **Cycle detection** in `recursive_make` prevents infinite loops at runtime
- **Stub generation**: `nonfig-stubgen src/` for perfect IDE support
- **Thread safe**: Concurrent config creation fully supported

## Best Practices

### Use Keyword-Only Parameters

Place hyperparameters after `*` to make them keyword-only:

```python
# ✅ Recommended
@configurable
def train(data, *, epochs: Hyper[int] = 10, lr: Hyper[float] = 0.01):
    ...

# ⚠️ Avoid: positional hypers can cause argument conflicts
@configurable
def train(data, epochs: Hyper[int] = 10, lr: Hyper[float] = 0.01):
    ...
```

## API Reference

### Decorators & Types

| Export | Description |
|:-------|:------------|
| `configurable` | Decorator to make classes/functions configurable |
| `Hyper[T, ...]` | Mark a parameter as a hyperparameter with constraints |
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