# nonfig

![CI](https://github.com/sencer/nonfig/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/sencer/nonfig/branch/master/graph/badge.svg)](https://app.codecov.io/github/sencer/nonfig)

**Automatic Pydantic config generation from class and function signatures.**

`nonfig` turns any class or function into a configurable, serializable, and validated
component. By generating Pydantic models directly from your code's signatures, it
eliminates boilerplate while enforcing type safety and validation.

## Quick Example

```python
from nonfig import configurable, DEFAULT, Hyper, Ge

@configurable
class Optimizer:
  def __init__(self, lr: Hyper[float, Ge(0)] = 0.01, momentum: float = 0.9):
    self.lr = lr
    self.momentum = momentum

@configurable
class Model:
  def __init__(
    self,
    hidden_size: int = 128,
    optimizer: Optimizer = DEFAULT,  # Automatically transforms to Optimizer.Config
  ):
    self.hidden_size = hidden_size
    self.optimizer = optimizer

# Instantiate via Config
config = Model.Config(hidden_size=256, optimizer={"lr": 0.001})
model = config.make()  # Returns a Model instance with a real Optimizer
```

## Installation

```bash
pip install nonfig
# Optional YAML support:
pip install nonfig[yaml]
```

## Core Concepts

### Configuring Classes (Greedy)

When you apply `@configurable` to a class, `nonfig` captures **all** parameters from
your `__init__` method (or dataclass fields) and generates a `.Config` class.

```python
@configurable
class Model:
  def __init__(self, layers: int = 3, dropout: float = 0.1):
    self.layers = layers
    self.dropout = dropout

# Direct instantiation still works as usual
m = Model(layers=5)

# Config provides validation and serialization
config = Model.Config(layers=10)
m = config.make()  # Returns Model instance
```

### Configuring Functions (Selective)

For functions, `@configurable` is **selective**. It only extracts parameters marked
with `Hyper[T]`, `DEFAULT`, or a Config object. Other parameters are treated as
"runtime arguments" that must be passed when calling the configured function.

```python
from nonfig import configurable, Hyper, Gt

@configurable
def train(
  dataset,  # Runtime argument (not in Config)
  *,
  epochs: Hyper[int, Gt(0)] = 10,  # Hyperparameter
  lr: Hyper[float] = 0.01          # Hyperparameter
):
  print(f"Training for {epochs} epochs with lr={lr}")

# Create a configured version of the function
trainer = train.Config(epochs=20).make()

# Call it with runtime arguments
trainer(my_dataset)  # Uses epochs=20, lr=0.01
```

### Nested Configuration & `DEFAULT`

The power of `nonfig` lies in its ability to compose configurations. Any
`@configurable` class used as a type hint is automatically transformed into a nested
configuration. Use `DEFAULT` to automatically instantiate the nested config with its
own defaults.

```python
@configurable
class Trainer:
  def __init__(self, optimizer: Optimizer = DEFAULT):
    self.optimizer = optimizer

# 'optimizer' can be passed as a dict, an Optimizer.Config, or an Optimizer instance
config = Trainer.Config(optimizer={"lr": 0.0001})
trainer = config.make()
assert isinstance(trainer.optimizer, Optimizer)
```

### Validation & Constraints

Use `Hyper[T, ...]` to attach Pydantic-style constraints to your parameters. These
are enforced whenever a `.Config` is created.

```python
from nonfig import configurable, Hyper, Ge, Le, Gt, MinLen, Pattern

@configurable
class Network:
  def __init__(
    self,
    lr: Hyper[float, Gt(0)] = 0.01,
    dropout: Hyper[float, Ge(0), Le(1)] = 0.5,
    name: Hyper[str, MinLen(3), Pattern(r"^[A-Z]")] = "Net"
  ): ...
```

**Available constraints:** `Ge` (>=), `Gt` (>), `Le` (<=), `Lt` (<), `MinLen`, `MaxLen`,
`MultipleOf`, `Pattern`.

---

## Features

### Inheritance & Smart Propagation

`nonfig` supports automatic inheritance. If a subclass of a `@configurable` class
includes `**kwargs` in its `__init__`, it automatically inherits all configurable
parameters from its parents.

```python
@configurable
class Base:
  def __init__(self, x: int = 1): self.x = x

class Sub(Base):
  def __init__(self, y: int = 2, **kwargs):
    super().__init__(**kwargs)
    self.y = y

# Sub.Config now has both 'x' and 'y'
config = Sub.Config(x=10, y=20)
```

### External Components (`wrap_external`)

Make third-party classes (e.g., from PyTorch or Scikit-learn) configurable without
modifying their code. `wrap_external` is always **greedy**, capturing every
parameter in the signature.

```python
from torch.optim import Adam
from nonfig import wrap_external

AdamConfig = wrap_external(Adam, overrides={"lr": Hyper[float, Gt(0)]})

@configurable
class Experiment:
  def __init__(self, opt: AdamConfig = DEFAULT):
    self.opt = opt
```

### Function Type Proxies (`.Type`)

When a `@configurable` function is used as a nested dependency, use `.Type` for
the type hint to enable proper configuration nesting and IDE support.

```python
@configurable
def preprocessor(data): ...

@configurable
class Pipeline:
  def __init__(self, prep: preprocessor.Type = DEFAULT):
    self.prep = prep
```

### Leaf Markers

Use `Leaf[T]` to disable "Config magic" for a specific parameter. This forces the
field to accept **only** raw instances of the class, preventing `nonfig` from
transforming it into a nested configuration.

```python
from nonfig import Leaf

@configurable
class Processor:
  def __init__(self, model: Leaf[MyModel]):
    # 'model' must be a MyModel instance.
    # Passing a dict or MyModel.Config will raise a ValidationError.
    self.model = model
```

**Leaf vs. DEFAULT:**
- **`DEFAULT`** is for **Composition**: It says "This is a sub-component; please
  auto-configure it using its own defaults."
- **`Leaf[T]`** is for **Inversion of Control**: It says "This is an external
  dependency that I (the user) will provide as a pre-instantiated object."

You should **not** use `Leaf[T] = DEFAULT`. They represent opposite intents: `Leaf`
blocks the configuration transformation that `DEFAULT` specifically requests.

### JAX, Flax & jaxtyping

`nonfig` is designed for machine learning research:
- **`jaxtyping`**: Preserves dimension metadata and works with `@jaxtyped`.
- **Flax**: Supports `nn.Module` classes natively.
- **Order**: Always place `@configurable` as the **outermost** decorator.

---

## Tooling & CLI

### CLI Overrides (`run_cli`)

Easily run your experiments with command-line overrides using dot-notation.

```python
from nonfig import run_cli

if __name__ == "__main__":
  # python train.py model.layers=5 optimizer.lr=0.001
  result = run_cli(train)
```

### IDE Support (`nonfig-stubgen`)

Because `nonfig` generates classes dynamically, IDEs might need help with
autocompletion. Generate `.pyi` stubs for your entire project:

```bash
nonfig-stubgen src/
```

### Config Loaders

Support for JSON, TOML, and YAML (requires `pyyaml`):

```python
from nonfig import load_yaml
config_data = load_yaml("config.yaml")
config = Model.Config(**config_data)
```

---

## Performance

`nonfig` is optimized for high-performance ML loops. `Config.make()` uses a cached
factory that is 70-80% faster than a standard Pydantic `model_validate`.

| Pattern | Latency | Note |
| :--- | :--- | :--- |
| Raw `Class()` | ~0.15µs | Baseline |
| `Config.make()` (Class) | ~0.29µs | Reused config instance |
| `Config(...).make()` | ~1.85µs | Full lifecycle |

**Best Practice:** Call `.make()` once outside your training loop.

---

## Comparison

| Feature | nonfig | gin-config | hydra | tyro |
| :--- | :--- | :--- | :--- | :--- |
| **Philosophy** | Code-first | DI | YAML-first | CLI-first |
| **Validation** | Pydantic | None | Runtime | Parse-time |
| **Nesting** | Automatic | Manual | Manual | Automatic |
| **IDE Support** | Stubs (.pyi) | None | Partial | Full |

---

## License

MIT License
