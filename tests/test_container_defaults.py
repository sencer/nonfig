from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from nonfig import DEFAULT, configurable


@configurable
@dataclass
class Item:
  name: str = "item"


@configurable
@dataclass
class ContainerConfig:
  items_tuple: tuple[Item, ...] = DEFAULT
  items_sequence: Sequence[Item] = DEFAULT
  items_deque: deque[Item] = DEFAULT
  items_mapping: Mapping[str, Item] = DEFAULT


def test_tuple_default_type() -> None:
  config = ContainerConfig.Config()
  container = config.make()
  assert isinstance(container.items_tuple, tuple)
  assert container.items_tuple == ()


def test_sequence_default_type() -> None:
  config = ContainerConfig.Config()
  container = config.make()
  # Abstract Sequence defaults to list
  assert isinstance(container.items_sequence, list)
  assert container.items_sequence == []


def test_deque_default_type() -> None:
  config = ContainerConfig.Config()
  container = config.make()

  # Ideally this should be a deque, but currently it likely defaults to list
  # We will assert the CURRENT behavior (list) first, then fix it
  assert isinstance(container.items_deque, deque)
  assert container.items_deque == deque()


def test_mapping_default_type() -> None:
  config = ContainerConfig.Config()
  container = config.make()
  # Abstract Mapping defaults to dict
  assert isinstance(container.items_mapping, dict)
  assert container.items_mapping == {}
