"""Name -> strategy. A strategy is an agent class plus the few facts the launcher needs.

Adding one is a new module under `refold/strategies/` that defines its agent and calls
`register_strategy`, plus one import line in `refold/strategies/__init__.py`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class StrategySpec:
    name: str
    agent: str
    """Dotted path of the agent class. A path, not the class, so that listing strategies
    does not import every strategy's dependencies."""
    summary: str
    """One line: what decides a removal, and what is left in its place ."""
    compact_agent: str | None = None
    """Agent class for `--compact`: the strategy beneath compaction (compact + ReFold)."""


_REGISTRY: dict[str, StrategySpec] = {}


def register_strategy(spec: StrategySpec) -> StrategySpec:
    if spec.name in _REGISTRY and _REGISTRY[spec.name] != spec:
        raise ValueError(f"strategy {spec.name!r} is already registered to {_REGISTRY[spec.name].agent}")
    _REGISTRY[spec.name] = spec
    return spec


def get_strategy(name: str) -> StrategySpec:
    import refold.strategies  # noqa: F401  -- importing registers the built-ins

    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}. Available: {', '.join(available())}") from None


def available() -> list[str]:
    import refold.strategies  # noqa: F401

    return sorted(_REGISTRY)


def import_class(dotted: str) -> type:
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)
