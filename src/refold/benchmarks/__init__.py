"""Benchmarks. A new one is a class with the `Benchmark` fields, decorated with
`register_benchmark`, in a module imported by `_load_builtins`."""

from __future__ import annotations

from refold.benchmarks.base import Benchmark, ScoreReport

_REGISTRY: dict[str, type] = {}


def register_benchmark(name: str):
    def deco(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"benchmark {name!r} is already registered to {_REGISTRY[name].__name__}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _load_builtins() -> None:
    # importing registers them
    import refold.benchmarks.swebench
    import refold.benchmarks.terminalbench  # noqa: F401


def get_benchmark(name: str) -> Benchmark:
    _load_builtins()
    if name not in _REGISTRY:
        raise KeyError(f"unknown benchmark {name!r}. Available: {', '.join(available())}")
    return _REGISTRY[name]()


def available() -> list[str]:
    _load_builtins()
    return sorted(_REGISTRY)


__all__ = ["Benchmark", "ScoreReport", "available", "get_benchmark", "register_benchmark"]
