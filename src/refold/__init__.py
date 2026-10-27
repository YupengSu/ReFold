"""refold -- ReFold, training-free reversible inter-turn context folding for long-horizon
agents, on mini-swe-agent, with the base and compact strategies it is compared with.

    refold.core        ReFold itself: a rendering layer over an append-only history, with
                       no agent framework underneath
    refold.strategies  base, compact and ReFold on mini-swe-agent, the harness of the
                       experiments
    refold.cli         refold run / score / report / dashboard

Extension points:
    a strategy   subclass RenderingAgent, override render(), register_strategy(...)
    a benchmark  refold.benchmarks.register_benchmark
    a model      --model / --api-base (litellm), or a Model class via --model-class
"""

__all__ = ["RenderingAgent", "StrategySpec", "available", "get_strategy", "register_strategy"]


def __getattr__(name: str):
    """Imported on first use, so that `refold.core` loads without mini-swe-agent."""
    if name == "RenderingAgent":
        from refold.agent import RenderingAgent

        return RenderingAgent
    if name in ("StrategySpec", "available", "get_strategy", "register_strategy"):
        from refold import registry

        return getattr(registry, name)
    raise AttributeError(f"module 'refold' has no attribute {name!r}")
