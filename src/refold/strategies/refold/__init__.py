"""ReFold on mini-swe-agent (agent.py), the adapter the paper's runs use. ReFold itself is
refold.core (engine, dedup, fold, parse, prompts)."""

from refold.registry import StrategySpec, register_strategy

register_strategy(StrategySpec(
    name="refold",
    agent="refold.strategies.refold.agent.ReFoldAgent",
    compact_agent="refold.strategies.compact.ReFoldCompactAgent",
    summary="repeats, and steps the agent reports STALE -> a stub or a fold note; `restore <step>` undoes either",
))
