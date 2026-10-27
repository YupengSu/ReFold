"""base -- nothing is removed.

mini-swe-agent's own loop: every turn is appended and all of it is sent. When the prompt
outgrows the window the server refuses the request and the task ends.
"""

from refold.registry import StrategySpec, register_strategy

register_strategy(StrategySpec(
    name="base",
    agent="refold.agent.RenderingAgent",
    summary="nothing is removed",
))
