"""Everything refold relies on from mini-swe-agent, checked in one place.

The dependency is a pinned release, not a vendored copy, which is only safe while the
surface is small and checked. Two names are private and pinned harder, by signature:
compaction calls `LitellmModel._query` and `_calculate_cost`, because every reported
compaction number came from that call.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "refold"


def test_the_symbols_we_use_are_importable():
    from minisweagent.agents.default import AgentConfig, DefaultAgent  # noqa: F401
    from minisweagent.config import get_config_from_spec  # noqa: F401
    from minisweagent.environments.docker import DockerEnvironment  # noqa: F401
    from minisweagent.exceptions import FormatError, LimitsExceeded  # noqa: F401
    from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig  # noqa: F401
    from minisweagent.run.benchmarks.swebench import (  # noqa: F401
        ProgressTrackingAgent,
        filter_instances,
        get_sb_environment,
        process_instance,
    )
    from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager  # noqa: F401
    from minisweagent.utils.serialize import UNSET, recursive_merge  # noqa: F401


def test_the_upstream_loop_is_still_append_only():
    """`base` is upstream's loop; if upstream starts removing messages, base changes."""
    from minisweagent.agents.default import DefaultAgent

    src = inspect.getsource(DefaultAgent)
    assert "self.messages.extend" in src
    assert not any(r in src for r in ("self.messages.pop", "self.messages.remove", "del self.messages"))


def test_process_instance_still_builds_the_patchable_agent_class():
    """The SWE-bench runner swaps the agent class by assigning ProgressTrackingAgent."""
    import minisweagent.run.benchmarks.swebench as sw

    src = inspect.getsource(sw.process_instance)
    assert "ProgressTrackingAgent(" in src and "get_sb_environment(" in src


@pytest.mark.parametrize("name,params,kwargs", [("_query", ["messages"], True), ("_calculate_cost", ["response"], False)])
def test_the_private_methods_compaction_needs_keep_their_shape(name, params, kwargs):
    from minisweagent.models.litellm_model import LitellmModel

    sig = inspect.signature(getattr(LitellmModel, name))
    assert all(p in sig.parameters for p in params)
    if kwargs:  # max_tokens rides **kwargs; without it every summary is uncapped
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_only_the_summariser_calls_a_private_upstream_method():
    offenders = []
    for f in SRC.rglob("*.py"):
        for fn in ast.walk(ast.parse(f.read_text())):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("_query", "_calculate_cost", "_parse_actions",
                                               "_prepare_messages_for_api")
                        and fn.name != "_summarise"):
                    offenders.append(f"{f.relative_to(SRC)}:{node.lineno} in {fn.name}()")
    assert not offenders
